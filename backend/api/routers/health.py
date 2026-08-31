"""Ops diagnostics: what the running process actually sees of its deployment.

These endpoints exist for deployment debugging rather than for the app — settings like a
disk-cache mount or a proxy chain are easy to misread from outside, so each route reports
what the process itself resolved instead of leaving it to be inferred from YAML or consoles.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi import APIRouter, Request

from backend.infra.headshot_cache import describe_cache
from backend.infra.rate_limit import (
    identify_rate_limit_client, get_rate_limiter, rate_limits_enabled, request_is_local,
)

router = APIRouter()


@router.get('/health/disk-cache')
def get_disk_cache_health_route():
    """What the running container actually sees of its disk cache.

    Deployment settings are easy to misread from outside — a mount path and the env var
    naming it can disagree, a revision's YAML can describe a revision that is not the one
    serving, and a bucket listing says nothing about where it is mounted. This reports the
    paths the process resolved and what is behind them, so the question is settled by the
    process rather than inferred. Paths and counts only; no secrets, no file contents."""
    def describe(path: Path | None) -> dict:
        if path is None:
            return {'configured': False}
        try:
            entries = sorted(child.name for child in path.iterdir())
        except OSError as exc:
            return {'configured': True, 'path': str(path), 'exists': path.exists(),
                    'readable': False, 'error': f'{type(exc).__name__}: {exc}'}
        return {
            'configured': True,
            'path':       str(path),
            'exists':     True,
            'readable':   True,
            'entries':    len(entries),
            'sample':     entries[:8],
        }

    disk_cache_dir = os.environ.get('DISK_CACHE_DIR')
    return {
        'DISK_CACHE_DIR_env':  disk_cache_dir,
        'disk_cache_root':     describe(Path(disk_cache_dir) if disk_cache_dir else None),
        # The headshot cache describes itself (headshot_dir + headshots_in_memory keys).
        **describe_cache(),
        # Checked explicitly because a mount can live somewhere other than the env var says.
        'other_mount_candidates': {
            candidate: describe(Path(candidate))
            for candidate in ('/cache', '/data-cache')
        },
    }


@router.get('/health/rate-limit')
def get_rate_limit_health_route(request: Request):
    """What the limiter makes of the caller it is looking at right now.

    The one genuinely uncertain input is X-Forwarded-For: how many proxies append to it decides
    which entry is the real client, and trusting the wrong one either lets a script rotate fake
    addresses past the limit or lumps every visitor into a single bucket. Rather than infer it,
    this echoes the chain as received alongside the address that was derived from it, so
    TRUSTED_PROXY_HOPS can be checked against reality from the deployment itself.

    Reports this caller's own usage only — never a list of who else has been here."""
    client_key = identify_rate_limit_client(request)
    return {
        'enabled':                rate_limits_enabled(),
        # True only for requests off the loopback interface with no proxy in front — local
        # development and the test suites. Should read false from anywhere on the internet.
        'exempt_as_local':        request_is_local(request),
        'x_forwarded_for':        request.headers.get('x-forwarded-for'),
        'direct_peer':            request.client.host if request.client else None,
        'trusted_proxy_hops':     int(os.environ.get('TRUSTED_PROXY_HOPS', '1')),
        # Identity only — the account hash is truncated and one-way, and an IP bucket names
        # the address the request already came from.
        'counted_as':             client_key,
        'signed_in':              client_key.startswith('user:'),
        'usage':                  get_rate_limiter().describe_client(client_key, time.monotonic()),
    }
