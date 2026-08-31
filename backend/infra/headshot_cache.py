"""Player headshots, proxied from the NBA CDN and cached in two tiers (memory + disk).

Lives in infra for the same reason snowflake_connection does: an external source fronted
by a process-lifetime cache. The api layer serves the bytes; this module owns fetching,
remembering, and describing them.

Public surface: get_headshot (bytes, or None for a confirmed no-image id),
HeadshotFetchError (the CDN could not be reached or errored — the message carries the
unwound cause chain), and describe_cache (for the disk-cache health endpoint).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import requests

_NBA_HEADSHOT_URL_TEMPLATE = 'https://cdn.nba.com/headshots/nba/latest/260x190/{nba_player_id}.png'

_logger = logging.getLogger('fbbo')


class HeadshotFetchError(Exception):
    """The NBA CDN could not be reached or answered with an error. The message is the
    diagnosis: for transport failures it is the unwound exception cause chain, since a DNS
    failure, a refused connection, a routing black hole and a TLS rejection are four
    different infrastructure problems that are indistinguishable once the error is
    swallowed."""


# nba_player_id -> PNG bytes, or None for a confirmed no-image id (the CDN 404s some
# historical players). ~600 active players x ~30KB ~= 20MB fully warm — fine in memory.
# Transient fetch errors are NOT cached, so a network blip doesn't permanently blank a face.
# The DISK layer (same DISK_CACHE_DIR convention as the Snowflake view cache) makes the
# warm set survive process restarts — dev auto-reloads and Cloud Run cold starts otherwise
# wipe it and the next render trickles in hundreds of CDN round-trips in arrival order.
_headshot_cache: dict[int, bytes | None] = {}
# Unlike the Snowflake view cache (data freshness concerns make persistence opt-in via
# DISK_CACHE_DIR), headshots are immutable public images — so without the env override
# the cache still persists, in a gitignored local directory. A dev server otherwise
# re-fetches hundreds of CDN images after every auto-reload.
_HEADSHOT_DISK_CACHE_DIR = (
    Path(os.environ['DISK_CACHE_DIR']) / 'headshots'
    if 'DISK_CACHE_DIR' in os.environ else Path('.cache') / 'headshots'
)


def _read_headshot_from_disk(nba_player_id: int) -> bytes | None:
    disk_path = _HEADSHOT_DISK_CACHE_DIR / f'{nba_player_id}.png'
    if not disk_path.exists():
        return None
    return disk_path.read_bytes()


def _write_headshot_to_disk(nba_player_id: int, image_bytes: bytes) -> None:
    """Best effort. In production this directory is a mounted bucket, which can be read-only,
    out of quota, or briefly unavailable through gcsfuse — and none of that is a reason to
    deny the caller an image we are already holding. Caching is the optimisation; serving is
    the job. Failures are logged, since a cache that never persists turns every cold start
    into a full re-fetch and should not do so silently."""
    try:
        _HEADSHOT_DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (_HEADSHOT_DISK_CACHE_DIR / f'{nba_player_id}.png').write_bytes(image_bytes)
    except OSError as exc:
        _logger.warning(
            'Could not cache headshot %s under %s: %s',
            nba_player_id, _HEADSHOT_DISK_CACHE_DIR, exc)


def get_headshot(nba_player_id: int) -> bytes | None:
    """The player's headshot PNG, or None when the CDN has confirmed there is no image.

    Checks memory, then disk, then fetches from the CDN at most once per id: a 200 is
    cached in both tiers, a 404 is cached in memory as None (an uncached miss would make
    every rebuilt <img> for the player refetch it), and anything else raises
    HeadshotFetchError without being cached."""
    if nba_player_id not in _headshot_cache:
        disk_bytes = _read_headshot_from_disk(nba_player_id)
        if disk_bytes is not None:
            _headshot_cache[nba_player_id] = disk_bytes

    if nba_player_id not in _headshot_cache:
        try:
            cdn_response = requests.get(
                _NBA_HEADSHOT_URL_TEMPLATE.format(nba_player_id=nba_player_id), timeout=5)
        except requests.RequestException as exc:
            # Say WHY, in the log and in the error. Nested causes matter — requests wraps
            # urllib3, which wraps socket.gaierror — so the chain is unwound rather than
            # just the surface.
            causes, cause = [], exc
            while cause is not None and len(causes) < 4:
                causes.append(f'{type(cause).__name__}: {cause}')
                cause = cause.__cause__ or cause.__context__
            reason = ' <- '.join(causes)
            _logger.warning('Headshot fetch failed for %s: %s', nba_player_id, reason)
            raise HeadshotFetchError(reason)
        if cdn_response.status_code == 200:
            _headshot_cache[nba_player_id] = cdn_response.content
            _write_headshot_to_disk(nba_player_id, cdn_response.content)
        elif cdn_response.status_code == 404:
            _headshot_cache[nba_player_id] = None
        else:
            raise HeadshotFetchError(f'HTTP {cdn_response.status_code} from the NBA CDN')

    return _headshot_cache[nba_player_id]


def describe_cache() -> dict:
    """The cache as this process sees it, for the disk-cache health endpoint.

    Paths and counts only; no file contents. The module describes itself so the health
    route never has to reach into cache internals."""
    try:
        entries = sorted(child.name for child in _HEADSHOT_DISK_CACHE_DIR.iterdir())
        directory = {
            'configured': True,
            'path':       str(_HEADSHOT_DISK_CACHE_DIR),
            'exists':     True,
            'readable':   True,
            'entries':    len(entries),
            'sample':     entries[:8],
        }
    except OSError as exc:
        directory = {'configured': True, 'path': str(_HEADSHOT_DISK_CACHE_DIR),
                     'exists': _HEADSHOT_DISK_CACHE_DIR.exists(),
                     'readable': False, 'error': f'{type(exc).__name__}: {exc}'}
    return {'headshot_dir': directory, 'headshots_in_memory': len(_headshot_cache)}
