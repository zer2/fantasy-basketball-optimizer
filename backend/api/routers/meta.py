"""Read-only reference endpoints: sport config, available historical seasons, and
proxied player headshot images."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import requests
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from backend.parameters import load_all_params
from backend.infra.rate_limit import (
    identify_rate_limit_client, get_rate_limiter, rate_limits_enabled, request_is_local,
)
from backend.api.helpers import fail
from backend.data_retrieval import get_available_seasons
from backend.infra.snowflake_connection import peek

router = APIRouter()


@router.get('/config/{sport}')
def get_config_route(sport: str):
    all_params = load_all_params()
    if sport not in all_params:
        raise HTTPException(status_code=400, detail=f'Unknown sport: {sport!r}')

    p = all_params[sport]

    # All selectable categories = ratio stat names + counting stat names
    ratio_names = list(p['ratio-statistics'].keys())
    counting_names = p.get('counting-statistics', [])
    all_categories = ratio_names + [c for c in counting_names if c not in ratio_names]

    # Options (min/max/default for each parameter), excluding positions
    raw_options = p.get('options', {})
    options = {k: v for k, v in raw_options.items() if k != 'positions'}

    pos_struct = p.get('position_structure', {})
    position_names = {}
    for abbr, info in pos_struct.get('base', {}).items():
        position_names[abbr] = info.get('full_str', abbr)
    for abbr, info in pos_struct.get('flex', {}).items():
        position_names[abbr] = info.get('full_str', abbr)

    return {
        'default_categories': p.get('default-categories', []),
        'all_categories': all_categories,
        'short_category_names': p.get('short-category-names', {}),
        'options': options,
        'positions': raw_options.get('positions', {}),
        'position_structure': {
            'base_list': pos_struct.get('base_list', []),
            'flex_list': pos_struct.get('flex_list', []),
        },
        'position_names': position_names,
    }


@router.get('/seasons')
def get_seasons_route():
    try:
        return {'seasons': get_available_seasons()}
    except Exception:
        raise fail(500, 'Could not load available seasons.')


@router.get('/players/pool-ids')
def get_pool_player_ids_route(data_type: str, season: str | None = None):
    """Best-effort NBA id list for a data source's player pool, so the frontend can
    prefetch headshots in parallel with a session build (image serving is pure I/O and
    never competes with the CPU-bound pipeline).

    Reads ONLY frames already in the view cache — a cold cache returns an empty list
    instead of triggering a Snowflake load that would race the session build's own.
    The registry-driven sweep after the build covers whatever this misses, so empty is
    always safe here (and only here — this endpoint is explicitly an optimization)."""

    if data_type == 'historical':
        if season is None:
            raise HTTPException(status_code=400, detail='season is required for historical pools.')
        historical_view = peek('HISTORICAL_SEASONAL_AVERAGES_VIEW')
        if historical_view is None:
            return {'player_ids': []}
        season_rows = historical_view[historical_view['SEASON'].astype(str) == season]
        return {'player_ids': sorted(int(i) for i in season_rows['NBA_PLAYER_ID'].dropna().unique())}

    if data_type in ('projections', 'csv'):
        # These pools resolve players through the unified table, so its current players
        # are a superset — fine to warm (a few extra ids, all cacheable).
        unified_players = peek('UNIFIED_PLAYER_TABLE')
        if unified_players is None:
            return {'player_ids': []}
        return {'player_ids': sorted(int(i) for i in unified_players['NBA_PLAYER_ID'].dropna().unique())}

    raise HTTPException(status_code=400, detail=f'Unknown pool data_type: {data_type!r}')


_NBA_HEADSHOT_URL_TEMPLATE = 'https://cdn.nba.com/headshots/nba/latest/260x190/{nba_player_id}.png'

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
    if _HEADSHOT_DISK_CACHE_DIR is None:
        return None
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
    if _HEADSHOT_DISK_CACHE_DIR is None:
        return
    try:
        _HEADSHOT_DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (_HEADSHOT_DISK_CACHE_DIR / f'{nba_player_id}.png').write_bytes(image_bytes)
    except OSError as exc:
        logging.getLogger('fbbo').warning(
            'Could not cache headshot %s under %s: %s',
            nba_player_id, _HEADSHOT_DISK_CACHE_DIR, exc)


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
        'headshot_dir':        describe(_HEADSHOT_DISK_CACHE_DIR),
        'headshots_in_memory': len(_headshot_cache),
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


@router.get('/players/headshots/{nba_player_id}.png')
def get_player_headshot_route(nba_player_id: int):
    """The NBA CDN headshot for a player, proxied and cached server-side (memory + disk).
    Serving these same-origin keeps the app independent of the CDN's hotlink/bot filtering
    (which rejects some clients outright) and fetches each image from the NBA at most once."""
    if nba_player_id not in _headshot_cache:
        disk_bytes = _read_headshot_from_disk(nba_player_id)
        if disk_bytes is not None:
            _headshot_cache[nba_player_id] = disk_bytes

    if nba_player_id not in _headshot_cache:
        try:
            cdn_response = requests.get(
                _NBA_HEADSHOT_URL_TEMPLATE.format(nba_player_id=nba_player_id), timeout=5)
        except requests.RequestException as exc:
            # Say WHY, in the log and in the response. The exception type is the whole
            # diagnosis when a deployment cannot reach the CDN and every id 502s: a DNS
            # failure, a refused connection, a routing black hole and a TLS rejection are
            # four different infrastructure problems that are indistinguishable once the
            # error is swallowed. Nested causes matter too — requests wraps urllib3, which
            # wraps socket.gaierror — so the chain is unwound rather than just the surface.
            causes, cause = [], exc
            while cause is not None and len(causes) < 4:
                causes.append(f'{type(cause).__name__}: {cause}')
                cause = cause.__cause__ or cause.__context__
            reason = ' <- '.join(causes)
            logging.getLogger('fbbo').warning(
                'Headshot fetch failed for %s: %s', nba_player_id, reason)
            raise fail(502, f'Headshot fetch failed: {reason}')
        if cdn_response.status_code == 200:
            _headshot_cache[nba_player_id] = cdn_response.content
            _write_headshot_to_disk(nba_player_id, cdn_response.content)
        elif cdn_response.status_code == 404:
            _headshot_cache[nba_player_id] = None
        else:
            raise fail(502, f'Headshot fetch failed ({cdn_response.status_code}).')

    image_bytes = _headshot_cache[nba_player_id]
    if image_bytes is None:
        # Cacheable like the hit: a no-image id stays that way, and an uncached 404 would
        # make every rebuilt <img> for the player refetch it.
        raise HTTPException(status_code=404, detail='No headshot for this player.',
                            headers={'Cache-Control': 'public, max-age=604800'})
    return Response(
        content=image_bytes,
        media_type='image/png',
        # The bytes are effectively immutable per id (a player's photo changes at most
        # yearly): a week keeps the browser cache warm across sessions, so the preloader
        # only ever pays the fetch cost for genuinely new players.
        headers={'Cache-Control': 'public, max-age=604800'},
    )
