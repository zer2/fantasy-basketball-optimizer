"""Read-only reference endpoints: sport config, available historical seasons, and
proxied player headshot images."""

from __future__ import annotations

import os
from pathlib import Path

import requests
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from backend.parameters import load_all_params
from backend.api.errors import fail

router = APIRouter()


@router.get('/config/{sport}')
def get_config_route(sport: str):
    all_params = load_all_params()
    if sport not in all_params:
        raise HTTPException(status_code=400, detail=f'Unknown sport: {sport!r}')

    p = all_params[sport]

    # All selectable categories = ratio stat names + counting stat names
    ratio_names = list(p.get('ratio-statistics', {}).keys())
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
        from backend.data_retrieval import get_available_seasons
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
    from backend.infra.snowflake_connection import peek

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
    if _HEADSHOT_DISK_CACHE_DIR is None:
        return
    _HEADSHOT_DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (_HEADSHOT_DISK_CACHE_DIR / f'{nba_player_id}.png').write_bytes(image_bytes)


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
        except requests.RequestException:
            raise fail(502, 'Headshot fetch failed.')
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
