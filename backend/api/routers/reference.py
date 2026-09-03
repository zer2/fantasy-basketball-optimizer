"""Read-only reference endpoints: sport config, available historical seasons, and the
player-media pair (pool-id prefetch listing + proxied headshots).

The headshot cache itself lives in infra.headshot_cache; the route here only maps its
results onto HTTP. Ops diagnostics live in routers.health.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from backend.parameters import load_all_params
from backend.api.helpers import fail
from backend.data_retrieval import get_available_seasons
from backend.infra.headshot_cache import HeadshotFetchError, get_headshot
from backend.infra.snowflake_connection import peek

router = APIRouter()


@router.get('/config/{sport}')
def get_config_route(sport: str):
    all_params = load_all_params()
    if sport not in all_params:
        raise HTTPException(status_code=400, detail=f'Unknown sport: {sport!r}')

    sport_params = all_params[sport]

    # All selectable categories = ratio stat names + counting stat names
    ratio_names = list(sport_params['ratio-statistics'].keys())
    counting_names = sport_params['counting-statistics']
    all_categories = ratio_names + [c for c in counting_names if c not in ratio_names]

    # Options (min/max/default for each parameter), excluding positions
    raw_options = sport_params['options']
    options = {k: v for k, v in raw_options.items() if k != 'positions'}

    position_structure = sport_params['position_structure']
    position_names = {}
    for abbreviation, position_info in position_structure['base'].items():
        position_names[abbreviation] = position_info['full_str']
    for abbreviation, position_info in position_structure['flex'].items():
        position_names[abbreviation] = position_info['full_str']

    return {
        'default_categories': sport_params['default-categories'],
        'all_categories': all_categories,
        'short_category_names': sport_params['short-category-names'],
        'options': options,
        'positions': raw_options['positions'],
        'position_structure': {
            'base_list': position_structure['base_list'],
            'flex_list': position_structure['flex_list'],
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


@router.get('/players/headshots/{nba_player_id}.png')
def get_player_headshot_route(nba_player_id: int):
    """The NBA CDN headshot for a player, proxied and cached server-side (memory + disk).
    Serving these same-origin keeps the app independent of the CDN's hotlink/bot filtering
    (which rejects some clients outright) and fetches each image from the NBA at most once."""
    try:
        image_bytes = get_headshot(nba_player_id)
    except HeadshotFetchError as exc:
        raise fail(502, f'Headshot fetch failed: {exc}')
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
