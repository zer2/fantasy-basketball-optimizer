"""Read-only reference endpoints: sport config and available historical seasons."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

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
