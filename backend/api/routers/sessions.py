"""Session lifecycle endpoints: create, patch, g-scores, delete, cache clear.

The routes here own the HTTP concerns — parse the request into plain dicts, resolve inputs
(raising 4xx), and shape responses — then hand the work to services.session_management.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, status, Response

from backend.infra.auth import current_user_key_optional
from backend.parameters import load_all_params
from backend.state.session import get_session, delete_session
from backend.services.session_management import build_session, apply_patch
from backend.services.build_agent import clear_v0_cache, parse_projection_csv, InsufficientPlayerPoolError
from backend.state.upload_store import get_upload
from backend.api.platform_helpers import resolve_platform_config
from backend.api.schemas import (
    SessionRequest, SessionResponse, PlayerGScore,
    PatchRequest, PatchResponse, GScoresResponse,
)
from backend.api.errors import fail

router = APIRouter()


# ── Request/response mapping helpers (transport-tier: request DTO <-> plain dicts) ──────

def _build_current_params(req: SessionRequest, all_params: dict) -> dict:
    """Flatten a SessionRequest into the flat current_params dict."""
    sport      = req.league.sport
    p          = req.parameters
    categories = req.league.categories or all_params[sport]['default-categories']
    n          = req.league.n_drafters

    return {
        'sport':            sport,
        'n_drafters':       n,
        'n_picks':          req.league.n_picks,
        'scoring_format':   req.league.scoring_format,
        'categories':       categories,
        'slot_counts':      req.slot_counts,
        'injured_players':  req.injured_players,
        'team_names':       [f'Drafter {i + 1}' for i in range(n)],
        # model parameters
        'omega':            p.omega,
        'gamma':            p.gamma,
        'beth':             p.beth,
        'upsilon':          p.upsilon,
        'psi':              p.psi,
        'chi':              p.chi,
        'aleph':            p.aleph,
        'reg_strength':     p.reg_strength,
        'n_iterations':     p.n_iterations,
        'streaming_noise':  p.streaming_noise,
        # auction
        'cash_per_team':    req.league.cash_per_team,
        # data source
        'data_source_type': req.data_source.type,
        'season':           req.data_source.season,
        'blend_weights':    req.data_source.blend_weights,
        'custom_data_ids':  req.data_source.custom_data_ids,
    }


def _build_patch(req: PatchRequest) -> dict:
    """Assemble the current_params patch from the non-None pieces of a PatchRequest."""
    patch: dict = {}
    if req.parameters is not None:
        patch.update(req.parameters.model_dump())
    if req.league is not None:
        for key, val in req.league.model_dump().items():
            if val is not None:
                patch[key] = val
    if req.slot_counts is not None:
        patch['slot_counts'] = req.slot_counts
    if req.injured_players is not None:
        patch['injured_players'] = req.injured_players
    if req.data_source is not None:
        patch['data_source_type'] = req.data_source.type
        if req.data_source.season is not None:
            patch['season'] = req.data_source.season
        if req.data_source.blend_weights is not None:
            patch['blend_weights'] = req.data_source.blend_weights
        if req.data_source.custom_data_ids is not None:
            patch['custom_data_ids'] = req.data_source.custom_data_ids
    return patch


def _resolve_csv(custom_data_ids: Optional[dict]) -> tuple[Optional[bytes], Optional[str]]:
    """Return (csv_bytes, file_type) for the first valid custom data_id ('csv' source)."""
    if not custom_data_ids:
        return None, None
    for _ft, did in custom_data_ids.items():
        if did is not None:
            entry = get_upload(did)
            if entry is None:
                raise HTTPException(status_code=404, detail=f'data_id {did!r} not found or expired.')
            return entry['bytes'], entry['file_type']
    return None, None


def _resolve_uploaded_dfs(custom_data_ids: Optional[dict], params: dict) -> dict:
    """Return {file_type: DataFrame} for all valid custom data_ids ('projections' source)."""
    if not custom_data_ids:
        return {}
    result = {}
    for file_type_key, did in custom_data_ids.items():
        if did is None:
            continue
        entry = get_upload(did)
        if entry is None:
            raise HTTPException(status_code=404, detail=f'data_id {did!r} not found or expired.')
        result[file_type_key.upper()] = parse_projection_csv(entry['bytes'], entry['file_type'], params)
    return result


def _serialize_g_scores(session) -> list[PlayerGScore]:
    """Serialize the session's G-scores DataFrame into a list of PlayerGScore objects."""
    categories = session.current_params['categories']
    g_scores_df = session.agent.info['G-scores']
    return [
        PlayerGScore(
            name=str(name),
            total=round(float(row['Total']), 2),
            values=[round(float(row[cat]), 2) for cat in categories],
        )
        for name, row in g_scores_df.iterrows()
    ]


# ── Routes ──────────────────────────────────────────────────────────────────────────────

@router.post('/sessions', response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
def create_session_route(req: SessionRequest, user_key: Optional[str] = Depends(current_user_key_optional)):
    all_params = load_all_params()
    if req.league.sport not in all_params:
        raise HTTPException(status_code=400, detail=f'Unknown sport: {req.league.sport!r}')

    params = all_params[req.league.sport]
    source_type = req.data_source.type
    if source_type == 'csv':
        csv_bytes, file_type_str = _resolve_csv(req.data_source.custom_data_ids)
        uploaded_dfs = None
    elif source_type == 'projections':
        csv_bytes, file_type_str = None, None
        uploaded_dfs = _resolve_uploaded_dfs(req.data_source.custom_data_ids, params)
    else:
        csv_bytes, file_type_str, uploaded_dfs = None, None, None

    # Resolve any live-platform connection up front so a bad league fails before the pipeline.
    platform_config = resolve_platform_config(req.platform, req.platform_config, user_key)
    current_params = _build_current_params(req, all_params)

    try:
        session = build_session(
            current_params  = current_params,
            platform_config = platform_config,
            csv_bytes       = csv_bytes,
            file_type       = file_type_str,
            uploaded_dfs    = uploaded_dfs,
        )
    except InsufficientPlayerPoolError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        raise fail(500, 'Failed to build projections for this session.')

    return SessionResponse(
        session_id=session.id,
        n_players_loaded=len(session.v0_clean),
        categories=list(session.current_params['categories']),
        g_scores=_serialize_g_scores(session),
        expires_at=(datetime.now(timezone.utc) + timedelta(seconds=4 * 3600)).strftime('%Y-%m-%dT%H:%M:%SZ'),
    )


@router.patch('/sessions/{session_id}', response_model=PatchResponse)
def patch_session_route(session_id: str, req: PatchRequest, user_key: Optional[str] = Depends(current_user_key_optional)):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail='Session not found or expired.')

    csv_bytes: Optional[bytes] = None
    file_type_str: Optional[str] = None
    uploaded_dfs: Optional[dict] = None
    if req.data_source is not None and req.data_source.custom_data_ids is not None:
        params = load_all_params()[session.current_params['sport']]
        if req.data_source.type == 'csv':
            csv_bytes, file_type_str = _resolve_csv(req.data_source.custom_data_ids)
        elif req.data_source.type == 'projections':
            uploaded_dfs = _resolve_uploaded_dfs(req.data_source.custom_data_ids, params)

    # Connecting/switching a live platform resolves up front so a bad league fails fast.
    platform_config = (
        resolve_platform_config(req.platform, req.platform_config, user_key)
        if req.platform is not None else None
    )

    try:
        apply_patch(
            session,
            patch           = _build_patch(req),
            from_step       = req.from_step,
            platform_config = platform_config,
            csv_bytes       = csv_bytes,
            file_type       = file_type_str,
            uploaded_dfs    = uploaded_dfs,
        )
    except InsufficientPlayerPoolError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return PatchResponse(ok=True, steps_rerun=list(range(req.from_step, 6)))


@router.get('/sessions/{session_id}/g-scores', response_model=GScoresResponse)
def get_g_scores_route(session_id: str):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail='Session not found or expired.')
    return GScoresResponse(g_scores=_serialize_g_scores(session))


@router.post('/cache/clear', status_code=status.HTTP_204_NO_CONTENT)
def clear_cache_route():
    clear_v0_cache()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete('/sessions/{session_id}', status_code=status.HTTP_204_NO_CONTENT)
def delete_session_route(session_id: str):
    found = delete_session(session_id)
    if not found:
        raise HTTPException(status_code=404, detail='Session not found or expired.')
    return Response(status_code=status.HTTP_204_NO_CONTENT)
