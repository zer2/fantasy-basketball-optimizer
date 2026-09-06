"""Session lifecycle endpoints: create, patch, g-scores, delete, cache clear.

The routes here own the HTTP concerns — parse the request into plain dicts, resolve inputs
(raising 4xx), and shape responses — then hand the work to services.session_management.
"""

from __future__ import annotations

import logging

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, status, Response

from backend.infra.auth import current_user_key_optional
from backend.infra.rate_limit import enforce_rate_limit, BUILD_POLICY, REBUILD_POLICY
from backend.parameters import load_all_params
from backend.api.helpers import fail, require_session, resolve_platform_config
from backend.state.session import Session, delete_session
from backend.services.session_management import build_session, apply_patch
from backend.services.build_agent import clear_v0_cache, derive_effective_objective, InsufficientPlayerPoolError
from backend.services.projection_parsing import parse_projection_upload
from backend.state.upload_store import get_upload
from backend.api.schemas import (
    SessionRequest, SessionResponse, PlayerGScore, PlayerRegistryEntry,
    PatchRequest, PatchResponse, GScoresResponse,
)

router = APIRouter()


# ── Request/response mapping helpers (transport-tier: request DTO <-> plain dicts) ──────

def _build_current_settings(req: SessionRequest, all_params: dict) -> dict:
    """Flatten a SessionRequest into the flat current_settings dict."""
    sport      = req.league.sport
    p          = req.model_settings
    categories = req.league.categories or all_params[sport]['default-categories']
    n          = req.league.n_drafters

    return {
        'sport':            sport,
        'is_auction':       req.is_auction,
        'n_drafters':       n,
        'n_picks':          req.league.n_picks,
        'scoring_format':   req.league.scoring_format,
        'most_categories_weight': req.league.most_categories_weight,
        'tiebreaker_category':    req.league.tiebreaker_category,
        'categories':       categories,
        'slot_counts':      req.slot_counts,
        'injured_players':  req.injured_players,
        'team_names':       [f'Drafter {i + 1}' for i in range(n)],
        # model parameters
        'pick_pool_size':   p.pick_pool_size,
        'beth':             p.beth,
        'upsilon':          p.upsilon,
        'psi':              p.psi,
        'chi':              p.chi,
        'aleph':            p.aleph,
        'lambda_c':         p.lambda_c,
        'lambda_p':         p.lambda_p,
        'opponent_model_confidence': p.opponent_model_confidence,
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
    """Assemble the current_settings patch from the non-None pieces of a PatchRequest."""
    patch: dict = {}
    if req.is_auction is not None:
        patch['is_auction'] = req.is_auction
    if req.model_settings is not None:
        patch.update(req.model_settings.model_dump())
    if req.league is not None:
        for key, val in req.league.model_dump().items():
            if val is not None:
                patch[key] = val
        # Every other field here uses null for "unchanged", but the tiebreaker's null IS the value
        # — it means no category breaks ties. Without this, clearing the selector sent a null that
        # was read as an omission and the session kept drafting to the old tiebreaker. Keyed off
        # what the client actually sent, so omitting the field still means "leave it alone".
        if 'tiebreaker_category' in req.league.model_fields_set:
            patch['tiebreaker_category'] = req.league.tiebreaker_category
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


def _require_upload(data_id: str) -> dict:
    """The stored upload for data_id, or a 404 when it is missing or TTL-expired."""
    entry = get_upload(data_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f'data_id {data_id!r} not found or expired.')
    return entry


def _resolve_csv(custom_data_ids: Optional[list[str]]) -> Optional[bytes]:
    """Return csv_bytes for the first custom data_id ('csv' source; format auto-detected later)."""
    for data_id in custom_data_ids or []:
        return _require_upload(data_id)['bytes']
    return None


def _resolve_uploaded_dfs(custom_data_ids: Optional[list[str]], sport_params: dict) -> dict:
    """Return {data_id: DataFrame} for all custom data_ids ('projections' source). The data_id
    is the upload's identity throughout the blend — it keys the blend weights too."""
    result = {}
    for data_id in custom_data_ids or []:
        result[data_id] = parse_projection_upload(_require_upload(data_id)['bytes'], sport_params)
    return result


def _serialize_g_scores(session) -> list[PlayerGScore]:
    """Serialize the session's G-scores DataFrame into a list of PlayerGScore objects."""
    categories, _ = derive_effective_objective(session)
    g_scores_df = session.agent.info['G-scores']
    return [
        PlayerGScore(
            player_id=int(player_id),
            total=round(float(row['Total']), 2),
            values=[round(float(row[cat]), 2) for cat in categories],
        )
        for player_id, row in g_scores_df.iterrows()
    ]


def _serialize_player_registry(session) -> list[PlayerRegistryEntry]:
    """The session's player identities, for the frontend registry (see PlayerRegistryEntry)."""
    return [
        PlayerRegistryEntry(
            player_id    = identity.player_id,
            name         = identity.name,
            last_name    = identity.last_name,
            positions    = identity.positions,
            has_headshot = identity.has_headshot,
        )
        for identity in session.player_registry.values()
    ]


# ── Routes ──────────────────────────────────────────────────────────────────────────────

@router.post('/sessions', response_model=SessionResponse, status_code=status.HTTP_201_CREATED,
             dependencies=[Depends(enforce_rate_limit(BUILD_POLICY))])
def create_session_route(req: SessionRequest, user_key: Optional[str] = Depends(current_user_key_optional)):
    all_params = load_all_params()
    # The full incoming request, so a session's behavior is always attributable to its exact
    # inputs (settings arrive from persisted client prefs, which are easy to misremember).
    logging.getLogger('fbbo').info('create_session request: %s', req.model_dump_json())
    if req.league.sport not in all_params:
        raise HTTPException(status_code=400, detail=f'Unknown sport: {req.league.sport!r}')

    sport_params = all_params[req.league.sport]
    source_type = req.data_source.type
    if source_type == 'csv':
        csv_bytes = _resolve_csv(req.data_source.custom_data_ids)
        uploaded_dfs = None
    elif source_type == 'projections':
        csv_bytes = None
        uploaded_dfs = _resolve_uploaded_dfs(req.data_source.custom_data_ids, sport_params)
    else:
        csv_bytes, uploaded_dfs = None, None

    # Resolve any live-platform connection up front so a bad league fails before the pipeline.
    platform_config = resolve_platform_config(req.platform, req.platform_config, user_key)
    current_settings = _build_current_settings(req, all_params)

    try:
        session = build_session(
            current_settings  = current_settings,
            platform_config = platform_config,
            csv_bytes       = csv_bytes,
            uploaded_dfs    = uploaded_dfs,
        )
    except InsufficientPlayerPoolError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        raise fail(500, 'Failed to build projections for this session.')

    return SessionResponse(
        session_id=session.id,
        n_players_loaded=len(session.v0_clean),
        categories=derive_effective_objective(session)[0],
        players=_serialize_player_registry(session),
        g_scores=_serialize_g_scores(session),
        expires_at=(datetime.now(timezone.utc) + timedelta(seconds=4 * 3600)).strftime('%Y-%m-%dT%H:%M:%SZ'),
    )


@router.patch('/sessions/{session_id}', response_model=PatchResponse,
              dependencies=[Depends(enforce_rate_limit(REBUILD_POLICY))])
def patch_session_route(req: PatchRequest, session: Session = Depends(require_session),
                        user_key: Optional[str] = Depends(current_user_key_optional)):
    # Full request logging, matching the session-create log (see create_session_route).
    logging.getLogger('fbbo').info('patch request: %s', req.model_dump_json())

    csv_bytes: Optional[bytes] = None
    uploaded_dfs: Optional[dict] = None
    if req.data_source is not None and req.data_source.custom_data_ids is not None:
        sport_params = load_all_params()[session.current_settings['sport']]
        if req.data_source.type == 'csv':
            csv_bytes = _resolve_csv(req.data_source.custom_data_ids)
        elif req.data_source.type == 'projections':
            uploaded_dfs = _resolve_uploaded_dfs(req.data_source.custom_data_ids, sport_params)

    # Connecting/switching a live platform resolves up front so a bad league fails fast.
    platform_config = (
        resolve_platform_config(req.platform, req.platform_config, user_key)
        if req.platform is not None else None
    )

    # A PATCH rebuilds the pipeline (and the agent) in place; hold the per-session lock so it cannot
    # overlap an in-flight evaluate on the same session, which would read a half-rebuilt agent and 500.
    try:
        with session.lock:
            apply_patch(
                session,
                patch           = _build_patch(req),
                from_step       = req.from_step,
                platform_config = platform_config,
                csv_bytes       = csv_bytes,
                uploaded_dfs    = uploaded_dfs,
            )
    except InsufficientPlayerPoolError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        # A patch that leaves the format and its objective dial inconsistent (see
        # normalize_objective_settings) — a malformed request, not a server fault.
        raise HTTPException(status_code=400, detail=str(exc))
    return PatchResponse(ok=True, steps_rerun=list(range(req.from_step, 6)))


@router.get('/sessions/{session_id}/g-scores', response_model=GScoresResponse)
def get_g_scores_route(session: Session = Depends(require_session)):
    # Reads agent/info state that a concurrent PATCH rebuilds in place — hold the lock so it never
    # observes a half-rebuilt pipeline.
    with session.lock:
        return GScoresResponse(
            players=_serialize_player_registry(session),
            g_scores=_serialize_g_scores(session),
        )


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
