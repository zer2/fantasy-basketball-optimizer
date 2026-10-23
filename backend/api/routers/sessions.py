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
from backend.services.build_agent import clear_v0_cache, clear_agent_cache, derive_effective_objective, InsufficientPlayerPoolError
from backend.services.projection_parsing import parse_projection_upload
from backend.state.upload_store import get_upload
from backend.api.schemas import (
    SessionRequest, SessionResponse, PlayerGScore, PlayerRegistryEntry,
    PatchRequest, PatchResponse, GScoresResponse,
)

router = APIRouter()


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
        uploaded_dfs = _resolve_uploaded_dfs(req.data_source.custom_data_ids, sport_params, req.league.points_config)
    else:
        csv_bytes, uploaded_dfs = None, None

    # Resolve any live-platform connection up front so a bad league fails before the pipeline.
    platform_config = resolve_platform_config(req.platform, req.platform_config, user_key)
    current_settings = _build_current_settings(req)

    try:
        session = build_session(
            current_settings  = current_settings,
            platform_config = platform_config,
            csv_bytes       = csv_bytes,
            uploaded_dfs    = uploaded_dfs,
        )
    except InsufficientPlayerPoolError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        # A settings combination the session cannot be built with (normalize_objective_settings,
        # validate_slot_structure) -- the request's fault, said in its own words.
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        raise fail(500, 'Failed to build projections for this session.')

    is_points = session.current_settings.get('scoring_system') == 'points'
    g_scores = [] if is_points else _serialize_g_scores(session)
    categories = [] if is_points else derive_effective_objective(session)[0]

    return SessionResponse(
        session_id=session.id,
        n_players_loaded=len(session.v0_clean),
        scoring_system=session.current_settings.get('scoring_system', 'categories'),
        categories=categories,
        players=_serialize_player_registry(session),
        g_scores=g_scores,
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
            uploaded_dfs = _resolve_uploaded_dfs(req.data_source.custom_data_ids, sport_params,
                req.league.points_config if req.league and req.league.points_config else session.current_settings.get('points_config'))

    rebuild_from_step = req.from_step
    def upload_totals(config):
        config = config.model_dump() if hasattr(config, 'model_dump') else (config or {})
        return config.get('points_input_mode') == 'raw_stats' and config.get('projection_basis') in ('season_total', 'rest_of_season_total')
    if req.league and req.league.points_config is not None:
        if upload_totals(req.league.points_config) != upload_totals(session.current_settings.get('points_config')):
            source = req.data_source.type if req.data_source else session.current_settings['data_source_type']
            ids = req.data_source.custom_data_ids if req.data_source and req.data_source.custom_data_ids is not None else session.current_settings.get('custom_data_ids')
            if source == 'csv':
                csv_bytes = _resolve_csv(ids)
                rebuild_from_step = 1
            elif source == 'projections' and ids:
                uploaded_dfs = _resolve_uploaded_dfs(ids, load_all_params()[session.current_settings['sport']], req.league.points_config)
                rebuild_from_step = 1

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
                from_step       = rebuild_from_step,
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
    return PatchResponse(ok=True, steps_rerun=list(range(rebuild_from_step, 6)),
                         players=_serialize_player_registry(session) if session.current_settings.get('scoring_system') == 'points' else None)


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
    clear_agent_cache()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete('/sessions/{session_id}', status_code=status.HTTP_204_NO_CONTENT)
def delete_session_route(session_id: str):
    found = delete_session(session_id)
    if not found:
        raise HTTPException(status_code=404, detail='Session not found or expired.')
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ── Request/response mapping helpers (transport-tier: request DTO <-> plain dicts) ──────

def _build_current_settings(req: SessionRequest) -> dict:
    """Flatten a SessionRequest into the flat current_settings dict."""
    sport      = req.league.sport
    p          = req.model_settings
    # Non-emptiness is guaranteed by the LeagueSettings validator; no silent default here.
    categories = req.league.categories
    n          = req.league.n_drafters

    return {
        'sport':            sport,
        'is_auction':       req.is_auction,
        'is_season_mode':   req.is_season_mode,
        'n_drafters':       n,
        'n_picks':          req.league.n_picks,
        'scoring_format':   req.league.scoring_format,
        'scoring_system':   req.league.scoring_system,
        'points_config':    req.league.points_config.model_dump(exclude_none=True) if req.league.points_config else None,
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
        'replacement_threshold': p.replacement_threshold,
        'cash_share_softness': p.cash_share_softness,
        'allow_force_weighting': p.allow_force_weighting,
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
    if req.is_season_mode is not None:
        patch['is_season_mode'] = req.is_season_mode
    if req.model_settings is not None:
        patch.update(req.model_settings.model_dump())
    if req.league is not None:
        for key, val in req.league.model_dump(exclude_none=True).items():
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


def _resolve_csv(custom_data_ids: Optional[list[str]]) -> Optional[bytes]:
    """Return csv_bytes for the first custom data_id ('csv' source; format auto-detected later)."""
    for data_id in custom_data_ids or []:
        return _require_upload(data_id)['bytes']
    return None


def _resolve_uploaded_dfs(custom_data_ids: Optional[list[str]], sport_params: dict, points_config=None) -> dict:
    """Return {data_id: DataFrame} for all custom data_ids ('projections' source). The data_id
    is the upload's identity throughout the blend — it keys the blend weights too."""
    result = {}
    config = points_config.model_dump() if hasattr(points_config, 'model_dump') else (points_config or {})
    basis = config.get('projection_basis') if config.get('points_input_mode') == 'raw_stats' else None
    for data_id in custom_data_ids or []:
        result[data_id] = parse_projection_upload(_require_upload(data_id)['bytes'], sport_params, raw_stats_basis=basis)
    return result


def _require_upload(data_id: str) -> dict:
    """The stored upload for data_id, or a 404 when it is missing or TTL-expired."""
    entry = get_upload(data_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f'data_id {data_id!r} not found or expired.')
    return entry


def _serialize_g_scores(session) -> list[PlayerGScore]:
    """Serialize the session's G-scores DataFrame into a list of PlayerGScore objects."""
    if session.current_settings.get('scoring_system') == 'points':
        return []
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


def _points_per_game(session, player_id):
    import math
    value = getattr(session.agent, 'per_game_fp', {}).get(player_id)
    return float(value) if value is not None and math.isfinite(value) else None


def _serialize_player_registry(session) -> list[PlayerRegistryEntry]:
    """The session's player identities, for the frontend registry (see PlayerRegistryEntry)."""
    return [
        PlayerRegistryEntry(
            player_id    = identity.player_id,
            name         = identity.name,
            last_name    = identity.last_name,
            positions    = identity.positions,
            has_headshot = identity.has_headshot,
            points_per_game = _points_per_game(session, identity.player_id),
        )
        for identity in session.player_registry.values()
    ]
