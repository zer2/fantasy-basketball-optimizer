"""
FastAPI application — Fantasy Basketball Optimizer backend.
"""

from __future__ import annotations

import time
import uuid
import threading
import traceback
from datetime import datetime, timezone
from typing import Optional

import yaml
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles

from backend.session import create_session, get_session, delete_session
from backend.pipeline import run_pipeline, _parse_projection_csv, clear_v0_cache
from backend.models import (
    UploadResponse,
    SessionRequest, SessionResponse, PlayerGScore,
    PatchRequest, PatchResponse, GScoresResponse,
    EvaluateRequest, EvaluateResponse,
    TradeAnalyzeRequest, TradeAnalyzeResponse,
    TradeSuggestRequest, TradeSuggestResponse,
    DivisionsResponse, ConnectRequest, ConnectResponse, DraftStateResponse,
)
from backend.evaluate import run_evaluate
from backend.math.trading import run_trade_analyze, run_trade_suggest
from backend.platform_integration.registry import get_integration, is_live_platform
from backend.platform_integration.base import PlatformConfig
from backend.platform_integration.helpers import build_platform_name_lookup
from backend.data_retrieval import get_player_mapping_view


# ── Upload store ──────────────────────────────────────────────────────────────

_UPLOAD_TTL     = 2 * 3600          # 2 hours
_MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB

_upload_store: dict[str, dict] = {}
_upload_lock = threading.Lock()


def _store_upload(data_id: str, csv_bytes: bytes, file_type: str, n_players: int) -> None:
    with _upload_lock:
        _upload_store[data_id] = {
            'bytes':      csv_bytes,
            'file_type':  file_type,
            'n_players':  n_players,
            'created_at': time.time(),
        }


def _get_upload(data_id: str) -> Optional[dict]:
    with _upload_lock:
        entry = _upload_store.get(data_id)
        if entry is None:
            return None
        if time.time() - entry['created_at'] > _UPLOAD_TTL:
            del _upload_store[data_id]
            return None
        return entry


# ── Small helpers ─────────────────────────────────────────────────────────────

_PARAMS_PATH = 'parameters.yaml'


def _load_all_params() -> dict:
    with open(_PARAMS_PATH) as f:
        return yaml.safe_load(f)


def _iso_expires(ttl_seconds: int) -> str:
    t = datetime.now(timezone.utc).timestamp() + ttl_seconds
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')



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


def _resolve_platform_config(req: SessionRequest) -> Optional[PlatformConfig]:
    """For a live platform, fetch the league shape and build a PlatformConfig to
    store on the session. Returns None for 'Enter your own data'."""
    if not is_live_platform(req.platform):
        return None
    if req.platform_config is None:
        raise HTTPException(
            status_code=400,
            detail=f'platform_config is required for platform {req.platform!r}.',
        )
    integration = get_integration(req.platform)
    try:
        shape = integration.fetch_league_shape(
            req.platform_config.league_id,
            req.platform_config.division_id,
        )
    except Exception:
        raise HTTPException(status_code=502, detail=traceback.format_exc())
    return PlatformConfig(
        platform           = req.platform,
        league_id          = req.platform_config.league_id,
        division_id        = req.platform_config.division_id,
        teams_dict         = shape.teams_dict,
        player_name_column = integration.player_name_column,
    )


def _refresh_platform_name_lookup(session) -> None:
    """Rebuild the session's platform name lookup from its current info.

    Lives here, not in run_pipeline, so the pipeline stays platform-agnostic. Call
    after the pipeline runs when the player set may have changed (session creation
    and data/injured patches, i.e. from_step <= 2); model/category/slot patches
    leave info['Positions'] untouched.
    """
    config = session.platform_config
    if config is None:
        return
    session.platform_name_lookup = build_platform_name_lookup(
        session.info, config.player_name_column, get_player_mapping_view(),
    )


def _resolve_csv(
    custom_data_ids: Optional[dict],
) -> tuple[Optional[bytes], Optional[str]]:
    """Return (csv_bytes, file_type) for the first valid custom data_id.
    Used for 'csv' data_source_type (single upload).
    """
    if not custom_data_ids:
        return None, None
    for ft, did in custom_data_ids.items():
        if did is not None:
            entry = _get_upload(did)
            if entry is None:
                raise HTTPException(
                    status_code=404,
                    detail=f'data_id {did!r} not found or expired.',
                )
            return entry['bytes'], entry['file_type']
    return None, None


def _resolve_uploaded_dfs(
    custom_data_ids: Optional[dict],
    params: dict,
) -> dict:
    """Return a {file_type: DataFrame} dict for all valid custom data_ids.
    Used for 'projections' data_source_type (multiple uploads).
    """
    if not custom_data_ids:
        return {}
    result = {}
    for ft, did in custom_data_ids.items():
        if did is None:
            continue
        entry = _get_upload(did)
        if entry is None:
            raise HTTPException(
                status_code=404,
                detail=f'data_id {did!r} not found or expired.',
            )
        result[ft.upper()] = _parse_projection_csv(entry['bytes'], entry['file_type'], params)
    return result


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title='Fantasy Basketball Optimizer', version='1.0.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


# ── GET /config/{sport} ───────────────────────────────────────────────────────

@app.get('/config/{sport}')
def get_config_route(sport: str):
    all_params = _load_all_params()
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


# ── POST /data/upload ─────────────────────────────────────────────────────────

@app.post('/data/upload', response_model=UploadResponse)
async def upload_projection(
    file: UploadFile = File(...),
    file_type: str = Form(...),
):
    ft = file_type.strip().upper()
    if ft not in ('HTB', 'BBM'):
        raise HTTPException(
            status_code=400,
            detail=f'Unsupported file_type {file_type!r}. Must be one of: HTB, BBM.',
        )

    csv_bytes = await file.read()
    if len(csv_bytes) > _MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail='File exceeds 10 MB limit.')

    all_params = _load_all_params()
    params = all_params.get('NBA', {})
    try:
        df = _parse_projection_csv(csv_bytes, ft, params)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f'Could not parse file: {exc}')

    data_id = uuid.uuid4().hex[:8]
    _store_upload(data_id, csv_bytes, ft, len(df))

    return UploadResponse(
        data_id=data_id,
        file_type=ft,
        n_players=len(df),
        expires_at=_iso_expires(_UPLOAD_TTL),
    )


# ── GET /seasons ──────────────────────────────────────────────────────────────

@app.get('/seasons')
def get_seasons_route():
    try:
        from backend.data_retrieval import get_available_seasons
        return {'seasons': get_available_seasons()}
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


# ── POST /sessions ────────────────────────────────────────────────────────────

@app.post('/sessions', response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
def create_session_route(req: SessionRequest):
    all_params = _load_all_params()

    if req.league.sport not in all_params:
        raise HTTPException(
            status_code=400,
            detail=f'Unknown sport: {req.league.sport!r}',
        )

    params = all_params[req.league.sport]
    source_type = req.data_source.type
    csv_bytes: Optional[bytes] = None
    file_type_str: Optional[str] = None
    uploaded_dfs: Optional[dict] = None

    if source_type == 'csv':
        csv_bytes, file_type_str = _resolve_csv(req.data_source.custom_data_ids)
    elif source_type == 'projections':
        uploaded_dfs = _resolve_uploaded_dfs(req.data_source.custom_data_ids, params)

    # Resolve any live-platform connection up front so a bad league fails before
    # the expensive pipeline runs.
    platform_config = _resolve_platform_config(req)

    session = create_session()
    session.current_params = _build_current_params(req, all_params)
    if platform_config is not None:
        session.platform_config = platform_config
        session.current_params['team_names'] = list(platform_config.teams_dict.keys())

    try:
        run_pipeline(session, from_step=1, csv_bytes=csv_bytes, file_type=file_type_str,
                     uploaded_dfs=uploaded_dfs)
        _refresh_platform_name_lookup(session)   # no-op unless a live platform is connected
    except Exception as exc:
        delete_session(session.id)
        raise HTTPException(status_code=500, detail=traceback.format_exc())

    categories = session.current_params['categories']

    # Serialize the G-scores DataFrame into a list of PlayerGScore objects
    g_scores_df = session.info['G-scores']
    g_scores_list = [
        PlayerGScore(
            name=str(name),
            total=round(float(row['Total']), 2),
            values=[round(float(row[cat]), 2) for cat in categories],
        )
        for name, row in g_scores_df.iterrows()
    ]

    return SessionResponse(
        session_id=session.id,
        n_players_loaded=len(session.v0_clean),
        categories=list(categories),
        g_scores=g_scores_list,
        expires_at=_iso_expires(4 * 3600),
    )


# ── PATCH /sessions/{session_id} ──────────────────────────────────────────────

@app.patch('/sessions/{session_id}', response_model=PatchResponse)
def patch_session_route(session_id: str, req: PatchRequest):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail='Session not found or expired.')

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

    csv_bytes: Optional[bytes] = None
    file_type_str: Optional[str] = None
    uploaded_dfs: Optional[dict] = None

    if req.data_source is not None:
        source_type = req.data_source.type
        patch['data_source_type'] = source_type
        if req.data_source.season is not None:
            patch['season'] = req.data_source.season
        if req.data_source.blend_weights is not None:
            patch['blend_weights'] = req.data_source.blend_weights
        if req.data_source.custom_data_ids is not None:
            patch['custom_data_ids'] = req.data_source.custom_data_ids
            _sport = session.current_params['sport']
            params = _load_all_params()[_sport]
            if source_type == 'csv':
                csv_bytes, file_type_str = _resolve_csv(req.data_source.custom_data_ids)
            elif source_type == 'projections':
                uploaded_dfs = _resolve_uploaded_dfs(req.data_source.custom_data_ids, params)

    session.current_params.update(patch)
    run_pipeline(session, from_step=req.from_step, csv_bytes=csv_bytes, file_type=file_type_str,
                 uploaded_dfs=uploaded_dfs)
    # The platform name lookup depends on the player set (info['Positions']), which
    # only changes on data/injured patches (from_step <= 2). Rebuilt outside
    # run_pipeline to keep the pipeline platform-agnostic.
    if req.from_step <= 2:
        _refresh_platform_name_lookup(session)
    return PatchResponse(ok=True, steps_rerun=list(range(req.from_step, 6)))


# ── GET /sessions/{session_id}/g-scores ───────────────────────────────────────

@app.get('/sessions/{session_id}/g-scores', response_model=GScoresResponse)
def get_g_scores_route(session_id: str):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail='Session not found or expired.')
    categories = session.current_params['categories']
    g_scores_df = session.info['G-scores']
    g_scores_list = [
        PlayerGScore(
            name=str(name),
            total=round(float(row['Total']), 2),
            values=[round(float(row[cat]), 2) for cat in categories],
        )
        for name, row in g_scores_df.iterrows()
    ]
    return GScoresResponse(g_scores=g_scores_list)


# ── POST /sessions/{session_id}/evaluate ──────────────────────────────────────

@app.post('/sessions/{session_id}/evaluate', response_model=EvaluateResponse)
def evaluate_route(session_id: str, req: EvaluateRequest):

    # Fetch once here so a missing/expired session returns a clean 404; the live
    # session object is handed to run_evaluate, which reads n_iterations from it.
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail='Session not found or expired.')

    try:
        result = run_evaluate(
            session            = session,
            player_assignments = req.player_assignments,
            my_team_id         = req.my_team_id,
            exclusion_list     = req.exclusion_list,
            remaining_cash     = req.remaining_cash,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


# ── POST /sessions/{session_id}/trade/analyze ────────────────────────────────

@app.post('/sessions/{session_id}/trade/analyze', response_model=TradeAnalyzeResponse)
def trade_analyze_route(session_id: str, req: TradeAnalyzeRequest):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail='Session not found or expired.')

    try:
        return run_trade_analyze(
            session,
            player_assignments=req.player_assignments,
            my_team=req.my_team,
            their_team=req.their_team,
            my_trade=req.my_trade,
            their_trade=req.their_trade,
            ignore_position_check=req.ignore_position_check,
        )
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


# ── POST /sessions/{session_id}/trade/suggest ────────────────────────────────

@app.post('/sessions/{session_id}/trade/suggest', response_model=TradeSuggestResponse)
def trade_suggest_route(session_id: str, req: TradeSuggestRequest):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail='Session not found or expired.')

    try:
        return run_trade_suggest(
            session,
            player_assignments=req.player_assignments,
            my_team=req.my_team,
            their_team=req.their_team,
            combo_params=req.combo_params,
            your_threshold=req.your_differential_threshold,
            their_threshold=req.their_differential_threshold,
            ignore_position_check=req.ignore_position_check,
        )
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


# ── DELETE /sessions/{session_id} ─────────────────────────────────────────────

@app.post('/cache/clear', status_code=status.HTTP_204_NO_CONTENT)
def clear_cache_route():
    clear_v0_cache()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.delete('/sessions/{session_id}', status_code=status.HTTP_204_NO_CONTENT)
def delete_session_route(session_id: str):
    found = delete_session(session_id)
    if not found:
        raise HTTPException(status_code=404, detail='Session not found or expired.')
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ── Platform integration (live: ESPN / Yahoo / Fantrax) ──────────────────────
# The {platform} path segment is the exact platform label the frontend sends
# (e.g. 'Retrieve from Fantrax'), URL-encoded; it is resolved via the registry.

def _resolve_live_integration(platform: str):
    if not is_live_platform(platform):
        raise HTTPException(status_code=400, detail=f'No live integration for platform {platform!r}.')
    return get_integration(platform)


@app.get('/platforms/{platform}/divisions', response_model=DivisionsResponse)
def get_platform_divisions_route(platform: str, league_id: str):
    integration = _resolve_live_integration(platform)
    try:
        divisions = integration.list_divisions(league_id)
    except Exception:
        raise HTTPException(status_code=502, detail=traceback.format_exc())
    return DivisionsResponse(divisions=divisions)


@app.post('/platforms/{platform}/connect', response_model=ConnectResponse)
def connect_platform_route(platform: str, req: ConnectRequest):
    integration = _resolve_live_integration(platform)
    try:
        shape = integration.fetch_league_shape(req.league_id, req.division_id)
    except Exception:
        raise HTTPException(status_code=502, detail=traceback.format_exc())
    return ConnectResponse(
        team_names      = shape.team_names,
        n_drafters      = shape.n_drafters,
        n_picks         = shape.n_picks,
        available_modes = integration.available_modes,
    )


@app.get('/sessions/{session_id}/draft-state', response_model=DraftStateResponse)
def get_draft_state_route(session_id: str, mode: str):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail='Session not found or expired.')
    if session.platform_config is None:
        raise HTTPException(status_code=400, detail='Session is not connected to a live platform.')
    integration = get_integration(session.platform_config.platform)
    # The lookup is built at session creation / data patches; rebuild defensively
    # if somehow absent so the poll never maps every player to 'RP'.
    if session.platform_name_lookup is None:
        _refresh_platform_name_lookup(session)
    try:
        state = integration.get_draft_results(session.platform_config, mode, session.platform_name_lookup)
    except Exception:
        raise HTTPException(status_code=502, detail=traceback.format_exc())
    return DraftStateResponse(
        player_assignments = state.player_assignments,
        injured_players    = state.injured_players,
        status             = state.status,
    )


# ── Static frontend ────────────────────────────────────────────────────────────

@app.get('/', include_in_schema=False)
def serve_index():
    return FileResponse('frontend/app.html')

app.mount('/styles', StaticFiles(directory='frontend/styles'), name='styles')
app.mount('/dist',   StaticFiles(directory='frontend/dist'),   name='dist')
