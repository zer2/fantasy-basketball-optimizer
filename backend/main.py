"""
FastAPI application — Fantasy Basketball Optimizer backend.
"""

from __future__ import annotations

import os
import time
import uuid
import logging
import threading
from datetime import datetime, timezone
from typing import Optional

import yaml
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, status, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from backend.auth import (
    oauth, current_user_key, current_user_key_optional, email_is_allowed,
    session_secret_key, session_https_only,
)
from backend.secret_config import get_secret

from backend.session import create_session, get_session, delete_session
from backend.server_timing import begin_timing, server_timing_header
from backend.pipeline import run_pipeline, _parse_projection_csv, clear_v0_cache
from backend.models import (
    UploadResponse,
    SessionRequest, SessionResponse, PlayerGScore,
    PatchRequest, PatchResponse, GScoresResponse,
    EvaluateRequest, EvaluateResponse,
    TradeAnalyzeRequest, TradeAnalyzeResponse,
    TradeSuggestRequest, TradeSuggestResponse,
    DivisionsResponse, ConnectRequest, ConnectResponse, DraftStateResponse,
    LeaguesResponse, YahooAuthUrlResponse, YahooTokenRequest, EspnCredentialsRequest,
    PlatformConfigRequest,
)
from backend.evaluate import run_evaluate
from backend.math.trading import run_trade_analyze, run_trade_suggest
from backend.platform_integration.registry import get_integration, is_live_platform
from backend.platform_integration.base import PlatformConfig
from backend.platform_integration.helpers import build_platform_name_lookup
from backend.platform_integration.credential_store import (
    yahoo_auth_dir, has_yahoo_credentials,
    store_espn_credentials, get_espn_credentials, has_espn_credentials,
)
from backend.platform_integration.integrations.yahoo import YahooIntegration
from backend.data_retrieval import get_player_mapping_view


logger = logging.getLogger('fbbo.api')
# Ensure app INFO logs (sign-ins) pass the level filter; handlers/formatting are left to the
# process's logging setup (uvicorn / imported libraries already configure stderr handlers).
logging.getLogger('fbbo').setLevel(logging.INFO)


def _fail(status_code: int, message: str) -> HTTPException:
    """Log the active exception server-side (with traceback) and return a client-safe error.
    Tracebacks/internal details must never be sent in the response body."""
    logger.exception(message)
    return HTTPException(status_code=status_code, detail=message)


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


def _resolve_platform_config(
    platform: str
    , platform_config_request: Optional[PlatformConfigRequest]
    , user_key: Optional[str]
) -> Optional[PlatformConfig]:
    """For a live platform, fetch the league shape and build a PlatformConfig to
    store on the session. Returns None for 'Enter your own data'. Shared by the
    create and patch routes (connecting a platform patches this onto the session)."""
    if not is_live_platform(platform):
        return None
    if user_key is None:
        raise HTTPException(status_code=401, detail='Sign in to connect a live platform.')
    if platform_config_request is None:
        raise HTTPException(
            status_code=400,
            detail=f'platform_config is required for platform {platform!r}.',
        )
    # Credentials are keyed by the authenticated user, never by anything in the request.
    integration = get_integration(platform, _credentials_for(platform, user_key))
    try:
        shape = integration.fetch_league_shape(
            platform_config_request.league_id,
            platform_config_request.division_id,
        )
    except Exception:
        raise _fail(502, 'Could not reach the fantasy platform for this league.')
    return PlatformConfig(
        platform           = platform,
        league_id          = platform_config_request.league_id,
        division_id        = platform_config_request.division_id,
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

# The app serves its own frontend (same-origin, BASE_URL=''), so normal use never triggers
# CORS. Restrict to a localhost allowlist (override with ALLOWED_ORIGINS, comma-separated, in
# deployment) so other origins can't script the API / credential endpoints.
_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        'ALLOWED_ORIGINS', 'http://localhost:8000,http://127.0.0.1:8000'
    ).split(',')
    if origin.strip()
]

# SessionMiddleware holds the signed login cookie (and Authlib's OAuth state). add_middleware
# prepends, so adding CORS last makes it outermost. Auth itself is enforced per-route via the
# current_user_key dependency, not globally — 'Enter your own data' usage needs no login.
app.add_middleware(
    SessionMiddleware,
    secret_key=session_secret_key(),
    https_only=session_https_only(),
    same_site='lax',
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.middleware('http')
async def _revalidate_app_assets(request: Request, call_next):
    """Force the browser to revalidate the app shell + static assets. StaticFiles sends no
    Cache-Control, so browsers heuristically cache styles.css / dist and serve stale copies —
    'no-cache' means 'revalidate before use' (cheap 304s when unchanged), so edits show up."""
    response = await call_next(request)
    path = request.url.path
    if path == '/' or path.startswith(('/dist', '/styles')):
        response.headers['Cache-Control'] = 'no-cache'
    return response


@app.middleware('http')
async def _server_timing(request: Request, call_next):
    """Append a `total` entry to Server-Timing for every request so each fetch's total server
    time shows in DevTools -> Network -> Timing. Instrumented handlers (e.g. /evaluate) set their
    own phase breakdown on the response first; this preserves it and adds the total."""
    start = time.perf_counter()
    response = await call_next(request)
    total_ms = (time.perf_counter() - start) * 1000
    existing = response.headers.get('Server-Timing')
    total_entry = f'total;dur={total_ms:.1f}'
    response.headers['Server-Timing'] = f'{existing}, {total_entry}' if existing else total_entry
    return response


# ── Auth (Google OIDC login) ──────────────────────────────────────────────────

@app.get('/auth/login')
async def auth_login_route(request: Request):
    """Kick off the Google sign-in redirect."""
    redirect_uri = os.environ.get('OAUTH_REDIRECT_URI') or str(request.url_for('auth_callback_route'))
    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get('/auth/callback', name='auth_callback_route')
async def auth_callback_route(request: Request):
    """Google redirects back here with a code; exchange it, verify, and start a session."""
    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception:
        raise _fail(401, 'Google sign-in failed.')
    userinfo = token.get('userinfo')
    if not userinfo or not userinfo.get('email_verified'):
        raise HTTPException(status_code=401, detail='Google account has no verified email.')
    email = userinfo['email']
    if not email_is_allowed(email):
        logger.warning('login denied (not allowlisted): %s', email)
        raise HTTPException(status_code=403, detail='This account is not permitted to sign in.')
    name = userinfo.get('given_name') or userinfo.get('name') or email
    request.session['user'] = {
        'sub': userinfo['sub'], 'email': email, 'name': name,
        'picture': userinfo.get('picture'),   # optional; Google provides it when available
    }
    logger.info('login: %s', email)
    return RedirectResponse(url='/')


@app.post('/auth/logout', status_code=status.HTTP_204_NO_CONTENT)
async def auth_logout_route(request: Request):
    request.session.clear()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get('/auth/me')
async def auth_me_route(request: Request):
    """Lightweight auth check for the frontend; 401 when not signed in."""
    # A valid session always carries name (set at login); a session missing it is stale /
    # malformed, so treat it as unauthenticated rather than silently degrading to the email.
    user = request.session.get('user')
    if not user or 'name' not in user:
        raise HTTPException(status_code=401, detail='Not authenticated.')
    return {'email': user['email'], 'name': user['name'], 'picture': user.get('picture')}


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
        raise _fail(500, 'Could not load available seasons.')


# ── POST /sessions ────────────────────────────────────────────────────────────

@app.post('/sessions', response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
def create_session_route(req: SessionRequest, user_key: Optional[str] = Depends(current_user_key_optional)):
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
    platform_config = _resolve_platform_config(req.platform, req.platform_config, user_key)

    session = create_session()
    session.current_params = _build_current_params(req, all_params)
    if platform_config is not None:
        session.platform_config = platform_config
        session.current_params['team_names'] = list(platform_config.teams_dict.keys())

    try:
        run_pipeline(session, from_step=1, csv_bytes=csv_bytes, file_type=file_type_str,
                     uploaded_dfs=uploaded_dfs)
        _refresh_platform_name_lookup(session)   # no-op unless a live platform is connected
    except Exception:
        delete_session(session.id)
        raise _fail(500, 'Failed to build projections for this session.')

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
def patch_session_route(session_id: str, req: PatchRequest, user_key: Optional[str] = Depends(current_user_key_optional)):
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

    # Connecting/switching a live platform patches its config onto the session. Resolve it
    # up front (before the pipeline) so a bad league fails fast. platform_config feeds only
    # the draft-state poll + name lookup, never the pipeline, so it needs no rerun of its own.
    platform_config = (
        _resolve_platform_config(req.platform, req.platform_config, user_key)
        if req.platform is not None else None
    )
    if platform_config is not None:
        session.platform_config = platform_config
        session.current_params['team_names'] = list(platform_config.teams_dict.keys())

    run_pipeline(session, from_step=req.from_step, csv_bytes=csv_bytes, file_type=file_type_str,
                 uploaded_dfs=uploaded_dfs)
    # Rebuild the platform name lookup when either of its inputs changed: the player set
    # (info['Positions'], on data/injured patches, from_step <= 2) or player_name_column
    # (when a platform_config was just set). Kept outside run_pipeline so the pipeline stays
    # platform-agnostic.
    if session.platform_config is not None and (req.from_step <= 2 or platform_config is not None):
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
def evaluate_route(session_id: str, req: EvaluateRequest, response: Response):
    begin_timing()

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
    except Exception:
        raise _fail(500, 'Evaluation failed.')

    response.headers['Server-Timing'] = server_timing_header()
    return result


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
        raise _fail(500, 'Trade analysis failed.')


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
        raise _fail(500, 'Trade suggestion failed.')


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

def _credentials_for(platform: str, client_id: Optional[str]) -> Optional[dict]:
    """Build the credentials an integration needs, or None. Yahoo needs a persisted OAuth
    token directory keyed by client_id; an unauthenticated client_id fails noisily."""
    if platform == 'Retrieve from Yahoo' and client_id:
        if not has_yahoo_credentials(client_id):
            raise HTTPException(status_code=401, detail='Not authenticated with Yahoo; complete the auth flow first.')
        return {'auth_dir': yahoo_auth_dir(client_id)}
    if platform == 'Retrieve from ESPN' and client_id:
        if not has_espn_credentials(client_id):
            raise HTTPException(status_code=401, detail='Not authenticated with ESPN; save your s2/SWID cookies first.')
        return get_espn_credentials(client_id)   # {'s2': ..., 'swid': ...} -> spread into ESPNIntegration
    return None


def _yahoo_app_credentials() -> tuple[str, str]:
    """The Yahoo *app* (developer) client id/secret, from the environment."""
    client_id = get_secret('YAHOO_CLIENT_ID')
    client_secret = get_secret('YAHOO_CLIENT_SECRET')
    if not client_id or not client_secret:
        raise HTTPException(status_code=500, detail='YAHOO_CLIENT_ID / YAHOO_CLIENT_SECRET are not configured.')
    return client_id, client_secret


def _resolve_live_integration(platform: str, client_id: Optional[str] = None):
    """Resolve a live integration, configured with credentials when the platform needs them
    (Yahoo). client_id=None gives an unauthenticated instance — fine for Fantrax and for Yahoo
    calls that don't hit the API (list_divisions)."""
    if not is_live_platform(platform):
        raise HTTPException(status_code=400, detail=f'No live integration for platform {platform!r}.')
    return get_integration(platform, _credentials_for(platform, client_id))


@app.get('/platforms/yahoo/auth-url', response_model=YahooAuthUrlResponse)
def yahoo_auth_url_route():
    client_id, _ = _yahoo_app_credentials()
    return YahooAuthUrlResponse(auth_url=YahooIntegration.build_auth_url(client_id))


@app.post('/platforms/yahoo/token', status_code=status.HTTP_204_NO_CONTENT)
def yahoo_token_route(req: YahooTokenRequest, user_key: str = Depends(current_user_key)):
    client_id, client_secret = _yahoo_app_credentials()
    try:
        YahooIntegration.exchange_auth_code(
            client_id, client_secret, req.auth_code, yahoo_auth_dir(user_key),
        )
    except Exception:
        raise _fail(502, 'Yahoo authorization failed.')
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post('/platforms/espn/credentials', status_code=status.HTTP_204_NO_CONTENT)
def espn_credentials_route(req: EspnCredentialsRequest, user_key: str = Depends(current_user_key)):
    # SWID is stored with braces stripped (as the Streamlit integration did).
    swid = req.swid.replace('{', '').replace('}', '')
    store_espn_credentials(user_key, req.s2, swid)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get('/platforms/{platform}/leagues', response_model=LeaguesResponse)
def get_platform_leagues_route(platform: str, user_key: str = Depends(current_user_key)):
    integration = _resolve_live_integration(platform, user_key)
    try:
        leagues = integration.list_leagues()
    except Exception:
        raise _fail(502, 'Failed to list leagues from the platform.')
    return LeaguesResponse(leagues=leagues)


@app.get('/platforms/{platform}/divisions', response_model=DivisionsResponse)
def get_platform_divisions_route(platform: str, league_id: str):
    integration = _resolve_live_integration(platform)
    try:
        divisions = integration.list_divisions(league_id)
    except Exception:
        raise _fail(502, 'Failed to list divisions from the platform.')
    return DivisionsResponse(divisions=divisions)


@app.post('/platforms/{platform}/connect', response_model=ConnectResponse)
def connect_platform_route(platform: str, req: ConnectRequest, user_key: str = Depends(current_user_key)):
    integration = _resolve_live_integration(platform, user_key)
    try:
        shape = integration.fetch_league_shape(req.league_id, req.division_id)
    except Exception:
        raise _fail(502, 'Failed to connect to the platform league.')
    return ConnectResponse(
        team_names      = shape.team_names,
        n_drafters      = shape.n_drafters,
        n_picks         = shape.n_picks,
        available_modes = integration.available_modes,
    )


@app.get('/sessions/{session_id}/draft-state', response_model=DraftStateResponse)
def get_draft_state_route(session_id: str, mode: str, user_key: str = Depends(current_user_key)):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail='Session not found or expired.')
    config = session.platform_config
    if config is None:
        raise HTTPException(status_code=400, detail='Session is not connected to a live platform.')
    # Poll with the requesting user's own credentials, never the key stored on the session.
    integration = get_integration(config.platform, _credentials_for(config.platform, user_key))
    # The lookup is built at session creation / data patches; rebuild defensively if
    # somehow absent so the poll never maps every player to 'RP'.
    if session.platform_name_lookup is None:
        _refresh_platform_name_lookup(session)
    try:
        if mode == 'Auction Mode':
            selections = integration.get_auction_results(config, mode, session.platform_name_lookup)
            if selections is None:
                raise HTTPException(status_code=400, detail=f'{config.platform} does not support auctions.')
        else:
            selections = integration.get_draft_results(config, mode, session.platform_name_lookup)
    except HTTPException:
        raise
    except Exception:
        raise _fail(502, 'Failed to fetch the live draft state.')

    # Auction selections carry per-player costs; turn them into per-team remaining cash.
    remaining_cash = None
    if selections.costs is not None:
        cash_per_team = session.current_params['cash_per_team']
        remaining_cash = {
            team: cash_per_team - sum(team_costs)
            for team, team_costs in selections.costs.items()
        }
    return DraftStateResponse(
        player_assignments = selections.player_assignments,
        injured_players    = selections.injured_players,
        status             = selections.status,
        remaining_cash     = remaining_cash,
    )


# ── Static frontend ────────────────────────────────────────────────────────────

@app.get('/', include_in_schema=False)
def serve_index():
    return FileResponse('frontend/app.html')

app.mount('/styles', StaticFiles(directory='frontend/styles'), name='styles')
app.mount('/dist',   StaticFiles(directory='frontend/dist'),   name='dist')
