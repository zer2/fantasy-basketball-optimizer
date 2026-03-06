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
from fastapi.responses import Response

from backend.session import create_session, get_session, delete_session
from backend.pipeline import run_pipeline, _parse_projection_csv
from backend.models import (
    UploadResponse,
    SessionRequest, SessionResponse,
    PatchRequest, PatchResponse,
    EvaluateRequest, EvaluateResponse,
)
from backend.evaluate import run_evaluate


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
    Used for 'blended' data_source_type (multiple uploads).
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
    source_type = req.data_source.type if req.data_source else 'mock'
    csv_bytes: Optional[bytes] = None
    file_type_str: Optional[str] = None
    uploaded_dfs: Optional[dict] = None

    if source_type == 'csv':
        csv_bytes, file_type_str = _resolve_csv(
            req.data_source.custom_data_ids if req.data_source else None
        )
    elif source_type == 'blended':
        uploaded_dfs = _resolve_uploaded_dfs(
            req.data_source.custom_data_ids if req.data_source else None, params
        )

    session = create_session()
    session.current_params = _build_current_params(req, all_params)

    try:
        run_pipeline(session, from_step=1, csv_bytes=csv_bytes, file_type=file_type_str,
                     uploaded_dfs=uploaded_dfs)
    except Exception as exc:
        delete_session(session.id)
        raise HTTPException(status_code=500, detail=traceback.format_exc())

    categories = session.current_params['categories']

    return SessionResponse(
        session_id=session.id,
        n_players_loaded=len(session.v0_clean),
        categories=list(categories),
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
            elif source_type == 'blended':
                uploaded_dfs = _resolve_uploaded_dfs(req.data_source.custom_data_ids, params)

    session.current_params.update(patch)
    run_pipeline(session, from_step=req.from_step, csv_bytes=csv_bytes, file_type=file_type_str,
                 uploaded_dfs=uploaded_dfs)
    return PatchResponse(ok=True, steps_rerun=list(range(req.from_step, 6)))


# ── POST /sessions/{session_id}/evaluate ──────────────────────────────────────

@app.post('/sessions/{session_id}/evaluate', response_model=EvaluateResponse)
def evaluate_route(session_id: str, req: EvaluateRequest):

    #ZR: What is the point of these lines? This function isn't using the session at all
    #just the session id.
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail='Session not found or expired.')

    n_iterations = session.current_params['n_iterations']

    try:
        result = run_evaluate(
            session_id        = session_id,
            player_assignments = req.player_assignments,
            my_team_id        = req.my_team_id,
            exclusion_list    = req.exclusion_list,
            remaining_cash    = req.remaining_cash,
            n_iterations      = n_iterations,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


# ── DELETE /sessions/{session_id} ─────────────────────────────────────────────

@app.delete('/sessions/{session_id}', status_code=status.HTTP_204_NO_CONTENT)
def delete_session_route(session_id: str):
    found = delete_session(session_id)
    if not found:
        raise HTTPException(status_code=404, detail='Session not found or expired.')
    return Response(status_code=status.HTTP_204_NO_CONTENT)
