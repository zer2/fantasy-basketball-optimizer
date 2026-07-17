"""Projection CSV upload endpoint (HTB / BBM)."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from backend.parameters import load_all_params
from backend.services.pipeline import _parse_projection_csv
from backend.state.upload_store import store_upload, UPLOAD_TTL, MAX_FILE_BYTES
from backend.api.schemas import UploadResponse
from backend.api.util import iso_expires

router = APIRouter()


@router.post('/data/upload', response_model=UploadResponse)
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
    if len(csv_bytes) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail='File exceeds 10 MB limit.')

    all_params = load_all_params()
    params = all_params.get('NBA', {})
    try:
        df = _parse_projection_csv(csv_bytes, ft, params)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f'Could not parse file: {exc}')

    data_id = uuid.uuid4().hex[:8]
    store_upload(data_id, csv_bytes, ft, len(df))

    return UploadResponse(
        data_id=data_id,
        file_type=ft,
        n_players=len(df),
        expires_at=iso_expires(UPLOAD_TTL),
    )
