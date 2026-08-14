"""Projection CSV upload endpoint. The export format is auto-detected from the headers."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, UploadFile, File

from backend.parameters import load_all_params
from backend.services.build_agent import parse_projection_csv, _CORE_PROJECTION_COLUMNS
from backend.state.upload_store import store_upload, UPLOAD_TTL, MAX_FILE_BYTES
from backend.api.schemas import UploadResponse

router = APIRouter()

# The stats worth calling out when an upload lacks them: the core counting stats plus the
# two ratio categories every standard league scores.
_REPORTABLE_STAT_COLUMNS = (*_CORE_PROJECTION_COLUMNS, 'Field Goal %', 'Free Throw %')


@router.post('/data/upload', response_model=UploadResponse)
async def upload_projection(
    file: UploadFile = File(...),
):
    csv_bytes = await file.read()
    if len(csv_bytes) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail='File exceeds 10 MB limit.')

    all_params = load_all_params()
    params = all_params.get('NBA', {})
    try:
        df, detected_format = parse_projection_csv(csv_bytes, params)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f'Could not parse file: {exc}')

    data_id = uuid.uuid4().hex[:8]
    store_upload(data_id, csv_bytes, detected_format, len(df))

    return UploadResponse(
        data_id=data_id,
        detected_format=detected_format,
        n_players=len(df),
        expires_at=(datetime.now(timezone.utc) + timedelta(seconds=UPLOAD_TTL)).strftime('%Y-%m-%dT%H:%M:%SZ'),
        missing_stats=[column for column in _REPORTABLE_STAT_COLUMNS if column not in df.columns],
    )
