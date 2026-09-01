"""Projection CSV upload endpoint. Headers are interpreted through the alias table, so no
particular export format is assumed (see projection_parsing.parse_projection_upload)."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File

from backend.infra.rate_limit import enforce_rate_limit, UPLOAD_POLICY
from backend.parameters import load_all_params
from backend.services.projection_parsing import parse_projection_upload, CORE_PROJECTION_COLUMNS
from backend.state.upload_store import store_upload, UPLOAD_TTL, MAX_FILE_BYTES
from backend.api.schemas import UploadResponse

router = APIRouter()

# The stats worth calling out when a file lacks them: the core counting stats plus the
# two ratio categories every standard league scores.
_REPORTABLE_STAT_COLUMNS = (*CORE_PROJECTION_COLUMNS, 'Field Goal %', 'Free Throw %')


def _find_missing_reportable_stats(parsed: pd.DataFrame, params: dict) -> list[str]:
    """Standard stats this file cannot contribute. A percentage whose attempts column is
    missing counts as absent too: the volume weights the percentage, so without it the
    category is dropped at build time — better to say so on the upload than to let it
    quietly disappear from the category list later."""
    missing = [column for column in _REPORTABLE_STAT_COLUMNS if column not in parsed.columns]
    for ratio_stat, ratio_info in params['ratio-statistics'].items():
        volume_statistic = ratio_info['volume-statistic']
        if (ratio_stat in parsed.columns
                and volume_statistic not in parsed.columns
                and volume_statistic not in missing):
            missing.append(volume_statistic)
    return missing


@router.post('/data/upload', response_model=UploadResponse,
             dependencies=[Depends(enforce_rate_limit(UPLOAD_POLICY))])
async def upload_projection_route(
    file: UploadFile = File(...),
):
    csv_bytes = await file.read()
    if len(csv_bytes) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail='File exceeds 10 MB limit.')

    all_params = load_all_params()
    params = all_params['NBA']
    try:
        df = parse_projection_upload(csv_bytes, params)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f'Could not parse file: {exc}')

    data_id = uuid.uuid4().hex[:8]
    store_upload(data_id, csv_bytes, len(df))

    return UploadResponse(
        data_id=data_id,
        n_players=len(df),
        expires_at=(datetime.now(timezone.utc) + timedelta(seconds=UPLOAD_TTL)).strftime('%Y-%m-%dT%H:%M:%SZ'),
        missing_stats=_find_missing_reportable_stats(df, params),
    )
