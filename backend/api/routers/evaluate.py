"""Candidate evaluation endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response

from backend.state.session import get_session
from backend.services.ranking import rank_candidates
from backend.infra.server_timing import begin_timing, server_timing_header
from backend.api.schemas import EvaluateRequest
from backend.models import EvaluateResponse
from backend.api.errors import fail

router = APIRouter()


@router.post('/sessions/{session_id}/evaluate', response_model=EvaluateResponse)
def evaluate_route(session_id: str, req: EvaluateRequest, response: Response):
    begin_timing()

    # Fetch once here so a missing/expired session returns a clean 404; the live
    # session object is handed to rank_candidates, which reads n_iterations from it.
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail='Session not found or expired.')

    try:
        result = rank_candidates(
            session            = session,
            player_assignments = req.player_assignments,
            my_team_id         = req.my_team_id,
            exclusion_list     = req.exclusion_list,
            remaining_cash     = req.remaining_cash,
            candidate_offset   = req.candidate_offset,
            candidate_limit    = req.candidate_limit,
        )
    except Exception:
        raise fail(500, 'Evaluation failed.')

    response.headers['Server-Timing'] = server_timing_header()
    return result
