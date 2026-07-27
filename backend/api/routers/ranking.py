"""Candidate ranking endpoint (fronts the ranking service)."""

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
def rank_candidates_route(session_id: str, req: EvaluateRequest, response: Response):
    begin_timing()

    # Fetch once here so a missing/expired session returns a clean 404; the live
    # session object is handed to rank_candidates, which reads n_iterations from it.
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail='Session not found or expired.')

    # Auction vs draft is all-or-nothing: an auction session must get per-team remaining_cash
    # on every evaluate, and any other session must never get it. Reject a request that mixes the
    # two rather than silently producing a draft-style result for an auction (or vice versa).
    # is_auction is patched whenever the user switches modes; a cash_per_team value left over
    # from an earlier auction lingers harmlessly — it is only consulted on auction sessions.
    is_auction_league = bool(session.current_params.get('is_auction'))
    if is_auction_league != (req.remaining_cash is not None):
        raise HTTPException(
            status_code=400,
            detail='remaining_cash is required for auction leagues and must be omitted for draft leagues.',
        )
    if is_auction_league and session.current_params.get('cash_per_team') is None:
        raise HTTPException(
            status_code=400,
            detail='cash_per_team must be set on the session for auction evaluates.',
        )

    # Hold the per-session lock for the whole evaluate: get_h_scores mutates shared agent state, so a
    # second evaluate (or a PATCH) overlapping this one on the same session would corrupt it and 500.
    # Serialised per session, so other sessions are unaffected.
    try:
        with session.lock:
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
