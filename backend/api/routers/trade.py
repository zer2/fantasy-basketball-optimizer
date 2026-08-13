"""Trade analysis and suggestion endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.state.session import get_session
from backend.services.trading import run_trade_analyze, run_trade_suggest
from backend.api.schemas import TradeAnalyzeRequest, TradeSuggestRequest
from backend.api.legacy_identity_shim import (
    convert_assignments_to_player_ids, convert_labels_to_player_ids,
)
from backend.models import TradeAnalyzeResponse, TradeSuggestResponse
from backend.api.errors import fail

router = APIRouter()


@router.post('/sessions/{session_id}/trade/analyze', response_model=TradeAnalyzeResponse)
def trade_analyze_route(session_id: str, req: TradeAnalyzeRequest):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail='Session not found or expired.')

    # Stage-A boundary conversion, before the try so its 400s survive the catch-all.
    player_assignments = convert_assignments_to_player_ids(
        session.player_registry, req.player_assignments)
    my_trade = convert_labels_to_player_ids(session.player_registry, req.my_trade)
    their_trade = convert_labels_to_player_ids(session.player_registry, req.their_trade)

    # Trade scoring runs get_h_scores, which mutates shared agent state; hold the per-session lock so
    # it cannot overlap an evaluate or another trade call on the same session (see Session.lock).
    try:
        with session.lock:
            return run_trade_analyze(
                session,
                player_assignments=player_assignments,
                my_team=req.my_team,
                their_team=req.their_team,
                my_trade=my_trade,
                their_trade=their_trade,
                ignore_position_check=req.ignore_position_check,
            )
    except Exception:
        raise fail(500, 'Trade analysis failed.')


@router.post('/sessions/{session_id}/trade/suggest', response_model=TradeSuggestResponse)
def trade_suggest_route(session_id: str, req: TradeSuggestRequest):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail='Session not found or expired.')

    player_assignments = convert_assignments_to_player_ids(
        session.player_registry, req.player_assignments)

    try:
        with session.lock:
            return run_trade_suggest(
                session,
                player_assignments=player_assignments,
                my_team=req.my_team,
                their_team=req.their_team,
                combo_params=req.combo_params,
                your_threshold=req.your_differential_threshold,
                their_threshold=req.their_differential_threshold,
                ignore_position_check=req.ignore_position_check,
            )
    except Exception:
        raise fail(500, 'Trade suggestion failed.')
