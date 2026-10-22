"""Candidate ranking endpoint (fronts the ranking service)."""

from __future__ import annotations

import logging

from typing import Union
from fastapi import APIRouter, Depends, HTTPException, Response

from backend.api.helpers import fail, require_session
from backend.state.session import Session
from backend.infra.rate_limit import enforce_rate_limit, COMPUTE_POLICY
from backend.math.algorithm_agents import ForcedWeightsInfeasibleError
from backend.services.ranking import rank_candidates, UnknownRosterPlayersError, UnknownTeamError
from backend.services.hypothetical_team import build_hypothetical_team, StaleBoardError
from backend.services.auction_autodraft import nominate_auction_player, value_auction_player
from backend.infra.server_timing import begin_timing, server_timing_header
from backend.api.schemas import (
    EvaluateRequest, HypotheticalTeamRequest, AuctionNominationRequest, AuctionNominationResponse,
    AuctionValuationRequest, AuctionValuationResponse,
)
from backend.models import EvaluateResponse, PointsEvaluateResponse, HypotheticalTeamResponse

router = APIRouter()


@router.post('/sessions/{session_id}/evaluate', response_model=Union[EvaluateResponse, PointsEvaluateResponse],
             dependencies=[Depends(enforce_rate_limit(COMPUTE_POLICY))])
def rank_candidates_route(req: EvaluateRequest, response: Response,
                          session: Session = Depends(require_session)):
    begin_timing()
    # Full request logging, matching the session-create log: a board is only attributable
    # to its inputs when every evaluate's exact payload is on record too.
    logging.getLogger('fbbo').info('evaluate request: %s', req.model_dump_json())

    # Auction vs draft is all-or-nothing: an auction session must get per-team remaining_cash
    # on every evaluate, and any other session must never get it. Reject a request that mixes the
    # two rather than silently producing a draft-style result for an auction (or vice versa).
    # is_auction is patched whenever the user switches modes; a cash_per_team value left over
    # from an earlier auction lingers harmlessly — it is only consulted on auction sessions.
    is_auction_league = bool(session.current_settings.get('is_auction'))
    if is_auction_league != (req.remaining_cash is not None):
        raise HTTPException(
            status_code=400,
            detail='remaining_cash is required for auction leagues and must be omitted for draft leagues.',
        )
    if is_auction_league and session.current_settings.get('cash_per_team') is None:
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
                forced_category_weights = req.forced_category_weights,
            )
    except (UnknownRosterPlayersError, UnknownTeamError, ForcedWeightsInfeasibleError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        raise fail(500, 'Evaluation failed.')

    response.headers['Server-Timing'] = server_timing_header()
    return result


@router.post('/sessions/{session_id}/hypothetical-team', response_model=HypotheticalTeamResponse,
             dependencies=[Depends(enforce_rate_limit(COMPUTE_POLICY))])
def build_hypothetical_team_route(req: HypotheticalTeamRequest, session: Session = Depends(require_session)):
    # Same all-or-nothing auction rule as the evaluate: the request names the board that evaluate scored.
    is_auction_league = bool(session.current_settings.get('is_auction'))
    if is_auction_league != (req.remaining_cash is not None):
        raise HTTPException(
            status_code=400,
            detail='remaining_cash is required for auction leagues and must be omitted for draft leagues.',
        )
    try:
        with session.lock:
            return build_hypothetical_team(
                session                 = session,
                player_assignments      = req.player_assignments,
                my_team_id              = req.my_team_id,
                exclusion_list          = req.exclusion_list,
                remaining_cash          = req.remaining_cash,
                forced_category_weights = req.forced_category_weights,
                candidate_player_id     = req.candidate_player_id,
            )
    except StaleBoardError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        # An infeasible team (e.g. the candidate's predicted price leaves too little cash) is the user's board, not
        # a server fault; the message says which.
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        raise fail(500, 'Finding a hypothetical team failed.')


@router.post('/sessions/{session_id}/auction-autodraft/nominate', response_model=AuctionNominationResponse,
             dependencies=[Depends(enforce_rate_limit(COMPUTE_POLICY))])
def nominate_auction_player_route(req: AuctionNominationRequest, response: Response,
                                  session: Session = Depends(require_session)):
    begin_timing()
    logging.getLogger('fbbo').info('auction nomination request: %s', req.model_dump_json())
    require_auction_league(session)
    try:
        with session.lock:
            nomination = nominate_auction_player(
                session             = session,
                player_assignments  = req.player_assignments,
                remaining_cash      = req.remaining_cash,
                nominator_id        = req.nominator_id,
                nominated_player_id = req.nominated_player_id,
                opening_bid         = req.opening_bid,
                valuation_team_ids  = req.valuation_team_ids,
            )
    except (UnknownRosterPlayersError, UnknownTeamError, ForcedWeightsInfeasibleError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logging.getLogger('fbbo').error('Auction nomination failed: %s', exc, exc_info=True)
        raise fail(500, f'Auction nomination failed: {exc}')

    response.headers['Server-Timing'] = server_timing_header()
    return AuctionNominationResponse(
        nominated_player_id   = nomination.nominated_player_id,
        nominated_player_name = nomination.nominated_player_name,
        opening_bid           = nomination.opening_bid,
        valuations            = nomination.valuations,
        max_allowed_bids      = nomination.max_allowed_bids,
    )


@router.post('/sessions/{session_id}/auction-autodraft/valuations', response_model=AuctionValuationResponse,
             dependencies=[Depends(enforce_rate_limit(COMPUTE_POLICY))])
def value_auction_player_route(req: AuctionValuationRequest, response: Response,
                               session: Session = Depends(require_session)):
    begin_timing()
    logging.getLogger('fbbo').info('auction valuation request: %s', req.model_dump_json())
    require_auction_league(session)
    try:
        with session.lock:
            valuations = value_auction_player(
                session            = session,
                player_assignments = req.player_assignments,
                remaining_cash     = req.remaining_cash,
                player_id          = req.player_id,
                valuation_team_ids = req.valuation_team_ids,
            )
    except (UnknownRosterPlayersError, UnknownTeamError, ForcedWeightsInfeasibleError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logging.getLogger('fbbo').error('Auction valuation failed: %s', exc, exc_info=True)
        raise fail(500, f'Auction valuation failed: {exc}')

    response.headers['Server-Timing'] = server_timing_header()
    return AuctionValuationResponse(valuations=valuations)


def require_auction_league(session: Session) -> None:
    if not session.current_settings.get('is_auction'):
        raise HTTPException(status_code=400, detail='Auction autodraft is only available for auction leagues.')
    if session.current_settings.get('cash_per_team') is None:
        raise HTTPException(status_code=400, detail='cash_per_team must be set on the session for auction leagues.')
