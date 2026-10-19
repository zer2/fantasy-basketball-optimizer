"""Streamer optimization endpoint for Fantasy Points leagues."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from fastapi import APIRouter, Depends, HTTPException

from backend.state.session import get_session
from backend.infra.rate_limit import enforce_rate_limit, COMPUTE_POLICY
from backend.api.schemas import StreamerOptimizeRequest
from backend.models import StreamerOptimizeResponse, StreamerPlanModel, StreamingMoveModel
from backend.points.streamer import optimize_streaming_schedule
from backend.points.schedule import ScheduleCalendar
from backend.api.helpers import fail

logger = logging.getLogger('fbbo.api.streamer')

router = APIRouter()


@router.post('/sessions/{session_id}/streamer/optimize', response_model=StreamerOptimizeResponse,
             dependencies=[Depends(enforce_rate_limit(COMPUTE_POLICY))])
def streamer_optimize_route(session_id: str, req: StreamerOptimizeRequest):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail='Session not found or expired.')

    try:
        with session.lock:
            # Roster and league configuration
            my_roster = [int(p) for p in req.player_assignments.get(req.my_team_id, []) if p is not None]

            # Available waivers
            rostered_players = {
                int(p)
                for team_players in req.player_assignments.values()
                for p in team_players
                if p is not None
            }

            # Resolve player metadata and projections from session
            player_fp_per_game: dict[int, float] = {}
            player_positions: dict[int, list[str]] = {}

            if hasattr(session, 'points_projections') and session.points_projections is not None:
                if hasattr(session.points_projections, 'per_game_fp') and session.points_projections.per_game_fp is not None:
                    player_fp_per_game = {int(k): float(v) for k, v in session.points_projections.per_game_fp.items() if v is not None}

            if hasattr(session, 'player_registry'):
                for pid, identity in session.player_registry.items():
                    player_positions[int(pid)] = list(identity.positions)

            # Fallback to points agent if available
            if not player_fp_per_game and hasattr(session, 'agent') and hasattr(session.agent, 'per_game_fp'):
                player_fp_per_game = {int(k): float(v) for k, v in session.agent.per_game_fp.items() if v is not None}
            if not player_positions and hasattr(session, 'agent') and hasattr(session.agent, 'position_eligibility'):
                player_positions = {int(k): list(v) for k, v in session.agent.position_eligibility.items()}

            # Available waiver candidates
            all_known = set(player_fp_per_game.keys()) | set(player_positions.keys())
            free_agents = sorted(
                [pid for pid in all_known if pid not in rostered_players and player_fp_per_game.get(pid, 0.0) > 0],
                key=lambda pid: player_fp_per_game.get(pid, 0.0),
                reverse=True,
            )
            if req.candidate_limit and req.candidate_limit > 0:
                free_agents = free_agents[:req.candidate_limit]

            # Resolve schedule
            schedule = getattr(session, 'schedule_calendar', None) or getattr(session.agent, 'schedule', None)
            if schedule is None:
                schedule = ScheduleCalendar()

            # Window
            today = date.today()
            window_dates = [(today + timedelta(days=i)).isoformat() for i in range(req.horizon_days)]

            # Active slots
            slot_counts = session.current_params.get('slot_counts', {}) or {}
            active_slots = {
                k: int(v) for k, v in slot_counts.items()
                if k not in ('Bench', 'IR', 'bench', 'ir') and int(v) > 0
            }
            if not active_slots:
                active_slots = {'PG': 1, 'SG': 1, 'SF': 1, 'PF': 1, 'C': 1, 'G': 1, 'F': 1, 'UTIL': 3}

            res = optimize_streaming_schedule(
                current_roster_ids=my_roster,
                available_waiver_ids=free_agents,
                schedule=schedule,
                evaluation_window=window_dates,
                player_positions=player_positions,
                player_fp_per_game=player_fp_per_game,
                active_slots=active_slots,
                max_acquisitions=req.max_acquisitions,
                mode=req.mode,
                streaming_slots=req.streaming_slots,
                drop_candidate_ids=req.designated_drops,
                bench_slots=session.current_params.get('bench_slots', 4),
                ir_slots=session.current_params.get('ir_slots', 3),
                solver_backend=req.solver,
            )

            # Convert result to StreamerOptimizeResponse
            def _convert_plan(p):
                return StreamerPlanModel(
                    strategy=p.strategy,
                    moves=[
                        StreamingMoveModel(
                            date=m.date,
                            add_player_id=m.add_player_id,
                            drop_player_id=m.drop_player_id,
                            games_gained=m.games_gained,
                            projected_fp=m.projected_fp,
                            dates_active=list(m.dates_active),
                            dates_played=list(m.dates_played),
                        )
                        for m in p.moves
                    ],
                    total_acquisitions=p.total_acquisitions,
                    total_games_streamed=p.total_games_streamed,
                    total_projected_fp=p.total_projected_fp,
                    points_per_add=p.points_per_add,
                    net_delta_fp=p.net_delta_fp,
                )

            return StreamerOptimizeResponse(
                optimal_plan=_convert_plan(res.optimal_plan),
                mode=res.mode,
                total_acquisitions_used=res.total_acquisitions_used,
                total_streamed_games=res.total_streamed_games,
                total_net_delta_fp=res.total_net_delta_fp,
                points_per_add=res.points_per_add,
                alternative_plans={k: _convert_plan(v) for k, v in res.alternative_plans.items()},
                daily_density_analysis=res.daily_density_analysis,
            )
    except Exception as exc:
        logger.exception("Streamer optimization failed: %s", exc)
        raise fail(500, f'Streamer optimization failed: {exc}')
