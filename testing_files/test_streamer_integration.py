"""
Integration test suite for fantasy_streamer package integrated into the Points stack.

Verifies:
1. Direct invocation of optimize_streaming_milp via Google OR-Tools and PuLP.
2. Mode 'milp' and mode 'auto' in optimize_streaming_schedule with fantasy_streamer.
3. Drop penalty deterrence in evaluate_waiver_candidates using DropPenaltyCalculator.
4. FastAPI /sessions/{id}/streamer/optimize REST endpoint.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date, timedelta
from fastapi.testclient import TestClient

from backend.main import app
from backend.points.schedule import ScheduleCalendar
from backend.points.streamer import optimize_streaming_schedule, optimize_streaming_milp
from backend.points.roster_schedule import evaluate_waiver_candidates
from backend.state.session import Session, create_session
from backend.points.config import build_points_scoring_config
from backend.points.agents import DynamicPointsAgent
from backend.player_identity import PlayerIdentity
from fantasy_streamer.models.penalties import DropPenaltyCalculator

client = TestClient(app)


@pytest.fixture
def streamer_test_context():
    """Create a realistic points league context for streaming tests."""
    window = ["2024-10-21", "2024-10-22", "2024-10-23", "2024-10-24", "2024-10-25", "2024-10-26", "2024-10-27"]
    schedule = ScheduleCalendar()

    # Roster of 13 players
    roster_ids = list(range(101, 114))
    teams = ["BOS", "LAL", "NYK", "MIA", "DEN", "GSW", "MIL", "DAL", "PHI", "PHX", "CLE", "ATL", "DET"]
    for pid, team in zip(roster_ids, teams):
        schedule.register_player_team(pid, team)

    player_positions = {}
    player_fp = {}

    positions_list = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL", "UTIL", "UTIL", "PG", "SF", "C"]
    fppg_list = [48.0, 42.0, 38.0, 35.0, 34.0, 30.0, 29.0, 27.0, 26.0, 25.0, 22.0, 20.0, 18.0]

    for i, pid in enumerate(roster_ids):
        pos = positions_list[i]
        player_positions[pid] = [pos, "UTIL"] if pos != "UTIL" else ["UTIL"]
        player_fp[pid] = fppg_list[i]

    # Available waiver candidates (high-yield streamers)
    waiver_ids = list(range(201, 215))
    fa_teams = ["BOS", "LAL", "NYK", "MIA", "DEN", "GSW", "MIL", "DAL", "PHI", "PHX", "CLE", "ATL", "DET", "IND"]
    for cid, team in zip(waiver_ids, fa_teams):
        schedule.register_player_team(cid, team)
        player_positions[cid] = ["PG", "SG", "G", "UTIL"] if cid % 2 == 0 else ["SF", "PF", "F", "UTIL"]
        player_fp[cid] = round(21.0 + (cid % 7), 1)

    active_slots = {"PG": 1, "SG": 1, "SF": 1, "PF": 1, "C": 1, "G": 1, "F": 1, "UTIL": 3}

    return {
        "roster_ids": roster_ids,
        "waiver_ids": waiver_ids,
        "schedule": schedule,
        "window": window,
        "player_positions": player_positions,
        "player_fp": player_fp,
        "active_slots": active_slots,
    }


def test_optimize_streaming_milp_ortools(streamer_test_context):
    """Verify optimize_streaming_milp runs and produces a valid MILP plan via OR-Tools."""
    ctx = streamer_test_context
    res = optimize_streaming_milp(
        current_roster_ids=ctx["roster_ids"],
        available_waiver_ids=ctx["waiver_ids"],
        schedule=ctx["schedule"],
        evaluation_window=ctx["window"],
        player_positions=ctx["player_positions"],
        player_fp_per_game=ctx["player_fp"],
        active_slots=ctx["active_slots"],
        max_acquisitions=5,
        solver_backend="ortools",
    )

    assert res.mode == "milp"
    assert res.optimal_plan.strategy == "milp"
    assert res.optimal_plan.total_acquisitions <= 5
    assert res.optimal_plan.total_projected_fp > 0.0
    assert "milp" in res.alternative_plans


def test_optimize_streaming_milp_pulp_fallback(streamer_test_context):
    """Verify optimize_streaming_milp runs and produces a valid plan via PuLP backend."""
    ctx = streamer_test_context
    res = optimize_streaming_milp(
        current_roster_ids=ctx["roster_ids"],
        available_waiver_ids=ctx["waiver_ids"][:5],
        schedule=ctx["schedule"],
        evaluation_window=ctx["window"][:4],
        player_positions=ctx["player_positions"],
        player_fp_per_game=ctx["player_fp"],
        active_slots=ctx["active_slots"],
        max_acquisitions=2,
        solver_backend="pulp",
    )

    assert res.mode == "milp"
    assert res.optimal_plan.total_acquisitions <= 2


def test_optimize_streaming_schedule_milp_mode(streamer_test_context):
    """Verify optimize_streaming_schedule seamlessly delegates to MILP when mode='milp'."""
    ctx = streamer_test_context
    res = optimize_streaming_schedule(
        current_roster_ids=ctx["roster_ids"],
        available_waiver_ids=ctx["waiver_ids"],
        schedule=ctx["schedule"],
        evaluation_window=ctx["window"],
        player_positions=ctx["player_positions"],
        player_fp_per_game=ctx["player_fp"],
        active_slots=ctx["active_slots"],
        max_acquisitions=4,
        mode="milp",
    )

    assert res.mode == "milp"
    assert res.optimal_plan.strategy == "milp"
    assert "daily_churn" in res.daily_density_analysis or len(res.daily_density_analysis) > 0


def test_optimize_streaming_schedule_auto_includes_milp(streamer_test_context):
    """Verify mode='auto' evaluates and compares the MILP plan alongside heuristics."""
    ctx = streamer_test_context
    res = optimize_streaming_schedule(
        current_roster_ids=ctx["roster_ids"],
        available_waiver_ids=ctx["waiver_ids"],
        schedule=ctx["schedule"],
        evaluation_window=ctx["window"],
        player_positions=ctx["player_positions"],
        player_fp_per_game=ctx["player_fp"],
        active_slots=ctx["active_slots"],
        max_acquisitions=4,
        mode="auto",
    )

    assert res.mode == "auto"
    assert "milp" in res.alternative_plans
    assert "daily_churn" in res.alternative_plans
    assert "tactical_multi" in res.alternative_plans
    assert res.optimal_plan is not None


def test_evaluate_waiver_candidates_with_drop_penalties(streamer_test_context):
    """Verify drop penalty calculator from fantasy_streamer correctly penalizes dropping valuable holds."""
    ctx = streamer_test_context
    calc = DropPenaltyCalculator(replacement_baseline=20.0)

    results = evaluate_waiver_candidates(
        current_roster_ids=ctx["roster_ids"],
        candidate_ids=[201, 202],
        schedule=ctx["schedule"],
        evaluation_window=ctx["window"],
        player_positions=ctx["player_positions"],
        player_fp_per_game=ctx["player_fp"],
        active_slots=ctx["active_slots"],
        drop_penalty_calculator=calc,
        consider_drop_penalties=True,
    )

    assert len(results) > 0
    for r in results:
        assert hasattr(r, "drop_penalty")
        assert hasattr(r, "net_delta_fp")
        assert r.net_delta_fp == round(r.delta_fp - r.drop_penalty, 4)


def test_streamer_optimize_api_endpoint(streamer_test_context):
    """Verify POST /sessions/{session_id}/streamer/optimize endpoint returns valid response."""
    ctx = streamer_test_context

    # Create mock session
    params = {
        "scoring_system": "points",
        "slot_counts": ctx["active_slots"],
        "bench_slots": 4,
        "ir_slots": 3,
    }
    session = create_session()
    session.current_params = params
    session_id = session.id

    # Seed projections and registry onto session
    config = build_points_scoring_config(
        points_input_mode="projected_points",
        projection_basis="per_game",
        scoring_rules={},
        active_slots=ctx["active_slots"],
    )
    session.agent = DynamicPointsAgent(
        config=config,
        per_game_fp=ctx["player_fp"],
        position_eligibility=ctx["player_positions"],
        slot_configs=[],
        total_roster_spots=13,
    )
    session.player_registry = {
        pid: PlayerIdentity(
            player_id=pid,
            name=f"Player_{pid}",
            last_name="Player",
            positions=list(ctx["player_positions"][pid]),
            has_headshot=True,
        )
        for pid in list(ctx["roster_ids"]) + list(ctx["waiver_ids"])
    }

    req_payload = {
        "player_assignments": {
            "team_user": ctx["roster_ids"],
            "team_opp": [301, 302, 303],
        },
        "my_team_id": "team_user",
        "horizon_days": 7,
        "max_acquisitions": 4,
        "mode": "auto",
        "streaming_slots": 1,
        "solver": "ortools",
    }

    resp = client.post(f"/sessions/{session_id}/streamer/optimize", json=req_payload)
    assert resp.status_code == 200, f"Error: {resp.text}"

    data = resp.json()
    assert "optimal_plan" in data
    assert "total_acquisitions_used" in data
    assert "total_net_delta_fp" in data
    assert "daily_density_analysis" in data
    assert data["optimal_plan"]["total_acquisitions"] <= 4
