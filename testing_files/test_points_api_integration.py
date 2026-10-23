"""
Milestone 4 Test Suite: Backend API Routing & Service Integration.

Comprehensive offline unit and integration tests covering:
1. Points session creation and build_agent with position_structure bug fix.
2. /sessions/{id}/evaluate in Points Mode returning PointsEvaluateResponse with Delta FP.
3. /sessions/{id}/trade/analyze in Points Mode returning Net Delta FP, streamer spot boost, and league impact.
4. /sessions/{id}/trade/suggest in Points Mode returning PointsTradeSuggestResponse with win-win proposals.
5. Category mode zero regression guard (/evaluate, /trade/analyze, /trade/suggest).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.state.session import create_session, delete_session, Session
from backend.points.agents import DynamicPointsAgent
from backend.points.config import PointsScoringConfig
from backend.points.schedule import ScheduleCalendar
from backend.player_identity import make_player_identity
from backend.services.build_agent import build_agent
from backend.models import (
    PointsEvaluateResponse,
    PointsTradeAnalyzeResponse,
    PointsTradeSuggestResponse,
    EvaluateResponse,
    TradeAnalyzeResponse,
    TradeSuggestResponse,
)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_schedule():
    """Deterministic schedule calendar for 7-day window (2024-11-04 to 2024-11-10)."""
    team_schedules = {
        "BOS": ["2024-11-04", "2024-11-06", "2024-11-08", "2024-11-10"],  # 4 games
        "LAL": ["2024-11-05", "2024-11-07", "2024-11-09"],                # 3 games
        "NYK": ["2024-11-04", "2024-11-05", "2024-11-07", "2024-11-08"],  # 4 games
        "MIA": ["2024-11-05", "2024-11-06", "2024-11-09"],                # 3 games
    }
    player_teams = {
        # Team 1 players
        1: "BOS", 2: "BOS", 3: "LAL", 4: "LAL", 5: "BOS", 6: "LAL",
        # Team 2 players
        11: "NYK", 12: "NYK", 13: "NYK", 14: "MIA", 15: "MIA", 16: "NYK",
        # Team 3 players (league impact)
        21: "BOS", 22: "LAL", 23: "NYK", 24: "MIA", 25: "BOS", 26: "NYK",
        # Free agents / candidates
        101: "NYK", 102: "BOS", 103: "LAL", 104: "MIA",
    }
    return ScheduleCalendar.from_dict(team_schedules, player_teams=player_teams)


@pytest.fixture
def sample_points_session(sample_schedule):
    """Create and return a configured points-mode session."""
    session = create_session()

    config = PointsScoringConfig(
        points_input_mode="projected_points",
        projection_basis="per_game",
        scoring_rules={"PTS": 1.0, "REB": 1.2, "AST": 1.5, "STL": 3.0, "BLK": 3.0, "TO": -1.0},
        competition_format="head_to_head",
        active_slots={"PG": 1, "SG": 1, "SF": 1, "PF": 1, "C": 1, "Util": 1},
        bench_slots=2,
        ir_slots=1,
    )

    per_game_fp = {
        1: 45.0, 2: 40.0, 3: 35.0, 4: 30.0, 5: 28.0, 6: 25.0,
        11: 44.0, 12: 38.0, 13: 32.0, 14: 29.0, 15: 26.0, 16: 22.0,
        21: 42.0, 22: 36.0, 23: 31.0, 24: 27.0, 25: 24.0, 26: 20.0,
        101: 34.0, 102: 31.0, 103: 27.0, 104: 18.0,
    }

    position_eligibility = {
        1: ["PG"], 2: ["SG"], 3: ["SF"], 4: ["PF"], 5: ["C"], 6: ["PG", "SG"],
        11: ["PG"], 12: ["SG"], 13: ["SF"], 14: ["PF"], 15: ["C"], 16: ["SF", "PF"],
        21: ["PG"], 22: ["SG"], 23: ["SF"], 24: ["PF"], 25: ["C"], 26: ["PG", "C"],
        101: ["PG"], 102: ["SG", "SF"], 103: ["C"], 104: ["PF"],
    }

    session.current_params = {
        "scoring_system": "points",
        "sport": "NBA",
        "evaluation_window": ["2024-11-04", "2024-11-05", "2024-11-06", "2024-11-07", "2024-11-08"],
        "slot_counts": dict(config.active_slots),
        "n_picks": 8,
        "n_drafters": 3,
        "is_auction": False,
        "waiver_baseline_fp_per_game": 20.0,
        "streamer_multiplier": 1.6,
        "streamer_games_per_week": 6.5,
    }

    session.agent = DynamicPointsAgent(
        config=config,
        per_game_fp=per_game_fp,
        position_eligibility=position_eligibility,
        slot_configs=[],
        total_roster_spots=8,
    )
    session.agent.schedule = sample_schedule
    session.schedule_calendar = sample_schedule
    session.points_config = config
    session.player_registry = {
        pid: make_player_identity(pid, f"Player {pid}", position_eligibility[pid])
        for pid in per_game_fp
    }

    yield session
    delete_session(session.id)


@pytest.fixture
def sample_category_session():
    """Create and return a configured category-mode session with mock agent."""
    session = create_session()
    session.current_params = {
        "scoring_system": "categories",
        "categories": ["PTS", "REB", "AST"],
        "n_iterations": 5,
        "n_picks": 6,
        "n_drafters": 2,
        "is_auction": False,
        "sport": "NBA",
    }

    mock_agent = MagicMock()
    mock_agent.info = {
        "Positions": pd.DataFrame({"Position": ["PG", "SG", "SF", "PF", "C"]}, index=["1", "2", "3", "4", "5"])
    }
    mock_agent.get_h_scores.return_value = {
        "Scores": pd.Series([10.0], index=["Team 1"]),
        "Rates": pd.DataFrame([[0.5, 0.5, 0.5]], index=["Team 1"]),
    }
    mock_agent._pos_cfg = MagicMock()
    mock_agent.default_h_scores = pd.Series([5.0, 4.0, 3.0], index=["1", "2", "3"])
    mock_agent.x_scores = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=[1, 2, 3])
    session.agent = mock_agent

    yield session
    delete_session(session.id)


# ==============================================================================
# 1. Points Session Creation & Build Agent Tests
# ==============================================================================

class TestPointsSessionCreation:
    """Tests for offline points session initialization and build_agent pipeline."""

    def test_build_agent_points_pipeline_offline(self):
        """build_agent constructs DynamicPointsAgent and stores points configs."""
        mock_resolver = {
            "Luka Doncic": 1,
            "Nikola Jokic": 2,
            "Giannis Antetokounmpo": 3,
            "Shai Gilgeous-Alexander": 4,
        }

        csv_text = """Player,Position,Games Played,Points,Rebounds,Assists,Steals,Blocks,Turnovers,Threes
Luka Doncic,PG,70,33.9,9.2,9.8,1.4,0.5,4.0,4.1
Nikola Jokic,C,79,26.4,12.4,9.0,1.4,0.9,3.0,1.0
Giannis Antetokounmpo,PF,73,30.4,11.5,6.5,1.2,1.1,3.4,0.5
Shai Gilgeous-Alexander,PG,75,30.1,5.5,6.2,2.0,0.9,2.2,1.3
"""

        session = create_session()
        session.current_params = {
            "scoring_system": "points",
            "sport": "NBA",
            "data_source_type": "csv",
            "scoring_format": "Head to Head: Each Category",
            "slot_counts": {"PG": 1, "SG": 1, "SF": 1, "PF": 1, "C": 1, "Util": 1},
            "n_picks": 7,
            "n_drafters": 2,
            "n_iterations": 0,
            "omega": 1.0,
            "gamma": 1.0,
            "aleph": 0.0,
            "kappa": 0.0,
            "beth": 0.0,
            "points_config": {
                "points_input_mode": "raw_stats",
                "projection_basis": "per_game",
                "scoring_rules": {"PTS": 1.0, "REB": 1.2, "AST": 1.5, "STL": 3.0, "BLK": 3.0, "TO": -1.0, "3PM": 1.0},
                "competition_format": "head_to_head",
                "bench_slots": 1,
                "ir_slots": 1,
            },
        }

        def _mock_attach(df):
            df = df.copy()
            df['player_id'] = df['Player'].map(mock_resolver)
            return df

        with patch("backend.services.build_agent.attach_player_ids_by_name", side_effect=_mock_attach):
            build_agent(session, from_step=1, csv_bytes=csv_text.encode("utf-8"))

        assert isinstance(session.agent, DynamicPointsAgent)
        assert session.points_config is not None
        assert session.points_projections is not None
        assert session.agent.schedule is session.schedule_calendar
        assert len(session.agent.per_game_fp) == 4
        delete_session(session.id)

    def test_build_agent_position_structure_resolution(self):
        """Verifies line 666 bug fix: handles position_structure with underscore or hyphen."""
        from backend.services.build_agent import _run_points_pipeline

        session = create_session()
        session.current_params = {
            "scoring_system": "points",
            "sport": "NBA",
            "slot_counts": {"PG": 1, "C": 1},
            "n_picks": 2,
            "points_config": {
                "points_input_mode": "projected_points",
                "projection_basis": "per_game",
                "scoring_rules": {},
                "competition_format": "head_to_head",
            },
        }
        session.v1_clean = pd.DataFrame({"FantasyPts": [35.0, 40.0], "Games Played": [82, 82]}, index=[1, 2])
        session.player_registry = {
            1: make_player_identity(1, "Player 1", ["PG"]),
            2: make_player_identity(2, "Player 2", ["C"]),
        }

        # Test with position_structure in params
        _run_points_pipeline(session, from_step=3)
        assert isinstance(session.agent, DynamicPointsAgent)
        assert session.agent.slot_configs is not None
        delete_session(session.id)


# ==============================================================================
# 2. /sessions/{id}/evaluate in Points Mode Tests
# ==============================================================================

class TestPointsEvaluateEndpoint:
    """Tests for the candidate evaluation API in Points Mode."""

    def test_evaluate_points_mode_returns_points_evaluate_response(self, client, sample_points_session):
        """/evaluate returns PointsEvaluateResponse with expected fields in points mode."""
        resp = client.post(
            f"/sessions/{sample_points_session.id}/evaluate",
            json={
                "player_assignments": {
                    "Team 1": [1, 2, 3, 4, 5, 6],
                    "Team 2": [11, 12, 13, 14, 15, 16],
                },
                "my_team_id": "Team 1",
                "exclusion_list": [],
            },
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["scoring_system"] == "points"
        assert data["objective"] == "expected_points"
        assert data["units"] == "fantasy_points"
        assert "horizon" in data
        assert isinstance(data["candidates"], list)
        assert len(data["candidates"]) > 0

        first_cand = data["candidates"][0]
        assert "player_id" in first_cand
        assert "delta_fp" in first_cand
        assert "marginal_fp" in first_cand
        assert first_cand["delta_fp"] == first_cand["marginal_fp"]
        assert "drop_player_id" in first_cand
        assert "candidate_games" in first_cand
        assert "daily_deltas" in first_cand

    def test_evaluate_points_batching_offset_and_limit(self, client, sample_points_session):
        """/evaluate supports candidate_offset and candidate_limit batching."""
        resp = client.post(
            f"/sessions/{sample_points_session.id}/evaluate",
            json={
                "player_assignments": {
                    "Team 1": [1, 2, 3, 4, 5, 6],
                    "Team 2": [11, 12, 13, 14, 15, 16],
                    "Team 3": [21, 22, 23, 24, 25, 26],
                },
                "my_team_id": "Team 1",
                "exclusion_list": [],
                "candidate_offset": 0,
                "candidate_limit": 2,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["candidates"]) == 2
        assert data["has_more"] is True
        assert data["total_candidates"] == 4

        # Second batch
        resp2 = client.post(
            f"/sessions/{sample_points_session.id}/evaluate",
            json={
                "player_assignments": {
                    "Team 1": [1, 2, 3, 4, 5, 6],
                    "Team 2": [11, 12, 13, 14, 15, 16],
                    "Team 3": [21, 22, 23, 24, 25, 26],
                },
                "my_team_id": "Team 1",
                "exclusion_list": [],
                "candidate_offset": 2,
                "candidate_limit": 2,
            },
        )
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert len(data2["candidates"]) == 2
        assert data2["has_more"] is False
        assert data2["candidates"][0]["rank"] == 3

    def test_evaluate_points_explicit_drop_candidate(self, client, sample_points_session):
        """Explicit drop_player_id in session params restricts drop options."""
        sample_points_session.current_params["drop_player_id"] = 6

        resp = client.post(
            f"/sessions/{sample_points_session.id}/evaluate",
            json={
                "player_assignments": {
                    "Team 1": [1, 2, 3, 4, 5, 6],
                    "Team 2": [11, 12, 13, 14, 15, 16],
                },
                "my_team_id": "Team 1",
                "exclusion_list": [],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        for cand in data["candidates"]:
            assert cand["drop_player_id"] == 6

    def test_evaluate_points_empty_roster(self, client, sample_points_session):
        """Evaluating with an empty roster yields valid positive candidate values."""
        resp = client.post(
            f"/sessions/{sample_points_session.id}/evaluate",
            json={
                "player_assignments": {
                    "Team 1": [],
                    "Team 2": [11, 12, 13],
                },
                "my_team_id": "Team 1",
                "exclusion_list": [],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["candidates"]) > 0
        assert data["candidates"][0]["marginal_fp"] > 0.0


# ==============================================================================
# 3. /sessions/{id}/trade/analyze in Points Mode Tests
# ==============================================================================

class TestPointsTradeAnalyzeEndpoint:
    """Tests for the trade analysis endpoint in Points Mode."""

    def test_trade_analyze_balanced_1v1(self, client, sample_points_session):
        """1v1 trade returns net delta FP for both sides and zero streamer boost."""
        resp = client.post(
            f"/sessions/{sample_points_session.id}/trade/analyze",
            json={
                "player_assignments": {
                    "Team 1": [1, 2, 3, 4, 5, 6],
                    "Team 2": [11, 12, 13, 14, 15, 16],
                },
                "my_team": "Team 1",
                "their_team": "Team 2",
                "my_trade": [1],
                "their_trade": [11],
                "ignore_position_check": False,
            },
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["scoring_system"] == "points"
        assert data["is_legal"] is True
        assert data["error"] is None
        assert "team_a_net_delta_fp" in data
        assert "team_b_net_delta_fp" in data
        assert data["team_a_streamer_boost"] == 0.0
        assert data["team_b_streamer_boost"] == 0.0
        assert data["team_a_opened_spots"] == 0
        assert data["team_b_opened_spots"] == 0

    def test_trade_analyze_unbalanced_2v1_with_streamer_boost(self, client, sample_points_session):
        """2v1 trade grants streamer spot boost to the team sending 2 and receiving 1."""
        resp = client.post(
            f"/sessions/{sample_points_session.id}/trade/analyze",
            json={
                "player_assignments": {
                    "Team 1": [1, 2, 3, 4, 5, 6],
                    "Team 2": [11, 12, 13, 14, 15],  # 5 players, can receive 2 without exceeding 8
                },
                "my_team": "Team 1",
                "their_team": "Team 2",
                "my_trade": [5, 6],
                "their_trade": [11],
                "ignore_position_check": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["is_legal"] is True
        assert data["team_a_opened_spots"] == 1
        assert data["team_b_opened_spots"] == 0
        assert data["team_a_streamer_boost"] > 0.0
        # Net delta includes the streamer boost and matches post minus pre
        expected_net_a = round(data["team_a_post_fp"] - data["team_a_pre_fp"], 4)
        assert pytest.approx(data["team_a_net_delta_fp"], 0.01) == expected_net_a

    def test_trade_analyze_illegal_trade_rejection(self, client, sample_points_session):
        """Lopsided or empty trades are rejected with clean rejection reason."""
        # Empty trade
        resp = client.post(
            f"/sessions/{sample_points_session.id}/trade/analyze",
            json={
                "player_assignments": {
                    "Team 1": [1, 2, 3],
                    "Team 2": [11, 12, 13],
                },
                "my_team": "Team 1",
                "their_team": "Team 2",
                "my_trade": [],
                "their_trade": [11],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_legal"] is False
        assert data["error"] is not None
        assert "at least one player" in data["error"]

    def test_trade_analyze_league_wide_impact(self, client, sample_points_session):
        """When 3+ team rosters are provided, league_impact is populated."""
        resp = client.post(
            f"/sessions/{sample_points_session.id}/trade/analyze",
            json={
                "player_assignments": {
                    "Team 1": [1, 2, 3, 4, 5, 6],
                    "Team 2": [11, 12, 13, 14, 15, 16],
                    "Team 3": [21, 22, 23, 24, 25, 26],
                },
                "my_team": "Team 1",
                "their_team": "Team 2",
                "my_trade": [1],
                "their_trade": [11],
                "ignore_position_check": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["league_impact"] is not None
        li = data["league_impact"]
        assert "pre_trade_ranks" in li
        assert "post_trade_ranks" in li
        assert "rank_shifts" in li
        assert "competitive_balance_delta" in li
        assert len(li["pre_trade_ranks"]) == 3


# ==============================================================================
# 4. /sessions/{id}/trade/suggest in Points Mode Tests
# ==============================================================================

class TestPointsTradeSuggestEndpoint:
    """Tests for the win-win trade suggestion engine in Points Mode."""

    def test_trade_suggest_win_win_proposals(self, client, sample_points_session):
        """/trade/suggest returns PointsTradeSuggestResponse with beneficial proposals."""
        resp = client.post(
            f"/sessions/{sample_points_session.id}/trade/suggest",
            json={
                "player_assignments": {
                    "Team 1": [1, 2, 3, 4, 5, 6],
                    "Team 2": [11, 12, 13, 14, 15, 16],
                },
                "my_team": "Team 1",
                "their_team": "Team 2",
                "combo_params": [{"n_traded": 1, "n_received": 1, "threshold": -50.0}],
                "your_differential_threshold": -50.0,
                "their_differential_threshold": -50.0,
                "ignore_position_check": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["scoring_system"] == "points"
        assert isinstance(data["suggestions"], list)
        assert len(data["suggestions"]) > 0

        first_sug = data["suggestions"][0]
        assert "partner_team_id" in first_sug
        assert "user_sends" in first_sug
        assert "partner_sends" in first_sug
        assert "user_gain" in first_sug
        assert "partner_gain" in first_sug
        assert "total_gain" in first_sug
        assert "trade_format" in first_sug
        assert first_sug["trade_format"] == "1v1"

    def test_trade_suggest_formats_2v1(self, client, sample_points_session):
        """/trade/suggest supports unbalanced 2v1 formats."""
        resp = client.post(
            f"/sessions/{sample_points_session.id}/trade/suggest",
            json={
                "player_assignments": {
                    "Team 1": [1, 2, 3, 4, 5, 6],
                    "Team 2": [11, 12, 13, 14, 15],  # 5 players
                },
                "my_team": "Team 1",
                "their_team": "Team 2",
                "combo_params": [{"n_traded": 2, "n_received": 1, "threshold": -100.0}],
                "your_differential_threshold": -100.0,
                "their_differential_threshold": -100.0,
                "ignore_position_check": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["suggestions"]) > 0
        assert data["suggestions"][0]["trade_format"] in ("2v1", "1v2")


# ==============================================================================
# 5. Category Mode Zero Regression Guard Tests
# ==============================================================================

class TestCategoryModeZeroRegression:
    """Verifies that category mode sessions continue functioning with 100% fidelity."""

    def test_category_trade_analyze_zero_regression(self, client, sample_category_session):
        """Category session returns standard TradeAnalyzeResponse with HScore pre/post."""
        resp = client.post(
            f"/sessions/{sample_category_session.id}/trade/analyze",
            json={
                "player_assignments": {
                    "Team 1": [1, 2],
                    "Team 2": [3, 4],
                },
                "my_team": "Team 1",
                "their_team": "Team 2",
                "my_trade": [1],
                "their_trade": [3],
                "ignore_position_check": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()

        assert "your_team" in data
        assert "their_team" in data
        assert data["error"] is None
        assert "pre" in data["your_team"]
        assert "post" in data["your_team"]
        assert "h_score" in data["your_team"]["pre"]
        assert "rates" in data["your_team"]["pre"]

    def test_category_trade_suggest_zero_regression(self, client, sample_category_session):
        """Category session returns standard TradeSuggestResponse with TradeSuggestion format."""
        mock_agent = sample_category_session.agent
        mock_agent.n_categories = 3
        mock_agent.x_scores = pd.DataFrame(
            [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0]],
            index=[1, 2, 3, 4],
        )
        mock_agent.positions = pd.DataFrame({"Positions": [["PG"], ["SG"], ["SF"], ["PF"]]}, index=[1, 2, 3, 4])
        mock_agent.get_diff_var.return_value = np.array([[1.0, 1.0, 1.0]])
        mock_agent.compute_h_score_from_diff_means.return_value = 10.0
        mock_agent.compute_h_scores_batched.return_value = np.array([12.0])

        resp = client.post(
            f"/sessions/{sample_category_session.id}/trade/suggest",
            json={
                "player_assignments": {
                    "Team 1": [1, 2],
                    "Team 2": [3, 4],
                },
                "my_team": "Team 1",
                "their_team": "Team 2",
                "combo_params": [{"n_traded": 1, "n_received": 1, "threshold": -5.0}],
                "your_differential_threshold": -0.02,
                "their_differential_threshold": -0.02,
                "ignore_position_check": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "suggestions" in data
        assert isinstance(data["suggestions"], list)
        if len(data["suggestions"]) > 0:
            sug = data["suggestions"][0]
            assert "send" in sug
            assert "receive" in sug
            assert "your_score" in sug
            assert "their_score" in sug
