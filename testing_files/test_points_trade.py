"""
Milestone 3 Test Suite: Trade Analyzer, Win-Win Suggestions & League Impact.

Comprehensive offline unit tests covering:
1. 1v1 trade analysis (Delta FP calculation for both sides, multi-day schedule displacement).
2. 2v1 trade analysis with streamer spot valuation adjustment (opened roster spot volume boost).
3. Illegal trade rejection (roster size limits, non-rostered players, positional caps, raise_on_illegal).
4. League-wide trade impact (standings changes, points totals, competitive balance std dev & spread).
5. Win-win trade suggestion search (1v1, 2v1, 2v2 formats, threshold filtering, pruning, sorting).
6. Boundary value analysis and real-world application scenarios.
"""

import pytest
import numpy as np

from backend.points.schedule import ScheduleCalendar
from backend.points.trade import (
    PointsTradeAnalysisResult,
    PointsTradeSuggestion,
    LeagueTradeImpact,
    IllegalTradeError,
    analyze_points_trade,
    search_win_win_points_trades,
    calculate_streamer_spot_boost,
    validate_trade_legality,
)


# ── Shared Test Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def sample_schedule():
    """Deterministic schedule: 7-day week (2024-11-04 to 2024-11-10)."""
    team_schedules = {
        "BOS": ["2024-11-04", "2024-11-06", "2024-11-08", "2024-11-10"],  # 4 games
        "LAL": ["2024-11-05", "2024-11-07", "2024-11-09"],                # 3 games
        "GSW": ["2024-11-04", "2024-11-05", "2024-11-08", "2024-11-10"],  # 4 games
        "MIA": ["2024-11-05", "2024-11-06", "2024-11-09"],                # 3 games
    }
    player_teams = {
        # Team A players (BOS & LAL)
        1: "BOS", 2: "BOS", 3: "LAL", 4: "LAL", 5: "BOS",
        # Team B players (GSW & MIA)
        10: "GSW", 11: "GSW", 12: "MIA", 13: "MIA", 14: "GSW",
        # League partner players
        20: "BOS", 21: "LAL", 22: "GSW", 23: "MIA",
        30: "BOS", 31: "LAL", 32: "GSW", 33: "MIA",
    }
    return ScheduleCalendar.from_dict(team_schedules, player_teams=player_teams)


@pytest.fixture
def eval_window():
    return ["2024-11-04", "2024-11-05", "2024-11-06", "2024-11-07", "2024-11-08", "2024-11-09", "2024-11-10"]


@pytest.fixture
def standard_positions():
    return {
        1: ["PG"], 2: ["SG"], 3: ["SF"], 4: ["PF"], 5: ["C"],
        10: ["PG"], 11: ["SG"], 12: ["SF"], 13: ["PF"], 14: ["C"],
        20: ["PG"], 21: ["SG"], 22: ["SF"], 23: ["C"],
        30: ["PG"], 31: ["SG"], 32: ["PF"], 33: ["C"],
    }


@pytest.fixture
def standard_fp():
    return {
        1: 45.0, 2: 35.0, 3: 30.0, 4: 25.0, 5: 20.0,
        10: 40.0, 11: 38.0, 12: 28.0, 13: 26.0, 14: 22.0,
        20: 34.0, 21: 29.0, 22: 27.0, 23: 24.0,
        30: 36.0, 31: 31.0, 32: 25.0, 33: 23.0,
    }


@pytest.fixture
def active_slots_5():
    return {"PG": 1, "SG": 1, "SF": 1, "PF": 1, "C": 1}


# ==============================================================================
# 1. 1v1 Trade Analysis Tests
# ==============================================================================

class TestPointsTrade1v1:
    """Tests for 1v1 trade evaluations and Delta FP calculations."""

    def test_1v1_balanced_starter_swap(
        self, sample_schedule, eval_window, standard_positions,
        standard_fp, active_slots_5,
    ):
        """Team A swaps Player 1 (PG, 45 FP, 4 games = 180 FP) for Team B's Player 10 (PG, 40 FP, 4 games = 160 FP)."""
        team_a = [1, 2, 3, 4, 5]
        team_b = [10, 11, 12, 13, 14]

        res = analyze_points_trade(
            team_a_roster=team_a,
            team_b_roster=team_b,
            team_a_sends=[1],
            team_b_sends=[10],
            schedule=sample_schedule,
            evaluation_window=eval_window,
            player_positions=standard_positions,
            player_fp_per_game=standard_fp,
            active_slots=active_slots_5,
            roster_size_limit=5,
        )

        assert res.is_legal is True
        assert res.rejection_reason is None
        assert res.team_a_opened_spots == 0
        assert res.team_b_opened_spots == 0
        assert res.team_a_streamer_boost == 0.0
        assert res.team_b_streamer_boost == 0.0

        # Team A lost 20 FP (160 - 180 = -20)
        assert res.team_a_delta_fp == -20.0
        assert res.team_a_net_delta_fp == -20.0

        # Team B gained 20 FP (180 - 160 = +20)
        assert res.team_b_delta_fp == 20.0
        assert res.team_b_net_delta_fp == 20.0

    def test_1v1_schedule_asymmetry_impact(
        self, sample_schedule, eval_window, standard_positions,
        standard_fp, active_slots_5,
    ):
        """Player 2 (BOS, 4 games, 35 FP = 140 FP) swapped for Player 12 (MIA, 3 games, 35 FP = 105 FP).

        Equal FP/G but unequal schedule volume: Team A loses 1 game = -35 FP, Team B gains 1 game = +35 FP.
        """
        # Give Player 12 and Player 2 dual eligibility so SF/SG remains covered for both teams
        pos = dict(standard_positions)
        pos[2] = ["SG", "SF"]
        pos[12] = ["SG", "SF"]
        fp = dict(standard_fp)
        fp[12] = 35.0

        team_a = [1, 2, 3, 4, 5]
        team_b = [10, 11, 12, 13, 14]

        res = analyze_points_trade(
            team_a_roster=team_a,
            team_b_roster=team_b,
            team_a_sends=[2],
            team_b_sends=[12],
            schedule=sample_schedule,
            evaluation_window=eval_window,
            player_positions=pos,
            player_fp_per_game=fp,
            active_slots=active_slots_5,
            roster_size_limit=5,
        )

        assert res.is_legal is True
        assert res.team_a_delta_fp == -35.0
        assert res.team_b_delta_fp == 35.0

    def test_1v1_lateral_identical_swap(
        self, sample_schedule, eval_window, standard_positions,
        standard_fp, active_slots_5,
    ):
        """Identical players produce 0.0 Delta FP on both sides."""
        fp = dict(standard_fp)
        fp[10] = fp[1]  # Same FP
        # Both BOS (4 games)
        sample_schedule.register_player_team(10, "BOS")

        res = analyze_points_trade(
            team_a_roster=[1, 2, 3, 4, 5],
            team_b_roster=[10, 11, 12, 13, 14],
            team_a_sends=[1],
            team_b_sends=[10],
            schedule=sample_schedule,
            evaluation_window=eval_window,
            player_positions=standard_positions,
            player_fp_per_game=fp,
            active_slots=active_slots_5,
            roster_size_limit=5,
        )

        assert res.team_a_delta_fp == 0.0
        assert res.team_b_delta_fp == 0.0

    def test_1v1_with_bench_displacement(self, sample_schedule, eval_window, standard_positions, standard_fp):
        """Lineup with bench: higher scoring acquired player displaces low scorer to bench."""
        # 2 active slots: PG, Util. Team A has Player 1 (45 FP), Player 2 (20 FP).
        # Team B has Player 10 (40 FP). Team A swaps Player 2 (20 FP) for Player 10 (40 FP).
        roster_a = [1, 2]
        roster_b = [10]
        slots = {"PG": 1, "Util": 1}
        pos = {1: ["PG"], 2: ["Util"], 10: ["Util"]}
        fp = {1: 45.0, 2: 20.0, 10: 40.0}

        res = analyze_points_trade(
            team_a_roster=roster_a,
            team_b_roster=roster_b,
            team_a_sends=[2],
            team_b_sends=[10],
            schedule=sample_schedule,
            evaluation_window=eval_window,
            player_positions=pos,
            player_fp_per_game=fp,
            active_slots=slots,
            roster_size_limit=2,
        )

        assert res.is_legal is True
        assert res.team_a_delta_fp > 0.0
        assert res.team_b_delta_fp < 0.0


# ==============================================================================
# 2. Streamer Spot Valuation Adjustment Tests (R2/R4)
# ==============================================================================

class TestStreamerSpotValuation:
    """Tests for opened streamer spot opportunity value in unbalanced trades."""

    def test_calculate_streamer_spot_boost_formula(self, eval_window):
        """Verify streamer boost calculation under standard 6.5 games/week and 20.0 waiver FP/G."""
        # 1 spot, 7 days = 1 week: 1 * 1.0 * 6.5 * 20.0 = 130.0 FP
        boost_1 = calculate_streamer_spot_boost(
            opened_spots=1,
            evaluation_window=eval_window,
            waiver_baseline_fp_per_game=20.0,
            streamer_games_per_week=6.5,
        )
        assert boost_1 == 130.0

        # 2 spots, 7 days: 2 * 1.0 * 6.5 * 20.0 = 260.0 FP
        boost_2 = calculate_streamer_spot_boost(
            opened_spots=2,
            evaluation_window=eval_window,
            waiver_baseline_fp_per_game=20.0,
            streamer_games_per_week=6.5,
        )
        assert boost_2 == 260.0

        # 0 spots: 0.0 FP
        boost_0 = calculate_streamer_spot_boost(
            opened_spots=0,
            evaluation_window=eval_window,
            waiver_baseline_fp_per_game=20.0,
        )
        assert boost_0 == 0.0

    def test_streamer_multiplier_over_baseline(self, eval_window):
        """Verify multiplier option (e.g. 1.75x baseline 3.4 games = 5.95 games/week)."""
        boost_mult = calculate_streamer_spot_boost(
            opened_spots=1,
            evaluation_window=eval_window,
            waiver_baseline_fp_per_game=20.0,
            streamer_multiplier=1.75,
            baseline_games_per_week=3.4,
        )
        # 1 * 1.0 * (1.75 * 3.4) * 20.0 = 5.95 * 20.0 = 119.0 FP
        assert boost_mult == 119.0

    def test_2v1_superstar_consolidation_with_streamer_boost(
        self, sample_schedule, eval_window, standard_positions,
        standard_fp, active_slots_5,
    ):
        """Team A sends two solid starters [1, 2] (45 + 35 FP/G) for Team B superstar [10] (60 FP/G).

        Team A opens 1 streamer spot, receiving the streaming volume boost.
        Without the streamer spot, Team A would lose points (1 superstar vs 2 starters).
        With the streamer spot, Team A accounts for the added ~6.5 streaming games.
        """
        # Setup: Team A has 6 players, Team B has 5 players. Roster limit = 7.
        team_a = [1, 2, 3, 4, 5, 20]
        team_b = [10, 11, 12, 13, 14]
        fp = dict(standard_fp)
        fp[10] = 60.0  # Superstar

        # Ensure all eligible for active slots
        pos = dict(standard_positions)
        pos[20] = ["PG", "SG"]

        res = analyze_points_trade(
            team_a_roster=team_a,
            team_b_roster=team_b,
            team_a_sends=[1, 2],
            team_b_sends=[10],
            schedule=sample_schedule,
            evaluation_window=eval_window,
            player_positions=pos,
            player_fp_per_game=fp,
            active_slots=active_slots_5,
            roster_size_limit=7,
            waiver_baseline_fp_per_game=20.0,
            streamer_games_per_week=6.5,
        )

        assert res.is_legal is True
        assert res.team_a_opened_spots == 1
        assert res.team_b_opened_spots == 0
        assert res.team_a_streamer_boost == 130.0
        assert res.team_b_streamer_boost == 0.0

        # Team A's net delta includes the 130.0 streamer boost
        assert res.team_a_net_delta_fp == res.team_a_delta_fp
        assert res.team_a_post_fp > 0.0

    def test_1v2_reverse_streamer_boost(
        self, sample_schedule, eval_window, standard_positions,
        standard_fp, active_slots_5,
    ):
        """Team B sends 2 players and receives 1, opening a streamer spot for Team B."""
        team_a = [1, 2, 3, 4, 5]
        team_b = [10, 11, 12, 13, 14, 22]
        pos = dict(standard_positions)
        pos[22] = ["PG", "SG"]

        res = analyze_points_trade(
            team_a_roster=team_a,
            team_b_roster=team_b,
            team_a_sends=[1],
            team_b_sends=[10, 11],
            schedule=sample_schedule,
            evaluation_window=eval_window,
            player_positions=pos,
            player_fp_per_game=standard_fp,
            active_slots=active_slots_5,
            roster_size_limit=7,
            waiver_baseline_fp_per_game=22.0,
            streamer_games_per_week=6.0,
        )

        assert res.team_a_opened_spots == 0
        assert res.team_b_opened_spots == 1
        assert res.team_a_streamer_boost == 0.0
        # Team B: 1 * 1.0 * 6.0 * 22.0 = 132.0 FP
        assert res.team_b_streamer_boost == 132.0

    def test_multi_week_streamer_scaling(self, sample_schedule, standard_positions, standard_fp, active_slots_5):
        """Evaluation window of 14 days (2 weeks) scales streamer boost proportionally."""
        window_14 = [f"2024-11-{d:02d}" for d in range(4, 18)]
        team_a = [1, 2, 3, 4, 5, 20]
        team_b = [10, 11, 12, 13, 14]
        pos = dict(standard_positions)
        pos[20] = ["PG", "SG"]

        res = analyze_points_trade(
            team_a_roster=team_a,
            team_b_roster=team_b,
            team_a_sends=[1, 2],
            team_b_sends=[10],
            schedule=sample_schedule,
            evaluation_window=window_14,
            player_positions=pos,
            player_fp_per_game=standard_fp,
            active_slots=active_slots_5,
            roster_size_limit=7,
            waiver_baseline_fp_per_game=20.0,
            streamer_games_per_week=6.5,
        )

        # 2 weeks * 6.5 games/wk * 20 FP = 260.0 FP
        assert res.team_a_streamer_boost == 260.0


# ==============================================================================
# 3. Illegal Trade Rejection Tests
# ==============================================================================

class TestIllegalTradeRejection:
    """Tests for validating legality and rejecting improper trade proposals."""

    def test_reject_non_rostered_player_team_a(
        self, sample_schedule, eval_window, standard_positions,
        standard_fp, active_slots_5,
    ):
        """Player 999 is not on Team A roster."""
        res = analyze_points_trade(
            team_a_roster=[1, 2, 3, 4, 5],
            team_b_roster=[10, 11, 12, 13, 14],
            team_a_sends=[999],
            team_b_sends=[10],
            schedule=sample_schedule,
            evaluation_window=eval_window,
            player_positions=standard_positions,
            player_fp_per_game=standard_fp,
            active_slots=active_slots_5,
            roster_size_limit=5,
        )

        assert res.is_legal is False
        assert "not on Team A roster" in res.rejection_reason

    def test_reject_non_rostered_player_team_b(
        self, sample_schedule, eval_window, standard_positions,
        standard_fp, active_slots_5,
    ):
        """Player 888 is not on Team B roster."""
        res = analyze_points_trade(
            team_a_roster=[1, 2, 3, 4, 5],
            team_b_roster=[10, 11, 12, 13, 14],
            team_a_sends=[1],
            team_b_sends=[888],
            schedule=sample_schedule,
            evaluation_window=eval_window,
            player_positions=standard_positions,
            player_fp_per_game=standard_fp,
            active_slots=active_slots_5,
            roster_size_limit=5,
        )

        assert res.is_legal is False
        assert "not on Team B roster" in res.rejection_reason

    def test_reject_roster_size_overflow(
        self, sample_schedule, eval_window, standard_positions,
        standard_fp, active_slots_5,
    ):
        """2-for-1 trade where receiving team already has full roster of 5 exceeds limit."""
        res = analyze_points_trade(
            team_a_roster=[1, 2, 3, 4, 5],
            team_b_roster=[10, 11, 12, 13, 14],
            team_a_sends=[1, 2],
            team_b_sends=[10],
            schedule=sample_schedule,
            evaluation_window=eval_window,
            player_positions=standard_positions,
            player_fp_per_game=standard_fp,
            active_slots=active_slots_5,
            roster_size_limit=5,  # Team B would have 6 players
        )

        assert res.is_legal is False
        assert "exceeds roster size limit" in res.rejection_reason

    def test_raise_on_illegal_flag(
        self, sample_schedule, eval_window, standard_positions,
        standard_fp, active_slots_5,
    ):
        """When raise_on_illegal=True, an IllegalTradeError (subclass of ValueError) is raised."""
        with pytest.raises(IllegalTradeError, match="exceeds roster size limit"):
            analyze_points_trade(
                team_a_roster=[1, 2, 3, 4, 5],
                team_b_roster=[10, 11, 12, 13, 14],
                team_a_sends=[1, 2],
                team_b_sends=[10],
                schedule=sample_schedule,
                evaluation_window=eval_window,
                player_positions=standard_positions,
                player_fp_per_game=standard_fp,
                active_slots=active_slots_5,
                roster_size_limit=5,
                raise_on_illegal=True,
            )

    def test_reject_duplicate_player_in_trade(
        self, sample_schedule, eval_window, standard_positions,
        standard_fp, active_slots_5,
    ):
        """Same player listed twice in team_a_sends."""
        res = analyze_points_trade(
            team_a_roster=[1, 2, 3, 4, 5],
            team_b_roster=[10, 11, 12, 13, 14],
            team_a_sends=[1, 1],
            team_b_sends=[10],
            schedule=sample_schedule,
            evaluation_window=eval_window,
            player_positions=standard_positions,
            player_fp_per_game=standard_fp,
            active_slots=active_slots_5,
            roster_size_limit=6,
        )

        assert res.is_legal is False
        assert "Duplicate player IDs" in res.rejection_reason

    def test_reject_empty_trade(
        self, sample_schedule, eval_window, standard_positions,
        standard_fp, active_slots_5,
    ):
        """Trade with empty send lists on both sides."""
        res = analyze_points_trade(
            team_a_roster=[1, 2, 3],
            team_b_roster=[10, 11, 12],
            team_a_sends=[],
            team_b_sends=[],
            schedule=sample_schedule,
            evaluation_window=eval_window,
            player_positions=standard_positions,
            player_fp_per_game=standard_fp,
            active_slots={"Util": 2},
            roster_size_limit=5,
        )

        assert res.is_legal is False
        assert "cannot be empty" in res.rejection_reason

    def test_reject_positional_cap_violation(
        self, sample_schedule, eval_window, standard_positions,
        standard_fp, active_slots_5,
    ):
        """Post-trade roster exceeds position cap (e.g. max 1 Center)."""
        pos = dict(standard_positions)
        pos[10] = ["C"]  # Player 10 is C
        # Team A already has Player 5 (C). If receiving Player 10 and sending Player 1 (PG), Team A has 2 C.
        res = analyze_points_trade(
            team_a_roster=[1, 2, 3, 4, 5],
            team_b_roster=[10, 11, 12, 13, 14],
            team_a_sends=[1],
            team_b_sends=[10],
            schedule=sample_schedule,
            evaluation_window=eval_window,
            player_positions=pos,
            player_fp_per_game=standard_fp,
            active_slots=active_slots_5,
            roster_size_limit=5,
            position_caps={"C": 1},
        )

        assert res.is_legal is False
        assert "exceeds positional cap for C" in res.rejection_reason

    def test_reject_unfillable_required_active_slot(
        self, sample_schedule, eval_window, standard_positions,
        standard_fp, active_slots_5,
    ):
        """Team A sends away only Center (Player 5) for PG, leaving 0 Centers on full roster."""
        res = analyze_points_trade(
            team_a_roster=[1, 2, 3, 4, 5],
            team_b_roster=[10, 11, 12, 13, 14],
            team_a_sends=[5],  # only C
            team_b_sends=[10],  # PG
            schedule=sample_schedule,
            evaluation_window=eval_window,
            player_positions=standard_positions,
            player_fp_per_game=standard_fp,
            active_slots=active_slots_5,
            roster_size_limit=5,
        )

        assert res.is_legal is False
        assert "cannot satisfy required active slot C" in res.rejection_reason

    def test_validate_trade_legality_helper_directly(self, standard_positions, active_slots_5):
        """Directly verify validate_trade_legality helper function."""
        is_legal, reason = validate_trade_legality(
            team_a_roster=[1, 2],
            team_b_roster=[10, 11],
            team_a_sends=[1],
            team_b_sends=[10],
            roster_size_limit=2,
            player_positions=standard_positions,
            active_slots=active_slots_5,
            ignore_position_check=True,
        )
        assert is_legal is True
        assert reason is None


# ==============================================================================
# 4. League-Wide Trade Impact Projections Tests (R2)
# ==============================================================================

class TestLeagueTradeImpact:
    """Tests for projecting standings shifts and competitive balance impact."""

    def test_standings_rank_shifts_and_points(
        self, sample_schedule, eval_window, standard_positions,
        standard_fp, active_slots_5,
    ):
        """4-team league: Team A trades with Team B, altering points and shifting ranks."""
        all_rosters = {
            "Team_A": [1, 2, 3, 4, 5],       # ~650 FP
            "Team_B": [10, 11, 12, 13, 14],   # ~600 FP
            "Team_C": [20, 21, 22, 23, 1],    # middle
            "Team_D": [30, 31, 32, 33, 2],    # middle
        }

        # Team B acquires Player 1 (star) from Team A for Player 14 (lower tier)
        # Give Player 1 C eligibility to keep rosters valid
        pos = dict(standard_positions)
        pos[1] = ["PG", "C"]
        pos[14] = ["C", "PG"]

        res = analyze_points_trade(
            team_a_roster=all_rosters["Team_A"],
            team_b_roster=all_rosters["Team_B"],
            team_a_sends=[1],
            team_b_sends=[14],
            schedule=sample_schedule,
            evaluation_window=eval_window,
            player_positions=pos,
            player_fp_per_game=standard_fp,
            active_slots=active_slots_5,
            roster_size_limit=5,
            all_team_rosters=all_rosters,
            team_a_id="Team_A",
            team_b_id="Team_B",
        )

        assert res.league_impact is not None
        impact = res.league_impact
        assert isinstance(impact, LeagueTradeImpact)

        # Check all teams exist in impact dictionaries
        assert set(impact.pre_trade_points.keys()) == set(all_rosters.keys())
        assert set(impact.post_trade_points.keys()) == set(all_rosters.keys())
        assert set(impact.pre_trade_ranks.keys()) == set(all_rosters.keys())
        assert set(impact.post_trade_ranks.keys()) == set(all_rosters.keys())
        assert set(impact.rank_shifts.keys()) == set(all_rosters.keys())

        # Uninvolved teams (Team C and Team D) have identical pre and post points
        assert impact.pre_trade_points["Team_C"] == impact.post_trade_points["Team_C"]
        assert impact.pre_trade_points["Team_D"] == impact.post_trade_points["Team_D"]

        # Team B points increased, Team A points decreased
        assert impact.post_trade_points["Team_B"] > impact.pre_trade_points["Team_B"]
        assert impact.post_trade_points["Team_A"] < impact.pre_trade_points["Team_A"]

        # Rank shifts: positive means improved rank
        assert impact.rank_shifts["Team_B"] >= 0

        # Competitive balance metrics
        assert impact.pre_trade_std > 0.0
        assert impact.post_trade_std > 0.0
        assert impact.pre_trade_spread >= 0.0
        assert impact.post_trade_spread >= 0.0
        assert impact.competitive_balance_delta == round(impact.post_trade_std - impact.pre_trade_std, 4)

    def test_competitive_parity_improvement_scenario(
        self, sample_schedule, eval_window, standard_positions,
        standard_fp, active_slots_5,
    ):
        """Trade transferring points from dominant leader to last place team narrows spread and reduces std."""
        # Leader: Team A (very high FP), Last: Team B (very low FP)
        fp = dict(standard_fp)
        fp[1] = 60.0
        fp[2] = 55.0
        fp[10] = 15.0

        all_rosters = {
            "Leader": [1, 2, 3, 4, 5],
            "Trailer": [10, 11, 12, 13, 14],
        }

        res = analyze_points_trade(
            team_a_roster=all_rosters["Leader"],
            team_b_roster=all_rosters["Trailer"],
            team_a_sends=[1],
            team_b_sends=[10],
            schedule=sample_schedule,
            evaluation_window=eval_window,
            player_positions=standard_positions,
            player_fp_per_game=fp,
            active_slots=active_slots_5,
            roster_size_limit=5,
            all_team_rosters=all_rosters,
            team_a_id="Leader",
            team_b_id="Trailer",
        )

        impact = res.league_impact
        assert impact is not None
        # Spread and standard deviation should decrease (more balanced)
        assert impact.post_trade_std < impact.pre_trade_std
        assert impact.post_trade_spread < impact.pre_trade_spread
        assert impact.competitive_balance_delta < 0.0  # std decreased -> parity improved


# ==============================================================================
# 5. Win-Win Trade Suggestion Engine Tests (R3)
# ==============================================================================

class TestWinWinTradeSuggestions:
    """Tests for searching and ranking win-win trade proposals."""

    def test_search_1v1_win_win_trade(self, sample_schedule, eval_window, standard_positions, standard_fp):
        """User needs a Center, Partner needs a Point Guard. 1v1 positional complement trade benefits both."""
        # Active slots: 1 PG, 1 C
        slots = {"PG": 1, "C": 1}
        # User has two PGs (Player 1 @ 40 FP, Player 2 @ 35 FP) and NO Center.
        # Partner has two Cs (Player 10 @ 40 FP, Player 11 @ 35 FP) and NO Point Guard.
        user_roster = [1, 2]
        partner_roster = [10, 11]

        pos = {
            1: ["PG"], 2: ["PG"],
            10: ["C"], 11: ["C"],
        }
        fp = {
            1: 40.0, 2: 35.0,
            10: 40.0, 11: 35.0,
        }

        all_rosters = {
            "UserTeam": user_roster,
            "PartnerTeam": partner_roster,
        }

        suggestions = search_win_win_points_trades(
            user_team_id="UserTeam",
            all_team_rosters=all_rosters,
            schedule=sample_schedule,
            evaluation_window=eval_window,
            player_positions=pos,
            player_fp_per_game=fp,
            active_slots=slots,
            roster_size_limit=2,
            min_user_gain=10.0,
            min_partner_gain=10.0,
            trade_formats=["1v1"],
        )

        assert len(suggestions) > 0
        top = suggestions[0]
        assert top.trade_format == "1v1"
        assert top.user_gain >= 10.0
        assert top.partner_gain >= 10.0
        assert top.total_gain >= 20.0
        assert top.partner_team_id == "PartnerTeam"
        assert len(top.user_sends) == 1
        assert len(top.partner_sends) == 1

    def test_search_2v1_consolidation_win_win(self, sample_schedule, eval_window, standard_positions, standard_fp):
        """2v1 trade search: User trades two bench players for partner starter, utilizing opened streamer spot."""
        slots = {"Util": 2}
        # User has 3 players: Player 1 (50 FP), Player 2 (25 FP), Player 3 (20 FP).
        # Partner has 2 players: Player 10 (35 FP), Player 11 (30 FP).
        # User sends [2, 3] (total 45 FP) for Partner's [10] (35 FP).
        # User opens 1 spot: streamer boost = 130 FP.
        # Partner now has 3 players to fill slots.
        user_roster = [1, 2, 3]
        partner_roster = [10, 11]

        pos = {1: ["Util"], 2: ["Util"], 3: ["Util"], 10: ["Util"], 11: ["Util"]}
        fp = {1: 50.0, 2: 25.0, 3: 20.0, 10: 35.0, 11: 30.0}

        all_rosters = {
            "UserTeam": user_roster,
            "PartnerTeam": partner_roster,
        }

        suggestions = search_win_win_points_trades(
            user_team_id="UserTeam",
            all_team_rosters=all_rosters,
            schedule=sample_schedule,
            evaluation_window=eval_window,
            player_positions=pos,
            player_fp_per_game=fp,
            active_slots=slots,
            roster_size_limit=4,
            min_user_gain=5.0,
            min_partner_gain=5.0,
            trade_formats=["2v1"],
            waiver_baseline_fp_per_game=20.0,
        )

        assert len(suggestions) > 0
        suggestion_formats = {s.trade_format for s in suggestions}
        assert "2v1" in suggestion_formats or "1v2" in suggestion_formats
        for s in suggestions:
            assert s.user_gain >= 5.0
            assert s.partner_gain >= 5.0

    def test_search_2v2_win_win_trade(self, sample_schedule, eval_window):
        """2v2 trade search finds multi-player complementary package."""
        slots = {"PG": 1, "C": 1}
        user_roster = [1, 2, 3]
        partner_roster = [10, 11, 12]

        pos = {
            1: ["PG"], 2: ["PG"], 3: ["C"],
            10: ["C"], 11: ["C"], 12: ["PG"],
        }
        fp = {
            1: 45.0, 2: 35.0, 3: 15.0,
            10: 45.0, 11: 35.0, 12: 15.0,
        }

        all_rosters = {
            "UserTeam": user_roster,
            "PartnerTeam": partner_roster,
        }

        suggestions = search_win_win_points_trades(
            user_team_id="UserTeam",
            all_team_rosters=all_rosters,
            schedule=sample_schedule,
            evaluation_window=eval_window,
            player_positions=pos,
            player_fp_per_game=fp,
            active_slots=slots,
            roster_size_limit=3,
            min_user_gain=5.0,
            min_partner_gain=5.0,
            trade_formats=["2v2"],
        )

        assert len(suggestions) > 0
        top = suggestions[0]
        assert top.trade_format == "2v2"
        assert len(top.user_sends) == 2
        assert len(top.partner_sends) == 2
        assert top.user_gain >= 5.0
        assert top.partner_gain >= 5.0

    def test_search_prunes_trades_violating_roster_limit(self, sample_schedule, eval_window):
        """Trades that would exceed roster size limit for partner are strictly pruned."""
        slots = {"Util": 1}
        # Partner already at max limit (3 players). A 2v1 trade would push partner to 4 players.
        user_roster = [1, 2, 3]
        partner_roster = [10, 11, 12]
        pos = {1: ["Util"], 2: ["Util"], 3: ["Util"], 10: ["Util"], 11: ["Util"], 12: ["Util"]}
        fp = {1: 50.0, 2: 40.0, 3: 30.0, 10: 25.0, 11: 20.0, 12: 15.0}

        all_rosters = {"User": user_roster, "Partner": partner_roster}

        suggestions = search_win_win_points_trades(
            user_team_id="User",
            all_team_rosters=all_rosters,
            schedule=sample_schedule,
            evaluation_window=eval_window,
            player_positions=pos,
            player_fp_per_game=fp,
            active_slots=slots,
            roster_size_limit=3,  # Strict limit = 3
            trade_formats=["2v1"],
        )

        # None of the 2-for-1 trades should be returned since partner cannot hold 4 players
        for s in suggestions:
            assert len(partner_roster) - len(s.partner_sends) + len(s.user_sends) <= 3

    def test_search_empty_when_no_win_win_exists(
        self, sample_schedule, eval_window, standard_positions,
        standard_fp, active_slots_5,
    ):
        """When threshold is impossibly high, search returns empty list."""
        all_rosters = {
            "User": [1, 2, 3, 4, 5],
            "Partner": [10, 11, 12, 13, 14],
        }

        suggestions = search_win_win_points_trades(
            user_team_id="User",
            all_team_rosters=all_rosters,
            schedule=sample_schedule,
            evaluation_window=eval_window,
            player_positions=standard_positions,
            player_fp_per_game=standard_fp,
            active_slots=active_slots_5,
            roster_size_limit=5,
            min_user_gain=10000.0,  # Impossibly high threshold
            min_partner_gain=10000.0,
        )

        assert suggestions == []

    def test_search_invalid_user_team_id_raises_value_error(
        self, sample_schedule, eval_window, standard_positions,
        standard_fp, active_slots_5,
    ):
        """Unknown user team raises ValueError."""
        with pytest.raises(ValueError, match="not found in all_team_rosters"):
            search_win_win_points_trades(
                user_team_id="NONEXISTENT",
                all_team_rosters={"TeamA": [1, 2]},
                schedule=sample_schedule,
                evaluation_window=eval_window,
                player_positions=standard_positions,
                player_fp_per_game=standard_fp,
                active_slots=active_slots_5,
                roster_size_limit=5,
            )


# ==============================================================================
# 6. Serialization & Property Helper Tests
# ==============================================================================

class TestTradeSerialization:
    """Tests for to_dict serialization and convenience properties."""

    def test_analysis_result_to_dict(
        self, sample_schedule, eval_window, standard_positions,
        standard_fp, active_slots_5,
    ):
        res = analyze_points_trade(
            team_a_roster=[1, 2, 3, 4, 5],
            team_b_roster=[10, 11, 12, 13, 14],
            team_a_sends=[1],
            team_b_sends=[10],
            schedule=sample_schedule,
            evaluation_window=eval_window,
            player_positions=standard_positions,
            player_fp_per_game=standard_fp,
            active_slots=active_slots_5,
            roster_size_limit=5,
        )

        d = res.to_dict()
        assert isinstance(res, PointsTradeAnalysisResult)
        assert isinstance(d, dict)
        assert "team_a_delta_fp" in d
        assert "team_b_delta_fp" in d
        assert "team_a_streamer_boost" in d
        assert d["is_legal"] is True

    def test_suggestion_properties_and_to_dict(self):
        s = PointsTradeSuggestion(
            partner_team_id="Team2",
            user_sends=[1, 2],
            partner_sends=[10],
            user_gain=15.5,
            partner_gain=8.2,
            trade_format="2v1",
        )

        assert s.send == [1, 2]
        assert s.receive == [10]
        assert s.user_delta_fp == 15.5
        assert s.partner_delta_fp == 8.2
        assert s.total_gain == 23.7

        d = s.to_dict()
        assert d["partner_team_id"] == "Team2"
        assert d["total_gain"] == 23.7
        assert d["trade_format"] == "2v1"


# ==============================================================================
# 7. Boundary, Multi-Spot & Real-World Application Scenarios (Tier 3 & 4)
# ==============================================================================

class TestBoundaryAndRealWorldScenarios:
    """Comprehensive boundary, pairwise, and real-world scenario tests."""

    def test_3v1_multi_spot_double_streamer_boost(
        self, sample_schedule, eval_window, standard_positions,
        standard_fp, active_slots_5,
    ):
        """Team A sends 3 players and receives 1, opening 2 streamer spots (2x volume boost = 260.0 FP)."""
        # Team A has 7 players, Team B has 5 players. Roster limit = 8.
        team_a = [1, 2, 3, 4, 5, 20, 21]
        team_b = [10, 11, 12, 13, 14]
        pos = dict(standard_positions)
        pos[20] = ["PG", "SG"]
        pos[21] = ["SF", "PF"]

        res = analyze_points_trade(
            team_a_roster=team_a,
            team_b_roster=team_b,
            team_a_sends=[1, 2, 3],
            team_b_sends=[10],
            schedule=sample_schedule,
            evaluation_window=eval_window,
            player_positions=pos,
            player_fp_per_game=standard_fp,
            active_slots=active_slots_5,
            roster_size_limit=8,
            waiver_baseline_fp_per_game=20.0,
            streamer_games_per_week=6.5,
        )

        assert res.is_legal is True
        assert res.team_a_opened_spots == 2
        assert res.team_b_opened_spots == 0
        # 2 spots * 1.0 wk * 6.5 * 20.0 = 260.0 FP
        assert res.team_a_streamer_boost == 260.0
        assert res.team_b_streamer_boost == 0.0

    def test_league_impact_none_when_omitted(
        self, sample_schedule, eval_window, standard_positions,
        standard_fp, active_slots_5,
    ):
        """When all_team_rosters is omitted/None, league_impact is cleanly None."""
        res = analyze_points_trade(
            team_a_roster=[1, 2, 3, 4, 5],
            team_b_roster=[10, 11, 12, 13, 14],
            team_a_sends=[1],
            team_b_sends=[10],
            schedule=sample_schedule,
            evaluation_window=eval_window,
            player_positions=standard_positions,
            player_fp_per_game=standard_fp,
            active_slots=active_slots_5,
            roster_size_limit=5,
            all_team_rosters=None,
        )

        assert res.league_impact is None

    def test_league_impact_tiebreak_deterministic_ranks(
        self, sample_schedule, eval_window, standard_positions,
        standard_fp, active_slots_5,
    ):
        """Teams with tied projected points have deterministic rank ordering by team ID."""
        all_rosters = {
            "Team_Z": [1, 2, 3, 4, 5],
            "Team_A": [1, 2, 3, 4, 5],
        }

        res = analyze_points_trade(
            team_a_roster=all_rosters["Team_Z"],
            team_b_roster=all_rosters["Team_A"],
            team_a_sends=[1],
            team_b_sends=[1],
            schedule=sample_schedule,
            evaluation_window=eval_window,
            player_positions=standard_positions,
            player_fp_per_game=standard_fp,
            active_slots=active_slots_5,
            roster_size_limit=5,
            all_team_rosters=all_rosters,
            team_a_id="Team_Z",
            team_b_id="Team_A",
        )

        impact = res.league_impact
        assert impact is not None
        # Tied scores: Team_A ranks ahead of Team_Z by alphabetical tiebreak
        assert impact.pre_trade_ranks["Team_A"] == 1
        assert impact.pre_trade_ranks["Team_Z"] == 2

    def test_search_format_filter_strictly_enforced(self, sample_schedule, eval_window):
        """Passing trade_formats=['1v1'] only returns 1v1 suggestions."""
        slots = {"PG": 1, "C": 1}
        user_roster = [1, 2, 3]
        partner_roster = [10, 11, 12]
        pos = {1: ["PG"], 2: ["PG"], 3: ["C"], 10: ["C"], 11: ["C"], 12: ["PG"]}
        fp = {1: 45.0, 2: 35.0, 3: 15.0, 10: 45.0, 11: 35.0, 12: 15.0}

        all_rosters = {"User": user_roster, "Partner": partner_roster}

        suggestions_1v1 = search_win_win_points_trades(
            user_team_id="User",
            all_team_rosters=all_rosters,
            schedule=sample_schedule,
            evaluation_window=eval_window,
            player_positions=pos,
            player_fp_per_game=fp,
            active_slots=slots,
            roster_size_limit=4,
            min_user_gain=1.0,
            min_partner_gain=1.0,
            trade_formats=["1v1"],
        )

        assert len(suggestions_1v1) > 0
        assert all(s.trade_format == "1v1" for s in suggestions_1v1)

    def test_search_max_suggestions_cap(self, sample_schedule, eval_window):
        """max_suggestions limits returned list length."""
        slots = {"Util": 1}
        user_roster = [1, 2, 3, 4]
        partner_roster = [10, 11, 12, 13]
        pos = {i: ["Util"] for i in user_roster + partner_roster}
        # Give diverse values so combinations yield varied gains
        fp = {1: 30.0, 2: 28.0, 3: 26.0, 4: 24.0, 10: 29.0, 11: 27.0, 12: 25.0, 13: 23.0}

        all_rosters = {"User": user_roster, "Partner": partner_roster}

        suggestions = search_win_win_points_trades(
            user_team_id="User",
            all_team_rosters=all_rosters,
            schedule=sample_schedule,
            evaluation_window=eval_window,
            player_positions=pos,
            player_fp_per_game=fp,
            active_slots=slots,
            roster_size_limit=5,
            min_user_gain=-50.0,  # permissive to get many matches
            min_partner_gain=-50.0,
            trade_formats=["1v1"],
            max_suggestions=3,
        )

        assert len(suggestions) <= 3

    def test_boundary_single_player_rosters(self, sample_schedule, eval_window):
        """Boundary test: 1-player rosters trade single player."""
        res = analyze_points_trade(
            team_a_roster=[1],
            team_b_roster=[10],
            team_a_sends=[1],
            team_b_sends=[10],
            schedule=sample_schedule,
            evaluation_window=eval_window,
            player_positions={1: ["Util"], 10: ["Util"]},
            player_fp_per_game={1: 30.0, 10: 40.0},
            active_slots={"Util": 1},
            roster_size_limit=1,
        )

        assert res.is_legal is True
        assert res.team_a_delta_fp > 0.0
        assert res.team_b_delta_fp < 0.0

    def test_boundary_unprojected_none_fp_safety(self, sample_schedule, eval_window):
        """Unprojected player with None FP is safely benched and does not crash trade analyzer."""
        res = analyze_points_trade(
            team_a_roster=[1, 2],
            team_b_roster=[10, 11],
            team_a_sends=[1],
            team_b_sends=[10],
            schedule=sample_schedule,
            evaluation_window=eval_window,
            player_positions={1: ["Util"], 2: ["Util"], 10: ["Util"], 11: ["Util"]},
            player_fp_per_game={1: None, 2: 30.0, 10: 25.0, 11: 20.0},
            active_slots={"Util": 1},
            roster_size_limit=2,
        )

        assert res.is_legal is True
        assert np.isfinite(res.team_a_delta_fp)
        assert np.isfinite(res.team_b_delta_fp)

    def test_real_world_superstar_consolidation_deadline_scenario(self, sample_schedule, eval_window):
        """Tier 4 Real-World Application Scenario 2:

        2-for-1 trade: User trades solid starters Player 1 (32.0 FP/G) and Player 2 (28.0 FP/G)
        for Superstar Player 10 (55.0 FP/G).
        Without streamer spot, trade would appear negative:
          Pre-trade Player 1 (4 games * 32 = 128) + Player 2 (4 games * 28 = 112) = 240 FP.
          Post-trade Player 10 (4 games * 55 = 220) = 220 FP (Raw loss of 20 FP).
        With streamer spot (opened 1 spot * 6.5 games/wk * 20.0 FP/G = 130.0 FP):
          Total post = 220 + 130 = 350 FP (+110 FP net gain!).
        Demonstrates why superstar consolidation wins points leagues when streaming is enabled.
        """
        user_roster = [1, 2, 3, 4, 5, 20]
        partner_roster = [10, 11, 12, 13, 14]
        slots = {"PG": 1, "SG": 1, "SF": 1, "PF": 1, "C": 1}

        pos = {
            1: ["PG"], 2: ["SG"], 3: ["SF"], 4: ["PF"], 5: ["C"], 20: ["PG", "SG"],
            10: ["PG"], 11: ["SG"], 12: ["SF"], 13: ["PF"], 14: ["C"],
        }
        fp = {
            1: 32.0, 2: 28.0, 3: 25.0, 4: 22.0, 5: 20.0, 20: 18.0,
            10: 55.0, 11: 30.0, 12: 26.0, 13: 24.0, 14: 21.0,
        }

        res = analyze_points_trade(
            team_a_roster=user_roster,
            team_b_roster=partner_roster,
            team_a_sends=[1, 2],
            team_b_sends=[10],
            schedule=sample_schedule,
            evaluation_window=eval_window,
            player_positions=pos,
            player_fp_per_game=fp,
            active_slots=slots,
            roster_size_limit=7,
            waiver_baseline_fp_per_game=20.0,
            streamer_games_per_week=6.5,
        )

        assert res.is_legal is True
        assert res.team_a_opened_spots == 1
        assert res.team_a_streamer_boost == 130.0
        assert res.team_a_net_delta_fp > 0.0

    def test_real_world_deadline_search_elevates_standings(self, sample_schedule, eval_window):
        """Tier 4 Real-World Application Scenario 3:

        League-wide search finds a win-win trade that elevates User Team in standings.
        """
        slots = {"PG": 1, "C": 1}
        pos = {
            1: ["PG"], 2: ["PG"], 3: ["C"],
            10: ["C"], 11: ["C"], 12: ["PG"],
            20: ["PG", "C"], 21: ["PG", "C"],
        }
        fp = {
            1: 45.0, 2: 35.0, 3: 15.0,
            10: 45.0, 11: 35.0, 12: 15.0,
            20: 30.0, 21: 30.0,
        }

        all_rosters = {
            "Team_Dominant": [20, 21],  # 60 FP
            "User_Team": [1, 2],        # 45 FP (only 1 PG slot active, C empty)
            "Partner_Team": [10, 11],   # 45 FP (only 1 C slot active, PG empty)
        }

        # Search win-win trades
        suggestions = search_win_win_points_trades(
            user_team_id="User_Team",
            all_team_rosters=all_rosters,
            schedule=sample_schedule,
            evaluation_window=eval_window,
            player_positions=pos,
            player_fp_per_game=fp,
            active_slots=slots,
            roster_size_limit=2,
            min_user_gain=10.0,
            min_partner_gain=10.0,
            trade_formats=["1v1"],
        )

        assert len(suggestions) > 0
        best_trade = suggestions[0]

        # Analyze best trade with league-wide impact
        analysis = analyze_points_trade(
            team_a_roster=all_rosters["User_Team"],
            team_b_roster=all_rosters["Partner_Team"],
            team_a_sends=best_trade.user_sends,
            team_b_sends=best_trade.partner_sends,
            schedule=sample_schedule,
            evaluation_window=eval_window,
            player_positions=pos,
            player_fp_per_game=fp,
            active_slots=slots,
            roster_size_limit=2,
            all_team_rosters=all_rosters,
            team_a_id="User_Team",
            team_b_id="Partner_Team",
        )

        impact = analysis.league_impact
        assert impact is not None
        # User team improved in points and ranks
        assert impact.post_trade_points["User_Team"] > impact.pre_trade_points["User_Team"]
        assert impact.rank_shifts["User_Team"] >= 0
