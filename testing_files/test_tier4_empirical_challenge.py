"""
Tier 4 Real-World Application Scenarios Adversarial Verification Suite.

Empirically challenges the end-to-end integrated system across Tier 4 scenarios:
1. Standard Weekly Matchup Streamer Churn vs Multi-Slot Hold:
   Simulate a team under an ESPN 7-add limit. Verify that optimize_streaming_acquisitions
   correctly balances daily single-slot churn against multi-slot tactical adds
   (back-to-backs, 3-in-4 nights) and off-night schedule density.
2. 2-for-1 Superstar Consolidation Trade with Streamer Spot Boost:
   Simulate an asymmetric trade where Team A trades two 30-FP/G players for one 50-FP/G
   superstar. Verify that calculate_streamer_spot_boost correctly credits the opened
   roster spot with ~6-7 games/week volume boost rather than dead/0 value.
3. League-Wide Trade Impact:
   Verify that compute_league_trade_impact recalculates projected points, standings rank
   shifts, and competitive balance delta across league teams.
4. Daily vs Weekly Lineup Lock Decisions:
   Verify that weekly optimization accounts for games played (W_i = FP/G * G_i) while
   daily optimization assigns active starters date-by-date.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.points.schedule import ScheduleCalendar
from backend.points.lineups import (
    assign_optimal_daily_lineup,
    assign_optimal_weekly_lineup,
    DailyLineupAssignment,
    WeeklyLineupAssignment,
)
from backend.points.streamer import (
    analyze_schedule_density,
    optimize_streaming_acquisitions,
    StreamerOptimizationResult,
    StreamerSchedulePlan,
    StreamingMove,
)
from backend.points.trade import (
    analyze_points_trade,
    calculate_streamer_spot_boost,
    compute_league_trade_impact,
    search_win_win_points_trades,
    IllegalTradeError,
    LeagueTradeImpact,
    PointsTradeAnalysisResult,
)


# ==============================================================================
# 1. Standard Weekly Matchup Streamer Churn vs Multi-Slot Hold
# ==============================================================================

class TestTier4Scenario1StreamerChurnVsHold:
    """Tier 4 Scenario 1: Streamer Churn vs Multi-Slot Hold under ESPN 7-add limit."""

    @pytest.fixture
    def espn_matchup_schedule(self):
        """Construct a realistic 7-day ESPN matchup week (Monday to Sunday).

        Schedule structure:
        - Monday (2024-11-18): Light off-night (2 NBA games: DEN, PHX)
        - Tuesday (2024-11-19): Light off-night (2 NBA games: DEN, LAL) -> DEN on Back-to-Back!
        - Wednesday (2024-11-20): Heavy congested night (12 NBA games: BOS, MIA, MIL, GSW, NYK, etc.)
        - Thursday (2024-11-21): Light off-night (2 NBA games: PHX, CHI)
        - Friday (2024-11-22): Heavy congested night (12 NBA games: BOS, MIA, MIL, GSW, NYK, etc.)
        - Saturday (2024-11-23): Light off-night (2 NBA games: PHX, MEM) -> PHX plays 3 in 4 (Thu, Fri off, Sat)!
        - Sunday (2024-11-24): Medium slate (4 NBA games: LAL, CHI)
        """
        sched_map = {
            "DEN": ["2024-11-18", "2024-11-19"],               # Mon-Tue Back-to-Back (2 games in 2 days)
            "PHX": ["2024-11-18", "2024-11-21", "2024-11-23"], # Mon, Thu, Sat (3 games across off-nights)
            "LAL": ["2024-11-19", "2024-11-24"],               # Tue, Sun
            "CHI": ["2024-11-21", "2024-11-24"],               # Thu, Sun
            "MEM": ["2024-11-23"],                             # Sat
            # Core roster teams playing on heavy congested nights (Wed & Fri)
            "BOS": ["2024-11-20", "2024-11-22"],
            "MIA": ["2024-11-20", "2024-11-22"],
            "MIL": ["2024-11-20", "2024-11-22"],
            "GSW": ["2024-11-20", "2024-11-22"],
            "NYK": ["2024-11-20", "2024-11-22"],
        }
        player_teams = {
            # Core starters (occupy all 5 slots on Wed & Fri)
            1: "BOS", 2: "MIA", 3: "MIL", 4: "GSW", 5: "NYK",
            # Droppable end-of-bench player (0 games or low impact)
            99: "MEM",
            # Available Free Agent Streamers
            101: "DEN",  # Mon-Tue B2B (2 games)
            102: "PHX",  # Mon, Thu, Sat (3 games)
            103: "LAL",  # Tue, Sun (2 games)
            104: "CHI",  # Thu, Sun (2 games)
            105: "MEM",  # Sat (1 game)
        }
        return ScheduleCalendar.from_dict(sched_map, player_teams=player_teams)

    def test_density_analysis_accurately_detects_off_nights_and_congested_slates(self, espn_matchup_schedule):
        """Schedule density correctly identifies off-nights with open slots vs heavy congested slates."""
        cal = espn_matchup_schedule
        window = [
            "2024-11-18", "2024-11-19", "2024-11-20", "2024-11-21",
            "2024-11-22", "2024-11-23", "2024-11-24",
        ]
        roster = [1, 2, 3, 4, 5, 99]
        positions = {i: ["Util"] for i in roster}
        fp = {1: 35.0, 2: 35.0, 3: 35.0, 4: 35.0, 5: 35.0, 99: 10.0}
        active_slots = {"Util": 5}

        density = analyze_schedule_density(
            current_roster_ids=roster,
            schedule=cal,
            evaluation_window=window,
            player_positions=positions,
            player_fp_per_game=fp,
            active_slots=active_slots,
        )

        # Mon (Nov 18): 0 core starters playing -> 5 open slots -> light off-night
        assert density["2024-11-18"]["open_slots_count"] == 5
        assert density["2024-11-18"]["is_light_slate"] is True

        # Wed (Nov 20): All 5 core starters playing -> 0 open slots -> congested heavy slate
        assert density["2024-11-20"]["open_slots_count"] == 0
        assert density["2024-11-20"]["active_starters_count"] == 5

        # Fri (Nov 22): All 5 core starters playing -> 0 open slots -> congested heavy slate
        assert density["2024-11-22"]["open_slots_count"] == 0

        # Sat (Nov 23): Player 99 playing -> 4 open slots -> light off-night
        assert density["2024-11-23"]["open_slots_count"] == 4
        assert density["2024-11-23"]["is_light_slate"] is True

    def test_streamer_churn_vs_tactical_hold_b2b_efficiency(self, espn_matchup_schedule):
        """Tactical multi-slot adds achieve superior Points Per Add (PPA) by leveraging B2B stints."""
        cal = espn_matchup_schedule
        window = [
            "2024-11-18", "2024-11-19", "2024-11-20", "2024-11-21",
            "2024-11-22", "2024-11-23", "2024-11-24",
        ]
        roster = [1, 2, 3, 4, 5, 99]
        positions = {
            1: ["Util"], 2: ["Util"], 3: ["Util"], 4: ["Util"], 5: ["Util"], 99: ["Util"],
            101: ["Util"], 102: ["Util"], 103: ["Util"], 104: ["Util"], 105: ["Util"],
        }
        fp = {
            1: 35.0, 2: 35.0, 3: 35.0, 4: 35.0, 5: 35.0, 99: 10.0,
            101: 25.0, 102: 25.0, 103: 22.0, 104: 22.0, 105: 20.0,
        }
        active_slots = {"Util": 5}

        # 1. Test Tactical Multi Mode
        res_tactical = optimize_streaming_acquisitions(
            current_roster_ids=roster,
            available_waiver_ids=[101, 102, 103, 104, 105],
            schedule=cal,
            evaluation_window=window,
            player_positions=positions,
            player_fp_per_game=fp,
            active_slots=active_slots,
            max_acquisitions=7,
            drop_candidate_ids=[99],
            mode="tactical_multi",
        )
        plan_tactical = res_tactical.optimal_plan

        # Verify tactical holds player 101 across Mon-Tue B2B (1 acquisition yielding 2 games)
        # or player 102 across Mon, Thu, Sat (1 acquisition yielding 3 games)
        assert plan_tactical.total_acquisitions <= 7
        assert plan_tactical.total_games_streamed > plan_tactical.total_acquisitions
        # Points per add must exceed average single-game FP (25.0) because of multi-game stints
        assert plan_tactical.points_per_add > 25.0
        assert plan_tactical.net_delta_fp > 0.0

        # 2. Test Daily Churn Mode
        res_churn = optimize_streaming_acquisitions(
            current_roster_ids=roster,
            available_waiver_ids=[101, 102, 103, 104, 105],
            schedule=cal,
            evaluation_window=window,
            player_positions=positions,
            player_fp_per_game=fp,
            active_slots=active_slots,
            max_acquisitions=7,
            drop_candidate_ids=[99],
            mode="daily_churn",
        )
        plan_churn = res_churn.optimal_plan
        assert plan_churn.total_games_streamed == plan_churn.total_acquisitions
        # In daily churn, each add yields exactly 1 game, so PPA == average FP/G of streamed games
        assert np.isclose(plan_churn.points_per_add, plan_churn.total_projected_fp / plan_churn.total_acquisitions)

        # 3. Compare: Tactical mode achieves higher PPA than churn
        assert plan_tactical.points_per_add > plan_churn.points_per_add

    def test_streamer_crowded_slate_avoidance_displaces_nobody(self, espn_matchup_schedule):
        """Streamers are NOT acquired for heavy slates when they cannot crack the active lineup."""
        cal = espn_matchup_schedule
        window = ["2024-11-20", "2024-11-21"]  # Wed (congested, 5 stars playing) and Thu (light, 0 stars playing)

        # On Wed: 5 stars (35.0 FP each) fill all 5 slots. Candidate 101 has 22.0 FP.
        # Candidate 101 cannot displace any 35.0 FP star on Wed.
        # On Thu: 0 stars playing. Candidate 101 has open slot.
        roster = [1, 2, 3, 4, 5, 99]
        positions = {i: ["Util"] for i in [1, 2, 3, 4, 5, 99, 101]}
        fp = {1: 35.0, 2: 35.0, 3: 35.0, 4: 35.0, 5: 35.0, 99: 10.0, 101: 22.0}
        active_slots = {"Util": 5}

        # Register candidate 101 playing on both Wed and Thu via calendar loader
        cal._load_from_dict({"CUSTOM": ["2024-11-20", "2024-11-21"]})
        cal.register_player_team(101, "CUSTOM")

        res = optimize_streaming_acquisitions(
            current_roster_ids=roster,
            available_waiver_ids=[101],
            schedule=cal,
            evaluation_window=window,
            player_positions=positions,
            player_fp_per_game=fp,
            active_slots=active_slots,
            max_acquisitions=2,
            drop_candidate_ids=[99],
            mode="daily_churn",
        )

        plan = res.optimal_plan
        # On Wed: candidate 101 provides 0 marginal gain because all 5 slots have 35.0 FP stars.
        # So churn should NOT waste an acquisition on Wed!
        assert plan.total_acquisitions == 1
        assert plan.moves[0].date == "2024-11-21"  # Only added on Thu off-night!
        assert plan.moves[0].projected_fp == 22.0

    def test_strict_acquisition_limit_enforcement(self, espn_matchup_schedule):
        """Strictly enforces ESPN max_acquisitions cap when many lucrative options exist."""
        cal = espn_matchup_schedule
        window = [
            "2024-11-18", "2024-11-19", "2024-11-20", "2024-11-21",
            "2024-11-22", "2024-11-23", "2024-11-24",
        ]
        roster = [1, 2, 3, 4, 5, 99]
        positions = {i: ["Util"] for i in range(1, 150)}
        fp = {i: 35.0 for i in range(1, 6)}
        fp[99] = 10.0
        for i in range(101, 115):
            fp[i] = 20.0 + (i % 5)

        candidates = list(range(101, 115))

        # Test varying caps: 1, 3, 5, 7
        for cap in [1, 3, 5, 7]:
            res = optimize_streaming_acquisitions(
                current_roster_ids=roster,
                available_waiver_ids=candidates,
                schedule=cal,
                evaluation_window=window,
                player_positions=positions,
                player_fp_per_game=fp,
                active_slots={"Util": 5},
                max_acquisitions=cap,
                drop_candidate_ids=[99],
                mode="auto",
            )
            assert res.total_acquisitions_used <= cap
            assert res.optimal_plan.total_acquisitions <= cap

    def test_multi_slot_concurrent_streaming_two_slots(self):
        """Supports concurrent multi-slot streaming (streaming_slots=2) without player collisions."""
        cal = ScheduleCalendar.from_dict({
            "DEN": ["2024-11-18", "2024-11-19"],  # Slot 1 target
            "PHX": ["2024-11-18", "2024-11-20"],  # Slot 2 target
            "MIA": [],
        }, player_teams={1: "MIA", 2: "MIA", 101: "DEN", 102: "PHX"})

        roster = [1, 2]  # Two droppable players
        positions = {1: ["Util"], 2: ["Util"], 101: ["Util"], 102: ["Util"]}
        fp = {1: 10.0, 2: 10.0, 101: 26.0, 102: 24.0}
        active_slots = {"Util": 2}
        window = ["2024-11-18", "2024-11-19", "2024-11-20"]

        res = optimize_streaming_acquisitions(
            current_roster_ids=roster,
            available_waiver_ids=[101, 102],
            schedule=cal,
            evaluation_window=window,
            player_positions=positions,
            player_fp_per_game=fp,
            active_slots=active_slots,
            max_acquisitions=2,
            streaming_slots=2,
            mode="tactical_multi",
        )

        plan = res.optimal_plan
        assert plan.total_acquisitions == 2
        assert plan.total_games_streamed == 4
        # Slot 1 and Slot 2 both active concurrently
        slots_used = {m.slot_index for m in plan.moves}
        assert 1 in slots_used
        assert 2 in slots_used
        # Total projected FP = (2 * 26) + (2 * 24) = 52 + 48 = 100.0
        assert plan.total_projected_fp == 100.0


# ==============================================================================
# 2. 2-for-1 Superstar Consolidation Trade with Streamer Spot Boost
# ==============================================================================

class TestTier4Scenario2SuperstarConsolidationTrade:
    """Tier 4 Scenario 2: 2-for-1 Superstar Consolidation Trade with Streamer Spot Boost."""

    @pytest.fixture
    def trade_schedule(self):
        """Synthetic schedule where every team plays 3.4 games per 7-day week."""
        sched = {
            "TEAM_A1": ["2024-11-01", "2024-11-03", "2024-11-05"],
            "TEAM_A2": ["2024-11-02", "2024-11-04", "2024-11-06", "2024-11-07"],
            "TEAM_SUPER": ["2024-11-01", "2024-11-03", "2024-11-05", "2024-11-07"],
            "TEAM_OTHER": ["2024-11-02", "2024-11-04", "2024-11-06"],
        }
        player_teams = {
            1: "TEAM_A1", 2: "TEAM_A2", 3: "TEAM_OTHER", 4: "TEAM_OTHER", 5: "TEAM_OTHER",
            10: "TEAM_SUPER", 11: "TEAM_OTHER", 12: "TEAM_OTHER", 13: "TEAM_OTHER", 14: "TEAM_OTHER",
        }
        return ScheduleCalendar.from_dict(sched, player_teams=player_teams)

    def test_calculate_streamer_spot_boost_formula_precision(self):
        """Direct mathematical verification of calculate_streamer_spot_boost formula."""
        window_7d = ["2024-11-01", "2024-11-02", "2024-11-03", "2024-11-04", "2024-11-05", "2024-11-06", "2024-11-07"]

        # 1. Standard 1 week, 1 spot opened, baseline 20.0 FP/G, 6.5 games/week:
        # Boost = 1 spot * 1.0 week * 6.5 games * 20.0 FP/G = 130.0 FP
        boost_1w = calculate_streamer_spot_boost(
            opened_spots=1,
            evaluation_window=window_7d,
            waiver_baseline_fp_per_game=20.0,
            streamer_games_per_week=6.5,
        )
        assert boost_1w == 130.0

        # 2. Multi-week: 14-day window = 2 weeks -> exactly double = 260.0 FP
        window_14d = window_7d + ["2024-11-08", "2024-11-09", "2024-11-10", "2024-11-11", "2024-11-12", "2024-11-13", "2024-11-14"]
        boost_2w = calculate_streamer_spot_boost(
            opened_spots=1,
            evaluation_window=window_14d,
            waiver_baseline_fp_per_game=20.0,
            streamer_games_per_week=6.5,
        )
        assert boost_2w == 260.0

        # 3. Double opened spot (e.g. 3-for-1 trade): 2 spots opened in 1 week = 260.0 FP
        boost_2spots = calculate_streamer_spot_boost(
            opened_spots=2,
            evaluation_window=window_7d,
            waiver_baseline_fp_per_game=20.0,
            streamer_games_per_week=6.5,
        )
        assert boost_2spots == 260.0

        # 4. Streamer multiplier test: multiplier=1.8 over baseline 3.4 games = 6.12 games/wk
        # Boost = 1 * 1.0 * (3.4 * 1.8) * 20.0 = 122.4 FP
        boost_mult = calculate_streamer_spot_boost(
            opened_spots=1,
            evaluation_window=window_7d,
            waiver_baseline_fp_per_game=20.0,
            streamer_multiplier=1.8,
            baseline_games_per_week=3.4,
        )
        assert np.isclose(boost_mult, 122.4)

        # 5. Boundary cases: 0 opened spots, negative opened spots, empty window -> 0.0
        assert calculate_streamer_spot_boost(0, window_7d) == 0.0
        assert calculate_streamer_spot_boost(-1, window_7d) == 0.0
        assert calculate_streamer_spot_boost(1, []) == 0.0

    def test_2v1_superstar_consolidation_trade_analysis_flips_valuation(self, trade_schedule):
        """2-for-1 trade appears negative on raw player FP, but strongly POSITIVE with streamer boost.

        Team A trades:
          Player 1 (30.0 FP/G, 3 games = 90.0 FP)
          Player 2 (30.0 FP/G, 4 games = 120.0 FP)
          Total pre-trade = 210.0 FP
        Team A receives:
          Superstar Player 10 (50.0 FP/G, 4 games = 200.0 FP)
        Raw line-up post = 200.0 FP (Raw deficit of -10.0 FP).

        With 1 opened streamer spot:
          Streamer spot boost = 1 * 1.0 * 6.5 * 20.0 = +130.0 FP!
          Total post-trade = 200.0 + 130.0 = 330.0 FP.
          Net Delta FP = 330.0 - 210.0 = +120.0 FP!
        """
        cal = trade_schedule
        window = [
            "2024-11-01", "2024-11-02", "2024-11-03", "2024-11-04",
            "2024-11-05", "2024-11-06", "2024-11-07",
        ]
        team_a_roster = [1, 2, 3, 4, 5]
        team_b_roster = [10, 11, 12, 13, 14]
        positions = {i: ["Util"] for i in range(1, 20)}
        fp = {
            1: 30.0, 2: 30.0, 3: 20.0, 4: 20.0, 5: 20.0,
            10: 50.0, 11: 22.0, 12: 22.0, 13: 22.0, 14: 22.0,
        }
        active_slots = {"Util": 5}

        res = analyze_points_trade(
            team_a_roster=team_a_roster,
            team_b_roster=team_b_roster,
            team_a_sends=[1, 2],
            team_b_sends=[10],
            schedule=cal,
            evaluation_window=window,
            player_positions=positions,
            player_fp_per_game=fp,
            active_slots=active_slots,
            roster_size_limit=6,
            waiver_baseline_fp_per_game=20.0,
            streamer_games_per_week=6.5,
        )

        assert res.is_legal is True
        # Team A opened spots = len([1, 2]) - len([10]) = 1
        assert res.team_a_opened_spots == 1
        assert res.team_a_streamer_boost == 130.0
        # Without streamer boost, raw post lineup would be less than pre
        raw_delta_a = (res.team_a_post_fp - res.team_a_streamer_boost) - res.team_a_pre_fp
        assert raw_delta_a < 0.0  # Raw loss of -10 FP
        # With streamer boost, net delta is strongly positive!
        assert res.team_a_net_delta_fp > 0.0
        assert res.team_a_net_delta_fp == raw_delta_a + 130.0

        # Team B received 2 players for 1 -> opened spots = 0, streamer boost = 0.0
        assert res.team_b_opened_spots == 0
        assert res.team_b_streamer_boost == 0.0

    def test_3v1_multi_spot_double_streamer_boost(self, trade_schedule):
        """3-for-1 trade opens 2 streaming spots, yielding 2x streamer boost (+260.0 FP)."""
        cal = trade_schedule
        window = [
            "2024-11-01", "2024-11-02", "2024-11-03", "2024-11-04",
            "2024-11-05", "2024-11-06", "2024-11-07",
        ]
        team_a_roster = [1, 2, 3, 4, 5]
        team_b_roster = [10, 11, 12, 13]
        positions = {i: ["Util"] for i in range(1, 20)}
        fp = {
            1: 25.0, 2: 25.0, 3: 25.0, 4: 20.0, 5: 20.0,
            10: 55.0, 11: 20.0, 12: 20.0, 13: 20.0,
        }
        active_slots = {"Util": 5}

        res = analyze_points_trade(
            team_a_roster=team_a_roster,
            team_b_roster=team_b_roster,
            team_a_sends=[1, 2, 3],
            team_b_sends=[10],
            schedule=cal,
            evaluation_window=window,
            player_positions=positions,
            player_fp_per_game=fp,
            active_slots=active_slots,
            roster_size_limit=6,
            waiver_baseline_fp_per_game=20.0,
            streamer_games_per_week=6.5,
        )

        assert res.is_legal is True
        assert res.team_a_opened_spots == 2
        assert res.team_a_streamer_boost == 260.0  # 2 * 130.0 FP
        assert res.team_b_opened_spots == 0
        assert res.team_b_streamer_boost == 0.0


# ==============================================================================
# 3. League-Wide Trade Impact Projections
# ==============================================================================

class TestTier4Scenario3LeagueWideTradeImpact:
    """Tier 4 Scenario 3: League-Wide Trade Impact Projections."""

    @pytest.fixture
    def six_team_league(self):
        """Construct a 6-team league with distinct point tiers."""
        cal = ScheduleCalendar.from_dict({
            "ALL": ["2024-11-01", "2024-11-02", "2024-11-03", "2024-11-04"],
        }, player_teams={i: "ALL" for i in range(1, 50)})

        # 6 teams, 2 players each, 1 active slot (highest FP plays each day)
        all_rosters = {
            "Team_Alpha": [1, 2],    # 45 FP
            "Team_Beta": [11, 12],   # 40 FP
            "Team_Gamma": [21, 22],  # 35 FP
            "Team_Delta": [31, 32],  # 30 FP
            "Team_Epsilon": [41, 42],# 25 FP
            "Team_Zeta": [51, 52],   # 20 FP
        }
        player_positions = {i: ["Util"] for i in range(1, 60)}
        player_fp = {
            1: 45.0, 2: 40.0,
            11: 40.0, 12: 35.0,
            21: 35.0, 22: 30.0,
            31: 30.0, 32: 25.0,
            41: 25.0, 42: 20.0,
            51: 20.0, 52: 15.0,
        }
        return {
            "schedule": cal,
            "window": ["2024-11-01", "2024-11-02", "2024-11-03", "2024-11-04"],
            "rosters": all_rosters,
            "positions": player_positions,
            "fp": player_fp,
            "active_slots": {"Util": 1},
        }

    def test_standings_rank_shifts_and_displaced_teams(self, six_team_league):
        """A blockbuster trade elevates a mid-tier team and displaces non-participating teams downward."""
        ctx = six_team_league
        # Pre-trade point totals over 4 days (1 active slot):
        # Alpha: 4 * 45 = 180 (Rank 1)
        # Beta: 4 * 40 = 160 (Rank 2)
        # Gamma: 4 * 35 = 140 (Rank 3)
        # Delta: 4 * 30 = 120 (Rank 4)
        # Epsilon: 4 * 25 = 100 (Rank 5)
        # Zeta: 4 * 20 = 80 (Rank 6)

        # Team_Delta (Rank 4, 120 pts) trades with Team_Alpha (Rank 1, 180 pts):
        # Team_Delta acquires Player 1 (45 FP) + streamer boost -> shoots up to 190 pts!
        # Team_Alpha drops to 150 pts.
        impact = compute_league_trade_impact(
            all_team_rosters=ctx["rosters"],
            team_a_id="Team_Delta",
            team_b_id="Team_Alpha",
            team_a_pre_fp=120.0,
            team_a_post_fp=190.0,  # Delta rises to 1st place!
            team_b_pre_fp=180.0,
            team_b_post_fp=150.0,  # Alpha drops below Beta (160) into 3rd place!
            schedule=ctx["schedule"],
            evaluation_window=ctx["window"],
            player_positions=ctx["positions"],
            player_fp_per_game=ctx["fp"],
            active_slots=ctx["active_slots"],
        )

        # Pre-trade ranks
        assert impact.pre_trade_ranks["Team_Alpha"] == 1
        assert impact.pre_trade_ranks["Team_Beta"] == 2
        assert impact.pre_trade_ranks["Team_Gamma"] == 3
        assert impact.pre_trade_ranks["Team_Delta"] == 4

        # Post-trade ranks:
        # 1. Delta (190)
        # 2. Beta (160)
        # 3. Alpha (150)
        # 4. Gamma (140)
        assert impact.post_trade_ranks["Team_Delta"] == 1
        assert impact.post_trade_ranks["Team_Beta"] == 2
        assert impact.post_trade_ranks["Team_Alpha"] == 3
        assert impact.post_trade_ranks["Team_Gamma"] == 4

        # Rank shifts:
        # Team_Delta: 4 -> 1 is +3 (rose 3 spots!)
        # Team_Alpha: 1 -> 3 is -2 (fell 2 spots)
        # Team_Beta: 2 -> 2 is 0
        # Team_Gamma: 3 -> 4 is -1 (displaced down 1 spot by Delta's rise!)
        assert impact.rank_shifts["Team_Delta"] == +3
        assert impact.rank_shifts["Team_Alpha"] == -2
        assert impact.rank_shifts["Team_Beta"] == 0
        assert impact.rank_shifts["Team_Gamma"] == -1

    def test_competitive_parity_improvement_vs_disparity_widening(self, six_team_league):
        """Competitive balance delta accurately reflects parity improvement vs disparity widening."""
        ctx = six_team_league

        # 1. Parity Improvement Scenario:
        # Last place Zeta (80 pts) trades with first place Alpha (180 pts), transferring 40 pts to Zeta.
        # Points become: Alpha: 140, Beta: 160, Gamma: 140, Delta: 120, Epsilon: 100, Zeta: 120.
        # Points standard deviation MUST shrink -> competitive_balance_delta < 0
        parity_impact = compute_league_trade_impact(
            all_team_rosters=ctx["rosters"],
            team_a_id="Team_Zeta",
            team_b_id="Team_Alpha",
            team_a_pre_fp=80.0,
            team_a_post_fp=120.0,
            team_b_pre_fp=180.0,
            team_b_post_fp=140.0,
            schedule=ctx["schedule"],
            evaluation_window=ctx["window"],
            player_positions=ctx["positions"],
            player_fp_per_game=ctx["fp"],
            active_slots=ctx["active_slots"],
        )
        assert parity_impact.post_trade_std < parity_impact.pre_trade_std
        assert parity_impact.competitive_balance_delta < 0.0
        assert parity_impact.post_trade_spread < parity_impact.pre_trade_spread

        # 2. Disparity Widening Scenario:
        # First place Alpha (180 pts) acquires more from last place Zeta (80 pts), rising to 220 pts while Zeta falls to 40 pts.
        # Standard deviation MUST grow -> competitive_balance_delta > 0
        disparity_impact = compute_league_trade_impact(
            all_team_rosters=ctx["rosters"],
            team_a_id="Team_Alpha",
            team_b_id="Team_Zeta",
            team_a_pre_fp=180.0,
            team_a_post_fp=220.0,
            team_b_pre_fp=80.0,
            team_b_post_fp=40.0,
            schedule=ctx["schedule"],
            evaluation_window=ctx["window"],
            player_positions=ctx["positions"],
            player_fp_per_game=ctx["fp"],
            active_slots=ctx["active_slots"],
        )
        assert disparity_impact.post_trade_std > disparity_impact.pre_trade_std
        assert disparity_impact.competitive_balance_delta > 0.0
        assert disparity_impact.post_trade_spread > disparity_impact.pre_trade_spread

    def test_deterministic_rank_tiebreaking_with_identical_points(self, six_team_league):
        """Tiebreaker between teams with identical points is deterministic (sorted by team key)."""
        ctx = six_team_league
        # Set two teams to identical 150.0 FP
        impact = compute_league_trade_impact(
            all_team_rosters=ctx["rosters"],
            team_a_id="Team_Delta",
            team_b_id="Team_Alpha",
            team_a_pre_fp=120.0,
            team_a_post_fp=160.0,  # Delta ties with Beta at 160.0 FP!
            team_b_pre_fp=180.0,
            team_b_post_fp=160.0,  # Alpha also ties at 160.0 FP!
            schedule=ctx["schedule"],
            evaluation_window=ctx["window"],
            player_positions=ctx["positions"],
            player_fp_per_game=ctx["fp"],
            active_slots=ctx["active_slots"],
        )
        # All three teams (Team_Alpha, Team_Beta, Team_Delta) have 160.0 points.
        # Ranks must be distinct integers (1, 2, 3) without collision or missing ranks.
        ranks = [impact.post_trade_ranks["Team_Alpha"], impact.post_trade_ranks["Team_Beta"], impact.post_trade_ranks["Team_Delta"]]
        assert sorted(ranks) == [1, 2, 3]


# ==============================================================================
# 4. Daily vs Weekly Lineup Lock Decisions
# ==============================================================================

class TestTier4Scenario4DailyVsWeeklyLineupLocks:
    """Tier 4 Scenario 4: Daily vs Weekly Lineup Lock Decisions."""

    def test_weekly_volume_vs_daily_efficiency_inversion(self):
        """Demonstrates mathematical divergence between Weekly Matchup Lock and Daily Lineup Locks.

        Setup:
        Active Slot: 1 PG slot.
        Player 1 (Elite efficiency, low volume): 45.0 FP/G, plays 2 games this week.
          Weekly volume = 45.0 * 2 = 90.0 FP.
        Player 2 (Good efficiency, high volume): 28.0 FP/G, plays 4 games this week.
          Weekly volume = 28.0 * 4 = 112.0 FP.

        1. Weekly Lock Mode (W_i = FP/G * G_i):
           Player 2 MUST be assigned starter (112.0 FP > 90.0 FP).
           Player 1 MUST be benched.
           Total Weekly FP = 112.0 FP.

        2. Daily Lock Mode:
           Day 1 (Both play): Player 1 (45.0) starts over Player 2 (28.0) -> +45.0 FP
           Day 2 (Only Player 2 plays): Player 2 (28.0) starts -> +28.0 FP
           Day 3 (Neither plays): Both 0.0 FP
           Day 4 (Both play): Player 1 (45.0) starts over Player 2 (28.0) -> +45.0 FP
           Day 5 (Only Player 2 plays): Player 2 (28.0) starts -> +28.0 FP
           Total Daily Starting FP captured across week = 45 + 28 + 0 + 45 + 28 = 146.0 FP!

        This proves the daily solver captures 146.0 FP while weekly lock locks in 112.0 FP.
        """
        roster = [1, 2]
        positions = {1: ["PG"], 2: ["PG"]}
        fp_per_game = {1: 45.0, 2: 28.0}
        games_in_week = {1: 2, 2: 4}
        slots = {"PG": 1}

        # ── 1. Weekly Lock Solver ─────────────────────────────────────────────
        weekly_res = assign_optimal_weekly_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp_per_game=fp_per_game,
            player_games_in_week=games_in_week,
            active_slots=slots,
        )

        assert isinstance(weekly_res, WeeklyLineupAssignment)
        # Player 2 (112.0 FP) MUST be selected for PG1 over Player 1 (90.0 FP)
        assert weekly_res.active_assignments["PG1"] == 2
        assert weekly_res.bench_players == [1]
        assert weekly_res.total_expected_fp == 112.0
        assert weekly_res.player_weekly_fp[1] == 90.0
        assert weekly_res.player_weekly_fp[2] == 112.0

        # ── 2. Daily Lock Solver across 5 days ────────────────────────────────
        daily_schedule = [
            {1, 2},  # Day 1: both play
            {2},     # Day 2: only Player 2 plays
            set(),   # Day 3: neither plays
            {1, 2},  # Day 4: both play
            {2},     # Day 5: only Player 2 plays
        ]

        daily_points_sum = 0.0
        starters_per_day = []

        for day_playing in daily_schedule:
            daily_res = assign_optimal_daily_lineup(
                roster_player_ids=roster,
                player_positions=positions,
                player_fp=fp_per_game,
                active_slots=slots,
                playing_today=day_playing,
            )
            daily_points_sum += daily_res.total_expected_fp
            starters_per_day.append(daily_res.active_assignments["PG1"])

        # Day 1: Player 1 starts
        assert starters_per_day[0] == 1
        # Day 2: Player 2 starts
        assert starters_per_day[1] == 2
        # Day 3: None starts
        assert starters_per_day[2] is None
        # Day 4: Player 1 starts
        assert starters_per_day[3] == 1
        # Day 5: Player 2 starts
        assert starters_per_day[4] == 2

        # Daily sum captures 146.0 FP vs Weekly's 112.0 FP!
        assert daily_points_sum == 146.0
        assert daily_points_sum > weekly_res.total_expected_fp

    def test_weekly_zero_game_superstar_rationally_benched(self):
        """A 60.0 FP/G superstar with 0 games in the week is benched for a 15.0 FP/G player with 2 games."""
        roster = [1, 2]
        positions = {1: ["C"], 2: ["C"]}
        fp_per_game = {1: 60.0, 2: 15.0}
        games_in_week = {1: 0, 2: 2}  # Player 1 has 0 games ($W=0$), Player 2 has 2 games ($W=30$)
        slots = {"C": 1}

        res = assign_optimal_weekly_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp_per_game=fp_per_game,
            player_games_in_week=games_in_week,
            active_slots=slots,
        )

        assert res.active_assignments["C1"] == 2
        assert 1 in res.bench_players
        assert res.total_expected_fp == 30.0

    def test_weekly_multi_position_flex_cascading(self):
        """Weekly optimizer correctly cascades multi-position players across base and flex slots."""
        roster = [1, 2, 3]
        # Player 1 is pure PG (35 FP/G, 3 games = 105 FP)
        # Player 2 is PG/SG dual (30 FP/G, 4 games = 120 FP)
        # Player 3 is pure SG (25 FP/G, 3 games = 75 FP)
        positions = {1: ["PG"], 2: ["PG", "SG"], 3: ["SG"]}
        fp = {1: 35.0, 2: 30.0, 3: 25.0}
        games = {1: 3, 2: 4, 3: 3}
        # 1 PG slot and 1 SG slot
        slots = {"PG": 1, "SG": 1}

        res = assign_optimal_weekly_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp_per_game=fp,
            player_games_in_week=games,
            active_slots=slots,
        )

        # Player 2 (dual PG/SG, 120 FP) can fill SG so Player 1 (PG, 105 FP) can start at PG.
        # Player 3 (SG, 75 FP) is benched.
        assert res.active_assignments["PG1"] == 1
        assert res.active_assignments["SG1"] == 2
        assert res.bench_players == [3]
        assert res.total_expected_fp == 105.0 + 120.0  # 225.0 FP

    def test_locked_suboptimal_player_enforced_in_weekly_lineup(self):
        """When a user locks a player into a weekly slot, that player is enforced and cannot be benched."""
        roster = [1, 2]
        positions = {1: ["PG"], 2: ["PG"]}
        fp = {1: 50.0, 2: 20.0}
        games = {1: 4, 2: 3}  # Player 1 = 200 FP, Player 2 = 60 FP
        slots = {"PG": 1}

        # Force lock Player 2 into PG1
        res = assign_optimal_weekly_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp_per_game=fp,
            player_games_in_week=games,
            active_slots=slots,
            locked_assignments={"PG1": 2},
        )

        assert res.active_assignments["PG1"] == 2
        assert res.bench_players == [1]
        assert res.total_expected_fp == 60.0


# ==============================================================================
# 5. Adversarial Stress & Boundary Challenge Tests
# ==============================================================================

class TestTier4AdversarialStressAndBoundaries:
    """Adversarial stress harness for extreme, hostile, and boundary inputs."""

    def test_adversarial_streamer_empty_and_ineligible_candidates(self):
        """Engine handles candidates with 0 games, negative FP, and ineligible positions gracefully."""
        cal = ScheduleCalendar.from_dict({
            "BOS": ["2024-11-01"],
            "LAL": ["2024-11-02"],
        }, player_teams={1: "BOS", 101: "LAL", 102: "LAL", 103: "BOS"})

        roster = [1]
        positions = {1: ["C"], 101: ["C"], 102: ["C"], 103: ["PG"]}
        # 101 has negative FP, 102 has 0 FP, 103 is PG (ineligible for C slot)
        fp = {1: 10.0, 101: -15.0, 102: 0.0, 103: 30.0}
        slots = {"C": 1}

        res = optimize_streaming_acquisitions(
            current_roster_ids=roster,
            available_waiver_ids=[101, 102, 103],
            schedule=cal,
            evaluation_window=["2024-11-01", "2024-11-02"],
            player_positions=positions,
            player_fp_per_game=fp,
            active_slots=slots,
            max_acquisitions=5,
            mode="auto",
        )

        # None of the candidates can produce positive starting points in C slot
        assert res.total_acquisitions_used == 0
        assert res.total_streamed_games == 0
        assert res.total_net_delta_fp == 0.0

    def test_adversarial_streamer_drop_candidates_invalid_ids(self):
        """Engine falls back safely when drop_candidate_ids contains IDs not on the roster."""
        cal = ScheduleCalendar.from_dict({"BOS": ["2024-11-01"]}, player_teams={1: "BOS", 101: "BOS"})
        roster = [1]
        positions = {1: ["PG"], 101: ["PG"]}
        fp = {1: 15.0, 101: 30.0}
        slots = {"PG": 1}

        res = optimize_streaming_acquisitions(
            current_roster_ids=roster,
            available_waiver_ids=[101],
            schedule=cal,
            evaluation_window=["2024-11-01"],
            player_positions=positions,
            player_fp_per_game=fp,
            active_slots=slots,
            max_acquisitions=2,
            drop_candidate_ids=[9999, 8888],  # Invalid IDs not on roster!
            mode="daily_churn",
        )

        assert res.optimal_plan.total_acquisitions == 1
        # Dropped player defaults to the core roster player 1
        assert res.optimal_plan.moves[0].drop_player_id == 1

    def test_adversarial_trade_analyzer_reverse_1v2_and_balanced_2v2(self):
        """Verifies streamer boost semantics in balanced 2v2 and reverse 1v2 trades."""
        cal = ScheduleCalendar.from_dict({"ALL": ["2024-11-01"]}, player_teams={i: "ALL" for i in range(1, 10)})
        positions = {i: ["Util"] for i in range(1, 10)}
        fp = {i: 25.0 for i in range(1, 10)}
        slots = {"Util": 5}
        window = ["2024-11-01"]

        # 1. Balanced 2v2 trade: 0 opened spots on both sides -> 0.0 boost on both sides
        res_2v2 = analyze_points_trade(
            team_a_roster=[1, 2, 3],
            team_b_roster=[4, 5, 6],
            team_a_sends=[1, 2],
            team_b_sends=[4, 5],
            schedule=cal,
            evaluation_window=window,
            player_positions=positions,
            player_fp_per_game=fp,
            active_slots=slots,
            roster_size_limit=5,
        )
        assert res_2v2.team_a_opened_spots == 0
        assert res_2v2.team_a_streamer_boost == 0.0
        assert res_2v2.team_b_opened_spots == 0
        assert res_2v2.team_b_streamer_boost == 0.0

        # 2. Reverse 1v2 trade: Team A sends 1, receives 2. Team B sends 2, receives 1.
        # Team A opened spots = 0, Team B opened spots = 1 -> Team B gets streamer boost!
        res_1v2 = analyze_points_trade(
            team_a_roster=[1, 2, 3],
            team_b_roster=[4, 5, 6],
            team_a_sends=[1],
            team_b_sends=[4, 5],
            schedule=cal,
            evaluation_window=window,
            player_positions=positions,
            player_fp_per_game=fp,
            active_slots=slots,
            roster_size_limit=5,
        )
        assert res_1v2.team_a_opened_spots == 0
        assert res_1v2.team_a_streamer_boost == 0.0
        assert res_1v2.team_b_opened_spots == 1
        assert res_1v2.team_b_streamer_boost > 0.0

    def test_adversarial_trade_unprojected_nan_and_none_players(self):
        """Trade evaluation is resilient against players with None or NaN projections."""
        cal = ScheduleCalendar.from_dict({"BOS": ["2024-11-01"]}, player_teams={1: "BOS", 2: "BOS", 3: "BOS"})
        positions = {1: ["Util"], 2: ["Util"], 3: ["Util"]}
        fp = {1: None, 2: np.nan, 3: 30.0}
        slots = {"Util": 2}

        res = analyze_points_trade(
            team_a_roster=[1, 2],
            team_b_roster=[3],
            team_a_sends=[1, 2],
            team_b_sends=[3],
            schedule=cal,
            evaluation_window=["2024-11-01"],
            player_positions=positions,
            player_fp_per_game=fp,
            active_slots=slots,
            roster_size_limit=5,
        )

        assert res.is_legal is True
        assert np.isfinite(res.team_a_net_delta_fp)
        assert np.isfinite(res.team_b_net_delta_fp)
        assert res.team_a_opened_spots == 1

    def test_adversarial_league_impact_minimal_two_team_league(self):
        """League trade impact functions correctly on a minimal 2-team league."""
        cal = ScheduleCalendar.from_dict({"BOS": ["2024-11-01"]}, player_teams={1: "BOS", 2: "BOS"})
        impact = compute_league_trade_impact(
            all_team_rosters={"Team_1": [1], "Team_2": [2]},
            team_a_id="Team_1",
            team_b_id="Team_2",
            team_a_pre_fp=30.0,
            team_a_post_fp=50.0,
            team_b_pre_fp=50.0,
            team_b_post_fp=30.0,
            schedule=cal,
            evaluation_window=["2024-11-01"],
            player_positions={1: ["Util"], 2: ["Util"]},
            player_fp_per_game={1: 30.0, 2: 50.0},
            active_slots={"Util": 1},
        )

        # Team 1 was 2nd (30), now 1st (50) -> rank shift = +1
        # Team 2 was 1st (50), now 2nd (30) -> rank shift = -1
        assert impact.pre_trade_ranks["Team_1"] == 2
        assert impact.post_trade_ranks["Team_1"] == 1
        assert impact.rank_shifts["Team_1"] == +1

        assert impact.pre_trade_ranks["Team_2"] == 1
        assert impact.post_trade_ranks["Team_2"] == 2
        assert impact.rank_shifts["Team_2"] == -1

        # Spread remains 20 (50 - 30) -> competitive_balance_delta is 0
        assert np.isclose(impact.competitive_balance_delta, 0.0)

    def test_adversarial_weekly_lineup_all_zero_games_and_nan_fp(self):
        """Weekly lineup optimizer safely benches players with 0 games or NaN projections."""
        roster = [1, 2, 3]
        positions = {1: ["PG"], 2: ["SG"], 3: ["C"]}
        fp = {1: np.nan, 2: -10.0, 3: 40.0}
        games = {1: 3, 2: 3, 3: 0}  # Player 3 has 40 FP/G but 0 games!
        slots = {"PG": 1, "SG": 1, "C": 1}

        res = assign_optimal_weekly_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp_per_game=fp,
            player_games_in_week=games,
            active_slots=slots,
        )

        # Player 3 with 0 games must have 0 weekly FP and get benched
        assert res.player_weekly_fp[3] == 0.0
        assert 3 in res.bench_players
        # Total expected FP is 0.0 (no positive projected players played)
        assert res.total_expected_fp == 0.0

