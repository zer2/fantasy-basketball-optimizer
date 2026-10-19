"""
Milestone 1 Adversarial Challenge & Stress Test Suite.

Authored by Challenger 1 (Empirical Challenger) to verify ScheduleCalendar and
Lineup Optimizer implementations against edge conditions and discover bugs empirically.
"""

import datetime
import time
import numpy as np
import pytest

from backend.points.schedule import (
    ScheduleCalendar,
    generate_synthetic_nba_schedule,
    NBAGame,
    normalize_date,
    normalize_team,
    date_range,
    NBA_TEAMS,
)
from backend.points.lineups import (
    assign_optimal_daily_lineup,
    assign_optimal_weekly_lineup,
    compute_lineup_fp,
    DailyLineupAssignment,
    WeeklyLineupAssignment,
)


class TestScheduleCalendarChallenges:
    """Adversarial challenge tests for ScheduleCalendar."""

    def test_leap_year_february_29_handling(self):
        """Calendar correctly handles leap year Feb 29 dates and back-to-backs."""
        cal = ScheduleCalendar(team_schedules={
            "BOS": ["2024-02-28", "2024-02-29", "2024-03-01"],
            "LAL": ["2025-02-28", "2025-03-01"],
        })
        leap_dates = cal.get_team_game_dates("BOS", "2024-02-27", "2024-03-02")
        assert leap_dates == ["2024-02-28", "2024-02-29", "2024-03-01"]
        assert cal.is_team_playing_on_date("BOS", "2024-02-29") is True
        b2b = cal.get_team_back_to_backs("BOS", "2024-02-27", "2024-03-02")
        assert b2b == [("2024-02-28", "2024-02-29"), ("2024-02-29", "2024-03-01")]

    def test_empty_and_inverted_date_ranges(self):
        """Inverted date ranges return empty list without crash."""
        cal = ScheduleCalendar(team_schedules={"BOS": ["2024-10-21"]})
        assert cal.get_team_game_dates("BOS", "2024-10-25", "2024-10-20") == []
        assert date_range("2024-10-25", "2024-10-20") == []
        assert cal.get_team_game_count("BOS", "2024-10-25", "2024-10-20") == 0

    def test_nonexistent_and_malformed_teams(self):
        """Non-existent team codes return empty results; malformed codes normalized."""
        cal = ScheduleCalendar(team_schedules={"BOS": ["2024-10-21"]})
        assert cal.get_team_game_dates("NON_EXISTENT", "2024-10-01", "2024-10-31") == []
        assert cal.is_team_playing_on_date("XYZ", "2024-10-21") is False
        assert cal.get_team_back_to_backs("FAKE", "2024-10-01", "2024-10-31") == []
        # Whitespace and lowercase normalization
        assert cal.get_team_game_dates("  bos  ", "2024-10-21", "2024-10-21") == ["2024-10-21"]

    def test_player_registry_edge_cases(self):
        """Unassigned, removed, or non-playing team players return safe defaults."""
        cal = ScheduleCalendar(team_schedules={"BOS": ["2024-10-21"]})
        cal.register_player_team(101, "BOS")
        cal.register_player_team(102, "NON_PLAYING")
        assert cal.is_player_playing_on_date(101, "2024-10-21") is True
        assert cal.is_player_playing_on_date(102, "2024-10-21") is False
        assert cal.is_player_playing_on_date(999, "2024-10-21") is False
        cal.remove_player(101)
        assert cal.is_player_playing_on_date(101, "2024-10-21") is False

    def test_synthetic_schedule_all_constraints(self):
        """Synthetic generator produces 30 teams, zero 3-in-3s, ~3.4 GPW, valid slates."""
        sched, games = generate_synthetic_nba_schedule(start_date="2024-10-21", weeks=25, seed=42)
        assert len(sched) == 30

        # Check: Zero 3-in-3s across all 30 teams for 25 weeks
        violations = []
        for team, dates in sched.items():
            dt_objs = [datetime.date.fromisoformat(d) for d in dates]
            for i in range(len(dt_objs) - 2):
                if (dt_objs[i + 1] - dt_objs[i]).days == 1 and (dt_objs[i + 2] - dt_objs[i + 1]).days == 1:
                    violations.append((team, dates[i], dates[i + 1], dates[i + 2]))
        assert len(violations) == 0, f"Found 3-in-3 violations: {violations}"

        # Check: Realistic density ~3.4 games per team per week
        for team, dates in sched.items():
            gpw = len(dates) / 25.0
            assert 3.2 <= gpw <= 3.6, f"Team {team} GPW {gpw} outside [3.2, 3.6]"

        # Check: Slate day-of-week distribution (heavy Wed/Fri > light Tue/Thu)
        dow_games = {i: 0 for i in range(7)}
        for g in games:
            dow = datetime.date.fromisoformat(g.date).weekday()
            dow_games[dow] += 1
        assert dow_games[2] > dow_games[1]  # Wed > Tue
        assert dow_games[4] > dow_games[3]  # Fri > Thu


class TestLineupOptimizerChallenges:
    """Adversarial challenge tests for Daily & Weekly Lineup Optimizers."""

    def test_degenerate_all_players_injured(self):
        """All injured roster leaves active slots None, caps IR, overflows to bench."""
        roster = [1, 2, 3, 4]
        positions = {i: ["Util"] for i in roster}
        fp = {1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan}
        slots = {"Util": 2}
        injured = {1, 2, 3, 4}

        assignment = assign_optimal_daily_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp=fp,
            active_slots=slots,
            bench_slots=2,
            ir_slots=2,
            injured_players=injured,
        )
        assert assignment.active_assignments == {"Util1": None, "Util2": None}
        assert len(assignment.ir_players) == 2
        assert len(assignment.bench_players) == 2
        assert assignment.total_expected_fp == 0.0

    def test_degenerate_roster_smaller_than_active_slots(self):
        """1 player on 14-slot roster assigned to 1 slot; remaining 13 are None."""
        roster = [10]
        positions = {10: ["PG"]}
        fp = {10: 45.0}
        slots = {"PG": 2, "SG": 2, "SF": 2, "PF": 2, "C": 2, "Util": 4}

        assignment = assign_optimal_daily_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp=fp,
            active_slots=slots,
        )
        non_null_slots = [k for k, v in assignment.items() if v is not None]
        assert len(non_null_slots) == 1
        assert assignment[non_null_slots[0]] == 10
        assert assignment.total_expected_fp == 45.0

    def test_negative_fp_players_strictly_benched(self):
        """Negative FP players left on bench to prevent lowering team score."""
        roster = [1, 2, 3, 4]
        positions = {1: ["PG"], 2: ["SG"], 3: ["SF"], 4: ["PF"]}
        fp = {1: 30.0, 2: 0.0, 3: -0.1, 4: -20.0}
        slots = {"PG": 1, "SG": 1, "SF": 1, "PF": 1}

        assignment = assign_optimal_daily_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp=fp,
            active_slots=slots,
        )
        assert assignment["PG1"] == 1
        assert assignment["SG1"] == 2
        assert assignment["SF1"] is None
        assert assignment["PF1"] is None
        assert set(assignment.bench_players) == {3, 4}
        assert assignment.total_expected_fp == 30.0

    def test_domino_flex_chain_with_frozen_lock(self):
        """Complex 5-position cascade correctly routes around locked intermediate slot."""
        roster = [1, 2, 3, 4, 5, 6]
        positions = {
            1: ["PG"],
            2: ["PG", "SG"],
            3: ["SG", "SF"],
            4: ["SF", "PF"],
            5: ["PF", "C"],
            6: ["C"],
        }
        fp = {1: 50.0, 2: 48.0, 3: 46.0, 4: 44.0, 5: 42.0, 6: 40.0}
        slots = {"PG": 1, "SG": 1, "SF": 1, "PF": 1, "C": 1}

        assignment = assign_optimal_daily_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp=fp,
            active_slots=slots,
            locked_assignments={"SF1": 4},
        )
        assert assignment["SF1"] == 4
        assert assignment["PG1"] == 1
        assert assignment["SG1"] == 2
        assert assignment["PF1"] == 5
        assert assignment["C1"] == 6
        assert assignment.bench_players == [3]
        assert assignment.total_expected_fp == 224.0

    def test_solver_scalability_50_players(self):
        """50-player / 15-slot bipartite matching solves in < 50ms."""
        n_players = 50
        large_roster = list(range(1, n_players + 1))
        pos_choices = [["PG"], ["SG"], ["SF"], ["PF"], ["C"], ["PG", "SG"], ["SG", "SF"], ["SF", "PF"], ["PF", "C"], ["Util"]]
        large_pos = {i: pos_choices[i % len(pos_choices)] for i in large_roster}
        large_fp = {i: float((i * 13) % 60 + 5) for i in large_roster}
        large_slots = {"PG": 2, "SG": 2, "SF": 2, "PF": 2, "C": 2, "G": 1, "F": 1, "Util": 3}

        t0 = time.perf_counter()
        large_assign = assign_optimal_daily_lineup(
            roster_player_ids=large_roster,
            player_positions=large_pos,
            player_fp=large_fp,
            active_slots=large_slots,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 50.0
        assert sum(1 for v in large_assign.values() if v is not None) == 15


class TestBugsAndDefectsReproductions:
    """Verifications of remediated defects found in Milestone 1 implementation."""

    def test_safe_handling_none_fp_projections(self):
        """None in player_fp, player_fp_per_game, or compute_lineup_fp handled safely without TypeError."""
        daily = assign_optimal_daily_lineup(
            roster_player_ids=[1],
            player_positions={1: ["PG"]},
            player_fp={1: None},
            active_slots={"PG": 1},
        )
        assert daily["PG1"] is None
        assert daily.total_expected_fp == 0.0

        weekly = assign_optimal_weekly_lineup(
            roster_player_ids=[1],
            player_positions={1: ["PG"]},
            player_fp_per_game={1: None},
            player_games_in_week={1: 3},
            active_slots={"PG": 1},
        )
        assert weekly["PG1"] is None
        assert weekly.total_expected_fp == 0.0

        total = compute_lineup_fp({"PG1": 1}, per_game_fp={1: None})
        assert total == 0.0

    def test_locked_ineligible_player_raises_value_error(self):
        """locked_assignments rejects assigning a player to an ineligible slot with ValueError."""
        roster = [1, 2]
        positions = {1: ["PG"], 2: ["C"]}
        fp = {1: 30.0, 2: 40.0}
        slots = {"PG": 1, "C": 1}

        with pytest.raises(ValueError, match="not eligible for locked slot"):
            assign_optimal_daily_lineup(
                roster_player_ids=roster,
                player_positions=positions,
                player_fp=fp,
                active_slots=slots,
                locked_assignments={"PG1": 2},  # Pure Center cannot be locked into PG slot
            )

    def test_locked_nonexistent_slot_raises_value_error(self):
        """locked_assignments rejects nonexistent slot keys with ValueError, preserving slot caps."""
        roster = [1, 2, 3]
        positions = {1: ["PG"], 2: ["SG"], 3: ["PG"]}
        fp = {1: 30.0, 2: 25.0, 3: 20.0}
        slots = {"PG": 1, "SG": 1}

        with pytest.raises(ValueError, match="does not exist"):
            assign_optimal_daily_lineup(
                roster_player_ids=roster,
                player_positions=positions,
                player_fp=fp,
                active_slots=slots,
                locked_assignments={"QB1": 1},
            )

    def test_duplicate_roster_player_ids_raises_value_error(self):
        """Passing duplicate IDs in roster_player_ids raises ValueError to prevent cloning."""
        roster = [1, 1]
        positions = {1: ["PG", "SG"]}
        fp = {1: 35.0}
        slots = {"PG": 1, "SG": 1}

        with pytest.raises(ValueError, match="Duplicate player IDs in roster"):
            assign_optimal_daily_lineup(
                roster_player_ids=roster,
                player_positions=positions,
                player_fp=fp,
                active_slots=slots,
            )

    def test_off_day_and_zero_game_player_benched(self):
        """Player with off-day or 0 games kept on bench rather than taking active slot over dummy."""
        roster = [1]
        positions = {1: ["PG"]}
        fp = {1: 35.0}
        slots = {"PG": 1}

        # Player 1 is NOT playing today (playing_today is empty)
        daily_assign = assign_optimal_daily_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp=fp,
            active_slots=slots,
            playing_today=set(),
        )
        assert daily_assign["PG1"] is None
        assert daily_assign.bench_players == [1]

        # Weekly: Player 1 has 0 games in week
        weekly_assign = assign_optimal_weekly_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp_per_game={1: 35.0},
            player_games_in_week={1: 0},
            active_slots=slots,
        )
        assert weekly_assign["PG1"] is None
        assert weekly_assign.bench_players == [1]
