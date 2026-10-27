"""
Empirical Challenge Test Suite for Milestone 1: Start/Sit Lineup Optimizer & Schedule Engine.
Written by Challenger 2 to stress-test:
1. Multi-position flex cascading where greedy heuristics fail and linear_sum_assignment finds global max.
2. Weekly vs Daily optimization under 3-game vs 4-game schedules (W_i = FP/G_i * G_i).
3. Pre-locking validation (duplicate locked players, locked players not on roster, non-existent slots).
"""

import random
import pytest
import numpy as np

from backend.points.lineups import (
    DailyLineupAssignment,
    WeeklyLineupAssignment,
    assign_optimal_daily_lineup,
    assign_optimal_weekly_lineup,
    build_slot_configs_from_position_config,
    _is_eligible,
)
from backend.points.schedule import (
    ScheduleCalendar,
    generate_synthetic_nba_schedule,
)


class TestMultiPositionFlexCascadingChallenges:
    """Empirical challenge tests verifying SciPy bipartite matching vs greedy heuristics."""

    def test_greedy_failure_guards_routing_pg_sg_g(self):
        """
        Scenario: 3 slots [PG1, G1, SG1]
        Players:
          P1: pure PG, 50.0 FP
          P2: PG/SG dual, 48.0 FP
          P3: pure PG, 46.0 FP
          P4: pure SG, 44.0 FP

        A greedy assignment that processes slots or players in natural order:
          PG1 <- P1 (50.0)
          G1  <- P2 (48.0, eligible for G)
          SG1 <- P4 (44.0, P3 cannot play SG)
          Sum = 50 + 48 + 44 = 142.0 (P3 with 46.0 is benched!)

        Optimal Global Matching (Hungarian):
          PG1 <- P1 (50.0)
          SG1 <- P2 (48.0, re-routed to SG)
          G1  <- P3 (46.0, pure PG takes G flex)
          Sum = 50 + 48 + 46 = 144.0 (P4 with 44.0 is benched!)
        """
        roster = [1, 2, 3, 4]
        positions = {1: ['PG'], 2: ['PG', 'SG'], 3: ['PG'], 4: ['SG']}
        fp = {1: 50.0, 2: 48.0, 3: 46.0, 4: 44.0}
        slots = [
            {'slot_id': 'PG1', 'eligible_positions': ['PG']},
            {'slot_id': 'G1', 'eligible_positions': ['PG', 'SG']},
            {'slot_id': 'SG1', 'eligible_positions': ['SG']},
        ]

        def run_greedy(roster_pids, pos_map, fp_map, slot_list):
            sorted_p = sorted(roster_pids, key=lambda p: fp_map.get(p, 0), reverse=True)
            assigned = {}
            used = set()
            for s in slot_list:
                s_id = s['slot_id']
                elig = set(s['eligible_positions'])
                for p in sorted_p:
                    if p not in used and (set(pos_map.get(p, [])) & elig):
                        assigned[s_id] = p
                        used.add(p)
                        break
            return sum(fp_map.get(p, 0) for p in assigned.values()), assigned

        greedy_fp, greedy_assign = run_greedy(roster, positions, fp, slots)
        assert greedy_fp == 142.0
        assert greedy_assign == {'PG1': 1, 'G1': 2, 'SG1': 4}

        opt = assign_optimal_daily_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp=fp,
            slot_configs=slots,
        )
        assert opt.total_expected_fp == 144.0
        assert opt['PG1'] == 1
        assert opt['SG1'] == 2
        assert opt['G1'] == 3
        assert opt.bench_players == [4]

    def test_greedy_failure_4way_domino_cascade(self):
        """
        Scenario: 4 slots [PG1, SG1, SF1, Util1]
        Players:
          P1: pure PG, 50.0 FP
          P2: pure SG, 45.0 FP
          P3: PG/SG/SF tri-eligible, 42.0 FP
          P4: pure PG, 40.0 FP
          P5: pure SF, 30.0 FP

        If P3 is greedily placed into Util1:
          PG1 <- P1 (50.0)
          SG1 <- P2 (45.0)
          Util1 <- P3 (42.0)
          SF1 <- P5 (30.0, because P4 cannot play SF)
          Sum = 50 + 45 + 42 + 30 = 167.0

        Optimal Global Matching:
          PG1 <- P1 (50.0)
          SG1 <- P2 (45.0)
          SF1 <- P3 (42.0, routed to SF)
          Util1 <- P4 (40.0, pure PG into Util)
          Sum = 50 + 45 + 42 + 40 = 177.0 (P5 is benched)
        """
        roster = [1, 2, 3, 4, 5]
        positions = {
            1: ['PG'],
            2: ['SG'],
            3: ['PG', 'SG', 'SF'],
            4: ['PG'],
            5: ['SF'],
        }
        fp = {1: 50.0, 2: 45.0, 3: 42.0, 4: 40.0, 5: 30.0}
        slots = {'PG': 1, 'SG': 1, 'SF': 1, 'Util': 1}

        opt = assign_optimal_daily_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp=fp,
            active_slots=slots,
        )
        assert opt.total_expected_fp == 177.0
        assert opt['PG1'] == 1
        assert opt['SG1'] == 2
        assert opt['SF1'] == 3
        assert opt['Util1'] == 4
        assert opt.bench_players == [5]

    def test_complex_full_roster_cascading(self):
        """
        Full 8-starter lineup (PG, SG, G, SF, PF, F, C, Util) with 12 players.
        Multiple overlapping multi-eligibilities requiring simultaneous re-routing.
        """
        roster = list(range(1, 13))
        positions = {
            1: ['PG'],
            2: ['PG', 'SG'],
            3: ['SG'],
            4: ['SG', 'SF'],
            5: ['SF'],
            6: ['SF', 'PF'],
            7: ['PF'],
            8: ['PF', 'C'],
            9: ['C'],
            10: ['PG'],
            11: ['SG'],
            12: ['PF'],
        }
        fp = {i: float(50 - 2 * i) for i in roster}
        slots = {'PG': 1, 'SG': 1, 'G': 1, 'SF': 1, 'PF': 1, 'F': 1, 'C': 1, 'Util': 1}

        opt = assign_optimal_daily_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp=fp,
            active_slots=slots,
        )
        starters = {pid for pid in opt.active_assignments.values() if pid is not None}
        assert starters == set(range(1, 9))
        assert set(opt.bench_players) == {9, 10, 11, 12}
        assert opt.total_expected_fp == sum(fp[i] for i in range(1, 9))

    def test_randomized_scenarios_hungarian_dominates_greedy(self):
        """
        Generate 100 randomized rosters with complex multi-position eligibility.
        Empirically verify:
        1. Hungarian optimizer total FP >= Greedy total FP in 100% of cases.
        2. At least one scenario shows strict superiority (Hungarian > Greedy).
        """
        rng = random.Random(2026)
        all_pos = ['PG', 'SG', 'SF', 'PF', 'C']
        slots = {'PG': 1, 'SG': 1, 'G': 1, 'SF': 1, 'PF': 1, 'F': 1, 'C': 1, 'Util': 2}
        slot_configs = build_slot_configs_from_position_config(slots)

        strict_improvements = 0
        for _ in range(100):
            roster = list(range(1, 16))
            positions = {}
            fp = {}
            for pid in roster:
                k = rng.choice([1, 2, 2, 3])
                positions[pid] = rng.sample(all_pos, k)
                fp[pid] = round(rng.uniform(10.0, 55.0), 1)

            opt = assign_optimal_daily_lineup(
                roster_player_ids=roster,
                player_positions=positions,
                player_fp=fp,
                slot_configs=slot_configs,
            )

            sorted_p = sorted(roster, key=lambda p: fp[p], reverse=True)
            greedy_assign = {}
            used_p = set()
            for s in slot_configs:
                s_id = s['slot_id']
                elig = s['eligible_positions']
                for p in sorted_p:
                    if p not in used_p and _is_eligible(positions[p], elig):
                        greedy_assign[s_id] = p
                        used_p.add(p)
                        break

            greedy_total = sum(fp[p] for p in greedy_assign.values())

            assert opt.total_expected_fp >= round(greedy_total - 0.0001, 4), (
                f"Hungarian ({opt.total_expected_fp}) failed to beat Greedy ({greedy_total})"
            )
            if opt.total_expected_fp > greedy_total + 0.01:
                strict_improvements += 1

        assert strict_improvements > 0


class TestWeeklyVsDailyScheduleChallenges:
    """Empirical challenge tests for weekly volume weighting W_i = FP/G_i * Games_i."""

    def test_weekly_prefers_lower_fpg_4g_over_higher_fpg_2g(self):
        """
        Prompt requirement:
        'verify weekly optimizer correctly prefers lower FP/G with 4 games over higher FP/G with 2 games
        when total expected points are higher (W_i = FP/G_i * G_i).'

        Player 1: 42.0 FP/G, 2 games -> 84.0 weekly FP
        Player 2: 23.0 FP/G, 4 games -> 92.0 weekly FP (92 > 84, Player 2 MUST start)
        """
        roster = [1, 2]
        positions = {1: ['Util'], 2: ['Util']}
        fpg = {1: 42.0, 2: 23.0}
        games = {1: 2, 2: 4}
        slots = {'Util': 1}

        # Daily: Player 1 starts
        daily = assign_optimal_daily_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp=fpg,
            active_slots=slots,
            playing_today={1, 2},
        )
        assert daily['Util1'] == 1
        assert daily.total_expected_fp == 42.0
        assert daily.bench_players == [2]

        # Weekly: Player 2 starts
        weekly = assign_optimal_weekly_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp_per_game=fpg,
            player_games_in_week=games,
            active_slots=slots,
        )
        assert weekly['Util1'] == 2
        assert weekly.total_expected_fp == 92.0
        assert weekly.bench_players == [1]

    def test_weekly_3g_vs_4g_close_margin(self):
        """
        3-game player with 31.0 FP/G = 93.0 FP
        4-game player with 24.0 FP/G = 96.0 FP
        4-game player wins by 3.0 FP.
        """
        roster = [1, 2]
        positions = {1: ['PG'], 2: ['PG']}
        fpg = {1: 31.0, 2: 24.0}
        games = {1: 3, 2: 4}
        slots = {'PG': 1}

        weekly = assign_optimal_weekly_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp_per_game=fpg,
            player_games_in_week=games,
            active_slots=slots,
        )
        assert weekly['PG1'] == 2
        assert weekly.total_expected_fp == 96.0
        assert weekly.bench_players == [1]

    def test_weekly_superstar_overcomes_game_deficit(self):
        """
        Superstar: 2 games at 55.0 FP/G = 110.0 FP
        Scrub: 4 games at 26.0 FP/G = 104.0 FP
        Superstar starts despite playing 2 fewer games.
        """
        roster = [1, 2]
        positions = {1: ['Util'], 2: ['Util']}
        fpg = {1: 55.0, 2: 26.0}
        games = {1: 2, 2: 4}
        slots = {'Util': 1}

        weekly = assign_optimal_weekly_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp_per_game=fpg,
            player_games_in_week=games,
            active_slots=slots,
        )
        assert weekly['Util1'] == 1
        assert weekly.total_expected_fp == 110.0
        assert weekly.bench_players == [2]

    def test_weekly_negative_fpg_with_games_benched(self):
        """Player with negative FP/G (-5.0) and 4 games (-20.0 FP) must be benched."""
        roster = [1, 2]
        positions = {1: ['Util'], 2: ['Util']}
        fpg = {1: 20.0, 2: -5.0}
        games = {1: 3, 2: 4}
        slots = {'Util': 2}

        weekly = assign_optimal_weekly_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp_per_game=fpg,
            player_games_in_week=games,
            active_slots=slots,
        )
        assert weekly['Util1'] == 1
        assert weekly['Util2'] is None
        assert weekly.bench_players == [2]
        assert weekly.total_expected_fp == 60.0

    def test_weekly_zero_games_benched_with_positive_fpg(self):
        """Player with 0 games in week must be benched even with elite 50.0 FP/G."""
        roster = [1, 2]
        positions = {1: ['PG'], 2: ['PG']}
        fpg = {1: 50.0, 2: 20.0}
        games = {1: 0, 2: 3}
        slots = {'PG': 1}

        weekly = assign_optimal_weekly_lineup(
            roster_player_ids=roster,
            player_positions=positions,
            player_fp_per_game=fpg,
            player_games_in_week=games,
            active_slots=slots,
        )
        assert weekly['PG1'] == 2
        assert 1 in weekly.bench_players or 1 in weekly.ir_players
        assert weekly.total_expected_fp == 60.0


class TestPreLockingValidationChallenges:
    """Empirical challenge tests for input validation of pre-locked slots."""

    def test_prelocking_duplicate_players_raises_value_error(self):
        """Prompt requirement: test duplicate locked players."""
        roster = [1, 2]
        positions = {1: ['PG'], 2: ['SG']}
        fp = {1: 30.0, 2: 25.0}
        slots = {'PG': 1, 'SG': 1}

        with pytest.raises(ValueError, match="Duplicate player IDs"):
            assign_optimal_daily_lineup(
                roster_player_ids=roster,
                player_positions=positions,
                player_fp=fp,
                active_slots=slots,
                locked_assignments={'PG1': 1, 'SG1': 1},
            )

    def test_prelocking_player_not_on_roster_raises_value_error(self):
        """Prompt requirement: test locked players not on roster."""
        roster = [1, 2]
        positions = {1: ['PG'], 2: ['SG']}
        fp = {1: 30.0, 2: 25.0}
        slots = {'PG': 1, 'SG': 1}

        with pytest.raises(ValueError, match="is not on roster"):
            assign_optimal_daily_lineup(
                roster_player_ids=roster,
                player_positions=positions,
                player_fp=fp,
                active_slots=slots,
                locked_assignments={'PG1': 999},
            )

    def test_prelocking_nonexistent_slot_raises_value_error(self):
        """
        Prompt requirement: test locked slots that don't exist.
        If locked slot does not exist in active slots (e.g. 'NONEXISTENT_SLOT' or 'PG99'),
        the solver MUST raise a ValueError rather than corrupting the lineup with phantom slots.
        """
        roster = [1, 2]
        positions = {1: ['PG'], 2: ['SG']}
        fp = {1: 30.0, 2: 25.0}
        slots = {'PG': 1, 'SG': 1}

        with pytest.raises(ValueError, match="does not exist|Invalid slot|unknown slot"):
            assign_optimal_daily_lineup(
                roster_player_ids=roster,
                player_positions=positions,
                player_fp=fp,
                active_slots=slots,
                locked_assignments={'NONEXISTENT_SLOT': 1},
            )

    def test_prelocking_weekly_nonexistent_slot_raises_value_error(self):
        """Verify assign_optimal_weekly_lineup also rejects non-existent locked slots."""
        roster = [1, 2]
        positions = {1: ['PG'], 2: ['SG']}
        fpg = {1: 30.0, 2: 25.0}
        games = {1: 3, 2: 4}
        slots = {'PG': 1, 'SG': 1}

        with pytest.raises(ValueError, match="does not exist|Invalid slot|unknown slot"):
            assign_optimal_weekly_lineup(
                roster_player_ids=roster,
                player_positions=positions,
                player_fp_per_game=fpg,
                player_games_in_week=games,
                active_slots=slots,
                locked_assignments={'G99': 1},
            )
