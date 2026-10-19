"""Adversarial and red-team stress tests for rolling-horizon MILP fantasy basketball streamer."""

import math
from datetime import date
import pytest

from fantasy_streamer.config import LeagueConfig
from fantasy_streamer.models.penalties import (
    calculate_drop_penalty,
    calculate_matchup_urgency,
)
from fantasy_streamer.models.projections import Player
from fantasy_streamer.models.schedule import (
    MatchupHorizon,
    ScheduleMatrix,
    create_synthetic_nba_schedule,
)
from fantasy_streamer.optimization.constraints import OptimizationProblemData
from fantasy_streamer.optimization.milp_engine import MILPStreamerEngine


def _create_base_roster() -> list[Player]:
    """Helper to create a standard 13-player fantasy roster."""
    return [
        Player(1, "PG Star", "LAL", ["PG"], 42.0, is_rostered=True, is_droppable=False),
        Player(2, "SG Star", "BOS", ["SG"], 38.0, is_rostered=True, is_droppable=False),
        Player(3, "SF Star", "MIA", ["SF"], 36.0, is_rostered=True, is_droppable=False),
        Player(4, "PF Star", "MIL", ["PF"], 40.0, is_rostered=True, is_droppable=False),
        Player(5, "C Star", "DEN", ["C"], 45.0, is_rostered=True, is_droppable=False),
        Player(6, "Guard Sub", "NYK", ["PG", "SG"], 30.0, is_rostered=True, is_droppable=False),
        Player(7, "Forward Sub", "GSW", ["SF", "PF"], 29.0, is_rostered=True, is_droppable=False),
        Player(8, "Util 1", "DAL", ["SG", "SF"], 28.0, is_rostered=True, is_droppable=False),
        Player(9, "Util 2", "PHX", ["PF", "C"], 27.0, is_rostered=True, is_droppable=False),
        Player(10, "Util 3", "PHI", ["PG", "SG"], 26.0, is_rostered=True, is_droppable=False),
        # 3 Bench Spots (candidates for streaming hold vs drop)
        Player(11, "Bench Hold 1", "CLE", ["SG", "SF"], 22.0, is_rostered=True, is_droppable=True, drop_penalty_c_p=25.0),
        Player(12, "Bench Hold 2", "ATL", ["PF", "C"], 20.0, is_rostered=True, is_droppable=True, drop_penalty_c_p=15.0),
        Player(13, "Bench Hold 3", "DET", ["PG"], 16.0, is_rostered=True, is_droppable=True, drop_penalty_c_p=0.0),
    ]


@pytest.mark.parametrize("backend", ["ortools", "pulp"])
def test_zero_add_budget_guarantees_holds_undisturbed(backend: str):
    """Adversarial Test 1: Zero-add budget constraint (rem_acquisitions_w1=0, cap_acquisitions_w2=0).

    Verifies that even with ultra-high FPPG free agents and 0-penalty bench players with no games,
    exactly 0 adds occur, 0 drops occur, and all roster holds are strictly undisturbed.
    """
    config = LeagueConfig()
    horizon = MatchupHorizon(start_date=date(2026, 11, 6), horizon_days=7)
    # Schedule where bench holds have no games, tempting the solver to discard them
    schedule = ScheduleMatrix(
        horizon=horizon,
        team_games={
            "LAL": {0, 2, 4, 6},
            "BOS": {0, 2, 4, 6},
            "MIA": {0, 2, 4, 6},
            "MIL": {0, 2, 4, 6},
            "DEN": {0, 2, 4, 6},
            "NYK": {1, 3, 5},
            "GSW": {1, 3, 5},
            "DAL": {1, 3, 5},
            "PHX": {1, 3, 5},
            "PHI": {1, 3, 5},
            "CLE": set(),  # Bench Hold 1 has NO games
            "ATL": set(),  # Bench Hold 2 has NO games
            "DET": set(),  # Bench Hold 3 has NO games
            "FA_TEAM": {0, 1, 2, 3, 4, 5, 6},
        },
    )

    roster = _create_base_roster()
    # Explicitly set all bench holds to 0.0 penalty
    for p in roster:
        if p.is_droppable:
            p.drop_penalty_c_p = 0.0

    # Irresistible free agents playing every single day with monster projections
    tempting_fas = [
        Player(101, "Mega FA 1", "FA_TEAM", ["PG", "SG", "SF", "PF", "C"], 65.0),
        Player(102, "Mega FA 2", "FA_TEAM", ["PG", "SG", "SF", "PF", "C"], 60.0),
    ]

    data = OptimizationProblemData(
        config=config,
        rostered_players=roster,
        candidate_free_agents=tempting_fas,
        schedule=schedule,
        horizon=horizon,
        rem_acquisitions_w1=0,  # Zero adds allowed in Week 1
        cap_acquisitions_w2=0,  # Zero adds allowed in Week 2
    )

    engine = MILPStreamerEngine(backend=backend)
    result = engine.solve(data)

    assert result.status == "OPTIMAL"
    assert result.total_adds_week1 == 0
    assert result.total_adds_week2 == 0
    assert len([a for a in result.actions if a.action_type == "ADD"]) == 0
    assert len([a for a in result.actions if a.action_type == "DROP"]) == 0
    assert len(result.dropped_bench_holds) == 0
    assert set(result.retained_bench_holds) == {"Bench Hold 1", "Bench Hold 2", "Bench Hold 3"}


def test_massive_blowout_and_deficit_urgency_clamping_and_numerical_stability():
    """Adversarial Test 2: Massive blowout / deficit matchup (|delta_S| >= 500).

    Verifies M_urgency correctly clamps at 3.0 without numerical overflow, float error,
    or negative differential breakdown across extreme bounds.
    """
    sigma_s = 65.0
    max_clamp = 3.0

    # Test extreme point differentials: blowout win, deficit loss, and extreme bounds
    test_differentials = [
        500.0,
        -500.0,
        1000.0,
        -1000.0,
        5000.0,
        -5000.0,
        1e6,
        -1e6,
    ]

    for delta in test_differentials:
        urgency = calculate_matchup_urgency(delta_s=delta, sigma_s=sigma_s, max_clamp=max_clamp)
        assert urgency == max_clamp, f"Failed clamping at delta_S={delta}: got {urgency}"

    # Verify neutral point differential
    assert pytest.approx(calculate_matchup_urgency(0.0, sigma_s=sigma_s), abs=1e-6) == 1.0

    # Verify non-positive sigma_s fallback
    assert calculate_matchup_urgency(100.0, sigma_s=0.0) == 1.0
    assert calculate_matchup_urgency(100.0, sigma_s=-10.0) == 1.0

    # Verify full drop penalty calculation under massive blowout
    res_blowout = calculate_drop_penalty(
        player_id=1,
        player_name="Test Player",
        mu_p=28.0,
        mu_rep=20.0,
        weeks_remaining=10,
        delta_s=500.0,  # Massive blowout
    )
    assert res_blowout.m_urgency == 3.0
    assert not math.isnan(res_blowout.c_p)
    assert not math.isinf(res_blowout.c_p)
    assert res_blowout.c_p > 0.0


def test_ineligible_position_streaming_cannot_slot_center_to_pg_sg():
    """Adversarial Test 3: Ineligible position streaming.

    Ensures that a Center-only streamer cannot be slotted into a PG/SG active slot,
    even when the Center has vastly superior FPPG and all Center/UTIL spots are filled.
    """
    config = LeagueConfig()
    horizon = MatchupHorizon(start_date=date(2026, 11, 6), horizon_days=1)
    schedule = ScheduleMatrix(
        horizon=horizon,
        team_games={
            "LOCKED_SLOTS": {0},
            "NO_GAME": set(),
            "FA_TEAM": {0},
        },
    )

    # Construct roster where SG, SF, PF, C, G, F, and all 3 UTIL slots are locked
    # by 70.0 FPPG non-droppable starters playing on Day 0.
    # ONLY the PG slot is unfilled because our PG has NO game.
    roster = [
        Player(1, "Resting PG", "NO_GAME", ["PG"], 40.0, is_rostered=True, is_droppable=False),
        Player(2, "SG Lock", "LOCKED_SLOTS", ["SG"], 50.0, is_rostered=True, is_droppable=False),
        Player(3, "SF Lock", "LOCKED_SLOTS", ["SF"], 50.0, is_rostered=True, is_droppable=False),
        Player(4, "PF Lock", "LOCKED_SLOTS", ["PF"], 50.0, is_rostered=True, is_droppable=False),
        Player(5, "C Lock 1", "LOCKED_SLOTS", ["C"], 70.0, is_rostered=True, is_droppable=False),
        Player(6, "G Lock", "LOCKED_SLOTS", ["SG"], 50.0, is_rostered=True, is_droppable=False),
        Player(7, "F Lock", "LOCKED_SLOTS", ["SF"], 50.0, is_rostered=True, is_droppable=False),
        Player(8, "C Lock 2", "LOCKED_SLOTS", ["C"], 70.0, is_rostered=True, is_droppable=False),
        Player(9, "C Lock 3", "LOCKED_SLOTS", ["C"], 70.0, is_rostered=True, is_droppable=False),
        Player(10, "C Lock 4", "LOCKED_SLOTS", ["C"], 70.0, is_rostered=True, is_droppable=False),
        # 3 droppable bench spots with no games
        Player(11, "B1", "NO_GAME", ["SG"], 10.0, is_rostered=True, is_droppable=True, drop_penalty_c_p=0.0),
        Player(12, "B2", "NO_GAME", ["SF"], 10.0, is_rostered=True, is_droppable=True, drop_penalty_c_p=0.0),
        Player(13, "B3", "NO_GAME", ["C"], 10.0, is_rostered=True, is_droppable=True, drop_penalty_c_p=0.0),
    ]

    # Available Free Agents:
    # 1. Monster Center Streamer with 75.0 FPPG, but ONLY eligible for ["C"]
    fa_center = Player(101, "Monster Center Streamer", "FA_TEAM", ["C"], 75.0)
    # 2. Modest PG Streamer with 22.0 FPPG, eligible for ["PG"]
    fa_pg = Player(102, "Modest PG Streamer", "FA_TEAM", ["PG"], 22.0)

    data = OptimizationProblemData(
        config=config,
        rostered_players=roster,
        candidate_free_agents=[fa_center, fa_pg],
        schedule=schedule,
        horizon=horizon,
        rem_acquisitions_w1=1,
        cap_acquisitions_w2=0,
    )

    engine = MILPStreamerEngine(backend="ortools")
    result = engine.solve(data)

    assert result.status == "OPTIMAL"
    day0_plan = result.daily_plans[0]

    # The PG slot must be occupied by the Modest PG Streamer, NOT the Center
    assert "PG" in day0_plan.starters
    assert "Modest PG Streamer" in day0_plan.starters["PG"]
    assert "Monster Center Streamer" not in day0_plan.starters.values()

    # Verify no illegal slot allocations across all starter positions
    for slot, starter_str in day0_plan.starters.items():
        if "Monster Center Streamer" in starter_str:
            assert slot in {"C", "UTIL1", "UTIL2", "UTIL3"}


def test_sunday_to_monday_boundary_transition_budget_partitioning():
    """Adversarial Test 4: Sunday-to-Monday boundary transition.

    Ensures Week 1 budget is never consumed on Monday, and Week 2 budget is never
    consumed on Sunday when the rolling horizon starts on a Sunday.
    """
    config = LeagueConfig()
    # 2026-11-08 is a Sunday (weekday 6)
    sunday = date(2026, 11, 8)
    assert sunday.weekday() == 6

    # Horizon: Day 0 (Sun = Week 1), Day 1 (Mon = Week 2), Day 2 (Tue = Week 2)
    horizon = MatchupHorizon(start_date=sunday, horizon_days=3)
    assert horizon.week1_days == [0]
    assert horizon.week2_days == [1, 2]

    # Schedule: Sunday game vs Monday game
    schedule = ScheduleMatrix(
        horizon=horizon,
        team_games={
            "ROST_PLAYING": {0, 1, 2},
            "NO_GAME": set(),
            "SUN_TEAM": {0},
            "MON_TEAM": {1},
        },
    )

    def _build_test_data(rem_w1: int, cap_w2: int) -> OptimizationProblemData:
        roster = [
            Player(i, f"P{i}", "ROST_PLAYING", ["PG", "SG", "SF", "PF", "C"], 30.0, is_rostered=True, is_droppable=False)
            for i in range(1, 10)
        ]
        # Open 1 starter slot on Day 0 and Day 1
        roster.append(Player(10, "P10", "NO_GAME", ["C"], 30.0, is_rostered=True, is_droppable=False))
        roster.extend([
            Player(i, f"B{i}", "NO_GAME", ["C"], 10.0, is_rostered=True, is_droppable=True, drop_penalty_c_p=0.0)
            for i in range(11, 14)
        ])

        fa_sun = Player(101, "FA Sunday", "SUN_TEAM", ["C"], 45.0)
        fa_mon = Player(102, "FA Monday", "MON_TEAM", ["C"], 45.0)

        return OptimizationProblemData(
            config=config,
            rostered_players=roster,
            candidate_free_agents=[fa_sun, fa_mon],
            schedule=schedule,
            horizon=horizon,
            rem_acquisitions_w1=rem_w1,
            cap_acquisitions_w2=cap_w2,
        )

    # Subtest 4A: Week 1 has 1 add, Week 2 has 0 adds -> Can add Sunday, CANNOT add Monday
    data_w1_only = _build_test_data(rem_w1=1, cap_w2=0)
    res_w1 = MILPStreamerEngine(backend="ortools").solve(data_w1_only)
    adds_w1 = [a.player_name for a in res_w1.actions if a.action_type == "ADD"]

    assert res_w1.total_adds_week1 == 1
    assert res_w1.total_adds_week2 == 0
    assert adds_w1 == ["FA Sunday"]

    # Subtest 4B: Week 1 has 0 adds, Week 2 has 1 add -> CANNOT add Sunday, Can add Monday
    data_w2_only = _build_test_data(rem_w1=0, cap_w2=1)
    res_w2 = MILPStreamerEngine(backend="ortools").solve(data_w2_only)
    adds_w2 = [a.player_name for a in res_w2.actions if a.action_type == "ADD"]

    assert res_w2.total_adds_week1 == 0
    assert res_w2.total_adds_week2 == 1
    assert adds_w2 == ["FA Monday"]


def test_solvers_parity_ortools_vs_pulp():
    """Adversarial Test 5: Solvers parity (OR-Tools vs PuLP).

    Verifies that Google OR-Tools and PuLP backends converge to identical objective
    values, transaction counts, and projected fantasy points on matching problem instances.
    """
    config = LeagueConfig()
    horizon = MatchupHorizon(start_date=date(2026, 11, 4), horizon_days=7)
    schedule = create_synthetic_nba_schedule(horizon)

    roster = [
        Player(1, "Luka Doncic", "DAL", ["PG"], 56.5, is_rostered=True, is_droppable=False),
        Player(2, "Donovan Mitchell", "CLE", ["SG"], 44.0, is_rostered=True, is_droppable=False),
        Player(3, "Jayson Tatum", "BOS", ["SF", "PF"], 48.0, is_rostered=True, is_droppable=False),
        Player(4, "Giannis Antetokounmpo", "MIL", ["PF"], 54.0, is_rostered=True, is_droppable=False),
        Player(5, "Nikola Jokic", "DEN", ["C"], 60.5, is_rostered=True, is_droppable=False),
        Player(6, "Jalen Brunson", "NYK", ["PG"], 43.5, is_rostered=True, is_droppable=False),
        Player(7, "Paul George", "PHI", ["SF", "PF"], 37.0, is_rostered=True, is_droppable=False),
        Player(8, "Chet Holmgren", "OKC", ["PF", "C"], 40.0, is_rostered=True, is_droppable=False),
        Player(9, "Derrick White", "BOS", ["PG", "SG"], 34.0, is_rostered=True, is_droppable=False),
        Player(10, "Mikal Bridges", "NYK", ["SG", "SF"], 31.5, is_rostered=True, is_droppable=False),
        Player(11, "Norman Powell", "LAC", ["SG", "SF"], 26.5, is_rostered=True, is_droppable=True, drop_penalty_c_p=35.0),
        Player(12, "Bobby Portis", "MIL", ["PF", "C"], 24.0, is_rostered=True, is_droppable=True, drop_penalty_c_p=20.0),
        Player(13, "Tre Jones", "SAS", ["PG"], 19.0, is_rostered=True, is_droppable=True, drop_penalty_c_p=0.0),
    ]

    free_agents = [
        Player(101, "Streamer A", "BOS", ["PG", "SG"], 28.0),
        Player(102, "Streamer B", "DEN", ["C"], 27.0),
        Player(103, "Streamer C", "GSW", ["SF", "PF"], 26.0),
    ]

    data = OptimizationProblemData(
        config=config,
        rostered_players=roster,
        candidate_free_agents=free_agents,
        schedule=schedule,
        horizon=horizon,
        rem_acquisitions_w1=2,
        cap_acquisitions_w2=2,
    )

    res_ortools = MILPStreamerEngine(backend="ortools").solve(data)
    res_pulp = MILPStreamerEngine(backend="pulp").solve(data)

    assert res_ortools.status == "OPTIMAL"
    assert res_pulp.status == "OPTIMAL"

    # Verify numerical convergence parity
    assert pytest.approx(res_ortools.total_projected_fp, abs=1e-2) == res_pulp.total_projected_fp
    assert pytest.approx(res_ortools.net_objective_value, abs=1e-2) == res_pulp.net_objective_value
    assert res_ortools.total_adds_week1 == res_pulp.total_adds_week1
    assert res_ortools.total_adds_week2 == res_pulp.total_adds_week2
    assert set(res_ortools.retained_bench_holds) == set(res_pulp.retained_bench_holds)
    assert set(res_ortools.dropped_bench_holds) == set(res_pulp.dropped_bench_holds)


def test_non_droppable_core_inviolability():
    """Adversarial Test 6: Non-droppable core players must never be dropped under any circumstance."""
    config = LeagueConfig()
    horizon = MatchupHorizon(start_date=date(2026, 11, 2), horizon_days=7)
    schedule = ScheduleMatrix(
        horizon=horizon,
        team_games={"ZERO_GAME": set(), "FA_TEAM": set(range(7))},
    )

    # Core player has 0 games, but is marked non-droppable
    roster = [
        Player(1, "Zero Game Superstar", "ZERO_GAME", ["PG"], 50.0, is_rostered=True, is_droppable=False),
    ]
    # Other 12 players
    for i in range(2, 14):
        roster.append(Player(i, f"P{i}", "ZERO_GAME", ["PG", "SG", "SF", "PF", "C"], 20.0, is_rostered=True, is_droppable=True, drop_penalty_c_p=0.0))

    fa = Player(100, "Everyday God", "FA_TEAM", ["PG"], 100.0)

    data = OptimizationProblemData(
        config=config,
        rostered_players=roster,
        candidate_free_agents=[fa],
        schedule=schedule,
        horizon=horizon,
        rem_acquisitions_w1=1,
        cap_acquisitions_w2=0,
    )

    res = MILPStreamerEngine(backend="ortools").solve(data)
    assert res.status == "OPTIMAL"

    # Core player must remain on roster for every single day
    dropped_names = [a.player_name for a in res.actions if a.action_type == "DROP"]
    assert "Zero Game Superstar" not in dropped_names


def test_monotonic_drop_rule():
    """Adversarial Test 7: Monotonic hold/drop rule verification.

    Verifies that once an initial roster hold is dropped on day d, they can never
    reappear or be rostered on day d' > d.
    """
    config = LeagueConfig()
    horizon = MatchupHorizon(start_date=date(2026, 11, 2), horizon_days=7)
    schedule = create_synthetic_nba_schedule(horizon)
    roster = _create_base_roster()

    fas = [
        Player(101, "Streamer 1", "LAL", ["PG"], 25.0),
        Player(102, "Streamer 2", "BOS", ["SG"], 25.0),
    ]

    data = OptimizationProblemData(
        config=config,
        rostered_players=roster,
        candidate_free_agents=fas,
        schedule=schedule,
        horizon=horizon,
        rem_acquisitions_w1=2,
        cap_acquisitions_w2=2,
    )

    res = MILPStreamerEngine(backend="ortools").solve(data)
    assert res.status == "OPTIMAL"

    # Verify that for any dropped bench hold, they never reappear in daily_plans after drop
    for dropped_name in res.dropped_bench_holds:
        first_missing_day = None
        for day_plan in res.daily_plans:
            is_present = any(dropped_name in s for s in day_plan.starters.values()) or any(dropped_name in b for b in day_plan.bench)
            if not is_present and first_missing_day is None:
                first_missing_day = day_plan.day_idx
            if first_missing_day is not None and day_plan.day_idx > first_missing_day:
                assert not is_present, f"{dropped_name} was dropped on day {first_missing_day} but reappeared on day {day_plan.day_idx}!"


def test_empty_candidate_pool_graceful_handling():
    """Adversarial Test 8: Empty candidate pool (no free agents available).

    Verifies the engine completes successfully with 0 transactions and optimal starter selection.
    """
    config = LeagueConfig()
    horizon = MatchupHorizon(start_date=date(2026, 11, 2), horizon_days=5)
    schedule = create_synthetic_nba_schedule(horizon)
    roster = _create_base_roster()

    data = OptimizationProblemData(
        config=config,
        rostered_players=roster,
        candidate_free_agents=[],  # Empty candidate pool
        schedule=schedule,
        horizon=horizon,
        rem_acquisitions_w1=2,
        cap_acquisitions_w2=2,
    )

    res = MILPStreamerEngine(backend="ortools").solve(data)
    assert res.status == "OPTIMAL"
    assert len(res.actions) == 0
    assert len(res.dropped_bench_holds) == 0
    assert len(res.retained_bench_holds) == 3
    assert len(res.daily_plans) == 5
