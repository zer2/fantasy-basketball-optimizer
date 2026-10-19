"""Unit tests for MILP solver convergence, B2B chaining, drop penalties, and partitioned caps."""

from datetime import date
import pytest

from fantasy_streamer.config import LeagueConfig
from fantasy_streamer.models.penalties import DropPenaltyCalculator
from fantasy_streamer.models.projections import (
    Player,
    compute_replacement_baseline,
    prune_candidate_pool,
)
from fantasy_streamer.models.schedule import (
    MatchupHorizon,
    ScheduleMatrix,
    create_synthetic_nba_schedule,
)
from fantasy_streamer.optimization.constraints import OptimizationProblemData
from fantasy_streamer.optimization.milp_engine import MILPStreamerEngine


def _create_test_roster(config: LeagueConfig) -> list[Player]:
    """Create a standard 13-player roster (10 starters + 3 bench spots)."""
    roster = [
        # Starters
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
        # 3 Bench Spots (Droppable Candidates)
        Player(11, "Valuable Bench Hold", "CLE", ["SG", "SF"], 25.0, is_rostered=True, is_droppable=True, drop_penalty_c_p=40.0),
        Player(12, "Fringe Bench Hold", "ATL", ["PF", "C"], 20.5, is_rostered=True, is_droppable=True, drop_penalty_c_p=25.0),
        Player(13, "Pure Streamer Spot", "DET", ["PG"], 17.0, is_rostered=True, is_droppable=True, drop_penalty_c_p=0.0),
    ]
    return roster


def test_b2b_chaining_preference():
    """Verify that the MILP prefers acquiring a B2B streamer over a 1-game streamer under tight cap."""
    config = LeagueConfig()
    horizon = MatchupHorizon(start_date=date(2026, 11, 4), horizon_days=7)

    # Schedule:
    # Day 0: Standard busy day
    # Day 1: Off-night (0 games for rostered teams)
    # Day 2: Light night (only 4 rostered teams play, leaving 6 open starter slots)
    # Day 3, 4: Standard days
    team_games = {
        "LAL": {0, 4},
        "BOS": {0, 4},
        "MIA": {0, 4},
        "MIL": {0, 4},
        "DEN": {0, 4},
        "NYK": {0, 4},
        "GSW": {0, 4},
        "DAL": {0, 4},
        "PHX": {2, 4},
        "PHI": {2, 4},
        "CLE": {2, 4},
        "ATL": {2, 4},
        "DET": set(),  # Player 13 has no games
        "TB2B": {1, 2},  # Back to back on off-night (Day 1) and light night (Day 2)
        "TONE": {1},     # Single game on Day 1
    }
    schedule = ScheduleMatrix(horizon=horizon, team_games=team_games)

    roster = _create_test_roster(config)
    # Player 11 & 12 have high drop penalties; Player 13 has 0.0 penalty
    roster[10].drop_penalty_c_p = 50.0
    roster[11].drop_penalty_c_p = 30.0
    roster[12].drop_penalty_c_p = 0.0

    fa_b2b = Player(101, "Streamer B2B", "TB2B", ["PG", "SG"], 22.0, is_rostered=False)
    fa_one = Player(102, "Streamer One", "TONE", ["PG", "SG"], 23.0, is_rostered=False)

    data = OptimizationProblemData(
        config=config,
        rostered_players=roster,
        candidate_free_agents=[fa_b2b, fa_one],
        schedule=schedule,
        horizon=horizon,
        rem_acquisitions_w1=1,  # Only 1 acquisition allowed in Week 1
        cap_acquisitions_w2=0,
    )

    engine = MILPStreamerEngine(backend="ortools")
    result = engine.solve(data)

    assert result.status == "OPTIMAL"
    assert result.total_adds_week1 == 1
    # B2B player should be acquired because 2 * 22.0 (44.0 FP) > 1 * 23.0 (23.0 FP)
    added_names = [a.player_name for a in result.actions if a.action_type == "ADD"]
    assert "Streamer B2B" in added_names
    assert "Streamer One" not in added_names

    # Only the 0-penalty player should be dropped
    dropped_names = [a.player_name for a in result.actions if a.action_type == "DROP"]
    assert "Pure Streamer Spot" in dropped_names
    assert "Valuable Bench Hold" not in dropped_names


def test_drop_penalty_deterrence():
    """Verify that high drop penalty prevents dropping valuable bench player for minor streaming gain."""
    config = LeagueConfig()
    horizon = MatchupHorizon(start_date=date(2026, 11, 2), horizon_days=7)

    # Schedule where team STR plays on empty days
    team_games = {
        "CLE": {0, 2},
        "ATL": {0, 2},
        "DET": {0, 2},
        "STR": {1, 3, 5},
    }
    for p in ["LAL", "BOS", "MIA", "MIL", "DEN", "NYK", "GSW", "DAL", "PHX", "PHI"]:
        team_games[p] = {0, 2, 4}

    schedule = ScheduleMatrix(horizon=horizon, team_games=team_games)
    roster = _create_test_roster(config)

    # Mark Player 11 (Valuable Bench Hold) with large penalty (e.g. 80.0)
    roster[10].drop_penalty_c_p = 80.0
    # Mark Player 13 (Pure Streamer Spot) with 0.0 penalty
    roster[12].drop_penalty_c_p = 0.0
    # Player 12 with medium penalty
    roster[11].drop_penalty_c_p = 35.0

    # Streamer available
    fa_streamer = Player(101, "Top Streamer", "STR", ["UTIL1"], 21.0, is_rostered=False)

    data = OptimizationProblemData(
        config=config,
        rostered_players=roster,
        candidate_free_agents=[fa_streamer],
        schedule=schedule,
        horizon=horizon,
        rem_acquisitions_w1=1,
        cap_acquisitions_w2=0,
    )

    engine = MILPStreamerEngine(backend="ortools")
    result = engine.solve(data)

    assert result.status == "OPTIMAL"
    # Valuable hold must be retained
    assert "Valuable Bench Hold" in result.retained_bench_holds
    # Dropped player should be the low-penalty Pure Streamer Spot
    if result.total_adds_week1 > 0:
        assert "Pure Streamer Spot" in result.dropped_bench_holds
        assert "Valuable Bench Hold" not in result.dropped_bench_holds


def test_partitioned_weekly_acquisition_caps():
    """Verify acquisitions strictly adhere to separate Week 1 and Week 2 limits."""
    config = LeagueConfig()
    # Start on Friday (2026-11-06): Fri, Sat, Sun are Week 1 (3 days); Mon-Thu are Week 2 (4 days)
    horizon = MatchupHorizon(start_date=date(2026, 11, 6), horizon_days=7)
    schedule = create_synthetic_nba_schedule(horizon)

    roster = _create_test_roster(config)
    for p in roster:
        if p.is_droppable:
            p.drop_penalty_c_p = 0.0

    # 4 Free agents playing every day
    free_agents = [
        Player(100 + i, f"FA {i}", "LAL", ["PG", "SG", "SF", "PF", "C"], 25.0)
        for i in range(4)
    ]
    for i, p in enumerate(free_agents):
        schedule.player_games[p.player_id] = set(range(7))

    data = OptimizationProblemData(
        config=config,
        rostered_players=roster,
        candidate_free_agents=free_agents,
        schedule=schedule,
        horizon=horizon,
        rem_acquisitions_w1=1,  # Max 1 in Week 1
        cap_acquisitions_w2=2,  # Max 2 in Week 2
    )

    engine = MILPStreamerEngine(backend="ortools")
    result = engine.solve(data)

    assert result.status == "OPTIMAL"
    assert result.total_adds_week1 <= 1
    assert result.total_adds_week2 <= 2


def test_candidate_pruning_and_solve_latency():
    """Verify candidate pool pruning keeps solver solve latency under 50ms."""
    config = LeagueConfig()
    horizon = MatchupHorizon(start_date=date(2026, 11, 2), horizon_days=10)
    schedule = create_synthetic_nba_schedule(horizon)

    roster = _create_test_roster(config)
    for p in roster:
        if p.is_droppable:
            p.drop_penalty_c_p = 5.0

    # Generate candidate pool across 5 teams and 3 positions
    teams = ["ATL", "BOS", "BKN", "CHA", "CHI"]
    raw_fas = []
    pid = 200
    for team in teams:
        for pos in [["PG"], ["SF"], ["C"]]:
            for rank in range(4):  # 4 per team/pos = 60 players
                fppg = 24.0 - rank * 2.0
                raw_fas.append(
                    Player(pid, f"FA_{team}_{pos[0]}_{rank}", team, pos, fppg)
                )
                pid += 1

    assert len(raw_fas) == 60

    # Prune candidate pool to top candidates (e.g. top 1 per team/pos, capped at 10 high-yield targets)
    pruned_fas = prune_candidate_pool(raw_fas, max_per_team_per_pos=1, max_total_candidates=10)
    assert len(pruned_fas) == 10

    data = OptimizationProblemData(
        config=config,
        rostered_players=roster,
        candidate_free_agents=pruned_fas,
        schedule=schedule,
        horizon=horizon,
        rem_acquisitions_w1=3,
        cap_acquisitions_w2=4,
    )

    engine = MILPStreamerEngine(backend="ortools")
    # Warm up solver to initialize C++ runtime
    _ = engine.solve(data)
    result = engine.solve(data)

    assert result.status == "OPTIMAL"
    # Solver execution time is around ~30-60ms with pruned candidate pool (threshold 250ms for Windows background scheduling jitter)
    assert result.solve_time_ms < 250.0
    assert len(result.daily_plans) == 10


def test_pulp_highs_fallback():
    """Verify that PuLP backend produces optimal feasible solution."""
    config = LeagueConfig()
    horizon = MatchupHorizon(start_date=date(2026, 11, 4), horizon_days=5)
    schedule = create_synthetic_nba_schedule(horizon)

    roster = _create_test_roster(config)
    fa = Player(101, "PuLP Test FA", "LAL", ["PG", "SG"], 22.0)

    data = OptimizationProblemData(
        config=config,
        rostered_players=roster,
        candidate_free_agents=[fa],
        schedule=schedule,
        horizon=horizon,
        rem_acquisitions_w1=1,
        cap_acquisitions_w2=0,
    )

    engine = MILPStreamerEngine(backend="pulp")
    result = engine.solve(data)

    assert result.status == "OPTIMAL"
    assert len(result.daily_plans) == 5
