"""CLI execution entrypoint and weekly streaming schedule output generator."""

import os
import sys
from datetime import date

# Ensure repo root is on sys.path for direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fantasy_streamer.config import LeagueConfig
from fantasy_streamer.models.penalties import DropPenaltyCalculator
from fantasy_streamer.models.projections import (
    Player,
    compute_replacement_baseline,
    prune_candidate_pool,
)
from fantasy_streamer.models.schedule import (
    MatchupHorizon,
    create_synthetic_nba_schedule,
)
from fantasy_streamer.optimization.constraints import OptimizationProblemData
from fantasy_streamer.optimization.milp_engine import MILPStreamerEngine


def run_sample_optimization():
    print("=" * 80)
    print(" ROLLING-HORIZON MILP FANTASY BASKETBALL STREAMER ENGINE")
    print("=" * 80)

    config = LeagueConfig()

    # Horizon: 10 days starting on Friday (e.g. 2026-11-06)
    # Week 1: Day 0 (Fri), Day 1 (Sat), Day 2 (Sun) -> 3 days
    # Week 2: Day 3 (Mon) through Day 9 (Sun) -> 7 days
    start_dt = date(2026, 11, 6)
    horizon = MatchupHorizon(start_date=start_dt, horizon_days=10)
    schedule = create_synthetic_nba_schedule(horizon)

    print(f"\n[1] Rolling Horizon Configuration (N={horizon.total_days} days):")
    print(f"    - Start Date: {start_dt}")
    print(f"    - Matchup Week 1 (Current): Days {horizon.week1_days} ({len(horizon.week1_days)} days)")
    print(f"    - Matchup Week 2 (Lookahead): Days {horizon.week2_days} ({len(horizon.week2_days)} days)")

    # 13-player Roster
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
        # 3 Bench Spots (Candidates for streaming hold vs drop)
        Player(11, "Norman Powell", "LAC", ["SG", "SF"], 26.5, is_rostered=True, is_droppable=True),
        Player(12, "Bobby Portis", "MIL", ["PF", "C"], 24.0, is_rostered=True, is_droppable=True),
        Player(13, "Tre Jones", "SAS", ["PG"], 19.0, is_rostered=True, is_droppable=True),
    ]

    # Candidate Free Agents
    teams = ["ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
             "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK"]
    raw_free_agents = []
    pid = 100
    for t in teams:
        raw_free_agents.extend([
            Player(pid, f"{t} Guard Streamer", t, ["PG", "SG"], 22.5),
            Player(pid + 1, f"{t} Wing Streamer", t, ["SF", "PF"], 21.0),
            Player(pid + 2, f"{t} Big Streamer", t, ["C"], 20.0),
            Player(pid + 3, f"{t} Bench FA", t, ["PG"], 16.0),
        ])
        pid += 4

    # Calculate Replacement Baseline mu_rep
    mu_rep = compute_replacement_baseline(raw_free_agents, k=config.top_k_replacement)
    print(f"\n[2] Waiver Replacement Baseline (Top {config.top_k_replacement} FAs): {mu_rep:.2f} FPPG")

    # Calculate Drop Penalty c_p for bench spots
    penalty_calc = DropPenaltyCalculator(
        mu_rep=mu_rep,
        weeks_remaining=14,
        delta_s=45.0,  # Projected tight matchup
        effective_games=config.effective_games_per_week,
        gamma=config.retention_rate_gamma,
        sigma_s=config.matchup_sigma,
    )

    print("\n[3] Bench Player Opportunity Cost & Drop Penalty Evaluation (c_p):")
    print(f"    {'Player':<20} {'FPPG':<8} {'VORP':<8} {'P_snipe':<10} {'H_eff':<8} {'c_p':<8}")
    print("    " + "-" * 62)
    for p in roster:
        if p.is_droppable:
            res = penalty_calc.evaluate_player(p.player_id, p.name, p.fppg)
            p.drop_penalty_c_p = res.c_p
            print(f"    {p.name:<20} {p.fppg:<8.1f} {res.weekly_vorp:<8.2f} {res.p_snipe:<10.3f} {res.h_eff:<8.2f} {res.c_p:<8.2f}")

    # Prune free agent candidate pool
    pruned_fas = prune_candidate_pool(raw_free_agents, max_per_team_per_pos=3)
    print(f"\n[4] Candidate Pool Pruning: Reduced from {len(raw_free_agents)} to {len(pruned_fas)} high-yield targets.")

    # Build Optimization Problem Data
    data = OptimizationProblemData(
        config=config,
        rostered_players=roster,
        candidate_free_agents=pruned_fas,
        schedule=schedule,
        horizon=horizon,
        rem_acquisitions_w1=2,  # 2 remaining adds for current week (Fri-Sun)
        cap_acquisitions_w2=4,  # 4 adds budgeted for next week
    )

    # Solve MILP
    print("\n[5] Invoking Google OR-Tools Rolling-Horizon MILP Solver...")
    engine = MILPStreamerEngine(backend="ortools")
    result = engine.solve(data)

    print(f"    - Solver Status: {result.status}")
    print(f"    - Solve Latency: {result.solve_time_ms:.2f} ms")
    print(f"    - Week 1 Acquisitions Used: {result.total_adds_week1} / {data.rem_acquisitions_w1}")
    print(f"    - Week 2 Acquisitions Used: {result.total_adds_week2} / {data.cap_acquisitions_w2}")
    print(f"    - Total Projected Points: {result.total_projected_fp:.1f} FP")
    print(f"    - Total Drop Penalties Incurred: {result.total_drop_penalties:.2f}")
    print(f"    - Net Objective Value: {result.net_objective_value:.2f}")

    print("\n[6] Recommended Transactions:")
    if not result.actions:
        print("    (No transactions recommended - maintaining current roster is optimal)")
    for act in result.actions:
        print(f"    - Day {act.day_idx} ({act.date_str}): [{act.action_type}] {act.player_name} ({act.player_team}) - {act.reason}")

    print("\n[7] Bench Retention Summary:")
    print(f"    - Retained Holds: {', '.join(result.retained_bench_holds) if result.retained_bench_holds else 'None'}")
    print(f"    - Dropped Holds: {', '.join(result.dropped_bench_holds) if result.dropped_bench_holds else 'None'}")
    print("=" * 80)


if __name__ == "__main__":
    run_sample_optimization()
