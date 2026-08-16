# benchmark/tests/test_sanity.py
import numpy as np
from benchmark.config import LeagueConfig, ExperimentConfig
from benchmark.data import load_fixture
from benchmark.bootstrap import bootstrap_session
from benchmark.agents import RandomAgent, GScoreAgent, HScoreAgent
from benchmark.draft import run_draft
from benchmark.evaluate import evaluate_rosters

def test_gscore_beats_random_field():
    averages, gamelogs = load_fixture('benchmark/fixtures/nba_2025-26.parquet')
    cfg = LeagueConfig()
    info = bootstrap_session(averages, cfg)
    exp = ExperimentConfig(n_season_sims=200)
    # Seat 0 = G-score hero; rest random
    agents = {s: RandomAgent(info) for s in range(cfg.n_drafters)}
    agents[0] = GScoreAgent(info, temperature=0.0)
    result = run_draft(agents, cfg.n_drafters, cfg.n_starters, np.random.default_rng(1))
    ev = evaluate_rosters(result, gamelogs, cfg, exp, np.random.default_rng(2))
    others = np.mean([ev[s]['EC'] for s in range(1, cfg.n_drafters)])
    assert ev[0]['EC'] > others   # skill beats randomness by EC win-rate
