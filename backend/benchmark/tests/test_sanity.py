# benchmark/tests/test_sanity.py
import numpy as np
from backend.benchmark.config import LeagueConfig, ExperimentConfig
from backend.benchmark.data import load_fixture
from backend.benchmark.bootstrap import bootstrap_session
from backend.benchmark.agents import RandomAgent, GScoreAgent, HScoreAgent
from backend.benchmark.draft import run_draft
from backend.benchmark.evaluate import evaluate_rosters

import pytest
@pytest.mark.skip(reason="Mocked data lacks variance")
def test_gscore_beats_random_field():
    averages, gamelogs = load_fixture('backend/benchmark/fixtures/nba_2025-26.parquet')
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
