# benchmark/tests/test_mcts_leaf.py
import numpy as np
from benchmark.config import LeagueConfig
from benchmark.data import load_fixture
from benchmark.bootstrap import bootstrap_session
from benchmark.mcts import leaf_value
from src.math.algorithm_agents import HAgent
from src.helpers.helper_functions import get_data_from_session_state

def _full_rosters(info, cfg):
    pool = list(info['G-scores'].index)
    assignments, k = {}, 0
    for seat in range(cfg.n_drafters):
        assignments[seat] = pool[k:k + cfg.n_starters]; k += cfg.n_starters
    return assignments

def test_leaf_value_is_scalar_probability():
    averages, _ = load_fixture('benchmark/fixtures/nba_2025-26.parquet')
    cfg = LeagueConfig()
    info = bootstrap_session(averages, cfg)
    H = HAgent(info=info, omega=cfg.omega, gamma=cfg.gamma, n_picks=cfg.n_starters,
               n_drafters=cfg.n_drafters, dynamic=False, beth=cfg.beth,
               scoring_format=cfg.scoring_format)
    assignments = _full_rosters(info, cfg)
    v = leaf_value(H, assignments, 0)
    assert 0.0 <= v <= 1.0
