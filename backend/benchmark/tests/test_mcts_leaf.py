# benchmark/tests/test_mcts_leaf.py
import numpy as np
from backend.benchmark.config import LeagueConfig
from backend.benchmark.data import load_fixture
from backend.benchmark.bootstrap import bootstrap_session
from backend.benchmark.mcts import leaf_value
from backend.math.algorithm_agents import HAgent
import yaml

def _full_rosters(info, cfg):
    pool = list(info['G-scores'].index)
    assignments, k = {}, 0
    for seat in range(cfg.n_drafters):
        assignments[seat] = pool[k:k + cfg.n_starters]; k += cfg.n_starters
    return assignments

def test_leaf_value_is_scalar_probability():
    averages, _ = load_fixture('backend/benchmark/fixtures/nba_2025-26.parquet')
    cfg = LeagueConfig()
    info = bootstrap_session(averages, cfg)
    with open('parameters.yaml', 'r') as f:
        _params_leaf = yaml.safe_load(f)[cfg.league]
    H = HAgent(info=info, omega=cfg.omega, gamma=cfg.gamma, n_picks=cfg.n_starters,
               n_drafters=cfg.n_drafters, dynamic=False,
               scoring_format=cfg.scoring_format,
               sport=cfg.league, params=_params_leaf,
               slot_counts=({'PG': 1, 'SG': 1, 'SF': 1, 'PF': 1, 'C': 1, 'G': 1, 'F': 1, 'Util': 2} if cfg.n_starters == 9 else {'Util': cfg.n_starters}))
    assignments = _full_rosters(info, cfg)
    v = leaf_value(H, assignments, 0)
    assert 0.0 <= v <= 1.0
