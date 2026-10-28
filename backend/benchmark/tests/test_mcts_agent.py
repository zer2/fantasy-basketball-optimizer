# benchmark/tests/test_mcts_agent.py
import numpy as np
from backend.benchmark.config import LeagueConfig, ExperimentConfig
from backend.benchmark.data import load_fixture
from backend.benchmark.bootstrap import bootstrap_session
from backend.benchmark.mcts import MCTSAgent

def test_mcts_returns_eligible_available_pick():
    averages, _ = load_fixture('backend/benchmark/fixtures/nba_2025-26.parquet')
    cfg = LeagueConfig()
    info = bootstrap_session(averages, cfg)
    agent = MCTSAgent(info, cfg, temperature=1.0, n_simulations=20, top_k=10, c_puct=1.4)
    empty = {i: [] for i in range(12)}
    pick = agent.make_pick(empty, 0, np.random.default_rng(0))
    assert pick in info['G-scores'].index

def test_mcts_is_deterministic_under_fixed_seed():
    averages, _ = load_fixture('backend/benchmark/fixtures/nba_2025-26.parquet')
    cfg = LeagueConfig()
    info = bootstrap_session(averages, cfg)
    a1 = MCTSAgent(info, cfg, temperature=1.0, n_simulations=20, top_k=10, c_puct=1.4)
    a2 = MCTSAgent(info, cfg, temperature=1.0, n_simulations=20, top_k=10, c_puct=1.4)
    empty = {i: [] for i in range(12)}
    p1 = a1.make_pick(empty, 0, np.random.default_rng(42))
    p2 = a2.make_pick(empty, 0, np.random.default_rng(42))
    assert p1 == p2
