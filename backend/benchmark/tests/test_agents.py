# benchmark/tests/test_agents.py
import numpy as np
import pandas as pd
from backend.benchmark.config import LeagueConfig
from backend.benchmark.data import load_fixture
from backend.benchmark.bootstrap import bootstrap_session
from backend.benchmark.agents import RandomAgent, GScoreAgent, HScoreAgent

def _setup():
    averages, _ = load_fixture('backend/benchmark/fixtures/nba_2025-26.parquet')
    info = bootstrap_session(averages, LeagueConfig())
    return info

def test_gscore_agent_picks_available_player():
    info = _setup()
    agent = GScoreAgent(info, temperature=0.0)
    empty = {i: [] for i in range(12)}
    pick = agent.make_pick(empty, 0, np.random.default_rng(0))
    # deterministic: highest total G-score
    top = info['G-scores'].sort_values('Total', ascending=False).index[0]
    assert pick == top

def test_agent_never_repicks_taken_player():
    info = _setup()
    agent = GScoreAgent(info, temperature=0.0)
    top = info['G-scores'].sort_values('Total', ascending=False).index[0]
    assignments = {0: [top], 1: [], 2: []}
    for i in range(3, 12):
        assignments[i] = []
    pick = agent.make_pick(assignments, 1, np.random.default_rng(0))
    assert pick != top

def test_hscore_agent_uses_cached_ordering():
    info = _setup()
    agent = HScoreAgent(info, LeagueConfig(), temperature=0.0)
    empty = {i: [] for i in range(12)}
    pick = agent.make_pick(empty, 0, np.random.default_rng(0))
    assert pick in info['G-scores'].index
