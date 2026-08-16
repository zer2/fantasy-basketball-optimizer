# benchmark/tests/test_bootstrap.py
from benchmark.config import LeagueConfig
from benchmark.data import load_fixture
from benchmark.bootstrap import bootstrap_session

def test_bootstrap_builds_info():
    averages, _ = load_fixture('benchmark/fixtures/nba_2025-26.parquet')
    info = bootstrap_session(averages, LeagueConfig())
    for key in ['G-scores', 'X-scores', 'Positions', 'v'] if 'v' in info else ['G-scores', 'X-scores', 'Positions']:
        assert key in info
    # G-scores has a Total column and 9 category columns
    gs = info['G-scores']
    assert 'Total' in gs.columns
    # at least a full league's worth of rankable players
    assert len(gs) >= 12 * 9

def test_bootstrap_getters_resolve():
    import streamlit as st
    from src.helpers.helper_functions import get_selected_categories, get_n_drafters, get_scoring_format
    averages, _ = load_fixture('benchmark/fixtures/nba_2025-26.parquet')
    bootstrap_session(averages, LeagueConfig())
    assert get_n_drafters() == 12
    assert get_scoring_format() == 'Head to Head: Each Category'
    assert len(get_selected_categories()) == 9
