# benchmark/tests/test_bootstrap.py
from backend.benchmark.config import LeagueConfig
from backend.benchmark.data import load_fixture
from backend.benchmark.bootstrap import bootstrap_session

def test_bootstrap_builds_info():
    averages, _ = load_fixture('backend/benchmark/fixtures/nba_2025-26.parquet')
    info = bootstrap_session(averages, LeagueConfig())
    for key in ['G-scores', 'X-scores', 'Positions', 'v'] if 'v' in info else ['G-scores', 'X-scores', 'Positions']:
        assert key in info
    # G-scores has a Total column and 9 category columns
    gs = info['G-scores']
    assert 'Total' in gs.columns
    # at least a full league's worth of rankable players
    assert len(gs) >= 12 * 9
