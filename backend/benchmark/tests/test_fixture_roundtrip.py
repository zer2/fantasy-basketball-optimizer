# benchmark/tests/test_fixture_roundtrip.py
import pandas as pd
from backend.benchmark.data import save_fixture, load_fixture

def test_fixture_roundtrip(tmp_path):
    averages = pd.DataFrame({'Points': [20.0], 'Position': ['C'], 'Games Played %': [1.0]},
                            index=pd.Index(['X (C)'], name='Player'))
    gamelogs = pd.DataFrame({'Player': ['X', 'X'], 'Points': [18, 22]})
    path = tmp_path / 'fix.parquet'
    save_fixture(averages, gamelogs, path)
    a2, g2 = load_fixture(path)
    pd.testing.assert_frame_equal(a2, averages)
    pd.testing.assert_frame_equal(g2, gamelogs)
