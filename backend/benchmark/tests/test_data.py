# benchmark/tests/test_data.py
import pandas as pd
from backend.benchmark.data import build_datasets, RENAMER

def _fake_raw():
    # two players, two games each; columns mirror nba_api PlayerGameLogs
    rows = [
        # PLAYER_NAME, GAME_DATE, MIN, PTS, REB, AST, STL, BLK, TOV, FG3M, FGM, FGA, FTM, FTA, OREB, DREB
        ('A Player', '2025-11-01T00:00:00', 30, 20, 10, 5, 1, 1, 3, 2, 8, 15, 2, 4, 3, 7),
        ('A Player', '2025-11-03T00:00:00', 32, 24, 12, 7, 2, 0, 2, 3, 9, 16, 3, 3, 4, 8),
        ('B Player', '2025-11-01T00:00:00', 25, 10, 4, 8, 0, 0, 4, 1, 4, 10, 1, 2, 1, 3),
        ('B Player', '2025-11-03T00:00:00', 28, 14, 5, 9, 1, 1, 1, 0, 5, 11, 4, 5, 2, 3),
    ]
    cols = ['PLAYER_NAME','GAME_DATE','MIN','PTS','REB','AST','STL','BLK','TOV','FG3M','FGM','FGA','FTM','FTA','OREB','DREB']
    return pd.DataFrame(rows, columns=cols)

def test_build_datasets_averages_shape():
    positions = pd.Series({'A Player': 'PG,SG', 'B Player': 'C'}, name='Position')
    averages, gamelogs = build_datasets(_fake_raw(), positions)

    # averages indexed by "Name (Position)" like the app does
    assert any(idx.startswith('A Player (') for idx in averages.index)
    # ratio components present, computed as means of per-game makes/attempts
    assert 'Field Goal Attempts' in averages.columns
    assert 'Free Throw Attempts' in averages.columns
    assert 'Position' in averages.columns
    assert 'Games Played %' in averages.columns

def test_build_datasets_gamelogs_are_per_game():
    positions = pd.Series({'A Player': 'PG,SG', 'B Player': 'C'}, name='Position')
    _, gamelogs = build_datasets(_fake_raw(), positions)
    # one row per player-game (4 rows), keeping made/attempt pairs for ratio evaluation
    assert len(gamelogs) == 4
    for col in ['Points','Rebounds','Assists','Steals','Blocks','Turnovers','Threes',
                'Field Goals Made','Field Goal Attempts','Free Throws Made','Free Throw Attempts']:
        assert col in gamelogs.columns
    assert set(gamelogs['Player'].unique()) == {'A Player', 'B Player'}
