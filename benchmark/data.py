# benchmark/data.py
"""Pull real NBA game logs and derive (a) season-average projections and
(b) per-game box scores for the independent evaluator. Read-only use of the engine."""
import os
import pandas as pd

from src.data_retrieval.get_data import process_game_level_data

# nba_api PlayerGameLogs -> app category names. Matches parameters.yaml current-season-api-renamer,
# extended with the columns the evaluator needs per game.
RENAMER = {
    'PLAYER_NAME': 'Player', 'GAME_DATE': 'Game Date', 'MIN': 'MIN',
    'PTS': 'Points', 'REB': 'Rebounds', 'AST': 'Assists', 'STL': 'Steals',
    'BLK': 'Blocks', 'TOV': 'Turnovers', 'FG3M': 'Threes',
    'FTA': 'Free Throw Attempts', 'FTM': 'Free Throws Made',
    'FGA': 'Field Goal Attempts', 'FGM': 'Field Goals Made',
    'OREB': 'Off Rebounds', 'DREB': 'Def Rebounds',
}

# Per-game columns the evaluator resamples (counting cats + ratio components).
GAMELOG_STAT_COLS = [
    'Points', 'Rebounds', 'Assists', 'Steals', 'Blocks', 'Turnovers', 'Threes',
    'Off Rebounds', 'Def Rebounds',
    'Field Goals Made', 'Field Goal Attempts', 'Free Throws Made', 'Free Throw Attempts',
]

def _rename(raw: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in RENAMER if c in raw.columns]
    return raw[keep].rename(columns={k: RENAMER[k] for k in keep})

def build_datasets(gamelogs_raw: pd.DataFrame, positions: pd.Series):
    """Return (averages_df, gamelogs_df).

    averages_df: one row per "Name (Position)", shape process_player_data expects.
    gamelogs_df: tidy per-game rows with counting cats and ratio components.
    """
    renamed = _rename(gamelogs_raw).fillna(0)

    # (a) Season averages via the engine's own converter (keeps ratio math consistent).
    metadata = positions.reindex(pd.unique(renamed['Player'])).fillna('NP')
    metadata.name = 'Position'
    # process_game_level_data does df.groupby('Player').mean(); a non-numeric 'Game Date'
    # column would raise inside its bare `except` and yield an empty frame, so drop it
    # here (mirrors the WNBA path which drops 'Game Date' before calling). Benchmark-side
    # only; src/ untouched.
    averages_input = renamed.drop(columns=['Game Date'], errors='ignore').copy()
    averages = process_game_level_data(averages_input, metadata)

    # process_game_level_data returns a player-name index without the "(Position)" suffix;
    # add it the way get_data / get_data_wnba callers do so the index matches the app's
    # "Name (Position)" convention that process_player_data expects.
    averages.index = averages.index + ' (' + averages['Position'] + ')'
    averages.index.name = 'Player'

    # (b) Per-game box scores for the evaluator (Player column retained, not indexed).
    stat_cols = [c for c in GAMELOG_STAT_COLS if c in renamed.columns]
    gamelogs = renamed[['Player'] + stat_cols].reset_index(drop=True)

    return averages, gamelogs


def save_fixture(averages: pd.DataFrame, gamelogs: pd.DataFrame, path) -> None:
    """Store both frames in one parquet file under distinct keys via a MultiIndex marker."""
    import os
    import pyarrow  # noqa: F401  (fail loudly if parquet engine missing)
    parent = os.path.dirname(str(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    averages.to_parquet(str(path).replace('.parquet', '.averages.parquet'))
    gamelogs.to_parquet(str(path).replace('.parquet', '.gamelogs.parquet'))

def load_fixture(path):
    averages = pd.read_parquet(str(path).replace('.parquet', '.averages.parquet'))
    gamelogs = pd.read_parquet(str(path).replace('.parquet', '.gamelogs.parquet'))
    return averages, gamelogs

def pull_and_snapshot(season: str, out_path: str):
    """One-time network pull of a full season. Not run in tests."""
    import yaml
    from nba_api.stats.endpoints import playergamelogs, playerindex
    # Reuse zer2's WNBA position mapper (read-only) to convert raw NBA letters
    # (G, F, C, G-F, ...) to the app's base_list convention (PG,SG,SF,PF,C).
    # Without this, check_single_player_eligibility matches nothing and the draft stalls.
    from src.data_retrieval.get_data_wnba import map_wnba_position

    raw = playergamelogs.PlayerGameLogs(
        league_id_nullable='00', season_nullable=season,
        season_type_nullable='Regular Season', timeout=60).get_data_frames()[0]
    idx = playerindex.PlayerIndex(
        league_id='00', season=season, timeout=60).get_data_frames()[0]
    idx['Player'] = idx['PLAYER_FIRST_NAME'] + ' ' + idx['PLAYER_LAST_NAME']
    raw_positions = idx.drop_duplicates('Player').set_index('Player')['POSITION'].fillna('NP')

    with open('parameters.yaml', 'r') as f:
        adjuster = yaml.safe_load(f)['NBA']['rotowire-position-adjuster']
    positions = raw_positions.map(lambda p: map_wnba_position(p, adjuster))
    positions.name = 'Position'

    averages, gamelogs = build_datasets(raw, positions)
    save_fixture(averages, gamelogs, out_path)
    return averages, gamelogs
