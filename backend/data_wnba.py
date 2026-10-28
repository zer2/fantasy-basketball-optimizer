import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playergamelogs, playerindex

WNBA_LEAGUE_ID = '10'

def map_wnba_position(position, adjuster: dict) -> str:
    if position is None or (isinstance(position, float) and np.isnan(position)) or position == '':
        return 'NP'
    mapped = []
    for part in str(position).split('-'):
        for code in adjuster.get(part.strip(), part.strip()).split(','):
            if code not in mapped:
                mapped.append(code)
    return ','.join(mapped)

def calculate_games_played_pct(game_counts: pd.Series) -> pd.Series:
    return (game_counts / game_counts.max()).clip(upper=1)

def process_game_level_data(df: pd.DataFrame, metadata: pd.Series) -> pd.DataFrame:
    agg_df = df.groupby('Player').mean(numeric_only=True).astype(float)
    if 'Free Throws Made' in agg_df.columns and 'Free Throw Attempts' in agg_df.columns:
        agg_df['Free Throw %'] = np.where(agg_df['Free Throw Attempts'] > 0,
                                          agg_df['Free Throws Made'] / agg_df['Free Throw Attempts'], 0)
    if 'Field Goals Made' in agg_df.columns and 'Field Goal Attempts' in agg_df.columns:
        agg_df['Field Goal %'] = np.where(agg_df['Field Goal Attempts'] > 0,
                                          agg_df['Field Goals Made'] / agg_df['Field Goal Attempts'], 0)
    if 'Threes' in agg_df.columns and 'Three Attempts' in agg_df.columns:
        agg_df['Three %'] = np.where(agg_df['Three Attempts'] > 0,
                                     agg_df['Threes'] / agg_df['Three Attempts'], 0)
    res = agg_df.join(metadata, how='inner')
    return res

def get_wnba_player_metadata(season: str, params: dict) -> pd.Series:
    index_df = playerindex.PlayerIndex(league_id=WNBA_LEAGUE_ID, season=season, historical_nullable='1', timeout=60).get_data_frames()[0]
    index_df['Player'] = index_df['PLAYER_FIRST_NAME'] + ' ' + index_df['PLAYER_LAST_NAME']
    if 'TO_YEAR' in index_df.columns:
        index_df = index_df.sort_values('TO_YEAR', ascending=False)
    index_df = index_df.drop_duplicates(subset='Player', keep='first')
    adjuster = params['rotowire-position-adjuster']
    positions = index_df.set_index('Player')['POSITION'].map(lambda p: map_wnba_position(p, adjuster))
    positions.name = 'Position'
    return positions

def get_wnba_current_season_data(season: str, params: dict) -> dict:
    try:
        pgl_df = playergamelogs.PlayerGameLogs(league_id_nullable=WNBA_LEAGUE_ID, season_nullable=season, season_type_nullable='Regular Season', timeout=60).get_data_frames()[0]
        metadata = get_wnba_player_metadata(season, params)
    except Exception as e:
        raise RuntimeError(f"Could not reach the WNBA stats API: {e}")

    if len(pgl_df) == 0:
        raise RuntimeError(f"No WNBA game data is available for the {season} season yet.")

    renamer = params['current-season-api-renamer']
    pgl_df = pgl_df.rename(columns=renamer)
    pgl_df = pgl_df[list(renamer.values())].fillna(0)

    metadata = metadata.reindex(pd.unique(pgl_df['Player'])).fillna('NP')

    game_dates = pd.to_datetime(pgl_df['Game Date'])
    four_weeks_ago = game_dates.max() - pd.Timedelta(days=28)
    two_weeks_ago = game_dates.max() - pd.Timedelta(days=14)

    data_dict = {
        f'{season}: Season to Date': pgl_df.drop(columns=['Game Date']),
        f'{season}: Four Week Average': pgl_df[game_dates >= four_weeks_ago].drop(columns=['Game Date']),
        f'{season}: Two Week Average': pgl_df[game_dates >= two_weeks_ago].drop(columns=['Game Date']),
    }

    games_played_pct = calculate_games_played_pct(data_dict[f'{season}: Season to Date'].groupby('Player').size())

    for dataset_name, subset in data_dict.items():
        df = process_game_level_data(subset, metadata)
        df['Games Played %'] = games_played_pct.reindex(df.index).fillna(0).astype(float)
        df.index = df.index + ' (' + df['Position'] + ')'
        df.index.name = 'Player'
        data_dict[dataset_name] = df

    return data_dict

def get_specified_wnba_stats(season: str, params: dict) -> pd.DataFrame:
    data_dict = get_wnba_current_season_data(season, params)
    dataset_name = f'{season}: Season to Date'
    return data_dict[dataset_name].copy()
