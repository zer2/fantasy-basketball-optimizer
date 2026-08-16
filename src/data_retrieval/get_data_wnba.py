import pandas as pd
import streamlit as st
import numpy as np

from nba_api.stats.endpoints import playergamelogs, playerindex

from src.helpers.helper_functions import gen_key, get_params
from src.data_retrieval.get_data import process_game_level_data

#the WNBA shares the stats.nba.com API with the NBA, distinguished by this league id
WNBA_LEAGUE_ID = '10'

def map_wnba_position(position, adjuster : dict) -> str:
  """Convert a WNBA API position string to the app's comma-separated convention

  The WNBA API reports positions as G/F/C or hyphenated combos like G-F. The
  adjuster maps each base letter to app positions, e.g. G -> 'PG,SG'

  Args:
      position: position string from the API. May be None/NaN
      adjuster: mapping of position letter -> comma-separated app positions
  Returns:
      Comma-separated position string, or 'NP' when the position is unknown
  """
  if position is None or (isinstance(position, float) and np.isnan(position)) or position == '':
     return 'NP'

  mapped = []
  for part in str(position).split('-'):
     for code in adjuster.get(part.strip(), part.strip()).split(','):
        if code not in mapped:
           mapped.append(code)

  return ','.join(mapped)

def calculate_games_played_pct(game_counts : pd.Series) -> pd.Series:
  """Calculate Games Played % relative to the most games played by any player

  Using the max within the dataset rather than the scheduled season length means
  this works mid-season and during the offseason alike

  Args:
      game_counts: Series of player -> games played
  Returns:
      Series of player -> games played fraction, in (0, 1]
  """
  return (game_counts/game_counts.max()).clip(upper = 1)

@st.cache_data(ttl = '1d')
def get_wnba_player_metadata(season : str) -> pd.Series:
  """Get player positions from the WNBA player index

  Args:
      season: season year as a string, e.g. '2026'
  Returns:
      Series of player -> position (comma-separated app convention)
  """
  index_df = playerindex.PlayerIndex(league_id = WNBA_LEAGUE_ID
                                     , season = season
                                     , historical_nullable = '1'
                                     , timeout = 60).get_data_frames()[0]

  index_df.loc[:,'Player'] = index_df['PLAYER_FIRST_NAME'] + ' ' + index_df['PLAYER_LAST_NAME']

  #when duplicate names exist, keep the most recently active player
  if 'TO_YEAR' in index_df.columns:
     index_df = index_df.sort_values('TO_YEAR', ascending = False)
  index_df = index_df.drop_duplicates(subset = 'Player', keep = 'first')

  adjuster = get_params()['rotowire-position-adjuster']
  positions = index_df.set_index('Player')['POSITION'].map(lambda p : map_wnba_position(p, adjuster))
  positions.name = 'Position'

  return positions

@st.cache_data(ttl = 3600, show_spinner = 'Fetching current season data from the WNBA stats API. This may take a moment')
def get_wnba_current_season_data(season : str) -> dict:
  """Get all box scores from the current WNBA season and calculate running averages

  Args:
      season: season year as a string, e.g. '2026'
  Returns:
      Dictionary of structure name of dataset -> dataframe, where the dataframes have fantasy-relevant player statistics
  """
  try:
    pgl_df = playergamelogs.PlayerGameLogs(league_id_nullable = WNBA_LEAGUE_ID
                                          , season_nullable = season
                                          , season_type_nullable = 'Regular Season'
                                          , timeout = 60).get_data_frames()[0]
    metadata = get_wnba_player_metadata(season)
  except Exception:
    st.error('Could not reach the WNBA stats API. Please try again later')
    st.stop()

  if len(pgl_df) == 0:
    st.error('No WNBA game data is available for the ' + season + ' season yet')
    st.stop()

  renamer = get_params()['current-season-api-renamer']
  pgl_df = pgl_df.rename(columns = renamer)
  pgl_df = pgl_df[list(renamer.values())].fillna(0)

  #make sure every player in the game logs has a position, even if the player index is missing them
  metadata = metadata.reindex(pd.unique(pgl_df['Player'])).fillna('NP')

  #windows are based on the latest game in the data rather than the current date,
  #so that the datasets remain usable during breaks and the offseason
  game_dates = pd.to_datetime(pgl_df['Game Date'])
  four_weeks_ago = game_dates.max() - pd.Timedelta(days = 28)
  two_weeks_ago = game_dates.max() - pd.Timedelta(days = 14)

  four_week_subset = pgl_df[game_dates >= four_weeks_ago].drop(columns = ['Game Date'])
  two_week_subset = pgl_df[game_dates >= two_weeks_ago].drop(columns = ['Game Date'])
  full_subset = pgl_df.drop(columns = ['Game Date'])

  games_played_pct = calculate_games_played_pct(full_subset.groupby('Player').size())

  data_dict = {season + ': Season to Date' : full_subset
               ,season + ': Four Week Average' : four_week_subset
               ,season + ': Two Week Average' : two_week_subset
              }

  for dataset_name, subset in data_dict.items():

    df = process_game_level_data(subset, metadata)

    #process_game_level_data sets a placeholder of 1; use the fraction of the season played instead
    df['Games Played %'] = games_played_pct.reindex(df.index).fillna(0).astype(float)

    df.index = df.index + ' (' + df['Position'] + ')'
    df.index.name = 'Player'

    data_dict[dataset_name] = df

  return data_dict

@st.cache_data(ttl = 3600, show_spinner = False)
def get_specified_wnba_stats(dataset_name : str, season : str) -> pd.DataFrame:
  """fetch the WNBA data subset which will be used for the algorithms

  Args:
    dataset_name: the name of the dataset to fetch
    season: season year as a string, used so the cache key changes with the season

  Returns:
    Dataframe of fantasy statistics, and a key for caching
  """
  data_dict = get_wnba_current_season_data(season)

  return data_dict[dataset_name].copy(), gen_key()
