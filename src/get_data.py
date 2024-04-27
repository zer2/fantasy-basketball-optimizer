import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder, playergamelogs
import streamlit as st
from datetime import datetime
from nba_api.stats import endpoints as nba_endpoints
import numpy as np
import requests
import os

#cache this globally so it doesn't have to be rerun constantly 
@st.cache_resource(ttl = '1d') 
def get_current_season_data(season : int = 2024) -> tuple:
  """Get all box scores from the current season and calculate various running averages. Currently 2-week, 4-week, and season to date

  Args:
      season: int, year of season 
              defined by the second half of the season. E.g. season that ends in 2024 is 2024
  Returns:
      1) Dictionary of structure name of dataset -> dataframe, where the dataframes have fantasy-relevant player statistics
      2) Series of player -> minutes, calculated as average minutes per game played 
  """
           
  season_str = str(season -1) + '-' + str(season -2000)
  pgl_df = pd.concat(
      [
          playergamelogs.PlayerGameLogs(
              season_nullable=season_str, season_type_nullable=season_type
          ).player_game_logs.get_data_frame()
          for season_type in ["Regular Season"]
      ]
  )

                            
  renamer = st.session_state.params['current-season-api-renamer']
  pgl_df = pgl_df.rename(columns = renamer)

  expected_minutes_long_term = process_minutes(pgl_df)
                            
  pgl_df = pgl_df[list(renamer.values())].fillna(0)  

  four_weeks_ago = datetime.now() - pd.Timedelta(days = 28)
  two_weeks_ago = datetime.now() - pd.Timedelta(days = 14)

  four_week_subset = pgl_df[pd.to_datetime(pgl_df['Game Date']) >= four_weeks_ago].drop(columns = ['Game Date'])
  two_week_subset = pgl_df[pd.to_datetime(pgl_df['Game Date']) >= two_weeks_ago].drop(columns = ['Game Date'])
  full_subset = pgl_df.drop(columns = ['Game Date'])

  player_metadata = get_player_metadata()

  season_display_string = str(season - 1) + '-' + str(season)

  data_dict = {season_display_string + ': Four Week Average' : process_game_level_data(four_week_subset, player_metadata)
               ,season_display_string + ': Two Week Average' : process_game_level_data(two_week_subset, player_metadata)
               ,season_display_string + ': Season to Date' :  process_game_level_data(full_subset, player_metadata)
              }
                              
  return data_dict, expected_minutes_long_term 

def process_minutes(pgl_df: pd.DataFrame) -> pd.Series:
  #helper function which calculates average minutes per player in a dataset
  
  agg = pgl_df.groupby('Player')['MIN'].mean()
  agg.name = 'Minutes'
  return agg

  
#no need to cache this since it only gets re-run when current_season_data is refreshed
def process_game_level_data(df : pd.DataFrame, metadata : pd.Series) -> pd.DataFrame:
  """Convert box scores to the format needed for fantasy

  Args:
      df: dataframe of game data
      metadata: Series of relevant metadata to join on. Currently just position
  Returns:
      Dataframe of player-level statistics needed for fantasy
  """
  #convert a game level dataframe to a week-level dataframe
           
  agg_df = df.groupby('Player').mean().astype(float)
  agg_df.loc[:,'Free Throw %'] = np.where(agg_df['Free Throw Attempts'] > 0
                                          , agg_df['Free Throws Made']/agg_df['Free Throw Attempts']
                                          ,0)
  agg_df.loc[:,'Field Goal %'] = np.where(agg_df['Field Goal Attempts'] > 0
                                          , agg_df['Field Goals Made']/agg_df['Field Goal Attempts']
                                          ,0) 

  agg_df = agg_df.fillna(0).merge(metadata, left_index = True, right_index = True)

  game_counts = df.groupby('Player').count()
  agg_df.loc[:,'Games Played %'] = 1
  
  return agg_df.drop(columns = ['Free Throws Made','Field Goals Made'])

#cache this globally so it doesn't have to be rerun constantly. No need for refreshes- it won't change
@st.cache_resource
def get_historical_data():
  #get the one-time load of historical data stored as a CSV. In the future, it would perhaps be better to get this from snowflake
  
  full_df = pd.read_csv('./data/stat_df.csv')

  renamer = st.session_state.params['stat-df-renamer']
  full_df = full_df.rename(columns = renamer)

  full_df['Season'] = (full_df['Season'] - 1).astype(str) + '-' + full_df['Season'].astype(str)

  full_df.loc[:,'Free Throw %'] = full_df.loc[:,'Free Throws Made']/full_df.loc[:,'Free Throw Attempts']
  full_df.loc[:,'Field Goal %'] = full_df.loc[:,'Field Goals Made']/full_df.loc[:,'Field Goal Attempts']

  full_df['Position'] = full_df['Position'].fillna('NP')

  full_df = full_df.set_index(['Season','Player']).sort_index().fillna(0)  

  #adjust for the fact that historical data is week-based on game-based
  all_counting_stats = st.session_state.params['counting-statistics'] + st.session_state.params['volume-statistics']
  full_df[all_counting_stats] = full_df[all_counting_stats]/3

  return full_df


@st.cache_resource(ttl = '1d') 
def get_player_metadata() -> pd.Series:
   """Get player data from the NBA api

   Args:
      none
   Returns:
      Currently: A series of the form Player Name -> Position
   """
  
   playerindex = nba_endpoints.playerindex.PlayerIndex()
   data = playerindex.data_sets[0].get_dict()['data']
   headers = playerindex.data_sets[0].get_dict()['headers']

   players_df = pd.DataFrame(
    data, columns=headers
           )

   simplified = players_df['POSITION'].str[0]
   simplified.index = players_df['PLAYER_FIRST_NAME'] + ' ' + players_df['PLAYER_LAST_NAME']
   simplified.index.name = 'Player'
   simplified.name = 'Position'

   return simplified

@st.cache_resource(ttl = '1d') 
def get_darko_data(expected_minutes : pd.Series) -> dict[pd.DataFrame]:
  """Get DARKO predictions from stored CSV files

  Args:
      expected_minutes: Series of expecteed minutes projections, used to build DAKRO-L
  Returns:
      Dictionary, {'DARKO-L' : DARKO-L dataframe, 'DARKO-S' : DARKO-S dataframe}
  """
  skill_projections = pd.read_csv('data/DARKO_player_talent_2024-03-13.csv')
  per_game_projections = pd.read_csv('data/DARKO_daily_projections_2024-03-13.csv')
  all_darko = skill_projections.merge(per_game_projections)

  player_renamer = st.session_state.params['darko-player-renamer']
  all_darko['Player'] = all_darko['Player'].replace(player_renamer)
  
  all_darko = all_darko.set_index('Player') 

  #get fg% from skill projections: fg2% * (1-FG3ARate%) + fg3% * Fg3ARate%
  fg3_pct = all_darko['FG3%']
  fg2_pct = all_darko['FG2%']
  fg3a_pct = all_darko['FG3ARate%']	
  fg3a = all_darko['FG3A']

  oreb = all_darko['OREB'] 
  dreb = all_darko['DREB'] 

  all_darko.loc[:,'FG%'] = fg2_pct * (1- fg3a_pct) + fg3_pct * fg3a_pct
  all_darko.loc[:,'FG3M'] = fg3_pct * fg3a
  all_darko.loc[:,'REB'] = dreb + oreb 

  renamer = st.session_state.params['darko-renamer']
  all_darko = all_darko.rename(columns = renamer)

  player_metadata = get_player_metadata()
  all_darko = all_darko.merge(player_metadata, left_index = True, right_index = True)
    
  required_columns = st.session_state.params['counting-statistics'] + \
                    st.session_state.params['percentage-statistics'] + \
                    st.session_state.params['volume-statistics'] + \
                    st.session_state.params['other-columns']
  darko_short_term = get_darko_short_term(all_darko)[required_columns]
  darko_long_term = get_darko_long_term(all_darko, expected_minutes)[required_columns]

  return {'DARKO-S' : darko_short_term
           ,'DARKO-L' : darko_long_term}

def get_darko_short_term(all_darko : pd.DataFrame) -> pd.DataFrame:
  """Get a short term version of darko, based on next-game predictions for counting statistics

  Args:
      all_darko: Datafrmae of all raw DARKO forecasts
      expected_minutes: Series of expecteed minutes projections, used to build DAKRO-L
  Returns:
      Dataframe of predictions
  """
  
  darko_short_term = all_darko.fillna(0)  
  darko_short_term.loc[:,'Games Played %'] = 1

  return darko_short_term


def get_darko_long_term(all_darko : pd.DataFrame, expected_minutes : pd.Series) -> pd.DataFrame:
    """Get a long term version of darko, based on skill statistics and expected minutes
  
    Args:
        all_darko: Datafrmae of all raw DARKO forecasts
        expected_minutes: series of expected minutes to use for DARKO recalibration
    Returns:
        Dataframe of predictions
    """
    all_darko = all_darko.drop(columns = ['Minutes'])
    darko_long_term = all_darko.merge(expected_minutes, left_index = True, right_index = True)
  
    possesions = darko_long_term['Minutes']/48 * darko_long_term['Pace']/100
  
    free_throw_attempts = possesions * darko_long_term.loc[:,'FTA/100'] 
    free_throws_made = free_throw_attempts * darko_long_term.loc[:,'Free Throw %']

    three_attempts = possesions * darko_long_term.loc[:,'FG3A/100']
    threes_made = three_attempts * darko_long_term['FG3%']

    two_attempts = possesions * (darko_long_term.loc[:,'FGA/100'] - darko_long_term.loc[:,'FG3A/100'] ) 
    twos_made = two_attempts * darko_long_term['FG2%']
 
    darko_long_term.loc[:,'Points'] = 3*threes_made + 2*twos_made + free_throws_made
    darko_long_term.loc[:,'Rebounds'] = possesions * darko_long_term.loc[:,'REB/100'] 
    darko_long_term.loc[:,'Assists'] = possesions * darko_long_term.loc[:,'AST/100'] 
    darko_long_term.loc[:,'Steals'] = possesions * darko_long_term.loc[:,'STL/100'] 
    darko_long_term.loc[:,'Blocks'] = possesions * darko_long_term.loc[:,'BLK/100'] 
    darko_long_term.loc[:,'Threes'] =  threes_made
    darko_long_term.loc[:,'Turnovers'] = possesions * darko_long_term.loc[:,'TOV/100'] 
    darko_long_term.loc[:,'Free Throw Attempts'] = free_throw_attempts
    darko_long_term.loc[:,'Field Goal Attempts'] = two_attempts + three_attempts

    darko_long_term.loc[:,'Field Goal %'] = (twos_made + threes_made)/(two_attempts + three_attempts)
    darko_long_term.loc[:,'Free Throw %'] = darko_long_term.loc[:,'Free Throw %']

    darko_long_term.loc[:,'Games Played %'] = 1

    return darko_long_term

#setting show spinner to false prevents flickering
#data is cached locally so that different users can have different cuts loaded
@st.cache_data(show_spinner = False)
def get_specified_stats(historical_df : pd.DataFrame
                     , current_data : dict
                     , darko_data : dict
                     , dataset_name : str) -> pd.DataFrame:
  """fetch the data subset which will be used for the algorithms
  Args:
    historical_df: Dataframe of raw historical fantasy metrics by player/season
    current_data: dictionary mapping to datasets based on the current season
    darko_data: dictionary mapping to datasets based on DARKO
    dataset_name: the name of the dataset to fetch

  Returns:
    Dataframe of fantasy statistics 
  """
  #not sure but I think copying the dataset instead of slicing it prevents issues with 
  #overwriting the version in the cache
  if dataset_name in list(current_data.keys()):
      df = current_data[dataset_name].copy()
  elif 'DARKO' in dataset_name:
      df = darko_data[dataset_name].copy()
      os.write(1,bytes(str(df),'utf-8'))
  else:
      df = historical_df.loc[dataset_name].copy()
  
  #adjust for the display
  df[r'Free Throw %'] = (df[r'Free Throw %'] * 100).round(1)
  df[r'Field Goal %'] = (df[r'Field Goal %'] * 100).round(1)
  df[r'Games Played %'] = (df[r'Games Played %'] * 100).round(1)

  df.index = df.index + ' (' + df['Position'] + ')'
  df.index.name = 'Player'
  return df.round(2) 

@st.cache_data(show_spinner = False)
def get_nba_schedule():
    nba_schedule = requests.get(st.session_state.params['schedule-url']).json()
    game_dates = nba_schedule['leagueSchedule']['gameDates']

    def get_all_teams_playing(game_date):
         return [game['homeTeam']['teamTricode'] for game in game_date['games']] + \
                [game['awayTeam']['teamTricode'] for game in game_date['games']]

    teams_playing = { game_date['gameDate'] : get_all_teams_playing(game_date)
                            for game_date in game_dates
    }

    return teams_playing

