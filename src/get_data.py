import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder, playergamelogs
import streamlit as st
from datetime import datetime
from nba_api.stats import endpoints as nba_endpoints
import numpy as np
import requests
import os
import snowflake.connector
from src.helper_functions import get_n_games
from unidecode import unidecode

@st.cache_resource()
def get_data_from_snowflake(table_name):
   
   con = snowflake.connector.connect(
        user=st.secrets['SNOWFLAKE_USER']
        ,password=st.secrets['SNOWFLAKE_PASSWORD']
        ,account='aib52055.us-east-1'
        ,database = 'FANTASYOPTIMIZER'
        ,schema = 'FANTASYBASKETBALLOPTIMIZER'
    )
   
   df = con.cursor().execute('SELECT * FROM ' + table_name).fetch_pandas_all()

   return df

def get_yahoo_key_to_name_mapper():
   return get_data_from_snowflake('YAHOO_ID_TO_NAME_VIEW')

def get_yahoo_key_to_position_eligibility(season = 2024):
   return get_data_from_snowflake('YAHOO_ID_TO_NAME_VIEW')


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

  #ZR: I believe this function currently does not work 
           
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

  try: 
           
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
  
  except:
     return pd.DataFrame()

#cache this globally so it doesn't have to be rerun constantly. No need for refreshes- it won't change
@st.cache_resource(ttl = '1d') 
def get_historical_data():  
  full_df = get_data_from_snowflake('AVERAGE_NUMBERS_VIEW_2')

  renamer = st.session_state.params['stat-df-renamer']
  full_df = full_df.rename(columns = renamer)

  full_df = full_df.apply(pd.to_numeric, errors='ignore')

  #full_df['Season'] = (full_df['Season'] - 1).astype(str) + '-' + full_df['Season'].astype(str)

  full_df.loc[:,'Free Throw %'] = full_df.loc[:,'Free Throws Made']/full_df.loc[:,'Free Throw Attempts']
  full_df.loc[:,'Field Goal %'] = full_df.loc[:,'Field Goals Made']/full_df.loc[:,'Field Goal Attempts']

  full_df['Position'] = full_df['Position'].fillna('NP')

  full_df = full_df.set_index(['Season','Player']).sort_index().fillna(0)  

  return full_df


@st.cache_resource(ttl = '1d') 
def get_player_metadata() -> pd.Series:
   """Get player data from the NBA api

   Args:
      none
   Returns:
      Currently: A series of the form Player Name -> Position
   """

   return st.session_state.player_stats['Position']

   '''
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
   '''

@st.cache_resource(ttl = '1d') 
def get_darko_data(expected_minutes : pd.Series) -> dict[pd.DataFrame]:
  """Get DARKO predictions from stored CSV files

  Args:
      expected_minutes: Series of expecteed minutes projections, used to build DAKRO-L
  Returns:
      Dictionary, {'DARKO-L' : DARKO-L dataframe, 'DARKO-S' : DARKO-S dataframe}
  """
  darko_df = get_data_from_snowflake('DARKO_VIEW')
  renamer = st.session_state.params['darko-renamer']
  darko_df = darko_df.rename(columns = renamer)
  darko_df = darko_df.apply(pd.to_numeric, errors='ignore')

  darko_df['Position'] = darko_df['Position'].fillna('NP')
  darko_df = darko_df.set_index(['Player']).sort_index().fillna(0)  
    
  required_columns = st.session_state.params['counting-statistics'] + \
                    list(st.session_state.params['ratio-statistics'].keys()) + \
                    [ratio_stat_info['volume-statistic'] for ratio_stat_info in st.session_state.params['ratio-statistics'].values()] + \
                    st.session_state.params['other-columns']
  darko_short_term = get_darko_short_term(darko_df)[required_columns]
  darko_long_term = get_darko_long_term(darko_df, expected_minutes)[required_columns]

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
    darko_long_term = all_darko.merge(expected_minutes, left_index = True, right_index = True)

    minutes_adjustment_factor = darko_long_term['Minutes']/darko_long_term['Forecasted Minutes']
 
    darko_long_term.loc[:,'Points'] = darko_long_term.loc[:,'Points'] * minutes_adjustment_factor
    darko_long_term.loc[:,'Rebounds'] = darko_long_term.loc[:,'Rebounds'] * minutes_adjustment_factor
    darko_long_term.loc[:,'Assists'] = darko_long_term.loc[:,'Assists'] * minutes_adjustment_factor
    darko_long_term.loc[:,'Steals'] = darko_long_term.loc[:,'Steals'] * minutes_adjustment_factor
    darko_long_term.loc[:,'Blocks'] = darko_long_term.loc[:,'Blocks'] * minutes_adjustment_factor
    darko_long_term.loc[:,'Threes'] =  darko_long_term.loc[:,'Threes'] * minutes_adjustment_factor
    darko_long_term.loc[:,'Turnovers'] = darko_long_term.loc[:,'Turnovers'] * minutes_adjustment_factor
    darko_long_term.loc[:,'Free Throw Attempts'] = darko_long_term.loc[:,'Free Throw Attempts'] * minutes_adjustment_factor
    darko_long_term.loc[:,'Field Goal Attempts'] = darko_long_term.loc[:,'Field Goal Attempts'] * minutes_adjustment_factor
    darko_long_term.loc[:,'Games Played %'] = 1

    return darko_long_term

#setting show spinner to false prevents flickering
#data is cached locally so that different users can have different cuts loaded
@st.cache_data(show_spinner = False, ttl = 3600)
def get_specified_stats(dataset_name : str
                     , default_key : int = None) -> pd.DataFrame:
  """fetch the data subset which will be used for the algorithms
  Args:
    dataset_name: the name of the dataset to fetch
    default_key: used for caching. Increments whenever the default dataset is changed, so that this function
                 gets rerun for sure whenever the default dataset changes

  Returns:
    Dataframe of fantasy statistics 
  """
  #not sure but I think copying the dataset instead of slicing it prevents issues with 
  #overwriting the version in the cache
  if st.session_state.league in ('NBA','WNBA'):

    historical_df = get_historical_data()
    current_data, expected_minutes = get_current_season_data()
    darko_data = get_darko_data(expected_minutes)
    htb_data = get_data_from_snowflake('HTB_PROJECTION_TABLE')

    if dataset_name in list(current_data.keys()):
        df = current_data[dataset_name].copy()
    elif 'DARKO' in dataset_name:
        df = darko_data[dataset_name].copy()
    elif 'Hashtag' in dataset_name:
        df = process_htb_data(htb_data)
    elif 'RotoWire' in dataset_name:
        if 'rotowire_data' in st.session_state:
            raw_df = st.session_state.rotowire_data
            df = process_basketball_rotowire_data(raw_df)
        else:
            st.error('Error! No rotowire data found: this should not happen')
    elif 'Basketball Monster' in dataset_name:
       if 'bbm_data' in st.session_state:
            raw_df = st.session_state.bbm_data
            df = process_basketball_monster_data(raw_df
                                                 , default_projections = process_htb_data(htb_data))
       else:
            st.error('Error! No Basketball Monster data found: this should not happen')

    else:
        df = historical_df.loc[dataset_name].copy()  
    #adjust for the display
    df[r'Free Throw %'] = (df[r'Free Throw %'] * 100).round(1)
    df[r'Field Goal %'] = (df[r'Field Goal %'] * 100).round(1)
    df[r'Games Played %'] = (df[r'Games Played %'] * 100).round(1)

    df.index = df.index + ' (' + df['Position'] + ')'
    df.index.name = 'Player'
    return df.round(2) 
  
  elif st.session_state.league in ('MLB'):
    if 'rotowire_data' in st.session_state:
        raw_df = st.session_state.rotowire_data
        df = process_baseball_rotowire_data(raw_df)   
    
    df[r'Batting Average'] = (df[r'Batting Average'] * 100).round(1)
    df[r'Games Played %'] = (df[r'Games Played %'] * 100).round(1)

    df.index = df.index + ' (' + df['Position'] + ')'
    df.index.name = 'Player'

    return df.round(2) 
  
@st.cache_data(ttl = 3600)
def combine_nba_projections(hashtag_upload
                            , rotowire_upload
                            , bbm_upload
                            , hashtag_slider
                            , rotowire_slider
                            , bbm_slider):
    slider_sum = hashtag_slider + rotowire_slider + bbm_slider

    st.write('BOOOOO')

    hashtag_stats = None if hashtag_upload is None else process_htb_data(hashtag_upload)
    rotowire_stats = None if rotowire_upload is None else process_basketball_rotowire_data(rotowire_upload)
    bbm_stats = None if bbm_upload is None else process_basketball_monster_data(bbm_upload)

    hashtag_weight = [hashtag_slider] if hashtag_upload is not None else []
    rotowire_weight = [rotowire_slider] if rotowire_upload is not None else []
    bbm_weight = [bbm_slider] if bbm_upload is not None else []

    weights = hashtag_weight + rotowire_weight + bbm_weight

    all_players = set([] if hashtag_stats is None else [p for p in hashtag_stats.index] + \
                  [] if rotowire_stats is None else [p for p in rotowire_stats.index] + \
                  [] if bbm_stats is None else [p for p in bbm_stats.index])
    
    df =  pd.concat({'HTB' : hashtag_stats 
                        ,'RotoWire' : rotowire_stats
                        ,'BBM' : bbm_stats}, names = ['Source'])
    
    new_index = pd.MultiIndex.from_product([['HTB','RotoWire','BBM'], all_players], names = ['Source','Player'])
    df = df.reindex(new_index)
    
    weights = [hashtag_slider, rotowire_slider, bbm_slider]
    
    df = df.groupby('Player') \
                .agg(lambda x: np.ma.average(np.ma.MaskedArray(x, mask=np.isnan(x)), weights = weights) \
                    if np.issubdtype(x.dtype, np.number) \
                    else x[0])
            
    df[r'Free Throw %'] = (df[r'Free Throw %'] * 100).round(1)
    df[r'Field Goal %'] = (df[r'Field Goal %'] * 100).round(1)
    df[r'Games Played %'] = (df[r'Games Played %'] * 100).round(1)

    df['Position'] = df['Position'].fillna('NP')
    df = df.fillna(0)

    df.index = df.index + ' (' + df['Position'] + ')'
    df.index.name = 'Player'
    return df.round(2) 
  
def process_baseball_rotowire_data(raw_df):
   
   raw_df.loc[:,'Games Played %'] = 1 #we need to fix this later
   raw_df['AVG'] = raw_df['AVG']/100
   raw_df.loc[:,'Pos'] = raw_df.loc[:,'Pos'].map(st.session_state.params['rotowire-position-adjuster'])

   raw_df = raw_df.rename(columns = st.session_state.params['rotowire-renamer'])

   #baseball has some duplicate player names, which we need to deal with
   is_duplicate_player = raw_df.groupby('Player')['Player'].transform('size') > 1
   raw_df.loc[:,'Player'] = np.where(is_duplicate_player
                                     ,raw_df['Player'] + ' (' + raw_df['Team'] + ')'
                                     ,raw_df['Player']
                                     )
   
   is_pitcher = raw_df['Position'].str.contains('P')
   pitcher_stats = st.session_state.params['pitcher_stats']
   batter_stats = st.session_state.params['batter_stats']

   raw_df.loc[is_pitcher,pitcher_stats] = raw_df[is_pitcher][pitcher_stats].fillna(0)
   raw_df.loc[~is_pitcher,batter_stats] = raw_df[~is_pitcher][batter_stats].fillna(0)

   raw_df = raw_df.set_index('Player')

   required_columns = st.session_state.params['counting-statistics'] + \
                    list(st.session_state.params['ratio-statistics'].keys()) + \
                    [ratio_stat_info['volume-statistic'] for ratio_stat_info in st.session_state.params['ratio-statistics'].values()] + \
                    st.session_state.params['other-columns']
   
   raw_df = raw_df[list(set(required_columns))]

   return raw_df

@st.cache_data()
def process_basketball_rotowire_data(raw_df):
   
   raw_df.loc[:,'Games Played %'] = raw_df['G']/get_n_games()
   raw_df['FG%'] = raw_df['FG%']/100
   raw_df['FT%'] = raw_df['FT%']/100
   raw_df.loc[:,'Pos'] = raw_df.loc[:,'Pos'].map(st.session_state.params['rotowire-position-adjuster'])

   raw_df = raw_df.rename(columns = st.session_state.params['rotowire-renamer'])
   
   raw_df = raw_df.set_index('Player')

   required_columns = st.session_state.params['counting-statistics'] + \
                    list(st.session_state.params['ratio-statistics'].keys()) + \
                    [ratio_stat_info['volume-statistic'] for ratio_stat_info in st.session_state.params['ratio-statistics'].values()] + \
                    st.session_state.params['other-columns']
   
   raw_df = raw_df[list(set(required_columns))]

   return raw_df

@st.cache_data()
def process_htb_data(raw_df):
   raw_df = raw_df.rename(columns = st.session_state.params['htb-renamer'])
   raw_df.loc[:,'Games Played %'] = raw_df['Games Played']/get_n_games()

   raw_df['Position'] = raw_df['Position'].str.replace('/',',')
   
   def name_renamer(name):
      name = unidecode(name)
      name = ' '.join(name.split(' ')[0:2])
      if name == 'Nicolas Claxton':
         name = 'Nic Claxton'
      if name == 'C.J. McCollum':
         name = 'CJ McCollum'
      if name == 'R.J. Barrett':
         name = 'RJ Barrett'
      if name == 'Herb Jones':
         name = 'Herbert Jones'
      if name == 'Cam Johnson':
         name = 'Cameron Johnson'
      if name == 'O.G. Anunoby':
         name = 'OG Anunoby'
      return name
      
   raw_df['Player'] = [name_renamer(name) for name in raw_df['Player']]

   raw_df = raw_df.set_index('Player')

   required_columns = st.session_state.params['counting-statistics'] + \
                    list(st.session_state.params['ratio-statistics'].keys()) + \
                    [ratio_stat_info['volume-statistic'] for ratio_stat_info in st.session_state.params['ratio-statistics'].values()] + \
                    st.session_state.params['other-columns']
   
   raw_df = raw_df[list(set(required_columns))]

   return raw_df
def process_basketball_monster_data(raw_df):
   
   raw_df = raw_df.rename(columns = st.session_state.params['bbm-renamer'])
   raw_df.loc[:,'Games Played %'] = raw_df['Games Played']/get_n_games()

   raw_df['Position'] = raw_df['Position'].str.replace('/',',')
   
   def name_renamer(name):
      name = ' '.join(name.split(' ')[0:2])
      if name == 'Nicolas Claxton':
         name = 'Nic Claxton'
      if name == 'C.J. McCollum':
         name = 'CJ McCollum'
      if name == 'R.J. Barrett':
         name = 'RJ Barrett'
      if name == 'Herb Jones':
         name = 'Herbert Jones'
      if name == 'Cam Johnson':
         name = 'Cameron Johnson'
      if name == 'O.G. Anunoby':
         name = 'OG Anunoby'
      return name
      
   raw_df['Player'] = [name_renamer(name) for name in raw_df['Player']]

   raw_df = raw_df.set_index('Player')

   required_columns = st.session_state.params['counting-statistics'] + \
                    list(st.session_state.params['ratio-statistics'].keys()) + \
                    [ratio_stat_info['volume-statistic'] for ratio_stat_info in st.session_state.params['ratio-statistics'].values()] + \
                    st.session_state.params['other-columns']
   
   raw_df = raw_df[list(set(required_columns))]

   return raw_df

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

