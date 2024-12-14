import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder, playergamelogs
import streamlit as st
from datetime import datetime
from nba_api.stats import endpoints as nba_endpoints
import numpy as np
import requests
import os
import snowflake.connector
from src.helpers.helper_functions import get_n_games, get_data_from_snowflake, get_league_type
from src.data_retrieval.get_data_baseball import process_baseball_rotowire_data, get_baseball_historical_data
from unidecode import unidecode

@st.cache_data()
def get_yahoo_key_to_name_mapper():
   return get_data_from_snowflake('PLAYER_MAPPING_VIEW')[['YAHOO_PLAYER_ID','PLAYER_NAME']].set_index('YAHOO_PLAYER_ID')

#ZR: This is obviously wrong?
def get_yahoo_key_to_position_eligibility(season = 2024):
   return get_data_from_snowflake('YAHOO_ID_TO_NAME_VIEW')

#cache this globally so it doesn't have to be rerun constantly 
@st.cache_data(ttl = '1d') 
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

  player_metadata = get_player_metadata(st.session_state.data_source)

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

def get_correlations():
   if get_league_type() == 'NBA':
    return pd.read_csv('src/data_retrieval/basketball_correlations.csv')
   
def get_max_table():
    return pd.read_csv('src/data_retrieval/max_table.csv')

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
@st.cache_data(ttl = '1d') 
def get_historical_data():  
  full_df = get_data_from_snowflake('AVERAGE_NUMBERS_VIEW_2')

  renamer = st.session_state.params['stat-df-renamer']
  full_df = full_df.rename(columns = renamer)

  full_df = full_df.apply(pd.to_numeric, errors='ignore')

  #full_df['Season'] = (full_df['Season'] - 1).astype(str) + '-' + full_df['Season'].astype(str)

  full_df.loc[:,'Free Throw %'] = full_df.loc[:,'Free Throws Made']/full_df.loc[:,'Free Throw Attempts']
  full_df.loc[:,'Field Goal %'] = full_df.loc[:,'Field Goals Made']/full_df.loc[:,'Field Goal Attempts']
  full_df.loc[:,'Three %'] = full_df.loc[:,'Threes']/full_df.loc[:,'Three Attempts']

  full_df.loc[:,'Assist to TO'] = full_df['Assists']/full_df['Turnovers']

  full_df['Position'] = full_df['Position'].fillna('NP')

  full_df = full_df.set_index(['Season','Player']).sort_index().fillna(0)  

  return full_df

@st.cache_data(ttl = 3600) 
def get_player_metadata(data_source) -> pd.Series:
   """Get player data from the NBA api

   Args:
      none
   Returns:
      Currently: A series of the form Player Name -> Position
   """

   print('I am here hi')
   print(data_source)
   df = get_htb_projections(data_source)

   return df['Position']


@st.cache_data(ttl = '1d') 
def get_darko_data(integration_source = None) -> dict[pd.DataFrame]:
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
  
  darko_df = map_player_names(darko_df, 'DARKO_NAME')

  darko_df['Position'] = 'NP'
  darko_df = darko_df.set_index(['Player']).sort_index().fillna(0)  

  #ZR: This should be simplified. We don't need to do this multiple times
  extra_info = get_data_from_snowflake('HTB_PROJECTION_TABLE')[['PLAYER','MPG','GP']]
  extra_info.loc[:,'GP'] = extra_info['GP']/get_n_games()
  extra_info.columns = ['Player','Minutes','Games Played %']

  extra_info = map_player_names(extra_info, 'BBM_NAME')

  extra_info = extra_info.set_index('Player')

  darko_long_term = darko_df.merge(extra_info, left_index = True, right_index = True)
  posessions_per_game = darko_long_term['Pace']/100 * darko_long_term['Minutes']/48

  darko_long_term.loc[:,'Points'] = darko_long_term.loc[:,'Points/100'] * posessions_per_game
  darko_long_term.loc[:,'Rebounds'] = darko_long_term.loc[:,'Rebounds/100'] * posessions_per_game
  darko_long_term.loc[:,'Assists'] = darko_long_term.loc[:,'Assists/100'] * posessions_per_game
  darko_long_term.loc[:,'Steals'] = darko_long_term.loc[:,'Steals/100'] * posessions_per_game
  darko_long_term.loc[:,'Blocks'] = darko_long_term.loc[:,'Blocks/100'] * posessions_per_game
  darko_long_term.loc[:,'Threes'] =  darko_long_term.loc[:,'Threes/100'] * posessions_per_game
  darko_long_term.loc[:,'Turnovers'] = darko_long_term.loc[:,'Turnovers/100'] * posessions_per_game
  darko_long_term.loc[:,'Free Throw Attempts'] = darko_long_term.loc[:,'Free Throw Attempts/100'] * posessions_per_game
  darko_long_term.loc[:,'Field Goal Attempts'] = darko_long_term.loc[:,'Field Goal Attempts/100'] * posessions_per_game

  darko_long_term.loc[:,'Three Attempts'] = darko_long_term.loc[:,'Three Attempts/100'] * posessions_per_game
  darko_long_term.loc[:,'Field Goals Made'] = darko_long_term.loc[:,'Field Goal Attempts/100'] * \
                                                darko_long_term.loc[:,'Field Goal %'] * \
                                                posessions_per_game
  darko_long_term.loc[:,'Field Goal Attempts'] = darko_long_term.loc[:,'Field Goal Attempts/100'] * posessions_per_game
  darko_long_term.loc[:,'Assist to TO'] = darko_long_term.loc[:,'Assists']/darko_long_term.loc[:,'Turnovers']

  darko_long_term.loc[:,'Def Rebounds'] = np.nan
  darko_long_term.loc[:,'Off Rebounds'] = np.nan
  darko_long_term.loc[:,'Double Doubles'] = np.nan

  required_columns = st.session_state.params['counting-statistics'] + \
                    list(st.session_state.params['ratio-statistics'].keys()) + \
                    [ratio_stat_info['volume-statistic'] for ratio_stat_info in st.session_state.params['ratio-statistics'].values()] + \
                    st.session_state.params['other-columns']
     
  required_columns =[x for x in required_columns if x in darko_long_term.columns]
      
  darko_long_term = darko_long_term[list(set(required_columns))]
   
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

    #current_data, expected_minutes = get_current_season_data()
    #darko_data = get_darko_data(expected_minutes)

    #if dataset_name in list(current_data.keys()):
    #    df = current_data[dataset_name].copy()
    #elif 'DARKO' in dataset_name:
    #    df = darko_data[dataset_name].copy()

    if 'Hashtag' in dataset_name:
        df = get_htb_projections()
    elif 'RotoWire' in dataset_name:
        if 'rotowire_data' in st.session_state:
            raw_df = st.session_state.rotowire_data
            df = process_basketball_rotowire_data(raw_df)
        else:
            st.error('Error! No rotowire data found: this should not happen')
            st.stop()
    elif 'Basketball Monster' in dataset_name:
       if 'bbm_data' in st.session_state:
            raw_df = st.session_state.bbm_data
            df = process_basketball_monster_data(raw_df
                                                 , default_projections = get_htb_projections())
       else:
            st.error('Error! No Basketball Monster data found: this should not happen')

    else:
        historical_df = get_historical_data()
        df = historical_df.loc[dataset_name].copy()  
    #adjust for the display
    df[r'Free Throw %'] = (df[r'Free Throw %'] * 100).round(1)
    df[r'Field Goal %'] = (df[r'Field Goal %'] * 100).round(1)
    df[r'Games Played %'] = (df[r'Games Played %'] * 100).round(1)

    df.index = df.index + ' (' + df['Position'] + ')'
    df.index.name = 'Player'
    return df.round(3) 
  
  elif st.session_state.league in ('MLB'):
    
    historical_df = get_baseball_historical_data()

    if 'rotowire_data' in st.session_state:
        raw_df = st.session_state.rotowire_data
        df = process_baseball_rotowire_data(raw_df)
    else: 
        df = historical_df.loc[dataset_name].copy()  
    
    df[r'Batting Average'] = (df[r'Batting Average'] * 100).round(1)
    df[r'Games Played %'] = (df[r'Games Played %'] * 100).round(1)

    df.index = df.index + ' (' + df['Position'] + ')'
    df.index.name = 'Player'

    return df.round(2) 
  
@st.cache_data()
def combine_nba_projections(rotowire_upload
                            , bbm_upload
                            , hashtag_slider
                            , bbm_slider
                            , darko_slider 
                            , rotowire_slider
                            , integration_source #just for caching purposes
                            ): 
    
    hashtag_stats = get_htb_projections(integration_source)
    rotowire_stats = None if rotowire_upload is None else process_basketball_rotowire_data(rotowire_upload, integration_source).fillna(0)
    bbm_stats = None if bbm_upload is None else process_basketball_monster_data(bbm_upload, integration_source)
    darko_stats = None if darko_slider == 0 else get_darko_data(integration_source)

    hashtag_weight = [hashtag_slider] 
    rotowire_weight = [rotowire_slider] if rotowire_upload is not None else []
    bbm_weight = [bbm_slider] if bbm_upload is not None else []
    darko_weight = [darko_slider] if darko_slider > 0 else []

    weights = hashtag_weight + rotowire_weight + bbm_weight + darko_weight

    all_players = set(
                  ([] if hashtag_stats is None else [p for p in hashtag_stats.index]) + \
                  ([] if rotowire_stats is None else [p for p in rotowire_stats.index]) + \
                  ([] if bbm_stats is None else [p for p in bbm_stats.index]) + \
                  ([] if darko_stats is None else [p for p in darko_stats.index])
                  )
        
    df =  pd.concat({'HTB' : hashtag_stats 
                        ,'RotoWire' : rotowire_stats
                        ,'BBM' : bbm_stats
                        ,'Darko' : darko_stats}, names = ['Source'])
            
    new_index = pd.MultiIndex.from_product([['HTB','RotoWire','BBM','Darko'], all_players], names = ['Source','Player'])

    df = df.reindex(new_index)
    
    weights = [hashtag_slider, rotowire_slider, bbm_slider, darko_slider]

    player_ineligible = (df.isna().groupby('Player').sum() == 4).sum(axis = 1) > 0
    inelegible_players = player_ineligible.index[player_ineligible]
    df = df[~df.index.get_level_values('Player').isin(inelegible_players)]
    
    df = df.groupby('Player') \
                .agg(lambda x: np.ma.average(np.ma.MaskedArray(x, mask=np.isnan(x)), weights = weights) \
                    if np.issubdtype(x.dtype, np.number) \
                    else x.dropna()[0])
            
    #Need to include this because not every source projects double doubles, which gets messy
    df['Double Doubles'] = [float(x) for x in df['Double Doubles']]

    df[r'Free Throw %'] = (df[r'Free Throw %'] * 100).round(1)
    df[r'Field Goal %'] = (df[r'Field Goal %'] * 100).round(1)
    df[r'Games Played %'] = (df[r'Games Played %'] * 100).round(1)
    df[r'Three %'] = (df[r'Three %'] * 100).round(1)

    df['Position'] = df['Position'].fillna('NP')
    df = df.fillna(0)

    df.index = df.index + ' (' + df['Position'] + ')'
    df.index.name = 'Player'
    return df.round(2) 


@st.cache_data()
def process_basketball_rotowire_data(raw_df, integration_source = None):
   
   raw_df.loc[:,'Games Played %'] = raw_df['G']/get_n_games()
   raw_df['FG%'] = raw_df['FG%']/100
   raw_df['FT%'] = raw_df['FT%']/100
   raw_df['3P%'] = raw_df['3P%']/100
   raw_df.loc[:,'Pos'] = raw_df.loc[:,'Pos'].map(st.session_state.params['rotowire-position-adjuster'])

   raw_df = raw_df.rename(columns = st.session_state.params['rotowire-renamer'])
   
   raw_df = raw_df.set_index('Player')

   #Rotowire doesn't forecast double doubles
   raw_df.loc[:,'Double Doubles'] = np.nan

   #Rotowire sometimes predicts Turnovers to be exactly 0, which is why we have this failsafe
   raw_df.loc[:,'Assist to TO'] = raw_df['Assists']/np.clip(raw_df['Turnovers'],0.1, None)
   raw_df.loc[:,'Field Goals Made'] = raw_df['Field Goal %'] * raw_df['Field Goal Attempts']

   required_columns = st.session_state.params['counting-statistics'] + \
                    list(st.session_state.params['ratio-statistics'].keys()) + \
                    [ratio_stat_info['volume-statistic'] for ratio_stat_info in st.session_state.params['ratio-statistics'].values()] + \
                    st.session_state.params['other-columns']
   
   required_columns =[x for x in required_columns if x in raw_df.columns]
      
   raw_df = raw_df[list(set(required_columns))]

   return raw_df

@st.cache_data(ttl = 3600)
def get_htb_adp():
   df = get_data_from_snowflake('HTB_PROJECTION_TABLE')
   df = df.rename(columns = st.session_state.params['htb-renamer'])

   df = map_player_names(df, 'HTB_NAME')
   df['Position'] = df['Position'].str.replace('/',',')

   df['Player'] = df['Player'] + ' (' + df['Position'] + ')'

   df = df[['Player','ADP']]
   return df.set_index('Player')

@st.cache_data(ttl = 3600)
def get_htb_projections(integration_source):

   raw_df = get_data_from_snowflake('HTB_PROJECTION_TABLE')

   raw_df = raw_df.rename(columns = st.session_state.params['htb-renamer'])
   raw_df.loc[:,'Games Played %'] = raw_df['Games Played']/get_n_games()

   raw_df['Position'] = raw_df['Position'].str.replace('/',',')

   raw_df = map_player_names(raw_df, 'HTB_NAME')

   #Rotowire sometimes predicts Turnovers to be exactly 0, which is why we have this failsafe
   raw_df.loc[:,'Assist to TO'] = raw_df['Assists']/np.clip(raw_df['Turnovers'],0.1, None)
   raw_df.loc[:,'Field Goals Made'] = raw_df['Field Goal %'] * raw_df['Field Goal Attempts']
   raw_df.loc[:,'Free Throws Made'] = raw_df['Free Throw %'] * raw_df['Free Throw Attempts']

   raw_df = raw_df.set_index('Player')

   required_columns = st.session_state.params['counting-statistics'] + \
                    list(st.session_state.params['ratio-statistics'].keys()) + \
                    [ratio_stat_info['volume-statistic'] for ratio_stat_info in st.session_state.params['ratio-statistics'].values()] + \
                    st.session_state.params['other-columns'] 
   
   raw_df = raw_df[list(set(required_columns))]

   return raw_df

@st.cache_data(ttl = 3600)
def process_basketball_monster_data(raw_df, integration_source = None):
   
   raw_df = raw_df.rename(columns = st.session_state.params['bbm-renamer'])

   #handling case where there is an extra column that gets interpreted as a missing value
   raw_df = raw_df.loc[:,[c for c in raw_df.columns if 'Unnamed' not in c]]
   raw_df.loc[:,'Games Played %'] = raw_df['Games Played']/get_n_games()
   raw_df['Position'] = raw_df['Position'].str.replace('/',',')
   raw_df = raw_df.dropna()

   raw_df = map_player_names(raw_df, 'BBM_NAME')

   raw_df = raw_df.set_index('Player')

   #Rotowire sometimes predicts Turnovers to be exactly 0, which is why we have this failsafe
   raw_df.loc[:,'Assist to TO'] = raw_df['Assists']/np.clip(raw_df['Turnovers'],0.1, None)
   raw_df.loc[:,'Field Goals Made'] = raw_df['Field Goal %'] * raw_df['Field Goal Attempts']

   required_columns = st.session_state.params['counting-statistics'] + \
                    list(st.session_state.params['ratio-statistics'].keys()) + \
                    [ratio_stat_info['volume-statistic'] for ratio_stat_info in st.session_state.params['ratio-statistics'].values()] + \
                    st.session_state.params['other-columns']
   
   required_columns =[x for x in required_columns if x in raw_df.columns]

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

def map_player_names(df, source_name):

   if 'integration' in st.session_state:
      player_name_column = st.session_state.integration.get_player_name_column()
   else:
      player_name_column = 'PLAYER_NAME'

   mapper_table = get_data_from_snowflake('PLAYER_MAPPING_VIEW').dropna(subset = [source_name]) \
                                                                  .set_index(source_name)[player_name_column]
   
   #does not get here
   df['Player'] = df['Player'].map(mapper_table).fillna(df['Player'])
   return df