import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder, playergamelogs
import streamlit as st
from datetime import datetime
from nba_api.stats import endpoints as nba_endpoints
import numpy as np
import os

#cache this globally so it doesn't have to be rerun constantly 
@st.cache_resource(ttl = '1d') 
def get_current_season_data(params
                            , season = 2024):
  #get all box scores from the current season and calculate various running averages 
           
  season_str = str(season -1) + '-' + str(season -2000)
  pgl_df = pd.concat(
      [
          playergamelogs.PlayerGameLogs(
              season_nullable=season_str, season_type_nullable=season_type
          ).player_game_logs.get_data_frame()
          for season_type in ["Regular Season"]
      ]
  )

  expected_minutes_long_term = process_minutes(pgl_df)
                            
  renamer = params['api-renamer']
  pgl_df = pgl_df.rename(columns = renamer)[list(renamer.values())].fillna(0)  

  four_weeks_ago = datetime.now() - pd.Timedelta(days = 28)
  two_weeks_ago = datetime.now() - pd.Timedelta(days = 14)

  four_week_subset = pgl_df[pd.to_datetime(pgl_df['Game Date']) >= four_weeks_ago].drop(columns = ['Game Date'])
  two_week_subset = pgl_df[pd.to_datetime(pgl_df['Game Date']) >= two_weeks_ago].drop(columns = ['Game Date'])
  full_subset = pgl_df.drop(columns = ['Game Date'])


  player_metadata = get_player_metadata()

  data_dict = {str(season) + '-Four Week Average' : process_game_level_data(four_week_subset, player_metadata)
               ,str(season) + '-Two Week Average' : process_game_level_data(two_week_subset, player_metadata)
               ,str(season) + '-Full Season' :  process_game_level_data(full_subset, player_metadata)
              }
                              
  return data_dict, expected_minutes_long_term 

def process_minutes(pgl_df):
  agg = pgl_df.groupby('Player')['MIN'].mean()
  agg.name = 'Minutes'
  return agg

  
#no need to cache this since it only gets re-run when current_season_data is refreshed
def process_game_level_data(df
                            , metadata):
  #convert a game level dataframe to a week-level dataframe
           
  agg_df = df.groupby('Player').mean().astype(float)
  agg_df.loc[:,'Free Throw %'] = np.where(agg_df['Free Throw Attempts'] > 0
                                          , agg_df['Free Throws Made']/agg_df['Free Throw Attempts']
                                          ,0)
  agg_df.loc[:,'Field Goal %'] = np.where(agg_df['Field Goal Attempts'] > 0
                                          , agg_df['Field Goals Made']/agg_df['Field Goal Attempts']
                                          ,0) 
  agg_df.loc[:,'No Play %'] = 0 #currently not implemented 

  agg_df = agg_df.fillna(0).merge(metadata, left_index = True, right_index = True)
  
  return agg_df.drop(columns = ['Free Throws Made','Field Goals Made'])

#cache this globally so it doesn't have to be rerun constantly. No need for refreshes- it won't change
@st.cache_resource
def get_historical_data(params):
  #get the one-time load of historical data stored as a CSV. In the future, it would perhaps be better to get this from snowflake
  
  full_df = pd.read_csv('./data/stat_df.csv').set_index(['Season','Player']).sort_index().fillna(0)  

  #adjust for the fact that historical data is week-based on game-based
  full_df[params['counting-statistics'] + params['volume-statistics'] ] = full_df[params['counting-statistics'] + params['volume-statistics']]/3
  return full_df


#no need to cache this since it only gets re-run when current_season_data is refreshed
def get_player_metadata():
   #get player positions from the NBA API
  
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
def get_darko_data(params):

  skill_projections = pd.read_csv('data/DARKO_player_talent_2024-01-19.csv')
  per_game_projections = pd.read_csv('data/DARKO_daily_projections_2024-01-19.csv')
  all_darko = skill_projections.merge(per_game_projections)

  all_darko['Player'] = np.where(all_darko['Player'] == 'Nicolas Claxton' 
                                 ,'Nic Claxton'
                                 ,all_darko['Player'])

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

  renamer = params['darko-renamer']
  all_darko = all_darko.rename(columns = renamer)[list(renamer.values())].fillna(0)  
  all_darko.loc[:,'No Play %'] = 0 #currently not implemented 

  player_metadata = get_player_metadata()
  all_darko = all_darko.merge(player_metadata, left_index = True, right_index = True)

  return all_darko, '1-19'
  

#setting show spinner to false prevents flickering
#data is cached locally so that different users can have different cuts loaded
@st.cache_data(show_spinner = False)
def get_partial_data(historical_df
                     , current_data
                     , darko_data
                     , dataset_name):
  #fetch the data subset which will be used for the algorithms

  #not sure but I think copying the dataset instead of slicing it prevents issues with 
  #overwriting the version in the cache
  if dataset_name in list(current_data.keys()):
      df = current_data[dataset_name].copy()
  elif 'DARKO' in dataset_name:
      df = darko_data.copy()
      os.write(1,bytes(str(df),'utf-8'))
  else:
      df = historical_df.loc[int(dataset_name)].copy()
  
  #adjust for the display
  df[r'Free Throw %'] = (df[r'Free Throw %'] * 100).round(1)
  df[r'Field Goal %'] = (df[r'Field Goal %'] * 100).round(1)
  df[r'No Play %'] = (df[r'No Play %'] * 100).round(1)
  return df.round(2) 

