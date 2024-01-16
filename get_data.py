import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder, playergamelogs
import streamlit as st
from datetime import datetime
from nba_api.stats import endpoints as nba_endpoints
import numpy as np
import os

renamer = {'PLAYER_NAME' : 'Player'
           ,'PTS' : 'Points'
           ,'REB' : 'Rebounds'
           ,'AST' : 'Assists'
           ,'STL': 'Steals'
           ,'BLK' : 'Blocks'
           ,'FG3M' : 'Threes'
           ,'TOV' : 'Turnovers'
           ,'FTA' : 'Free Throw Attempts'
           ,'FTM' : 'Free Throws Made'
           ,'FGA' : 'Field Goal Attempts'
           ,'FGM' : 'Field Goals Made'
           ,'GAME_DATE' : 'Game Date'}


#cache this globally so it doesn't have to be rerun constantly 
@st.cache_resource(ttl = '1d') 
def get_current_season_data(params, season = 2024):
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
  return data_dict 

@st.cache_resource
def get_historical_data(params):

  full_df = pd.read_csv('./stat_df.csv').set_index(['Season','Player']).sort_index().fillna(0)  
  full_df[params['counting-statistics'] + params['volume-statistics'] ] = full_df[params['counting-statistics'] + params['volume-statistics']]/3
  return full_df


def get_player_metadata():
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

  
def process_game_level_data(df, metadata):
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

#setting show spinner to false prevents flickering
@st.cache_data(show_spinner = False)
def get_partial_data(historical_df, current_data, dataset_name):

  #not sure but I think copying the dataset instead of slicing it prevents issues with 
  #overwriting the version in the cache
  if dataset_name in list(current_data.keys()):
      df = current_data[dataset_name].copy()
  else:
      df = historical_df.loc[int(dataset_name)].copy()

  #adjust for the display
  df[r'Free Throw %'] = (df[r'Free Throw %'] * 100).round(1)
  df[r'Field Goal %'] = (df[r'Field Goal %'] * 100).round(1)
  df[r'No Play %'] = (df[r'No Play %'] * 100).round(1)
  return df.round(2) 
