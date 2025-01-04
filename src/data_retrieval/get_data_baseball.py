import pandas as pd
import streamlit as st
from datetime import datetime
import numpy as np
import requests
import os
import snowflake.connector
from src.helpers.helper_functions import get_n_games, get_data_from_snowflake
from unidecode import unidecode


@st.cache_data()
def process_baseball_rotowire_data(raw_df, integration_source):
   """Turn a raw csv baseball projection from rotowire into a data format that can be understood by the app 
   Rotowire projections can be found here: https://www.rotowire.com/baseball/projections-ros.php

   Args:
         Dataframe, created directly from a rotowire csv file
   Returns:
         Dataframe post formatting
   """
   raw_df.loc[:,'Games Played %'] = 1 #we need to fix this later
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
   raw_df['Effective At Bats'] = raw_df['At Bats'] - raw_df['Walks']

   raw_df.loc[is_pitcher,pitcher_stats] = raw_df[is_pitcher][pitcher_stats].fillna(0)
   raw_df.loc[~is_pitcher,batter_stats] = raw_df[~is_pitcher][batter_stats].fillna(0)

   raw_df = raw_df.set_index('Player')

   required_columns = st.session_state.params['counting-statistics'] + \
                    list(st.session_state.params['ratio-statistics'].keys()) + \
                    [ratio_stat_info['volume-statistic'] for ratio_stat_info in st.session_state.params['ratio-statistics'].values()] + \
                    st.session_state.params['other-columns']
   
   raw_df = raw_df[list(set(required_columns))].fillna(0)

   return raw_df

@st.cache_data(ttl = '1d') 
def get_baseball_historical_data():  
  """Get historical baseball data, which is stored in Snowflake, and process it

  Args:
      None
  Returns:
      Dataframe of historical baseball data- weekly player averages per season
  """
  full_df = get_data_from_snowflake('AVERAGE_NUMBERS_VIEW', 'FANTASYBASEBALLOPTIMIZER')

  renamer = st.session_state.params['stat-df-renamer']
  full_df = full_df.rename(columns = renamer)

  full_df = full_df.apply(pd.to_numeric, errors='ignore')

  #full_df['Season'] = (full_df['Season'] - 1).astype(str) + '-' + full_df['Season'].astype(str)

  full_df.loc[:,'WHIP'] = full_df.loc[:,'Walka and Hits']/full_df.loc[:,'Innings Pitched']
  full_df.loc[:,'ERA'] = full_df.loc[:,'Innings Pitched']/full_df.loc[:,'Innings Pitched']
  full_df.loc[:,'Batting Average'] = full_df.loc[:,'Hits']/full_df.loc[:,'At Bats ']

  full_df['Position'] = 'NP'

  full_df = full_df.set_index(['Season','Player']).sort_index().fillna(0)  

  return full_df

@st.cache_data(ttl = 3600)
def get_baseball_htb_projections(integration_source):

   raw_df = get_data_from_snowflake('HTB_PROJECTION_TABLE', schema = 'FANTASYBASEBALLOPTIMIZER')

   raw_df = raw_df.rename(columns = st.session_state.params['htb-renamer'])
   raw_df.loc[:,'Games Played %'] = raw_df['Games Played']/get_n_games()

   raw_df['Position'] = raw_df['Position'].str.replace('/',',')

   raw_df['Position'] = raw_df['Position'].replace('Util','DH')

   #raw_df = map_player_names(raw_df, 'HTB_NAME')

   #Rotowire sometimes predicts Turnovers to be exactly 0, which is why we have this failsafe
   #raw_df.loc[:,'Assist to TO'] = raw_df['Assists']/np.clip(raw_df['Turnovers'],0.1, None)
   #raw_df.loc[:,'Field Goals Made'] = raw_df['Field Goal %'] * raw_df['Field Goal Attempts']
   #raw_df.loc[:,'Free Throws Made'] = raw_df['Free Throw %'] * raw_df['Free Throw Attempts']

   #baseball has some duplicate player names, which we need to deal with
   is_duplicate_player = raw_df.groupby('Player')['Player'].transform('size') > 1
   raw_df.loc[:,'Player'] = np.where(is_duplicate_player
                                     ,raw_df['Player'] + ' (' + raw_df['Team'] + ')'
                                     ,raw_df['Player']
                                     )


   raw_df = raw_df.set_index('Player')


   required_columns = st.session_state.params['counting-statistics'] + \
                    list(st.session_state.params['ratio-statistics'].keys()) + \
                    [ratio_stat_info['volume-statistic'] for ratio_stat_info in st.session_state.params['ratio-statistics'].values()] + \
                    st.session_state.params['other-columns'] 
   
   raw_df = raw_df[list(set(required_columns))]

   return raw_df


@st.cache_data()
def combine_baseball_projections(rotowire_upload
                            , hashtag_slider
                            , rotowire_slider
                            , integration_source #just for caching purposes
                            ): 
            
    hashtag_stats = get_baseball_htb_projections(integration_source)
    rotowire_stats = None if rotowire_upload is None else process_baseball_rotowire_data(rotowire_upload, integration_source).fillna(0)

    hashtag_weight = [hashtag_slider] 
    rotowire_weight = [rotowire_slider] if rotowire_upload is not None else []

    weights = hashtag_weight + rotowire_weight

    all_players = set(
                  ([] if hashtag_stats is None else [p for p in hashtag_stats.index]) + \
                  ([] if rotowire_stats is None else [p for p in rotowire_stats.index])
                  )
        
    df =  pd.concat({'HTB' : hashtag_stats 
                        ,'RotoWire' : rotowire_stats}, names = ['Source'])
                
    new_index = pd.MultiIndex.from_product([['HTB','RotoWire'], all_players], names = ['Source','Player'])

    df = df.reindex(new_index)
    
    weights = [hashtag_slider, rotowire_slider]

    player_ineligible = (df.isna().groupby('Player').sum() == 4).sum(axis = 1) > 0
    inelegible_players = player_ineligible.index[player_ineligible]
    df = df[~df.index.get_level_values('Player').isin(inelegible_players)]
    
    df = df.groupby('Player') \
                .agg(lambda x: np.ma.average(np.ma.MaskedArray(x, mask=np.isnan(x)), weights = weights) \
                    if np.issubdtype(x.dtype, np.number) \
                    else x.dropna()[0])
    
    df[r'Batting Average'] = (df[r'Batting Average'] * 100).round(1)

    df['Position'] = df['Position'].fillna('NP')
    df = df.fillna(0)

    df.index = df.index + ' (' + df['Position'] + ')'
    df.index.name = 'Player'

    st.write(df)

    return df.round(2) 