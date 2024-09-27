import pandas as pd
import streamlit as st
from datetime import datetime
import numpy as np
import requests
import os
import snowflake.connector
from src.helper_functions import get_n_games, get_data_from_snowflake
from unidecode import unidecode


@st.cache_data()
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

@st.cache_resource(ttl = '1d') 
def get_baseball_historical_data():  
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