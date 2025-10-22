import pandas as pd
import streamlit as st
import numpy as np
import requests
from src.helpers.helper_functions import gen_key, get_data_from_session_state, get_params, get_n_games, get_data_from_snowflake, get_league_type, store_dataset_in_session_state
from src.data_retrieval.get_data_baseball import process_baseball_rotowire_data, get_baseball_historical_data

@st.cache_data(ttl = '1d')
def get_yahoo_key_to_name_mapper():
   return get_data_from_snowflake('PLAYER_MAPPING_VIEW')[['YAHOO_PLAYER_ID','PLAYER_NAME']].set_index('YAHOO_PLAYER_ID')

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
  
@st.cache_data(ttl = '1d')
def get_espn_projections(player_name_column : str):
  espn_df = get_data_from_snowflake('ESPN_PROJECTION_VIEW')
  renamer = get_params()['espn-renamer']
  espn_df = espn_df.rename(columns = renamer)

  espn_df = map_player_names(espn_df, 'ESPN_NAME', player_name_column)
  espn_df.loc[:,'Games Played %'] = espn_df['Games Played']/get_n_games()

  espn_df = espn_df.set_index('Player')

  return espn_df

@st.cache_data(ttl = '1d') 
def get_darko_data(player_name_column : str) -> dict[pd.DataFrame]:
  """Get DARKO predictions from stored CSV files

  Args:
      expected_minutes: Series of expecteed minutes projections, used to build DAKRO-L
  Returns:
      Dictionary, {'DARKO-L' : DARKO-L dataframe, 'DARKO-S' : DARKO-S dataframe}
  """
  darko_df = get_data_from_snowflake('DARKO_VIEW')
  renamer = get_params()['darko-renamer']
  darko_df = darko_df.rename(columns = renamer)
  darko_df = darko_df.apply(pd.to_numeric, errors='ignore')
  
  darko_df = map_player_names(darko_df, 'DARKO_NAME', player_name_column)

  darko_df = darko_df.set_index(['Player']).sort_index().fillna(0)  

  #ZR: This should be simplified. We don't need to do this multiple times
  extra_info = get_data_from_snowflake('ESPN_PROJECTION_TABLE')[['ESPN_NAME','MINUTES_PLAYED','GAMES_PLAYED','POSITION']]
  extra_info.loc[:,'GAMES_PLAYED'] = extra_info['GAMES_PLAYED'].astype(float)/get_n_games()
  extra_info.columns = ['Player','Minutes','Games Played %','Position']

  extra_info = map_player_names(extra_info, 'BBM_NAME', player_name_column)

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

  params = get_params()

  required_columns = params['counting-statistics'] + \
                    list(params['ratio-statistics'].keys()) + \
                    [ratio_stat_info['volume-statistic'] for ratio_stat_info in params['ratio-statistics'].values()] + \
                    params['other-columns']
     
  required_columns =[x for x in required_columns if x in darko_long_term.columns]
      
  darko_long_term = darko_long_term[list(set(required_columns))]
   
  return darko_long_term

#cache this globally so it doesn't have to be rerun constantly. No need for refreshes- it won't change
@st.cache_data(ttl = '1d') 
def get_historical_data():  
  full_df = get_data_from_snowflake('AVERAGE_NUMBERS_VIEW_2')

  renamer = get_params()['stat-df-renamer']
  full_df = full_df.rename(columns = renamer)

  full_df = full_df.apply(pd.to_numeric, errors='ignore')

  #full_df['Season'] = (full_df['Season'] - 1).astype(str) + '-' + full_df['Season'].astype(str)

  full_df.loc[:,'Free Throw %'] = full_df.loc[:,'Free Throws Made']/full_df.loc[:,'Free Throw Attempts']
  full_df.loc[:,'Field Goal %'] = full_df.loc[:,'Field Goals Made']/full_df.loc[:,'Field Goal Attempts']
  full_df.loc[:,'Three %'] = full_df.loc[:,'Threes']/full_df.loc[:,'Three Attempts']

  full_df.loc[:,'Assist to TO'] = full_df['Assists']/full_df['Turnovers']

  full_df['Position'] = full_df['Position'].fillna('NP')

  #ZR: Hack for now because there is an extra OG Anunoby in the reference table
  full_df = full_df[full_df['Player'] != 'OG Anunoby']

  full_df = full_df.set_index(['Season','Player']).sort_index().fillna(0)  

  return full_df

#setting show spinner to false prevents flickering
@st.cache_data(show_spinner = False, ttl = 3600)
def get_specified_historical_stats(dataset_name : str, league : str) -> pd.DataFrame:
  """fetch the data subset which will be used for the algorithms
  Args:
    dataset_name: the name of the dataset to fetch
    league: used as an input rather than taken from session state so that the function re-runs when the league is changed
            (the names of datasets can be the same across leagues)

  Returns:
    None
  """
  #not sure but I think copying the dataset instead of slicing it prevents issues with 
  #overwriting the version in the cache
  if league in ('NBA','WNBA'):

    historical_df = get_historical_data()
    df = historical_df.loc[dataset_name].copy()  

    df.index = df.index + ' (' + df['Position'] + ')'
    df.index.name = 'Player'

    return df, gen_key()
  
  elif league in ('MLB'):
    
    historical_df = get_baseball_historical_data()

    if 'rotowire_data' in st.session_state:
        raw_df = st.session_state.rotowire_data
        df = process_baseball_rotowire_data(raw_df)
    else: 
        df = historical_df.loc[dataset_name].copy()  

    df.index = df.index + ' (' + df['Position'] + ')'
    df.index.name = 'Player'

    return df, gen_key()
  
#ZR: BBM upload and HTB upload should have keys attached
@st.cache_data(ttl = '1d')
def combine_nba_projections(hashtag_slider : float
                            , bbm_slider : float
                            , darko_slider : float
                            , espn_slider : float
                            , player_name_column : str
                            , player_stat_key : str
                            ): 
    hashtag_upload = get_data_from_session_state('HTB')
    bbm_upload = get_data_from_session_state('BBM')
      
    hashtag_stats = None if hashtag_upload is None else process_basketball_htb_data(hashtag_upload, player_name_column).fillna(0)
    bbm_stats = None if bbm_upload is None else process_basketball_monster_data(bbm_upload, player_name_column)
    darko_stats = None if darko_slider == 0 else get_darko_data(player_name_column)
    espn_stats = None if espn_slider ==0 else get_espn_projections(player_name_column)

    all_players = set(
                  ([] if hashtag_stats is None else [p for p in hashtag_stats.index]) + \
                  ([] if espn_stats is None else [p for p in espn_stats.index]) + \
                  ([] if bbm_stats is None else [p for p in bbm_stats.index]) + \
                  ([] if darko_stats is None else [p for p in darko_stats.index])
                  )
            
    df =  pd.concat({'HTB' : hashtag_stats 
                        ,'BBM' : bbm_stats
                        ,'Darko' : darko_stats
                        ,'ESPN' : espn_stats
                      }, names = ['Source'])
                    
    new_index = pd.MultiIndex.from_product([['HTB','BBM','Darko','ESPN'], all_players], names = ['Source','Player'])

    df = df.reindex(new_index)
    
    weights = [hashtag_slider, bbm_slider, darko_slider, espn_slider]

    player_ineligible = (df.isna().groupby('Player').sum() == 4).sum(axis = 1) > 0
    inelegible_players = player_ineligible.index[player_ineligible]
    df = df[~df.index.get_level_values('Player').isin(inelegible_players)]
    
    df = df.groupby('Player') \
                .agg(lambda x: np.ma.average(np.ma.MaskedArray(x, mask=np.isnan(x)), weights = weights) \
                    if np.issubdtype(x.dtype, np.number) \
                    else x.dropna()[0])
            
    #Need to include this because not every source projects double doubles, which gets messy
    if 'Double Doubles' in df.columns:
        df['Double Doubles'] = [float(x) for x in df['Double Doubles']]

    df['Position'] = df['Position'].fillna('NP')
    df = df.fillna(0)

    df.index = df.index + ' (' + df['Position'] + ')'
    df.index.name = 'Player'
    
    return df, gen_key()


@st.cache_data()
def process_basketball_htb_data(htb : pd.DataFrame, player_name_column : str):
    htb = htb[htb['PLAYER'] != 'PLAYER']

    htb.loc[:,'ADP'] = -1

    fg_split = htb['FG%'].str[0:-1].str.split('(').str[1].str.split('/')
    htb.loc[:,'FGM'] = fg_split.str[0] #the str here is a python hack. Its weird but it works 
    htb.loc[:,'FGA'] = fg_split.str[1] #the str here is a python hack. Its weird but it works 
    htb.loc[:,'FG%'] = htb.loc[:,'FG%'].str.split('(').str[0].astype(float)

    ft_split = htb['FT%'].str[0:-1].str.split('(').str[1].str.split('/')
    htb.loc[:,'FTM'] = ft_split.str[0] #the str here is a python hack. Its weird but it works 
    htb.loc[:,'FTA'] = ft_split.str[1] #the str here is a python hack. Its weird but it works 
    htb.loc[:,'FT%'] = htb.loc[:,'FT%'].str.split('(').str[0].astype(float)

    if '3P%' in htb.columns:
        three_split = htb['3P%'].str[0:-1].str.split('(').str[1].str.split('/')
        htb.loc[:,'3PA'] = three_split.str[1] #the str here is a python hack. Its weird but it works 
        htb.loc[:,'3P%'] = htb.loc[:,'3P%'].str.split('(').str[0].astype(float)

    params = get_params()

    htb = htb.rename(columns = params['htb-renamer'])
    htb = map_player_names(htb, 'HTB_NAME', player_name_column)

    stat_columns = set(params['counting-statistics'] + \
                        list(params['ratio-statistics'].keys()) + \
                        [ratio_stat_info['volume-statistic'] for ratio_stat_info in params['ratio-statistics'].values()]
                            )
    required_numeric_columns = [x for x in stat_columns if x in htb.columns] + ['Games Played']
    required_other_columns = params['other-columns'] 

    htb[required_numeric_columns] = htb[required_numeric_columns].astype(float)

    htb.loc[:,'Games Played %'] = htb['Games Played']/get_n_games()
    htb['Position'] = htb['Position'].fillna('NP').str.replace('/',',')

    #Rotowire sometimes predicts Turnovers to be exactly 0, which is why we have this failsafe. Might not be necessary for HTB
    htb.loc[:,'Assist to TO'] = htb['Assists']/np.clip(htb['Turnovers'],0.1, None)
    htb.loc[:,'Field Goals Made'] = htb['Field Goal %'] * htb['Field Goal Attempts']
    htb.loc[:,'Free Throws Made'] = htb['Free Throw %'] * htb['Free Throw Attempts']

    htb = htb.set_index('Player')

    htb = htb[list(set(required_numeric_columns + required_other_columns))]

    return htb

@st.cache_data(ttl = 3600)
def get_htb_adp():
   
   return -1 

@st.cache_data()
def process_basketball_monster_data(raw_df : pd.DataFrame
                                    , player_name_column : str):

   raw_df = raw_df.rename(columns = get_params()['bbm-renamer'])

   #handling case where there is an extra column that gets interpreted as a missing value
   raw_df = raw_df.loc[:,[c for c in raw_df.columns if 'Unnamed' not in c]]
   raw_df.loc[:,'Games Played %'] = raw_df['Games Played']/get_n_games()
   raw_df['Position'] = raw_df['Position'].str.replace('/',',')
   #raw_df = raw_df.dropna(required_columns)

   raw_df = map_player_names(raw_df, 'BBM_NAME', player_name_column)

   raw_df = raw_df.set_index('Player')

   #Rotowire sometimes predicts Turnovers to be exactly 0, which is why we have this failsafe
   raw_df.loc[:,'Assist to TO'] = raw_df['Assists']/np.clip(raw_df['Turnovers'],0.1, None)
   raw_df.loc[:,'Field Goals Made'] = raw_df['Field Goal %'] * raw_df['Field Goal Attempts']

   params = get_params()

   required_columns = params['counting-statistics'] + \
                    list(params['ratio-statistics'].keys()) + \
                    [ratio_stat_info['volume-statistic'] for ratio_stat_info in params['ratio-statistics'].values()] + \
                    params['other-columns']
   
   required_columns =[x for x in required_columns if x in raw_df.columns]

   raw_df = raw_df[list(set(required_columns))]

   return raw_df


def map_player_names(df
                     , source_name : str
                     , player_name_column : str):

   #Only change player name if it is necessary. E.g. if the data source and platform are both ESPN, no mapping is necessary
   if player_name_column != source_name:

    mapper_table = get_data_from_snowflake('PLAYER_MAPPING_VIEW').dropna(subset = [source_name]) \
                                                                    .set_index(source_name)[player_name_column]
        
    df['Player'] = df['Player'].map(mapper_table).fillna(df['Player'])

   return df

