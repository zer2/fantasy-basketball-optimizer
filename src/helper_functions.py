import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import itertools
import streamlit as st
import numexpr as ne
from datetime import datetime
from functools import reduce 
from unidecode import unidecode
import snowflake.connector
import os 

def get_categories():
    #convenience function to get the list of categories used for fantasy basketball
    return get_ratio_statistics() + get_counting_statistics()

def get_selected_volume_statistics():
   return [st.session_state['params']['ratio-statistics'][x]['volume-statistic'] for x in get_selected_ratio_statistics()] 
    
def get_counting_statistics():
    #convenience function to get the list of categories used for fantasy basketball
    if st.session_state:
      return st.session_state['params']['counting-statistics']
    else: 
      if os.environ['SPORT'] == 'NBA':
        return ['Threes','Points','Rebounds','Assists','Steals','Blocks'
                ,'Turnovers','Double Doubles','Off Rebounds','Def Rebounds','Field Goals Made']
      elif os.environ['SPORT'] == 'MLB':
        return ['Runs','Home Runs', 'RBI', 'Stolen Bases', 'Wins', 'Saves', 'Strikeouts'] 
          
def get_ratio_statistics():
    #convenience function to get the list of categories used for fantasy basketball
    if st.session_state:
      return list(st.session_state['params']['ratio-statistics'].keys()) 
    else: 
      if os.environ['SPORT'] == 'NBA':
        return ['Field Goal %','Free Throw %','Three %','Assist to TO']
      elif os.environ['SPORT'] == 'MLB':
        return ['Batting Average', 'Slugging %', 'OBP', 'ERA','WHIP']
    
def get_selected_categories():
    if st.session_state:
       return st.session_state['selected_categories']
    else: 
       return get_selected_counting_statistics() + get_selected_ratio_statistics()
    
def get_selected_counting_statistics():
   if st.session_state:
      return [category for category in get_counting_statistics() if category in st.session_state['selected_categories']]
   else:
      if os.environ['SPORT'] == 'NBA':
        return  ['Threes','Points','Rebounds','Assists','Steals','Blocks','Turnovers']
      elif os.environ['SPORT'] == 'MLB':
        return ['Runs','Home Runs', 'RBI', 'Stolen Bases', 'Wins', 'Saves', 'Strikeouts'] 

def get_selected_ratio_statistics():
   if st.session_state:
      return [category for category in get_ratio_statistics() if category in st.session_state['selected_categories']]
   else:
      if os.environ['SPORT'] == 'NBA':
        return ['Field Goal %','Free Throw %']
      elif os.environ['SPORT'] == 'MLB':
        return ['Batting Average','ERA','WHIP']

def get_position_numbers():

    if st.session_state:
        res = {}
        for position_code in st.session_state.params['position_structure']['base_list']:  
           res[position_code] = st.session_state['n_' + position_code]

        for position_code in st.session_state.params['position_structure']['flex_list']:  
           res[position_code] = st.session_state['n_' + position_code]

        return res
    else:
       
        return {'Util' :3
                ,'C' : 2
                ,'G' : 2
                ,'PG' : 1
                ,'SG' : 1
                ,'F' : 2
                ,'PF' : 1
                ,'SF' : 1
                }
    
def get_position_numbers_unwound():
   position_numbers = get_position_numbers()
   return reduce(lambda x, y: x + y ,
                  [[position_code + str(i+1) if position_number >1 else position_code for i in range(position_number)] 
                   for position_code, position_number in position_numbers.items()]
                 )
    
def get_position_structure():
    if st.session_state:
       return st.session_state.params['position_structure']
    else:
       return { 'base_list' :
                        ['C','PG','SG','PF','SF']
               ,'base' : {'C' : {'full_str' : 'Centers'}
                         ,'PG' :{'full_str' : 'Point Guards'}
                         ,'SG' : {'full_str' : 'Shooting Guards'}
                         ,'PF' : {'full_str' : 'Power Forwards'}
                         ,'SF' : {'full_str' : 'Small Forwards'}}
               ,'flex_list' : ['Util','G','F']
               ,'flex' : {'Util' : 
                          {'bases' : ['C','PG','SG','PF','SF']
                           ,'full_str' : 'Utilities'
                          }
                        ,'G' : {'bases' : ['PG','SG']
                           ,'full_str' : 'Guards'
                          }
                        ,'F' : {'bases' : ['PF','SF']
                           ,'full_str' : 'Forwards'
                          }
                            }
                 }

def get_position_ranges():

    if st.session_state:

        start = 0
        end = 0
        res = {}

        position_structure = st.session_state.params['position_structure']
        
        all_positions = position_structure['base_list'] + position_structure['flex_list']

        for position_code in all_positions:
           end += st.session_state['n_' + position_code]
           res[position_code] = {'start' : start, 'end': end}
           start = end

        return res 

    else:
       
       #default to the standard position requirements
       center = {'start' : 0, 'end' : 2}
       point_guard = {'start' : 2, 'end' : 3}
       shooting_guard = {'start' : 3, 'end' : 4}
       power_forward = {'start' : 4, 'end' : 5}
       shooting_forward = {'start' : 5, 'end' : 6}
       utility = {'start' : 6, 'end' : 9}
       guard =  {'start' : 9, 'end' : 11}
       forward = {'start' : 11, 'end' : 13}

       return {
                'C' : center
                ,'PG' : point_guard
                ,'SG' : shooting_guard
                ,'PF' : power_forward
                ,'SF' : shooting_forward
                ,'Util' :utility
                ,'G' : guard
                ,'F' : forward

                }
    
def get_position_indices(position_structure):
   
    flex_info =  position_structure['flex']
    base_position_list = position_structure['base_list']

    return {position_code : 
                            [i for i, val in enumerate(base_position_list) if val in position_info['bases']]
                                    for position_code, position_info in flex_info.items()
            }

def adjust_teams_dict_for_duplicate_names(teams_dict):
    all_names = []
    for k, v in teams_dict.items():
        i = 1
        new_name = v 

        while new_name in all_names:
            i = i + 1
            new_name = v + ' ' + str(i)

        all_names = all_names + [new_name]

        if i != 1:
            teams_dict[k] = new_name

    return teams_dict

def get_L_weights() -> pd.Series:
   #calculate a default weighting for L
   #this assumes that all flex positions are weighted evenly among their bases 
   position_structure = get_position_structure()
   position_numbers = get_position_numbers()
   flex_positions = position_structure['flex_list']
   base_positions = position_structure['base_list']
   n_slots = sum([v for v in position_numbers.values()])

   shares = pd.Series({position_code : position_numbers[position_code]/n_slots for position_code in base_positions})

   for position_code in flex_positions:
      bases = position_structure['flex'][position_code]['bases']

      for base in bases:
         shares[base] += position_numbers[position_code]/(n_slots * len(bases)) 

   return shares 

def get_n_games():
   if st.session_state:
      return st.session_state.params['n_games']
      
def get_games_per_week():
   if st.session_state:
      return st.session_state.params['n_games_per_week']
   
def standardize_name(name):
   name = ' '.join(name.split(' ')[0:2])
   return unidecode(name)


def listify(x : pd.DataFrame) -> list:
    #get all values from a dataframe into a list. Useful for listing all chosen players 
    #Goes row by row- very important! 

    x = x.values.tolist()
    return [item for row in x for item in row]

#ZR: Ideally this should merge with the h percentage styler, so it can handle anything
def static_score_styler(df : pd.DataFrame, multiplier : float, total_multiplier : float = None) -> pd.DataFrame:
  """Helper function for styling tables of Z or G scores

  Args:
    df: DataFrame with columns per category and total. Additional columns optional
    
  Returns:
    Styled dataframe
  """

  agg_columns = [col for col in ['H-score','$ Value','Total'] if col in df.columns]
  index_columns = [col for col in ['Rank','Player'] if col in df.columns]
  perc_columns = ['H-score'] if 'H-score' in df.columns else []

  colored_total_column = ['Total'] if (('H-score' in df.columns) and ('Total' in df.columns)) else []

  if colored_total_column:
    total_middle = df[colored_total_column[0]].nlargest(10).iloc[-1]
  else:
    total_middle = 0

  df = df[index_columns + agg_columns + get_selected_categories()]

  df_styled = df.style.format("{:.2f}"
                              , subset = pd.IndexSlice[:,agg_columns + get_selected_categories()]) \
                              .format("{:.1%}"
                                , subset = pd.IndexSlice[:,perc_columns] ) \
                            .map(styler_a
                                ,subset = pd.IndexSlice[:,agg_columns]) \
                            .map(stat_styler
                              , subset = pd.IndexSlice[:,get_selected_categories()]
                              , multiplier = multiplier) \
                            .map(stat_styler
                                 , subset = pd.IndexSlice[:,colored_total_column]
                                 , multiplier = total_multiplier
                                 , mode = 'yellow'
                                 , middle = total_middle
                                 )
  return df_styled

def h_percentage_styler(df : pd.DataFrame
                        , middle : float = 0.5
                        , multiplier : float = 300) -> pd.DataFrame:
  """Helper function for styling tables of H-score results

  Args:
    df: DataFrame with columns per category and overall H-score. Additional columns optional
        Values are ratios between 0 and 1
  Returns:
    Styled dataframe
  """
  perc_column = 'H-score' if 'H-score' in df.columns else 'Overall'

  df_styled = df.style.format("{:.2%}"
                                , subset = pd.IndexSlice[:,[perc_column]] ) \
                          .format("{:.1%}"
                                , subset = pd.IndexSlice[:,get_selected_categories()]) \
                          .map(styler_a
                                , subset = pd.IndexSlice[:,[perc_column]]) \
                          .map(stat_styler
                              , middle = middle
                              , multiplier = multiplier
                              , subset = get_selected_categories())
  return df_styled

def stat_styler(value : float, multiplier : float = 50, middle : float = 0, mode = 'rgb') -> str:
  """Styler function used for coloring stat values red/green with varying intensities 

  Args:
    value: DataFrame of shape (n,9) representing probabilities of winning each of the 9 categories 
    multiplier: degree to which intensity of color scales relative to input value 
    middle: value that should map to pure white 
  Returns:
    String describing format for a pandas styler object
  """
         
  if value != value:
    return f"background-color:white;color:white" 
  
  if mode == 'rgb':

    intensity = min(int(abs((value-middle)*multiplier)), 255)

    if (value - middle)*multiplier > 0:
      rgb = (255 -  intensity,255 , 255 -  intensity)
    else:
      rgb = (255, 255 - intensity, 255 - intensity)
        
  elif mode == 'yellow': 

    intensity = min(int(abs((value-middle)*multiplier)), 255)

    if (value - middle)*multiplier > 0:
      rgb = (255,255 , 255 - intensity)
    else:
      rgb = (255,255 - intensity,255)

  bgc = '#%02x%02x%02x' % rgb

  #formula adapted from
  #https://stackoverflow.com/questions/3942878/how-to-decide-font-color-in-white-or-black-depending-on-background-color
  darkness_value = rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114
  tc = 'black' if darkness_value > 150 else 'white'

  return f"background-color: " + str(bgc) + ";color:" + tc + ";" 

def styler_a(value : float) -> str:
    return f"background-color: grey; color:white;" 

def styler_b(value : float) -> str:
    return f"background-color: lightgrey; color:black;" 

def styler_c(value : float) -> str:
    return f"background-color: darkgrey; color:black;" 

def rotate(l, n):
  #rotate list l by n positions 
  return l[-n:] + l[:-n]

def make_progress_chart(res : list[pd.DataFrame]):
    """Chart the progress of gradient descent in action, for the top 10 players 

    Args:
        res: List of dataframes of H-scoring results

    Returns:
        Line chart showing scores per player by iteration

    """
    
    data = pd.concat([pd.DataFrame({'H-score' : [r.loc[player] for r in res]
                                , 'Player' : player
                               , 'Iteration' : list(range(len(res)))})
        for player in res[-1].sort_values(ascending = False).index[0:10]])
    
    fig = px.line(data
                  , x = "Iteration"
                  , y = 'H-score'
                  , color = "Player")
    
    fig.update_layout(legend=dict(
        yanchor="bottom",
        y=0.01,
        xanchor="right",
        x=0.99
                ))

    fig.update_layout(yaxis={'visible': False, 'showticklabels': True})

    fig.update_layout(margin=go.layout.Margin(
                                    l=0, #left margin
                                    r=0, #right margin
                                    b=0, #bottom margin
                                    t=0, #top margin
                                            )
                     )

    return fig

def weighted_cov_matrix(df, weights):
    weighted_means = np.average(df, axis=0, weights=weights)
    deviations = df - weighted_means
    weighted_cov = np.dot(weights * deviations.T, deviations) / weights.sum()
    return pd.DataFrame(weighted_cov, columns=df.columns, index=df.columns)

def increment_player_stats_version():
  if st.session_state:
    st.session_state.player_stats_editable_version += 1

def increment_info_key():
  if st.session_state:
    st.session_state.info_key += 1

def increment_default_key():
  if st.session_state:
    st.session_state.player_stats_default_key += 1

@st.cache_data()
def get_selections_default(n_picks, n_drafters):
   return pd.DataFrame(
            {'Drafter ' + str(n+1) : [None] * n_picks for n in range(n_drafters)}
            )


def clear_draft_board():
  if 'draft_results' in st.session_state:
    del st.session_state.draft_results 

  if 'live_draft_active' in st.session_state:
    del st.session_state.live_draft_active

  if 'selections_df' in st.session_state:
    del st.session_state.selections_df

def increment_default_key_and_clear_draft_board():
  increment_default_key()
  clear_draft_board()


def autodraft(autodraft_df, g_scores):

  #print("AUTODRAFTING")
   
  row = 0
  drafter = 0

  while not ((autodraft_df.columns[drafter] not in st.session_state.autodrafters) and \
                (autodraft_df.iloc[row,drafter] != autodraft_df.iloc[row,drafter])):
    top_player = g_scores.index[0]

    row, drafter = move_forward_one_pick(row, drafter, autodraft_df.shape[1])

    if (autodraft_df.iloc[row,drafter] != autodraft_df.iloc[row,drafter]):
      autodraft_df.iloc[row, drafter] = top_player
      g_scores = g_scores[1:]

  st.session_state.selections_df = autodraft_df

  return row, drafter
      
def move_forward_one_pick(row, drafter, n):
    if row % 2 == 1:
      if drafter == 0:
        row = row + 1
      else:
        drafter = drafter - 1
    else:
      if (drafter == n - 1):
        row = row + 1
      else:
        drafter = drafter + 1

    return row, drafter 

def move_back_one_pick(row, drafter, n):
    if row % 2 == 1:
      if drafter == (n-1):
        row = row - 1
      else:
        drafter = drafter + 1
    else:
      if (drafter == 0):
        row = row - 1
      else:
        drafter = drafter - 1

    return row, drafter 

@st.cache_resource()
def get_data_from_snowflake(table_name
                            , schema = 'FANTASYBASKETBALLOPTIMIZER'):
   
   con = snowflake.connector.connect(
        user=st.secrets['SNOWFLAKE_USER']
        ,password=st.secrets['SNOWFLAKE_PASSWORD']
        ,account='aib52055.us-east-1'
        ,database = 'FANTASYOPTIMIZER'
        ,schema = schema
    )
   
   df = con.cursor().execute('SELECT * FROM ' + table_name).fetch_pandas_all()

   return df