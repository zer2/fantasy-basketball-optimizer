import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import itertools
import streamlit as st
import numexpr as ne
from datetime import datetime
from functools import reduce 

def get_categories():
    #convenience function to get the list of categories used for fantasy basketball
    return get_ratio_statistics() + get_counting_statistics()
    
def get_counting_statistics():
    #convenience function to get the list of categories used for fantasy basketball
    if st.session_state:
      return st.session_state['params']['counting-statistics']
    else: 
      return ['Threes','Points','Rebounds','Assists','Steals','Blocks','Turnovers']
    
def get_ratio_statistics():
    #convenience function to get the list of categories used for fantasy basketball
    if st.session_state:
      return list(st.session_state['params']['ratio-statistics'].keys()) 
    else: 
      return ['Field Goal %','Free Throw %']
    
def get_selected_categories():
    return get_selected_ratio_statistics() + get_selected_counting_statistics()
    
def get_selected_counting_statistics():
   if st.session_state:
      return [category for category in st.session_state['selected_categories'] if category in get_counting_statistics()]
   else:
      return get_counting_statistics()
   
def get_selected_ratio_statistics():
   if st.session_state:
      return [category for category in st.session_state['selected_categories'] if category in get_ratio_statistics()]
   else:
      return get_ratio_statistics()

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
                  [[position_code + str(i+1) for i in range(position_number)] 
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

def listify(x : pd.DataFrame) -> list:
    #get all values from a dataframe into a list. Useful for listing all chosen players 
    #Goes row by row- very important! 

    x = x.values.tolist()
    return [item for row in x for item in row]

def static_score_styler(df : pd.DataFrame, multiplier : float) -> pd.DataFrame:
  """Helper function for styling tables of Z or G scores

  Args:
    df: DataFrame with columns per category and total. Additional columns optional
    
  Returns:
    Styled dataframe
  """

  agg_columns = [col for col in ['$ Value','Total'] if col in df.columns]
  index_columns = [col for col in ['Rank','Player'] if col in df.columns]

  df = df[index_columns + agg_columns + get_selected_categories()]

  df_styled = df.style.format("{:.2f}"
                              , subset = pd.IndexSlice[:,agg_columns + get_selected_categories()]) \
                            .map(styler_a
                                ,subset = pd.IndexSlice[:,agg_columns]) \
                            .map(stat_styler
                              , subset = pd.IndexSlice[:,get_selected_categories()]
                              , multiplier = multiplier)
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

def stat_styler(value : float, multiplier : float = 50, middle : float = 0) -> str:
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

  intensity = min(int(abs((value-middle)*multiplier)), 255)

  if (value - middle)*multiplier > 0:
    rgb = (255 -  intensity,255 , 255 -  intensity)
  else:
    rgb = (255, 255 - intensity, 255 - intensity)
      
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

#deprecated below 
'''
#this function should use scipy.special.linear_sum_assignment
def check_team_eligibility(players):    
    """Checks if a team is eligible or not, based on the players' possible positions

    The function works by setting up an optimization problem for assigning players to team positions
    If the optimization problem is infeasible, the team is not eligible
    
    Args:
        players:Lists of players, which are themselves lists of eligible positions. E.g. 
                [['SF','PF'],['C'],['SF']]

    Returns:
        True or False, depending on if the team is found to be eligible or not

    """

    n_players = len(players)
    
    #we need 8 columns for the 8 positions. We are defining them as 
    #C, PG, SG, G, SF, PF, F, U 
    X = cvxpy.Variable(shape = (n_players,8)) #we could set boolean = True, but it takes much longer

    eligibility = np.concatenate([get_eligibility_row(player) for player in players])

    #each player gets 1 position
    one_position_constraint = cvxpy.sum(X,axis = 1) == 1
    
    #total number of players in each category cannot exceed the maximum for the category
    available_positions_constraint = cvxpy.sum(X,axis = 0) <= [2,1,1,2,1,1,2,3]    
    
    #players can only play at positions they are eligible for 
    eligibility_constraint = X <= eligibility 
    
    positivity_constraint = X >= 0
    
    constraints = [one_position_constraint, available_positions_constraint, eligibility_constraint, positivity_constraint]
    problem = cvxpy.Problem(cvxpy.Minimize(0), constraints)
    problem.solve(solver=cvxpy.ECOS)
            
    return not problem.status == "infeasible"

def get_eligibility_row(pos):
    """Converts a list of player positions into a binary vector of length 8, for the 8 team positions"""
    eligibility = {7}
    if 'C' in pos:
        eligibility.add(0)
    if 'PG' in pos: 
        eligibility.update((1,3))
    if 'SG' in pos: 
        eligibility.update((2,3))
    if 'SF' in pos: 
        eligibility.update((4,6))
    if 'PF' in pos: 
        eligibility.update((5,6))
    return np.array([[i in eligibility for i in range(8)]])

def get_eligibility_row_simplified(pos):
    """Converts a list of player positions into a binary vector of length 5, for the 5 base positions"""
    eligibility = set()
    if 'C' in pos:
        eligibility.add(0)
    if 'PG' in pos: 
        eligibility.add(1)
    if 'SG' in pos: 
        eligibility.add((2))
    if 'SF' in pos: 
        eligibility.add((3))
    if 'PF' in pos: 
        eligibility.add((4))
    return np.array([[i in eligibility for i in range(5)]])
'''

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
      if drafter == 0:
        row = row - 1
      else:
        drafter = drafter + 1
    else:
      if (drafter == n - 1):
        row = row - 1
      else:
        drafter = drafter - 1

    return row, drafter 