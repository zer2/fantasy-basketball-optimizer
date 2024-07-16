import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import itertools
import streamlit as st
import numexpr as ne
from datetime import datetime
import cvxpy

def get_categories(params = None):
    #convenience function to get the list of categories used for fantasy basketball
    if params: 
      return params['percentage-statistics'] + params['counting-statistics']
    else: 
      return st.session_state.params['percentage-statistics'] + st.session_state.params['counting-statistics']

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

  df = df[index_columns + agg_columns + get_categories()]

  df_styled = df.style.format("{:.2f}"
                              , subset = pd.IndexSlice[:,agg_columns + get_categories()]) \
                            .map(styler_a
                                ,subset = pd.IndexSlice[:,agg_columns]) \
                            .map(stat_styler
                              , subset = pd.IndexSlice[:,get_categories()]
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
                                , subset = pd.IndexSlice[:,get_categories()]) \
                          .map(styler_a
                                , subset = pd.IndexSlice[:,[perc_column]]) \
                          .map(stat_styler
                              , middle = middle
                              , multiplier = multiplier
                              , subset = get_categories())
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

def combinatorial_calculation(c : np.array
                              , c_comp : np.array
                              , data = 1 #the latest probabilities. Defaults to 1 at start
                              , level : int = 0 #the number of categories that have been worked into the probability
                              , n_false : int = 0 #the number of category losses that have been tracked so far
                             ):
    """This recursive functions enumerates winning probabilities for the Gaussian optimizer

    The function's recursive structure creates a binary tree, where each split is based on whether the next category is 
    won or lost. At the high level it looks like 
    
                                            (start) 
                                    |                   |
                                won rebounds      lost rebounds
                             |          |           |            |
                          won pts    lost pts   won pts     lost pts
                          
    The probabilities of winning scenarios are then added along the tree. This is more efficient than brute force calculation
    of each possibility, because it doesn't repeat multiplication steps for similar scenarios like (won 9) and (won 8 then 
    lost the last 1). Ultimately it is about five times faster than the equivalent with list comprehension
    
    Args:
        c: Array of all category winning probabilities. Dimensions are (player, category, opponent)
        c_comp: 1 - c
        data: probability of the node's scenario. Defaults to 1 because no categories are required at first
        level: the number of categories that have been worked into the probability
        n_false: the number of category losses that have been tracked so far. When it gets high enough 
                 we write off the node; the remaining scenarios do not contribute to winning chances

    Returns:
        DataFrame with probability of winning a majority of categories for each player 

        axis 0: player 
        axis 1: opponent

    """
    if n_false > (c.shape[1] -1)/2: #scenarios where a majority of categories are losses are overall losses
        return 0 
    elif level < c.shape[1] :
        #find the total winning probability of both branches from this point- if we win or lose the current category 
        return combinatorial_calculation(c, c_comp, data * c[:,level,:], level + 1, n_false) + \
                combinatorial_calculation(c, c_comp, data * c_comp[:,level,:], level + 1, n_false + 1)
    else: #a series where all 9 categories has been processed, and n_false <= the cutoff, can be added to the total %
        return data

@st.cache_data()
def get_grid():
    #create a grid representing 126 scenarios where 5 categories are won and 4 are lost

    which = np.array([list(itertools.combinations(range(9), 5))] )
    grid = np.zeros((126, 9), dtype="bool")     
    grid[np.arange(126)[None].T, which] = True

    grid = np.expand_dims(grid, axis = 2)

    return grid

def calculate_tipping_points(x : np.array) -> np.array:
    """Calculate the probability of each category being a tipping point, assuming independence

    Args:
        x: Array of shape (n,9,m) representing probabilities of winning each of the 9 categories 

    Returns:
        DataFrame of shape (n,9,m) representing probabilities of each category being a tipping point
        m is number of opponents
    """

    grid = get_grid()

    #copy grid for each row in x 
    grid = np.array([grid] * x.shape[0])

    x = x.reshape(x.shape[0],1,9, x.shape[2])

    #get the probabilities of the scenarios and filter them by which categories they apply to
    #the categories that are won all become tipping points

    first_part = ne.evaluate('grid * x + (1-grid) * (1-x)') \
                                   .prod(axis = 2).reshape(x.shape[0],126,1,x.shape[3])
    positive_case_probabilities = ne.evaluate('first_part * grid').sum(axis = 1)

    #do the same but for the inverse scenarios, where 5 categories are lost and 4 are won
    #in this case the lost categories become tipping points 
    first_part = ne.evaluate('(1 - grid) * x + grid * (1-x)') \
                                  .prod(axis = 2).reshape(x.shape[0],126,1,x.shape[3])
    negative_case_probabilities = ne.evaluate('first_part * grid').sum(axis = 1)

    final_probabilities = ne.evaluate('positive_case_probabilities + negative_case_probabilities')

    return final_probabilities

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