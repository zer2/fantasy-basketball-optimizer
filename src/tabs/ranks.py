import streamlit as st
import pandas as pd 
import numpy as np
from src.helpers.helper_functions import static_score_styler, h_percentage_styler, get_selected_categories, \
                                styler_a, stat_styler
from src.math.algorithm_agents import HAgent
from src.math.algorithm_helpers import savor_calculation


@st.cache_data(show_spinner = False, ttl = 3600)
def make_rank_tab(scores : pd.DataFrame
                      , player_multiplier : float
                      , info_key : int):
  """Show rankings by general value

  Args:
      scores: Dataframe of floats, rows by player and columns by category
      player_multipliers: scaling factor to use for color-coded display of player stats
      info_key: for detecting changes
  
  Returns:
      None
  """
  scores_copy = scores.copy().drop('RP')

  scores_copy.loc[:,'Rank'] = np.arange(scores_copy.shape[0]) + 1
  scores_copy.loc[:,'Player'] = scores_copy.index

  if '$ Value' in scores_copy.columns:
    scores_copy = scores_copy[['Rank','Player','$ Value','Total'] + get_selected_categories()]
  else:
    scores_copy = scores_copy[['Rank','Player','Total'] + get_selected_categories()]
  
  scores_styled = static_score_styler(scores_copy,player_multiplier)
      
  rank_display = st.dataframe(scores_styled, hide_index = True, use_container_width = True)

@st.cache_data(show_spinner = False, ttl = 3600)
def make_h_rank_tab(info : dict
                  , omega : float
                  , gamma : float
                  , n_picks : int
                  , n_drafters : int
                  , n_iterations : int
                  , scoring_format : str
                  , mode : str
                  , psi : float
                  , upsilon : float
                  , chi : float
                  , info_key : int):
  """Make ranks by H-score

  Args:
    _info: dictionary with info related to player statistics etc. 
    omega: float, parameter as described in the paper
    gamma: float, parameter as described in the paper
    n_picks: int, number of picks each drafter gets 
    n_drafters: int, number of drafters
    n_iterations: int, number of gradient descent steps
    scoring_format: 
    mode: 
    info_key: key to info data, used to detect changes

  Returns:
      None
  """

  H = HAgent(info = info
    , omega = omega
    , gamma = gamma
    , n_picks = n_picks
    , n_drafters = n_drafters
    , dynamic = n_iterations > 0
    , scoring_format = scoring_format
    , chi = chi
    , team_names = [n for n in range(n_drafters)])
  
  if st.session_state['mode'] == 'Auction Mode':
    cash_remaining_per_team = {n : 200 for n in range(n_drafters)}
  else:
    cash_remaining_per_team = None

  generator = H.get_h_scores(player_assignments = {n : [] for n in range(n_drafters)}
                          , drafter = 0
                          , cash_remaining_per_team = cash_remaining_per_team)

  for i in range(max(1,n_iterations)):
    res = next(generator)

  h_res = res['Scores']
  c = res['Weights']
  cdf_estimates = res['Rates']
    
  cdf_estimates.columns = get_selected_categories()
  rate_df = cdf_estimates.loc[h_res.index].dropna()

  h_res = h_res.sort_values(ascending = False)

  h_res = pd.DataFrame({'Rank' : np.arange(len(h_res)) + 1
                        ,'Player' : h_res.index
                        ,'H-score' : h_res.values
                      })

  h_res = h_res.merge(rate_df
                      , left_on = 'Player'
                      ,right_index = True)
  
  if st.session_state['mode'] == 'Auction Mode':

    h_res.loc[:,'$ Value'] = savor_calculation(h_res['H-score']
                                                    , n_picks * n_drafters
                                                    , 200*n_drafters
                                                    , st.session_state['streaming_noise_h'])
    
    h_res = h_res[['Rank','Player','$ Value','H-score'] + get_selected_categories()]

    h_res_styled = h_res.style.format("{:.1%}"
                      ,subset = pd.IndexSlice[:,['H-score']]) \
                    .format("{:.1f}"
                      ,subset = pd.IndexSlice[:,['$ Value']]) \
              .map(styler_a
                    , subset = pd.IndexSlice[:,['H-score','$ Value']]) \
              .map(stat_styler, middle = 0.5, multiplier = 300, subset = rate_df.columns) \
              .format('{:,.1%}', subset = rate_df.columns)
    
  else:
    h_res_styled = h_percentage_styler(h_res)

  st.dataframe(h_res_styled, hide_index = True, use_container_width = True)
  return h_res