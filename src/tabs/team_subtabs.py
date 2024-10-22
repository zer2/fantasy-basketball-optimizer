import streamlit as st
import pandas as pd 
import numpy as np
from src.helpers.helper_functions import get_position_numbers_unwound, static_score_styler, h_percentage_styler, get_selected_categories, \
                                styler_a, styler_b, styler_c, stat_styler, \
                                get_selected_counting_statistics, get_selected_volume_statistics
from src.math.algorithm_agents import HAgent
from src.math.algorithm_helpers import savor_calculation
from src.math.process_player_data import process_player_data
from src.data_retrieval.get_data import get_htb_adp
import os
import itertools
from pathlib import Path
import gc
from src.math.algorithm_helpers import combinatorial_calculation
from src.math.algorithm_agents import get_base_h_score

@st.fragment
def roster_inspection(selections_df, info, omega, gamma, scoring_format, chi, player_assignments):

    roster_inspection_seat = st.selectbox(f'Which team do you want to get aggregated statistics for?'
                                        , selections_df.columns
                                        , index = 0)

    inspection_players = selections_df[roster_inspection_seat].dropna()

    if len(inspection_players) == st.session_state.n_picks:

        base_h_res = get_base_h_score(info
                                        ,omega
                                        ,gamma
                                        ,st.session_state.n_picks
                                        ,st.session_state.n_drafters
                                        ,scoring_format
                                        ,chi
                                        ,player_assignments
                                        ,roster_inspection_seat
                                        ,st.session_state.info_key)


    else:

        base_h_res = None

    make_full_team_tab(st.session_state.z_scores
                        ,st.session_state.g_scores
                        ,inspection_players
                        ,st.session_state.n_drafters
                        ,st.session_state.n_picks
                        ,base_h_res
                        ,st.session_state.info_key
                        ,roster_inspection_seat
                        )

#ZR: Uncaching because having this cached messes up the display somehow
#@st.cache_data(show_spinner = False, ttl = 3600)
def make_team_tab(_scores : pd.DataFrame
              , my_players : list[str]
              , n_drafters : int
              , player_multiplier : float
              , team_multiplier : float
              , info_key : int
              ) -> pd.DataFrame:
  """Make a tab summarizing your team as it currently stands

  Args:
      scores: Dataframe of floats, rows by player and columns by category\
      my_players: list of players on 'your' team
      n_drafters: number of drafters in the relevant league
      player_multiplier: scaling factor to use for color-coded display of player stats
      team_multiplier: scaling factor to use for color-coded display of team stats
      info_key: to detect changes

  Returns:
      DataFrame of team stats, to use in other tabs
  """

  team_stats = _scores[_scores.index.isin(my_players)]

  team_stats.loc['Total', :] = team_stats.sum(axis = 0)

  n_players_on_team = team_stats.shape[0] - 1

  team_stats_styled = team_stats.style.format("{:.2f}").map(styler_a) \
                                              .map(styler_c, subset = pd.IndexSlice[['Total'], get_selected_categories()]) \
                                              .map(styler_b, subset = pd.IndexSlice[['Total'], ['Total']]) \
                                              .map(stat_styler, subset = pd.IndexSlice[my_players, get_selected_categories()], multiplier = player_multiplier) \
                                              .applymap(stat_styler, subset = pd.IndexSlice['Total', get_selected_categories()], multiplier = team_multiplier)
  
  
  st.dataframe(team_stats_styled
                      , use_container_width = True
                      , height = len(team_stats) * 35 + 38
                      , key = str(info_key) + '_player_df' 
                                                )     

@st.cache_data(show_spinner = False, ttl = 3600)
def make_team_h_tab( base_h_score : float
                  , base_win_rates: pd.Series 
                  ) -> pd.DataFrame:
  """Display the H-score for your team

  Args:
      base_h_score: The H-score of your full team
      base_win_rates: expected win rates for each category

  Returns:
      None
  """
  st.markdown('The H-score of your team is ' + str((base_h_score * 100).round(1).values[0]) + '%')

  base_win_rates_copy = base_win_rates.copy()
  base_win_rates_copy.insert(0, 'H-score', base_h_score)

  base_win_rates_formatted = h_percentage_styler(base_win_rates_copy)
  st.dataframe(base_win_rates_formatted, hide_index = True)
  
@st.cache_data(show_spinner = False, ttl = 3600)
def make_full_team_tab(z_scores : pd.DataFrame
                  ,g_scores : pd.DataFrame
                  ,my_players : list[str]
                  ,n_drafters : int
                  ,n_picks : int
                  ,base_h_res : dict 
                  ,info_key : int
                  ,team_name : str
                  ):
  """Make a tab summarizing your team as it currently stands

  Args:
      z_scores: Dataframe of floats, rows by player and columns by category
      g_scores: Dataframe of floats, rows by player and columns by category\
      my_players: list of players on 'your' team
      n_drafters: number of drafters in the relevant league
      n_picks: number of picks per drafter
      base_h_score: The H-score of your full team
      base_win_rates: expected win rates for each category
      info_key : used to detect changes

  Returns:
      None
  """

  if len(my_players) == 0:
     st.write('This team has no players')

  else:
    z_team_tab, g_team_tab, matchups_tab = st.tabs(["Z-score", "G-score","Matchups"])

    with z_team_tab:

        my_real_players = [x for x in my_players if x != 'RP']

        make_team_tab(z_scores
                      , my_real_players
                      , n_drafters
                      , st.session_state.params['z-score-player-multiplier']
                      , st.session_state.params['z-score-team-multiplier']
                      , info_key)

    with g_team_tab:

        make_team_tab(g_scores
                        , my_real_players
                        , n_drafters
                        , st.session_state.params['g-score-player-multiplier']
                        , st.session_state.params['g-score-team-multiplier']
                        , info_key + 1)    

    with matchups_tab:
        if base_h_res is not None:

            make_team_matchup_tab(base_h_res['Scores']
                                    ,base_h_res['CDFs']
                                    , team_name)
        else:
            st.markdown('Team H-score not defined until team is full') 

def make_team_matchup_tab(base_h_score
                        , cdfs
                        , team_name):

    st.markdown('The H-score of your team is ' + str((base_h_score * 100).round(1).values[0]) + '%')

    cdfs_consolidated = pd.concat(cdfs, axis = 0)
    cdfs_expanded = np.expand_dims(cdfs_consolidated, axis = 2)

    cdfs_consolidated.loc[:,'EC'] = cdfs_consolidated.mean(axis = 1)

    #We've already calculated this but it is not retained by the algorithm agent

    cdfs_consolidated.loc[:,'MC'] = combinatorial_calculation(cdfs_expanded
                                                                , 1 - cdfs_expanded)



    cdfs_consolidated = cdfs_consolidated[['MC','EC'] + get_selected_categories()]

    cdfs_consolidated.index = [team for team in st.session_state.team_names if team != team_name]

    averages = cdfs_consolidated.mean(axis = 0)
    averages.name = '✨ Average ✨'

    cdfs_consolidated = pd.concat([averages.to_frame().T
                                        ,cdfs_consolidated])

    cdfs_styled = cdfs_consolidated.style.format("{:.1%}"
                                , subset = pd.IndexSlice[:,['MC','EC']] ) \
                          .format("{:.1%}"
                                , subset = pd.IndexSlice[:,get_selected_categories()]) \
                          .map(styler_a
                                , subset = pd.IndexSlice[:,['MC','EC']]) \
                          .map(stat_styler
                              , middle = 0.5
                              , multiplier = 300
                              , subset = get_selected_categories())


    st.dataframe(cdfs_styled
                , height = len(cdfs_consolidated) * 35 + 38)

    st.caption('MC score is probability of winning a matchup overall. EC score is average winning probability across categories')


