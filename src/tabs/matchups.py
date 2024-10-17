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
from scipy.stats import norm

@st.cache_data(show_spinner = False, ttl = 3600)
def make_matchup_matrix(_x_scores : pd.DataFrame
                  , selections : pd.DataFrame
                  , scoring_format : str
                  , info_key : int):
    """Make a tab for a matchup matrix, showing how likely you are to win against any particular opponent

    Args:
        x_scores: Dataframe of floats, rows by player and columns by category
        selections: Dataframe of which teams have which players
        scoring_format: format to use for analysis 
        info_key: key for the info dict, used to detect changes

    Returns:
        None
    """

    selections_full = selections.dropna(axis = 1)

    if selections_full.shape[1] > 1:

      n_picks = selections_full.shape[0]

      team_stats = {team : _x_scores.loc[players].sum(axis = 0) for team,players in selections_full.items()}

      selections_combos = itertools.combinations(list(selections_full.columns), 2)

      matchup_df = pd.DataFrame(index = selections_full.columns
                                    , columns = selections_full.columns)

      for combo in selections_combos:

        team_1_x_scores = team_stats[combo[0]]
        team_2_x_scores = team_stats[combo[1]]

        result, _  = estimate_matchup_result(team_1_x_scores
                              , team_2_x_scores
                              , n_picks
                              , scoring_format)

        matchup_df.loc[combo[1],combo[0]] = 1- result
        matchup_df.loc[combo[0],combo[1]] = result

      for team in selections_full.columns:
        matchup_df.loc[team,team] = 0.5

      #A hack to make all of the columns similar width
      max_len_team_name = max([len(x) for x in matchup_df.columns])
      new_team_names = [x + (max_len_team_name - len(x)) * ' ' for x in matchup_df.columns]

      matchup_df.columns = new_team_names
      matchup_df.index = new_team_names


      matchup_df_styled = matchup_df.style.format("{:.1%}") \
                            .highlight_null(props="color: transparent;") \
                            .map(stat_styler
                                , middle = 0.5
                                , multiplier = 500) 
      
      def highlight_diag(df):
        a = np.full(df.shape, '', dtype='<U24')
        np.fill_diagonal(a, f"background-color:grey;")

        return pd.DataFrame(a, index=df.index, columns=df.columns)

      def highlight_diag_2(df):
        a = np.full(df.shape, '', dtype='<U24')
        np.fill_diagonal(a, f"color:grey;")

        return pd.DataFrame(a, index=df.index, columns=df.columns)     

      matchup_df_styled = matchup_df_styled.apply(highlight_diag, axis = None) \
                                          .apply(highlight_diag_2, axis = None)
      st.dataframe(matchup_df_styled)
    
    else: 

      st.markdown("""Not enough full teams yet! Make sure at least two teams are full on the
            "Drafting & Teams" page then come back here""")

@st.cache_data(show_spinner = False, ttl = 3600)
def make_matchup_tab(player_stats
                  , selections
                  , matchup_seat
                  , opponent_seat
                  , matchup_week
                  , n_picks
                  , n_drafters
                  , conversion_factors
                  , psi
                  , scoring_format):
  
  potential_games = st.session_state['schedule'][matchup_week].reindex(player_stats.index).fillna(3)
  week_number = int(matchup_week.split(':')[0].split(' ')[1])

  week_specific_player_stats = player_stats.copy()

  effective_games_played_percent = 1 - psi * (1- player_stats['Games Played %']/100)

  for col in get_selected_counting_statistics() + get_selected_volume_statistics() :
    week_specific_player_stats[col] = week_specific_player_stats[col] * effective_games_played_percent * \
                                                                        potential_games/3
  #ZR: WE should really clean up this keying mechanism
  week_specific_info = process_player_data(None
                        ,week_specific_player_stats
                        ,conversion_factors
                        ,0 #Upsilon isn't needed for this function anymore- should be removed 
                        ,psi
                        ,n_drafters
                        ,n_picks
                        ,st.session_state.params
                        ,st.session_state.player_stats_editable_version + week_number*100)
  
  week_specific_x_scores = week_specific_info['X-scores']

  team_1_x_scores = week_specific_x_scores.loc[selections[matchup_seat]].sum(axis = 0)
  team_2_x_scores = week_specific_x_scores.loc[selections[opponent_seat]].sum(axis = 0)

  result, win_probabilities = estimate_matchup_result(team_1_x_scores
                        , team_2_x_scores
                        , n_picks
                        , scoring_format)

  win_probabilities.loc[:,'Overall'] = result
  win_probabilities = win_probabilities[['Overall'] + get_selected_categories()]
  win_probabilities_styled = h_percentage_styler(win_probabilities)
  st.dataframe(win_probabilities_styled, hide_index = True)


@st.cache_data(show_spinner = False, ttl = 3600)
def estimate_matchup_result(team_1_x_scores : pd.Series
                            , team_2_x_scores : pd.Series
                            , n_picks : int
                            , scoring_format : str) -> float:
    """Based on X scores, estimates the result of a matchup. Chance that team 1 will beat team 2

    Args:
      team_1_x_scores: Series of x-scores for one team
      team_2_x_scores: Series of x-scores for other team
      n_picks: number of players on each team
      scoring_format: format to use for analysis

    Returns:
      Dictionary with results of the trade
    """

    cdf_estimates = pd.DataFrame(norm.cdf(team_1_x_scores - team_2_x_scores
                                        , scale = np.sqrt(n_picks*2)
                                        )
                            ).T

    cdf_array = np.expand_dims(np.array(cdf_estimates),2)

    if scoring_format == 'Head to Head: Most Categories':
        score = combinatorial_calculation(cdf_array
                                                    , 1 - cdf_array
                        )

    else:
        score = cdf_array.mean() 

    cdf_estimates.columns = get_selected_categories()
    return float(score), cdf_estimates