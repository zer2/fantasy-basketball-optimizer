import streamlit as st
import pandas as pd 
from src.helpers.helper_functions import static_score_styler, h_percentage_styler, get_selected_categories
from src.math.algorithm_agents import get_base_h_score
from src.tabs.candidate_subtabs import make_h_cand_tab

@st.fragment
def make_full_waiver_tab(H
                         ,selections_df : pd.DataFrame
                         ,player_assignments : dict
                         ,selection_list : list):
  """Make a tab showing how the team will change given a waiver substitution

  Args:
    H: H-scoring agent, which can be used to calculate H-score 
    selections_df: The selections df from the rosters page- potentially modified by the user
    player_assignments: Dictionary form of the selections df
    selection_list: List of chosen players, for convenience 

  Returns:
      DataFrame of stats of unselected players, to use in other tabs
  """

  
  c1, c2 = st.columns([0.5,0.5])

  with c1: 
    waiver_inspection_seat = st.selectbox(f'Which team so you want to drop a player from?'
        , st.session_state.selections_df.columns
        , index = 0)

  with c2: 
      waiver_players = [x for x in selections_df[waiver_inspection_seat] if x != 'RP']

      if len(waiver_players) < st.session_state.n_picks:
          st.markdown("""This team is not full yet!""")

      else:

        #Could we modify this to align with the candidates tab? Perhaps the "make candidate tab full"
        #function could take an optional argument for the default player and add them 
        #with the blue highlight everywhere

        waiver_team_stats_z = st.session_state.z_scores[st.session_state.z_scores.index.isin(waiver_players)]
        waiver_team_stats_z.loc['Total', :] = waiver_team_stats_z.sum(axis = 0)

        waiver_team_stats_g = st.session_state.g_scores[st.session_state.g_scores.index.isin(waiver_players)]
        waiver_team_stats_g.loc['Total', :] = waiver_team_stats_g.sum(axis = 0)

        if st.session_state.scoring_format == 'Rotisserie':
          worst_player = list(st.session_state.z_scores.index[st.session_state.z_scores.index.isin(waiver_players)])[-1]
        else:
          worst_player = list(st.session_state.g_scores.index[st.session_state.g_scores.index.isin(waiver_players)])[-1]

        default_index = list(waiver_players).index(worst_player)

        drop_player = st.selectbox(
          'Which player are you considering dropping?'
          ,waiver_players
          ,index = default_index
        )

        make_h_cand_tab(H
                ,st.session_state.g_scores
                ,st.session_state.g_scores
                ,player_assignments
                ,waiver_inspection_seat
                ,1
                ,st.session_state.v
                ,1
                ,None
                ,None
                ,st.session_state.n_picks * st.session_state.n_drafters
                ,drop_player)

        