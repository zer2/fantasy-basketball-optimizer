import streamlit as st
import pandas as pd 
from src.helpers.helper_functions import get_selected_categories, \
                                styler_a, styler_b, styler_c, stat_styler
from src.tabs.trading import make_trade_tab
from pandas.api.types import CategoricalDtype
from src.helpers.helper_functions import listify
from src.tabs.candidate_subtabs import make_cand_tab


def make_season_mode_tabs(H):
  """Create the season mode display, which has three tabs: waiver players, trades, and rosters 

  Args:
      H : for calculating H-scores for waiver selection and trading purposes 

  Returns:
      None
  """
  main_tabs = st.tabs(["⛹️‍♂️ Waiver Wire & Free Agents"
                  ,"📋 Trading"
                  ,"🏟️ Rosters"])

  waiver_tab = main_tabs[0]
  trade_tab = main_tabs[1]
  rosters_tab = main_tabs[2]

  with rosters_tab:

    left, right = st.columns([0.5,0.5])
      
    with left:

      st.caption("""Enter which player is on which team below""")
      player_category_type = CategoricalDtype(categories=list(st.session_state.player_stats.index) + ['RP']
                                                , ordered=True)

      with st.form('manual_rosters'):

        selections_df = st.data_editor(st.session_state.selections_df.astype(player_category_type)
                                            , hide_index = True
                                            , height = st.session_state.n_picks * 35 + 50
                                            , key = 'selections_df_edited').fillna('RP')
        
        c1, c2 = st.columns([0.2,0.8])
            
        with c1: 
            submit = st.form_submit_button("Lock in")
        with c2:
            st.warning('Lock in to update rosters')

        selection_list = listify(selections_df)

        player_assignments = selections_df.to_dict('list')

        g_scores_unselected = st.session_state.info['G-scores'][~st.session_state.info['G-scores'].index.isin(selection_list)]

      with right: 

        roster_inspection(selections_df.fillna('RP'))  
  
  with waiver_tab:

      make_full_waiver_tab(H
                           ,selections_df
                           ,player_assignments)
      
  with trade_tab:

    make_trade_tab(H
                   , selections_df
                   , player_assignments
                   , g_scores_unselected)         
    
@st.fragment
def roster_inspection(selections_df : pd.DataFrame):
    """Create a component of the UI which allows the user to select a team and view statistics for that team

    Args:
        selections_df : Dataframe representing which team selected which player

    Returns:
        None
    """

    roster_inspection_seat = st.selectbox(f'Which team do you want to get aggregated statistics for?'
                                        , selections_df.columns
                                        , index = 0)

    inspection_players = selections_df[roster_inspection_seat].dropna()

    make_team_display(st.session_state.info['G-scores']
                        ,inspection_players
                        ,st.session_state.info_key
                        )

@st.cache_data(ttl = 3600)
def make_team_display(_g_scores : pd.DataFrame
                  ,my_players : list[str]
                  ,info_key
                  ):
  """Make a table summarizing a team as it currently stands

  Args:
      g_scores: Dataframe of floats, rows by player and columns by category\
      my_players: list of players on the team to be inspected
      info_key : used to detect changes

  Returns:
      None
  """

  if len(my_players) > 0:

    st.divider()

    my_real_players = [x for x in my_players if x != 'RP']

    team_stats = _g_scores[_g_scores.index.isin(my_real_players)]

    team_stats.loc['Total', :] = team_stats.sum(axis = 0)

    team_stats = team_stats.loc[['Total'] + list(my_real_players)]

    team_stats_styled = team_stats.style.format("{:.2f}").map(styler_a) \
                                                .map(styler_c, subset = pd.IndexSlice[['Total'], get_selected_categories()]) \
                                                .map(styler_b, subset = pd.IndexSlice[['Total'], ['Total']]) \
                                                .map(stat_styler, subset = pd.IndexSlice[my_real_players, get_selected_categories()]
                                                     , multiplier = st.session_state.params['g-score-player-multiplier']) \
                                                .applymap(stat_styler, subset = pd.IndexSlice['Total', get_selected_categories()]
                                                    , multiplier = st.session_state.params['g-score-team-multiplier']) \
    
    
    st.dataframe(team_stats_styled
                        , use_container_width = True
                        , height = len(team_stats) * 35 + 38
                                                    )     
    
@st.fragment
def make_full_waiver_tab(H
                         ,selections_df : pd.DataFrame
                         ,player_assignments : dict):
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
    waiver_inspection_seat = st.selectbox(f'Which team do you want to drop a player from?'
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


        waiver_team_stats_g = st.session_state.info['G-scores'][st.session_state.info['G-scores'].index.isin(waiver_players)]
        waiver_team_stats_g.loc['Total', :] = waiver_team_stats_g.sum(axis = 0)

        worst_player = list(st.session_state.info['G-scores'].index[st.session_state.info['G-scores'].index.isin(waiver_players)])[-1]

        default_index = list(waiver_players).index(worst_player)

        drop_player = st.selectbox(
          'Which player are you considering dropping?'
          ,waiver_players
          ,index = default_index
        )

  if len(waiver_players) >= st.session_state.n_picks:
        make_cand_tab(H
                ,st.session_state.info['G-scores']
                ,player_assignments
                ,waiver_inspection_seat
                ,1
                ,1
                ,None
                ,None
                ,None
                ,st.session_state.n_picks * st.session_state.n_drafters
                ,drop_player)
