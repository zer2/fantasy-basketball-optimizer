import streamlit as st
import pandas as pd 
from src.helpers.helper_functions import static_score_styler, h_percentage_styler, get_selected_categories
from src.math.algorithm_agents import get_base_h_score

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

  if len(waiver_players) == st.session_state.n_picks:

        mod_waiver_players = [x for x in waiver_players if x != drop_player]

        z_waiver_tab, g_waiver_tab, h_waiver_tab = st.tabs(['Z-score','G-score','H-score'])

        with z_waiver_tab:

            st.markdown('Projected team stats with chosen player:')

            make_waiver_tab(st.session_state.z_scores
                          , selection_list
                          , waiver_team_stats_z
                          , drop_player
                          , st.session_state.params['z-score-team-multiplier']
                          , st.session_state.info_key)

        with g_waiver_tab:

            st.markdown('Projected team stats with chosen player:')
            make_waiver_tab(st.session_state.g_scores
                          , selection_list
                          , waiver_team_stats_g
                          , drop_player
                          , st.session_state.params['g-score-team-multiplier']
                          , st.session_state.info_key)

        with h_waiver_tab:

            base_h_res = get_base_h_score(st.session_state.info
                            ,st.session_state.omega
                            ,st.session_state.gamma
                            ,st.session_state.n_picks
                            ,st.session_state.n_drafters
                            ,st.session_state.scoring_format
                            ,st.session_state.chi
                            ,player_assignments
                            ,waiver_inspection_seat
                            ,st.session_state.info_key)

            waiver_base_h_score = base_h_res['Scores']
            waiver_base_win_rates = base_h_res['Rates']

            make_h_waiver_df(H
                        , st.session_state.player_stats
                        , mod_waiver_players
                        , drop_player
                        , player_assignments
                        , waiver_inspection_seat
                        , waiver_base_h_score
                        , waiver_base_win_rates
                        , st.session_state.info_key)
            
@st.cache_data(show_spinner = False, ttl = 3600)
def make_waiver_tab(_scores : pd.DataFrame
                , selection_list : list
                , team_stats : pd.Series
                , drop_player : str
                , team_multiplier : float
                , info_key : int):
  """Display how your team will change based on a waiver wire move 

  Args:
      scores: Dataframe of floats, rows by player and columns by category
      selection_list: list of unavailable players
      team_stats: Your team's stats as they stand currently
      drop_player: Candidate player to replace on the waiver wire
      team_multiplier: scaling factor to use for color-coded display of team stats
      info_key: for detecting changes

  Returns:
      None
  """
  scores_unselected = _scores[~_scores.index.isin(selection_list + ['RP'])]

  no_drop = team_stats.loc[['Total'],:]
  no_drop.index = [drop_player]
  
  drop_player_stats = _scores.loc[drop_player]
  new =  team_stats.loc['Total',:] + scores_unselected - drop_player_stats

  new = pd.concat([no_drop,new])
  new.index.name = 'Player'
  new = new.sort_values('Total', ascending = False).reset_index()

  new_styled = static_score_styler(new, team_multiplier)

  def color_blue(label):
    return "background-color: lightblue; color:black" if label == drop_player else None

  new_styled = new_styled.map(color_blue, subset = pd.IndexSlice[:,['Player']])

  st.dataframe(new_styled, use_container_width = True, hide_index = True) 

@st.cache_data(show_spinner = False, ttl = 3600)
def make_h_waiver_df(_H
                  , _player_stats : pd.DataFrame
                  , mod_my_players : list[str]
                  , drop_player : str
                  , player_assignments : dict[list[str]]
                  , team : str
                  , base_h_score : float
                  , base_win_rates : pd.Series
                  , info_key : int):

  """Show how your H-score would change based on waiver wire moves 

  Args:
    _H: H-scoring agent, which can be used to calculate H-score 
    mod_my_players: list of your players, excluding the player who is a candidate to be dropped
    drop_player: Candidate to be dropped
    player_assignments:
    base_h_score: H-score of your team before modification
    base_win_rates: expected win rates before modifications 

  Returns:
      None
  """

  player_assignments_post_drop = player_assignments.copy()
  player_assignments_post_drop[team] = mod_my_players
  h_score_results = next(_H.get_h_scores(player_assignments_post_drop, team))

  res = h_score_results['Scores']

  res = res.drop(index = [drop_player])
  win_rates = h_score_results['Rates'] 

  win_rates = win_rates.drop(index = [drop_player]) #we can think about other ways to do this


  res = res.sort_values(ascending = False)
  win_rates = win_rates.loc[res.index]

  win_rates.columns = get_selected_categories()
  res.name = 'H-score'

  base_h_score_copy = base_h_score.copy()
  base_h_score.index = [drop_player]
  base_h_score.name = 'H-score'

  base_win_rates_copy = base_win_rates.copy()
  base_win_rates_copy.index = [drop_player]

  win_rates_all = pd.concat([base_win_rates_copy, win_rates])

  scores_all = pd.concat([pd.DataFrame(base_h_score), pd.DataFrame(res)])

  h_display = pd.DataFrame(scores_all).merge(win_rates_all
                                        , left_index = True
                                        , right_index = True)
  h_display.index.name = 'Player'

  h_display = h_display.sort_values('H-score', ascending = False).reset_index()

  def color_blue(label):
    return "background-color: blue; color:white" if label == drop_player else None

  h_display = h_percentage_styler(h_display)
  h_display = h_display.map(color_blue, subset = pd.IndexSlice[:,['Player']])

  st.dataframe(h_display, use_container_width = True, hide_index = True)