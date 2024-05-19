import streamlit as st
import pandas as pd 
import numpy as np
from src.helper_functions import  static_score_styler, h_percentage_styler, get_categories, styler_a, styler_b, styler_c, stat_styler
from src.algorithm_agents import HAgent
from src.h_score_analysis import estimate_matchup_result, analyze_trade, analyze_trade_value
from src.algorithm_helpers import savor_calculation
from src.process_player_data import process_player_data
import os
import itertools
from pathlib import Path
  
@st.cache_data(show_spinner = False)
def make_about_tab(md_path : str):
    """Make one of the tabs on the about page

    Args:
      md_path : string representing the path to the relevant markdown file for display
    Returns:
      None
    """
    c2,c2,c3 = st.columns([0.1,0.8,0.1])
    with c2:
        intro_md = Path('about/' + md_path).read_text()
        st.markdown(intro_md, unsafe_allow_html=True)

### Team tabs 

@st.cache_data(show_spinner = False)
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

  if n_players_on_team > 0:

      team_stats_styled = team_stats.style.format("{:.2f}").map(styler_a) \
                                                  .map(styler_c, subset = pd.IndexSlice[['Total'], get_categories()]) \
                                                  .map(styler_b, subset = pd.IndexSlice[['Total'], ['Total']]) \
                                                  .map(stat_styler, subset = pd.IndexSlice[my_players, get_categories()], multiplier = player_multiplier) \
                                                  .applymap(stat_styler, subset = pd.IndexSlice['Total', get_categories()], multiplier = team_multiplier)
      display = st.dataframe(team_stats_styled
                          , use_container_width = True
                          , height = len(team_stats) * 35 + 38
                                                    )     
  else:
    st.markdown('This team does not have any players yet!')
  return team_stats
  
@st.cache_data(show_spinner = False)
def make_team_h_tab(my_players : list[str]
                  , n_picks : int
                  , base_h_score : float
                  , base_win_rates: pd.Series 
                  ) -> pd.DataFrame:
  """Display the H-score for your team

  Args:
      my_players: list of players on 'your' team
      n_picks: number of players per drafter
      base_h_score: The H-score of your full team
      base_win_rates: expected win rates for each category

  Returns:
      None
  """
  if len(my_players) < n_picks:
        st.markdown('This team is not full yet! Come back here when it is a full team')
  else:
        st.markdown('The H-score of your team is ' + str((base_h_score * 100).round(1).values[0]) + '%')

        base_win_rates_copy = base_win_rates.copy()
        base_win_rates_copy.insert(0, 'H-score', base_h_score)

        base_win_rates_formatted = h_percentage_styler(base_win_rates_copy)
        st.dataframe(base_win_rates_formatted, hide_index = True)
  
def make_full_team_tab(z_scores : pd.DataFrame
                  ,g_scores : pd.DataFrame
                  ,my_players : list[str]
                  ,n_drafters : int
                  ,n_picks : int
                  ,base_h_score : float
                  ,base_win_rates : float
                  ,info_key : int
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
  z_tab, g_tab, h_tab = st.tabs(["Z-score", "G-score","H-score"])

  with z_tab:

      make_team_tab(z_scores
                              , my_players
                              , n_drafters
                              , st.session_state.params['z-score-player-multiplier']
                              , st.session_state.params['z-score-team-multiplier']
                              , info_key)

  with g_tab:

      make_team_tab(g_scores
                              , my_players
                              , n_drafters
                              , st.session_state.params['g-score-player-multiplier']
                              , st.session_state.params['g-score-team-multiplier']
                              , info_key)    
  with h_tab:
    if len(my_players) == n_picks:

      make_team_h_tab(my_players
                    ,n_picks
                    ,base_h_score
                    ,base_win_rates)
    else:
      st.markdown('Team H-score not defined until team is full') 
### Candidate tabs 

@st.cache_data(show_spinner = False)
def make_cand_tab(_scores : pd.DataFrame
              , selection_list : list[str]
              , player_multiplier : float
              , remaining_cash : int = None 
              , total_players : int = None 
              , info_key : int = None) :
  """Make a tab showing stats for players that have not yet been drafted

  Args:
      scores: Dataframe of floats, rows by player and columns by category
      selection_list: list of players that have already been selected
      player_multiplier: scaling factor to use for color-coded display of player stats
      remaining_cash: for auction calculation
      total_players: for auction calculation
      info_key: for detecting changes

  Returns:
      DataFrame of stats of unselected players, to use in other tabs
  """
              
  counting_statistics = st.session_state.params['counting-statistics'] 
  percentage_statistics = st.session_state.params['percentage-statistics'] 

  scores_unselected = _scores[~_scores.index.isin(selection_list)]

  if remaining_cash:

    scores_unselected.loc[:,'$ Value'] = savor_calculation(scores_unselected['Total']
                                                          , total_players - len(selection_list)
                                                          , remaining_cash
                                                          , st.session_state['streaming_noise'])
    
  scores_unselected_styled = static_score_styler(scores_unselected, player_multiplier)
  scores_display = st.dataframe(scores_unselected_styled, use_container_width = True)

  return scores_unselected

def make_h_cand_tab(H
                    ,player_assignments
                    ,draft_seat
                    ,n_iterations
                    ,v
                    ,cash_remaining_per_team : dict[int] = None
                    ,generic_player_value : pd.Series = None
                    ,total_players : int = None):
  """Make a tab showing H-scores for the current draft situation

  Args:
      H:
      player_assignments: dict of who has drafted what player
      draft_seat: seat from which to calculate H-score
      n_iterations:
      v:
      cash_remaining_per_team:

  Returns:
      DataFrame of stats of unselected players, to use in other tabs
  """
          
  generator = H.get_h_scores(player_assignments, draft_seat, cash_remaining_per_team)

  placeholder = st.empty()

  #if n_iterations is 0 we run just once
  for i in range(max(1,n_iterations)):

    res = next(generator)
    score = res['Scores']
    weights = res['Weights']
    win_rates = res['Rates']

    #normalize weights by what we expect from other drafters
    weights = pd.DataFrame(weights
                  , index = score.index
                  , columns = get_categories())/v
    
    win_rates.columns = get_categories()
    
    with placeholder.container():

      if cash_remaining_per_team:
        target_tab, rate_tab, weight_tab,  = st.tabs(['Targets','Expected Win Rates', 'Weights'])
      else:
        rate_tab, weight_tab = st.tabs(['Expected Win Rates', 'Weights'])
          
      score = score.sort_values(ascending = False)
      score.name = 'H-score'
      score_df = pd.DataFrame(score)

      with rate_tab:
        rate_df = win_rates.loc[score_df.index].dropna()
        rate_display = score_df.merge(rate_df, left_index = True, right_index = True)

        if cash_remaining_per_team:

          players_chosen = [x for v in player_assignments.values() for x in v if x == x]
          total_cash_remaining = np.sum([v for k, v in cash_remaining_per_team.items()])

          rate_display.loc[:,'$ Value'] = savor_calculation(score_df
                                                          , total_players - len(players_chosen)
                                                          , total_cash_remaining
                                                          , st.session_state['streaming_noise_h'])

          rate_display = rate_display[['$ Value','H-score'] + get_categories()]

          rate_display_styled = rate_display.style.format("{:.1%}"
                            ,subset = pd.IndexSlice[:,['H-score']]) \
                          .format("{:.1f}"
                            ,subset = pd.IndexSlice[:,['$ Value']]) \
                    .map(styler_a
                          , subset = pd.IndexSlice[:,['H-score','$ Value']]) \
                    .map(stat_styler, middle = 0.5, multiplier = 300, subset = rate_df.columns) \
                    .format('{:,.1%}', subset = rate_df.columns)
        else:
          rate_display_styled = rate_display.style.format("{:.1%}"
                            ,subset = pd.IndexSlice[:,['H-score']]) \
                    .map(styler_a
                          , subset = pd.IndexSlice[:,['H-score']]) \
                    .map(stat_styler, middle = 0.5, multiplier = 300, subset = rate_df.columns) \
                    .format('{:,.1%}', subset = rate_df.columns)
        st.dataframe(rate_display_styled, use_container_width = True)
      with weight_tab:
        weight_df = weights.loc[score_df.index].dropna()
        weight_display = score_df.merge(weight_df
                              , left_index = True
                              , right_index = True)
        weight_display_styled = weight_display.style.format("{:.0%}"
                                                    , subset = weight_df.columns)\
                  .format("{:.1%}"
                          ,subset = pd.IndexSlice[:,['H-score']]) \
                  .map(styler_a
                        , subset = pd.IndexSlice[:,['H-score']]) \
                  .background_gradient(axis = None,subset = weight_df.columns) 
        st.dataframe(weight_display_styled, use_container_width = True)

      if cash_remaining_per_team:
         
         with target_tab:
                  
          comparison_df = pd.DataFrame({'Your $ Value' : rate_display['$ Value']
                                        , '$ Value' : generic_player_value.loc[rate_display.index]})
          comparison_df.loc[:,'Difference'] = comparison_df['Your $ Value'] - comparison_df['$ Value']

          comparison_df = comparison_df.join(rate_df)

          comparison_df = comparison_df[['Difference','Your $ Value','$ Value'] + list(rate_df.columns)]

          comparison_df_styled = comparison_df.style.format("{:.1f}"
                                                            , subset = ['Your $ Value', '$ Value','Difference']) \
                    .map(styler_a
                        , subset = ['Your $ Value', '$ Value']) \
                    .background_gradient(axis = None
                                        ,cmap = 'PiYG'
                                        ,subset = ['Difference']) \
                    .map(stat_styler, middle = 0.5, multiplier = 300, subset = rate_df.columns) \
                    .format('{:,.1%}', subset = rate_df.columns)
          
          st.dataframe(comparison_df_styled)
### Waiver tabs 

@st.cache_data(show_spinner = False)
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

  scores_unselected = _scores[~_scores.index.isin(selection_list)]

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

@st.cache_data(show_spinner = False)
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

@st.cache_data(show_spinner = False)
def make_matchup_tab(player_stats
                  , selections
                  , matchup_seat
                  , opponent_seat
                  , matchup_week
                  , n_picks
                  , n_drafters
                  , conversion_factors
                  , multipliers
                  , psi
                  , nu
                  , scoring_format):
  ### BELOW HERE SHOULD BE IN A CACHED TAB
  potential_games = st.session_state['schedule'][matchup_week].reindex(player_stats.index).fillna(3)
  week_number = int(matchup_week.split(':')[0].split(' ')[1])

  week_specific_player_stats = player_stats.copy()

  effective_games_played_percent = 1 - psi * (1- player_stats['Games Played %']/100)

  for col in st.session_state.params['counting-statistics']  + st.session_state.params['volume-statistics'] :
    week_specific_player_stats[col] = week_specific_player_stats[col] * effective_games_played_percent * \
                                                                        potential_games/3
  #ZR: WE should really clean up this keying mechanism
  week_specific_info = process_player_data(week_specific_player_stats
                        ,conversion_factors
                        ,multipliers
                        ,psi
                        ,nu
                        ,n_drafters
                        ,n_picks
                        ,st.session_state.player_stats_editable_version + week_number*100)
  
  week_specific_x_scores = week_specific_info['X-scores']

  team_1_x_scores = week_specific_x_scores.loc[selections[matchup_seat]].sum(axis = 0)
  team_2_x_scores = week_specific_x_scores.loc[selections[opponent_seat]].sum(axis = 0)

  result, win_probabilities = estimate_matchup_result(team_1_x_scores
                        , team_2_x_scores
                        , n_picks
                        , scoring_format)

  win_probabilities.loc[:,'Overall'] = result
  win_probabilities = win_probabilities[['Overall'] + get_categories()]
  win_probabilities_styled = h_percentage_styler(win_probabilities)
  st.dataframe(win_probabilities_styled, hide_index = True)

@st.cache_data(show_spinner = False)
def get_base_h_score(_info : dict
                , omega : float
                , gamma : float
                , alpha : float
                , beta : float
                , n_picks : int
                , n_drafters : int
                , scoring_format : str
                , chi : float
                , player_assignments : dict[list[str]]
                , team : str
                , info_key : int):
  """Calculate your team's H-score

  Args:
    info: dictionary with info related to player statistics etc. 
    omega: float, parameter as described in the paper
    gamma: float, parameter as described in the paper
    alpha: float, step size parameter for gradient descent 
    beta: float, decay parameter for gradient descent 
    n_picks: int, number of picks each drafter gets 
    n_drafters: int, number of drafters
    scoring_format: 
    player_assignments : player assignment dictionary
    team: name of team to evaluate

  Returns:
      None
  """

  H = HAgent(info = _info
    , omega = omega
    , gamma = gamma
    , alpha = alpha
    , beta = beta
    , n_picks = n_picks
    , n_drafters = n_drafters
    , dynamic = False
    , scoring_format = scoring_format
    , chi = chi)

  return next(H.get_h_scores(player_assignments, team))   

@st.cache_data(show_spinner = False)
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

  win_rates.columns = get_categories()
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

### Trade tabs

@st.cache_data(show_spinner = False)
def make_trade_score_tab(_scores : pd.DataFrame
              , players_sent : list[str]
              , players_received : list[str]
              , player_multiplier : float
              , team_multiplier : float
              , info_key : int
              ) -> None:
  """Make a tab summarizing a trade by total scores

  Args:
      scores: Dataframe of floats, rows by player and columns by category
      players_sent: Players you are sending in the trade
      players_received: Players you are receiving in the trade
      player_multiplier: scaling factor to use for color-coded display of player stats
      team_multiplier: scaling factor to use for color-coded display of team stats
      info_key: for keeping track of changes

  Returns:
      None
  """

  sent_stats = _scores[_scores.index.isin(players_sent)]
  sent_stats.loc['Total Sent', :] = sent_stats.sum(axis = 0)

  received_stats = _scores[_scores.index.isin(players_received)]
  received_stats.loc['Total Received', :] = received_stats.sum(axis = 0)

  full_frame = pd.concat([sent_stats,received_stats])
  full_frame.loc['Total Difference', :] = received_stats.loc['Total Received', :] - sent_stats.loc['Total Sent', :]

  full_frame_styled = full_frame.style.format("{:.2f}").map(styler_a, subset = pd.IndexSlice[['Total Difference']
                                                                                    , ['Total']]) \
                                              .map(styler_b, subset = pd.IndexSlice[players_sent + players_received
                                                                                    , ['Total']]) \
                                              .map(styler_c, subset = pd.IndexSlice[['Total Sent','Total Received']
                                                                                , ['Total'] + get_categories()]) \
                                              .map(stat_styler, subset = pd.IndexSlice[players_sent + players_received
                                                                                  , get_categories()]
                                                                                  , multiplier = player_multiplier) \
                                              .map(stat_styler, subset = pd.IndexSlice[['Total Difference']
                                                                                  , get_categories()]
                                                                                  , multiplier = player_multiplier)  
  display = st.dataframe(full_frame_styled
                      , use_container_width = True
                      , height = len(full_frame) * 35 + 38
                                                )     

@st.cache_data(show_spinner = False)
def make_trade_destination_display(_H
                  , _player_stats : pd.DataFrame
                  , player_assignments : dict[list[str]]
                  , my_team : list[str]
                  , scoring_format : str
                  , info_key : int
                        ):
  """Make a dataframe showing which of your players would be good candidates to send to which other teams

  Args:
    _H: H-scoring agent, which can be used to calculate H-score 
    player_stats: DataFrame of player statistics 
    player_assignments:
    my_team:
    their_team:
    scoring_format: Name of format. Included as input because it it an input to H
            and the cache should be re-calculated when format changes
    info_key: for detecting changes
)
  Returns:
      None
  """
  my_players = player_assignments[my_team]

  their_players_dict = player_assignments.copy()
  their_players_dict.pop(my_team)

  teams = list(their_players_dict.keys())
  for team in teams:
    if any([p != p for p in their_players_dict[team]]):
      their_players_dict.pop(team)

  values_to_me = pd.Series([analyze_trade_value(player
                                      , my_team
                                      , _H
                                      , _player_stats
                                      , player_assignments) for player in my_players
                    ]
                    , index = my_players)
  values_to_me = np.clip(values_to_me, 0, 1)

  values_to_team = pd.DataFrame(
                              {their_team : [analyze_trade_value(player
                                                    , their_team
                                                    , _H
                                                    , _player_stats
                                                    , player_assignments) 
                                        for player in my_players]
                                for their_team, their_players in their_players_dict.items()
                                }
                                , index = my_players
                                )
  values_to_team = np.clip(values_to_team, 0, 1)

  for col, vals in values_to_team.items():
    
    values_to_team[col] = values_to_team[col] - values_to_team[col].mean() - \
                            (values_to_me - values_to_me.mean())

  values_to_team_styled = values_to_team.T.style.format("{:.2%}") \
                          .map(stat_styler
                              , middle = 0
                              , multiplier = 15000
                          )
  st.dataframe(values_to_team_styled, use_container_width = True)

  return values_to_team

@st.cache_data(show_spinner = False)
def make_trade_target_display(_H
                  , _player_stats : pd.DataFrame
                  , my_team : str
                  , their_team : str
                  , player_assignments : dict[list[str]]
                  , values_to_team : pd.Series
                  , scoring_format : str
                  , info_key : int
                        ):
  """Make a dataframe showing which of your players would be good candidates to send to which other teams

  Args:
    _H: H-scoring agent, which can be used to calculate H-score 
    player_stats: DataFrame of player statistics 
    my_team: 
    their_team: 
    player_assignments:
    values_to_team: value of your own players to the selected team
    scoring_format: Name of format. Included as input because it it an input to H
            and the cache should be re-calculated when format changes
    info_key : for detecting changes
)
  Returns:
      None
  """
  their_players = player_assignments[their_team]

  values_to_me = pd.Series([analyze_trade_value(player
                                      , my_team
                                      , _H
                                      , _player_stats
                                      , player_assignments) for player in their_players
                    ]
                    , index = their_players)
  values_to_me = np.clip(values_to_me,0, 1)

  #make this into a team-wise dict
  value_to_them = pd.Series([analyze_trade_value(player
                                      , their_team
                                      , _H
                                      , _player_stats
                                      , player_assignments) for player in their_players
                    ]
                    , index = their_players)

  values_to_them = np.clip(value_to_them, 0, 1)

  values_to_me = values_to_me - values_to_me.mean() - \
                (values_to_them - values_to_them.mean())

  values_to_me.name = 'Relative Trade Value'
  values_to_team.name = 'Relative Trade Value'

  values_to_me = values_to_me.sort_values(ascending = False)
  values_to_team = values_to_team.sort_values(ascending = False)

  c1, c2 = st.columns([0.5,0.5])

  with c1: 
    values_to_me_styled = values_to_me.to_frame().style.format("{:.2%}") \
                            .map(stat_styler
                                , middle = 0
                                , multiplier = 15000
                            )
    st.dataframe(values_to_me_styled
              , use_container_width = True
              , height = len(values_to_me) * 35 + 38
)  

  with c2: 
    values_to_team_styled = values_to_team.to_frame().style.format("{:.2%}") \
                            .map(stat_styler
                                , middle = 0
                                , multiplier = 15000
                            )
    st.dataframe(values_to_team_styled
              , use_container_width = True
              , height = len(values_to_team) * 35 + 38
)  

  return values_to_me

@st.cache_data(show_spinner = False)
def get_cross_combos(n : int
                      , m : int
                      , my_players : list[str]
                      , their_players : list[str]
                      , general_values : pd.Series
                      , replacement_value : float
                      , values_to_me : pd.Series
                      , values_to_them : pd.Series
                      , heuristic_differential_threshold : float
                      , value_threshold : float) -> pd.DataFrame :
  """Helper function for trade suggesions. Makes a dataframe of viable trades 

  Args:
    n: number of players to send
    m: number of players to receive 
    my_players: initial list of players on your team
    their_players: initial list of players on other team 
    general_values : series representing general values, for filtering purposes
    replacement_value : generic value of the top replacement player
    values_to_me : targetedness of counterparty players to you
    values_to_them : targetedness of your players to counterparty
    heuristic_differential_threshold : only consider players above the heuristic differential threshold in trades 
    value_threshold : only consider trades with absolute value of G-score difference below the value threshold 

  Returns:
      Dataframe of viable trades according to the criteria
  """

  #helper function for getting trades between combos. Creates a dataframe for vectorized filtering
  my_candidate_players = [p for p in my_players if values_to_them[p] > heuristic_differential_threshold ]
  their_candidate_players = [p for p in their_players if values_to_me[p] > heuristic_differential_threshold ]

  my_players_with_weight = [(p,general_values[p]) for p in my_candidate_players]
  their_players_with_weight = [(p,general_values[p])  for p in their_candidate_players]

  cross_combos = list(itertools.product(get_combos(my_players_with_weight, n)
                                        ,get_combos(their_players_with_weight, m)
                                        )
                              )

  full_dataframe = pd.DataFrame(cross_combos)
  full_dataframe_separated = pd.concat([pd.DataFrame(full_dataframe[0].tolist()
                                                      , index=full_dataframe.index)
                                  ,pd.DataFrame(full_dataframe[1].tolist()
                                                , index=full_dataframe.index)], axis = 1
                                    )
  full_dataframe_separated.columns = ['My Trade','My Value','Their Trade','Their Value']
  
  if n!= m:
      full_dataframe_separated['My Value'] += replacement_value * (m-n)
  
  value_differential = full_dataframe_separated['My Value'] - full_dataframe_separated['Their Value']
  meets_threshold = abs(value_differential) <= value_threshold
  
  return full_dataframe_separated[meets_threshold]

def get_combos(players_with_weight : list[tuple]
            , n : int) -> list[tuple]:
  #helper function just for getting all 1,2,3 combos etc. from a set of candidates
  player_combos_with_weight = list(itertools.combinations(players_with_weight,n))
  player_combos_with_total_weight = [(list(z[0] for z in m), sum(z[1] for z in m)) 
                                      for m in player_combos_with_weight]
  return player_combos_with_total_weight

@st.cache_data(show_spinner = False)
def make_combo_df(all_combos : list
                  , player_stats : pd.DataFrame
                  , my_team : str
                  , their_team : str
                  , _H
                  , player_assignments : dict[list[str]]
                  , scoring_format : str) -> pd.DataFrame:
  """Makes a dataframe of all trade possibilities according to specifications

  Args:
    combos: list of trades to try. These are tuples where the first specifies players to send, and the second to receive 
    player_stats: DataFrame of player statistics 
    my_players: initial list of players on your team
    their_players: initial list of players on other team 
    _H: H-scoring agent, which can be used to calculate H-score 
    player_assignments: 
    scoring_format: Name of format. Included as input because it is an input to H
            and the cache should be re-calculated when format changes
  Returns:
      None
  """
  
  full_dataframe = pd.DataFrame()
    
  for key, row in all_combos.iterrows():

      my_trade = row['My Trade']
      their_trade = row['Their Trade']

      my_general_value = row['My Value']
      their_general_value = row['Their Value']
      #check if the general value disparity is extreme. If it is, pass 

      trade_results = analyze_trade(my_team
                                , my_trade
                                , their_team
                                , their_trade
                                , _H
                                , player_stats
                                , player_assignments
                                , 1)
      your_score_pre_trade = trade_results[1]['pre']['H-score']
      your_score_post_trade = trade_results[1]['post']['H-score']
      their_score_pre_trade = trade_results[2]['pre']['H-score']
      their_score_post_trade = trade_results[2]['post']['H-score']

      your_differential = your_score_post_trade - your_score_pre_trade
      their_differential = their_score_post_trade - their_score_pre_trade

      new_row = pd.DataFrame({ 'Send' : [my_trade]
                                ,'Receive' : [their_trade]
                                ,'Your H-score Differential' : [your_differential]
                                ,'Their H-score Differential' : [their_differential]
                                })
      full_dataframe = pd.concat([full_dataframe, new_row])

  full_dataframe = full_dataframe.sort_values('Your H-score Differential', ascending = False)

  return full_dataframe

@st.cache_data(show_spinner = """Finding suggested trades. How long this will take depends on 
                                  the trade parameters""")
def make_trade_suggestion_display(_H
                  , _player_stats : pd.DataFrame
                  , player_assignments : dict[list[str]]
                  , my_team : str
                  , their_team : str
                  , general_values : pd.Series
                  , replacement_value : float
                  , values_to_me : pd.Series
                  , values_to_them : pd.Series
                  , your_differential_threshold : float
                  , their_differential_threshold : float
                  , combo_params : list[tuple]
                  , trade_filter : list[tuple[int]]
                  , scoring_format : str
                  , info_key : int):
  """Shows automatic trade suggestions 

  Args:
    _H: H-scoring agent, which can be used to calculate H-score 
    player_stats: DataFrame of player statistics 
    player_assignments: 
    my_players: initial list of players on your team
    their_players: initial list of players on other team 
    general_values : series representing general values, for filtering purposes
    replacement_value : generic value of the top replacement player
    values_to_me : targetedness of counterparty players to you
    values_to_them : targetedness of your players to counterparty
    your_differential_threshold : for display, only include trades above this level of value for you
    their_differential_threshold : for display, only include trades above this level of value for counterparty
    combo_params : list of parameter sets for combos to try. See options page for details 
    trade_filter : show only trades with this number of players involved
    scoring_format: Name of format. Included as input because it is an input to H
            and the cache should be re-calculated when format changes
    info_key : for detecting changes
  Returns:
      None
  """

  my_players = player_assignments[my_team]
  their_players = player_assignments[their_team]

  all_combos = pd.concat([get_cross_combos(n
                                , m
                                , my_players 
                                , their_players 
                                , general_values 
                                , replacement_value 
                                , values_to_me 
                                , values_to_them 
                                , hdt
                                , vt) for n,m,hdt,vt in combo_params])

  full_dataframe = make_combo_df(all_combos
                  , _player_stats 
                  , my_team
                  , their_team
                  , _H
                  , player_assignments 
                  , scoring_format) 

  my_threshold_criteria = full_dataframe['Your H-score Differential'] > your_differential_threshold
  their_threshold_criteria = full_dataframe['Their H-score Differential'] > their_differential_threshold
  
  lens = pd.Series(zip(full_dataframe['Send'].map(len), full_dataframe['Receive'].map(len))
                    , index = full_dataframe.index)

  lens_criteria = lens.isin(trade_filter)

  full_dataframe = full_dataframe[my_threshold_criteria & their_threshold_criteria & \
                                  lens_criteria]

  if len(full_dataframe) > 0:

    full_dataframe_styled = full_dataframe.reset_index(drop = True).style.format("{:.2%}"
                                      , subset = ['Your H-score Differential'
                                                ,'Their H-score Differential']) \
                            .map(stat_styler
                                , middle = 0
                                , multiplier = 15000
                                , subset = ['Your H-score Differential'
                                          ,'Their H-score Differential']
                            ).set_properties(**{
                                  'font-size': '12pt',
                              })
    st.dataframe(full_dataframe_styled
                , hide_index = True
                , column_config={
                            "Send": st.column_config.ListColumn("Send", width='large')
                            ,"Receive": st.column_config.ListColumn("Receive", width='large')
                                                    })
  else: 
    st.markdown('No promising trades found')

@st.cache_data(show_spinner = False)
def make_trade_h_tab(_H
                  , _player_stats : pd.DataFrame
                  , player_assignments : dict[list[str]]
                  , n_iterations : int
                  , my_team : str
                  , my_trade : list[str]
                  , their_team : str
                  , their_trade : list[str]
                  , scoring_format : str
                  , info_key : int):
  """show the results of a potential trade

  Args:
    _H: H-scoring agent, which can be used to calculate H-score 
    player_stats: DataFrame of player statistics 
    player_assignments: 
    n_iterations: int, number of gradient descent steps
    my_trade: player(s) to be traded from your team
    their_trade: player(s) to be traded for
    my_players: initial list of players on your team
    their_players: initial list of players on other team 
    their_team_name: name of counterparty team
    scoring_format: Name of format. Included as input because it it an input to H
            and the cache should be re-calculated when format changes
    info_key: for detecting changes
)
  Returns:
      None
  """
  my_trade_len = len(my_trade)
  their_trade_len = len(their_trade)

  if (my_trade_len == 0) | (their_trade_len == 0):
      st.markdown('Need to trade at least one player')
  elif abs(my_trade_len - their_trade_len) > 6:
      st.markdown("Too lopsided of a trade! The computer can't handle it :frowning:")
  else:

      trade_results = analyze_trade(my_team
                                , my_trade
                                , their_team
                                , their_trade
                                , _H
                                , _player_stats
                                , player_assignments
                                ,n_iterations)
      your_team_pre_trade = trade_results[1]['pre']
      your_team_post_trade = trade_results[1]['post']
      their_team_pre_trade = trade_results[2]['pre']
      their_team_post_trade = trade_results[2]['post']

      if your_team_pre_trade['H-score'] < your_team_post_trade['H-score']:
          st.markdown('This trade benefits your team :slightly_smiling_face:')
      else:
          st.markdown('This trade does not benefit your team :slightly_frowning_face:')
      
      pre_to_post = pd.concat([your_team_pre_trade,your_team_post_trade], axis = 1).T
      pre_to_post.index = ['Pre-trade','Post-trade']
      pre_to_post_styled = h_percentage_styler(pre_to_post)
      st.dataframe(pre_to_post_styled, use_container_width = True, height = 108)
    
      if their_team_pre_trade['H-score'] < their_team_post_trade['H-score']:
          st.markdown('This trade benefits their team :slightly_smiling_face:')
      else:
          st.markdown('This trade does not benefit ' + their_team + ' :slightly_frowning_face:')
                  
      pre_to_post = pd.concat([their_team_pre_trade,their_team_post_trade], axis = 1).T
      pre_to_post.index = ['Pre-trade','Post-trade']
      pre_to_post_styled = h_percentage_styler(pre_to_post)
      st.dataframe(pre_to_post_styled, use_container_width = True, height = 108)

### Rank tabs 

#@st.cache_data(show_spinner = False)
def make_rank_tab(_scores : pd.DataFrame
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
  scores_copy = _scores.copy()

  scores_copy.loc[:,'Rank'] = np.arange(scores_copy.shape[0]) + 1
  scores_copy.loc[:,'Player'] = scores_copy.index
  scores_copy = scores_copy[['Rank','Player','Total'] + get_categories()]
  
  scores_styled = static_score_styler(scores_copy,player_multiplier)
      
  rank_display = st.dataframe(scores_styled, hide_index = True, use_container_width = True)

#@st.cache_data(show_spinner = False)
def make_h_rank_tab(_info : dict
                  , omega : float
                  , gamma : float
                  , alpha : float
                  , beta : float
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
    alpha: float, step size parameter for gradient descent 
    beta: float, decay parameter for gradient descent 
    n_picks: int, number of picks each drafter gets 
    n_drafters: int, number of drafters
    n_iterations: int, number of gradient descent steps
    scoring_format: 
    mode: 
    info_key: key to info data, used to detect changes

  Returns:
      None
  """

  H = HAgent(info = _info
    , omega = omega
    , gamma = gamma
    , alpha = alpha
    , beta = beta
    , n_picks = n_picks
    , n_drafters = n_drafters
    , dynamic = n_iterations > 0
    , scoring_format = scoring_format
    , chi = chi)
  
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
    
  cdf_estimates.columns = get_categories()
  rate_df = cdf_estimates.loc[h_res.index].dropna()

  h_res = h_res.sort_values(ascending = False)
  h_res = pd.DataFrame({'Rank' : np.arange(len(h_res)) + 1
                        ,'Player' : h_res.index
                        ,'H-score' : h_res.values
                      })

  h_res = h_res.merge(rate_df
                      , left_on = 'Player'
                      ,right_index = True)

  h_res_styled = h_percentage_styler(h_res)
  st.dataframe(h_res_styled, hide_index = True, use_container_width = True)
  return h_res
