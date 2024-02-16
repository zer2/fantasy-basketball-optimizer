import streamlit as st
import pandas as pd 
import numpy as np
from src.helper_functions import  stat_styler, styler_a,styler_b, styler_c, get_categories
from src.run_algorithm import HAgent, analyze_trade

### Team tabs 

@st.cache_data()
def make_team_tab(scores : pd.DataFrame
              , my_players : list
              , n_drafters : int
              , player_multiplier : float
              , team_multiplier : float) -> pd.DataFrame:
  """Make a tab summarizing your team as it currently stands

  Args:
      scores: Dataframe of floats, rows by player and columns by category\
      my_players: list of players on 'your' team
      n_drafters: number of drafters in the relevant league
      player_multiplier: scaling factor to use for color-coded display of player stats
      team_multiplier: scaling factor to use for color-coded display of team stats

  Returns:
      DataFrame of team stats, to use in other tabs
  """
  counting_statistics = st.session_state.params['counting-statistics'] 
  percentage_statistics = st.session_state.params['percentage-statistics'] 

  team_stats = scores[scores.index.isin(my_players)]
  expected = scores[0:len(my_players)*n_drafters].mean() * len(my_players)

  team_stats.loc['Total', :] = team_stats.sum(axis = 0)

  team_stats.loc['Expected', :] = expected
  team_stats.loc['Difference', :] = team_stats.loc['Total',:] - team_stats.loc['Expected',:]

  n_players_on_team = team_stats.shape[0] - 3

  if n_players_on_team > 0:

      team_stats_styled = team_stats.style.format("{:.2f}").map(styler_a) \
                                                  .map(styler_b, subset = pd.IndexSlice[['Expected','Total'], counting_statistics + percentage_statistics]) \
                                                  .map(styler_c, subset = pd.IndexSlice[['Expected','Total'], ['Total']]) \
                                                  .map(stat_styler, subset = pd.IndexSlice[my_players, counting_statistics + percentage_statistics], multiplier = player_multiplier) \
                                                  .applymap(stat_styler, subset = pd.IndexSlice['Difference', counting_statistics + percentage_statistics], multiplier = team_multiplier)
      display = st.dataframe(team_stats_styled, use_container_width = True)     
  else:
    st.markdown('Your team does not have any players yet!')
  return team_stats
  
@st.cache_data()
def make_team_h_tab(my_players : list
                  , n_picks : int
                  , base_h_score : float
                  , base_win_rates: pd.Series ) -> pd.DataFrame:
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
        st.markdown('Your team is not full yet! Come back here when you have a full team')
  else:
        st.markdown('The H-score of your team is ' + str((base_h_score * 100).round(1).values[0]) + '%')
        base_win_rates_formatted = base_win_rates.T.style.map(stat_styler
                                                        , middle = 0.5
                                                        , multiplier = 300).format('{:,.1%}')
        st.dataframe(base_win_rates_formatted, hide_index = True)
  
### Candidate tabs 

@st.cache_data()
def make_cand_tab(scores
              , selection_list
              , player_multiplier):
  """Make a tab showing stats for players that have not yet been drafted

  Args:
      scores: Dataframe of floats, rows by player and columns by category
      selection_list: list of all players that have already been drafted
      player_multiplier: scaling factor to use for color-coded display of player stats

  Returns:
      DataFrame of stats of unselected players, to use in other tabs
  """
              

  counting_statistics = st.session_state.params['counting-statistics'] 
  percentage_statistics = st.session_state.params['percentage-statistics'] 

  scores_unselected = scores[~scores.index.isin(selection_list)]

  scores_unselected_styled = scores_unselected.style.format("{:.2f}").map(styler_a).map(stat_styler, subset = pd.IndexSlice[:,counting_statistics + percentage_statistics], multiplier = player_multiplier)
  scores_display = st.dataframe(scores_unselected_styled, use_container_width = True)

  return scores_unselected

### Waiver tabs 

@st.cache_data()
def make_waiver_tab(scores : pd.DataFrame
                , scores_unselected : pd.DataFrame
                , team_stats : pd.Series
                , drop_player : str
                , team_multiplier : float):
  """Display how your team will change based on a waiver wire move 

  Args:
      scores: Dataframe of floats, rows by player and columns by category
      scores_unselected: scores, but filtered for only available players 
      team_stats: Your team's stats as they stand currently
      drop_player: Candidate player to replace on the waiver wire
      team_multiplier: scaling factor to use for color-coded display of team stats

  Returns:
      None
  """
            
  counting_statistics = st.session_state.params['counting-statistics'] 
  percentage_statistics = st.session_state.params['percentage-statistics'] 

  no_drop = team_stats.loc[['Total'],:]
  no_drop.index = [drop_player]
  
  drop_player_stats = scores.loc[drop_player]
  new =  team_stats.loc['Total',:] + scores_unselected - drop_player_stats

  new = pd.concat([no_drop,new])

  new_styled = new.style.format("{:.2f}").map(styler_a).map(stat_styler, subset = pd.IndexSlice[:,counting_statistics + percentage_statistics], multiplier = team_multiplier)
  st.dataframe(new_styled, use_container_width = True) 

@st.cache_data()
def get_base_h_score(info : dict
                , omega : float
                , gamma : float
                , alpha : float
                , beta : float
                , n_picks : int
                , winner_take_all : bool
                , punting : bool
                , player_stats : pd.DataFrame
                , my_players : list
                , players_chosen : list):
  """Calculate your team's H-score

  Args:
    info: dictionary with info related to player statistics etc. 
    omega: float, parameter as described in the paper
    gamma: float, parameter as described in the paper
    alpha: float, step size parameter for gradient descent 
    beta: float, decay parameter for gradient descent 
    n_picks: int, number of picks each drafter gets 
    winner_take_all: Boolean of whether to optimize for the winner-take-all format
                      If False, optimizes for total categories
    punting: boolean for whether to adjust expectation of future picks by formulating a punting strategy
    player_stats: DataFrame of player statistics 
    my_players: list of the players you have chosen
    players_chosen: list of all chosen players

  Returns:
      None
  """
  H = HAgent(info = info
    , omega = omega
    , gamma = gamma
    , alpha = alpha
    , beta = beta
    , n_picks = n_picks
    , winner_take_all = winner_take_all
    , punting = punting)

  return next(H.get_h_scores(player_stats, my_players, players_chosen))   

@st.cache_data()
def make_h_waiver_df(_H
                  , player_stats : pd.DataFrame
                  , mod_my_players : list
                  , drop_player : str
                  , players_chosen : list
                  , base_h_score : float
                  , base_win_rates : pd.Series):

  """Show how your H-score would change based on waiver wire moves 

  Args:
    _H: H-scoring agent, which can be used to calculate H-score 
    mod_my_players: list of your players, excluding the player who is a candidate to be dropped
    drop_player: Candidate to be dropped
    players_chosen: list of all chosen players
    base_h_score: H-score of your team before modification
    base_win_rates: expected win rates before modifications 

  Returns:
      None
  """
  res, _, win_rates = next(_H.get_h_scores(player_stats, mod_my_players, players_chosen))

  res = res.sort_values(ascending = False)
  win_rates = win_rates.loc[res.index]

  win_rates.columns = get_categories()
  res.name = 'H-score'

  base_h_score_copy = base_h_score.copy()
  base_h_score.index = [drop_player]
  base_h_score.name = 'H-score'

  base_win_rates_copy = base_win_rates.copy().T
  base_win_rates_copy.index = [drop_player]

  win_rates_all = pd.concat([base_win_rates_copy, win_rates])

  scores_all = pd.concat([pd.DataFrame(base_h_score), pd.DataFrame(res)])

  h_display = pd.DataFrame(scores_all).merge(win_rates_all, left_index = True, right_index = True)

  h_display = h_display.style.format("{:.1%}"
                    ,subset = pd.IndexSlice[:,['H-score']]) \
            .map(styler_a
                  , subset = pd.IndexSlice[:,['H-score']]) \
            .map(stat_styler, middle = 0.5, multiplier = 300, subset = win_rates_all.columns) \
            .format('{:,.1%}', subset = win_rates_all.columns)

  st.dataframe(h_display, use_container_width = True)

### Trade tabs

@st.cache_data()
def make_trade_display(_H
                  , player_stats : pd.DataFrame
                  , players_chosen : list
                  , n_iterations : int
                  , my_trade : list
                  , their_trade : list
                  , my_players : list
                  , their_players : list):
  """Analyze a trade 

  Args:
    _H: H-scoring agent, which can be used to calculate H-score 
    player_stats: DataFrame of player statistics 
    players_chosen: list of all chosen players
    n_iterations: int, number of gradient descent steps
    my_trade: player(s) to be traded from your team
    their_trade: player(s) to be traded for
    my_players: initial list of players on your team
    their_players: initial list of players on other team 
                                  ,H
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
      my_others = [x for x in my_players if x not in my_trade]
      their_others = [x for x in their_players if x not in their_trade]

      trade_results = analyze_trade(my_others
                                , my_trade
                                , their_others
                                , their_trade
                                , _H
                                , player_stats
                                , players_chosen
                                ,n_iterations)
      your_team_pre_trade = trade_results[1]['pre']
      your_team_post_trade = trade_results[1]['post']
      their_team_pre_trade = trade_results[2]['pre']
      their_team_post_trade = trade_results[2]['post']

      if your_team_pre_trade < your_team_post_trade:
          st.markdown('This trade benefits your team. H-score goes from ' + str(np.round(your_team_pre_trade[0]*100,1)) + '% to ' + str(np.round(your_team_post_trade[0]*100,1)) + '%')
      else:
          st.markdown('This trade does not benefit your team. H-score goes from ' + str(np.round(your_team_pre_trade[0]*100,1)) + '% to ' + str(np.round(your_team_post_trade[0]*100,1)) + '%')
      
      pre_to_post = pd.concat([your_team_pre_trade[1],your_team_post_trade[1]], axis = 1).T
      pre_to_post.index = ['Pre-trade','Post-trade']
      pre_to_post_styled = pre_to_post.style.map(stat_styler, middle = 0.5, multiplier = 300).format('{:,.1%}')
      st.dataframe(pre_to_post_styled, use_container_width = True)
    
      if their_team_pre_trade < their_team_post_trade:
          st.markdown('This trade benefits their team. H-score goes from ' + str(np.round(their_team_pre_trade[0]*100,1)) + '% to ' + str(np.round(their_team_post_trade[0]*100,1)) + '%')
      else:
          st.markdown('This trade does not benefit their team. H-score goes from ' + str(np.round(their_team_pre_trade[0]*100,1)) + '% to ' + str(np.round(their_team_post_trade[0]*100,1)) + '%')
                  
      pre_to_post = pd.concat([their_team_pre_trade[1],their_team_post_trade[1]], axis = 1).T
      pre_to_post.index = ['Pre-trade','Post-trade']
      pre_to_post_styled = pre_to_post.style.map(stat_styler, middle = 0.5, multiplier = 300).format('{:,.1%}')
      st.dataframe(pre_to_post_styled, use_container_width = True)

### Rank tabs 

@st.cache_data()
def make_rank_tab(scores : pd.DataFrame
                , player_multiplier : float):
  """Show rankings my general value

  Args:
      scores: Dataframe of floats, rows by player and columns by category
      player_multipliers: scaling factor to use for color-coded display of player stats
  
  Returns:
      None
  """
  counting_statistics = st.session_state.params['counting-statistics'] 
  percentage_statistics = st.session_state.params['percentage-statistics'] 

  scores.loc[:,'Rank'] = np.arange(scores.shape[0]) + 1
  scores.loc[:,'Player'] = scores.index
  scores = scores[['Rank','Player','Total'] + counting_statistics + percentage_statistics]
  
  scores_styled = scores.style.format("{:.2f}"
                                        ,subset = pd.IndexSlice[:,counting_statistics + percentage_statistics + ['Total']]) \
                                    .map(styler_a
                                        , subset = pd.IndexSlice[:,['Total']]) \
                                    .map(stat_styler
                                      , subset = pd.IndexSlice[:,counting_statistics + percentage_statistics]
                                      , multiplier = player_multiplier)
      
  rank_display = st.dataframe(scores_styled, hide_index = True)

@st.cache_data()
def make_h_rank_tab( info
                  , omega
                  , gamma
                  , alpha
                  , beta
                  , n_picks
                  , n_iterations
                  , winner_take_all
                  , punting
                  , player_stats):
  """Make ranks by H-score

  Args:
    info: dictionary with info related to player statistics etc. 
    omega: float, parameter as described in the paper
    gamma: float, parameter as described in the paper
    alpha: float, step size parameter for gradient descent 
    beta: float, decay parameter for gradient descent 
    n_picks: int, number of picks each drafter gets 
    n_iterations: int, number of gradient descent steps
    winner_take_all: Boolean of whether to optimize for the winner-take-all format
                      If False, optimizes for total categories
    punting: boolean for whether to adjust expectation of future picks by formulating a punting strategy
    player_stats: DataFrame of player statistics 

  Returns:
      None
  """
  H = HAgent(info = info
    , omega = omega
    , gamma = gamma
    , alpha = alpha
    , beta = beta
    , n_picks = n_picks
    , winner_take_all = winner_take_all
    , punting = punting)

  generator = H.get_h_scores(player_stats, [], [])
  for i in range(max(1,n_iterations)):
    h_res, c, cdf_estimates = next(generator)
    
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

  h_res = h_res.style.format("{:.1%}"
                              ,subset = pd.IndexSlice[:,['H-score']]) \
                      .map(styler_a
                            , subset = pd.IndexSlice[:,['H-score']]) \
                      .map(stat_styler, middle = 0.5, multiplier = 300, subset = rate_df.columns) \
                      .format('{:,.1%}', subset = rate_df.columns)
  h_score_display = st.dataframe(h_res, hide_index = True)