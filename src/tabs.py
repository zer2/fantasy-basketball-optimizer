import streamlit as st
import pandas as pd 
import numpy as np
from src.helper_functions import  static_score_styler, h_percentage_styler, get_categories, styler_a, styler_b, styler_c, stat_styler
from src.run_algorithm import HAgent, analyze_trade, analyze_trade_value
import os
import itertools
  
### Team tabs 

@st.cache_data()
def make_team_tab(scores : pd.DataFrame
              , my_players : list[str]
              , n_drafters : int
              , player_multiplier : float
              , team_multiplier : float
              ) -> pd.DataFrame:
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

  team_stats = scores[scores.index.isin(my_players)]
  expected = scores[0:len(my_players)*n_drafters].mean() * len(my_players)

  team_stats.loc['Total', :] = team_stats.sum(axis = 0)

  team_stats.loc['Expected', :] = expected
  team_stats.loc['Difference', :] = team_stats.loc['Total',:] - team_stats.loc['Expected',:]

  n_players_on_team = team_stats.shape[0] - 3

  if n_players_on_team > 0:

      team_stats_styled = team_stats.style.format("{:.2f}").map(styler_a) \
                                                  .map(styler_b, subset = pd.IndexSlice[['Expected','Total'], get_categories()]) \
                                                  .map(styler_c, subset = pd.IndexSlice[['Expected','Total'], ['Total']]) \
                                                  .map(stat_styler, subset = pd.IndexSlice[my_players, get_categories()], multiplier = player_multiplier) \
                                                  .applymap(stat_styler, subset = pd.IndexSlice['Difference', get_categories()], multiplier = team_multiplier)
      display = st.dataframe(team_stats_styled
                          , use_container_width = True
                                                    )     
  else:
    st.markdown('Your team does not have any players yet!')
  return team_stats
  
@st.cache_data()
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
        st.markdown('Your team is not full yet! Come back here when you have a full team')
  else:
        st.markdown('The H-score of your team is ' + str((base_h_score * 100).round(1).values[0]) + '%')

        base_win_rates_copy = base_win_rates.copy()
        base_win_rates_copy.insert(0, 'H-score', base_h_score)

        base_win_rates_formatted = h_percentage_styler(base_win_rates_copy)
        st.dataframe(base_win_rates_formatted, hide_index = True)
  
### Candidate tabs 

@st.cache_data()
def make_cand_tab(scores : pd.DataFrame
              , selection_list : list[str]
              , player_multiplier : float):
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

  scores_unselected_styled = static_score_styler(scores_unselected, player_multiplier)
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

  no_drop = team_stats.loc[['Total'],:]
  no_drop.index = [drop_player]
  
  drop_player_stats = scores.loc[drop_player]
  new =  team_stats.loc['Total',:] + scores_unselected - drop_player_stats

  new = pd.concat([no_drop,new])

  new_styled = static_score_styler(new, team_multiplier)
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
                , my_players : list[str]
                , players_chosen : list[str]):
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

  return next(H.get_h_scores(my_players, players_chosen))   

@st.cache_data()
def make_h_waiver_df(_H
                  , player_stats : pd.DataFrame
                  , mod_my_players : list[str]
                  , drop_player : str
                  , players_chosen : list[str]
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
  res, _, win_rates = next(_H.get_h_scores(mod_my_players, players_chosen))

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

  h_display = h_percentage_styler(h_display)

  st.dataframe(h_display, use_container_width = True)

### Trade tabs

@st.cache_data()
def make_trade_destination_display(_H
                  , player_stats : pd.DataFrame
                  , my_players : list[str]
                  , their_players_dict : dict[list[str]]
                  , players_chosen : list[str]
                  , format : str
                        ):
  """Make a dataframe showing which of your players would be good candidates to send to which other teams

  Args:
    _H: H-scoring agent, which can be used to calculate H-score 
    player_stats: DataFrame of player statistics 
    my_players: list of players on your team
    their_players_dict: dict relating other team names to their players 
    players_chosen: list of all chosen players
    format: Name of format. Included as input because it it an input to H
            and the cache should be re-calculated when format changes
)
  Returns:
      None
  """
  values_to_me = pd.Series([analyze_trade_value(player
                                      , [other_player for other_player in my_players if other_player != player]
                                      , _H
                                      , player_stats
                                      , players_chosen) for player in my_players
                    ]
                    , index = my_players)
  values_to_me = np.clip(values_to_me, 0, 1)

  values_to_team = pd.DataFrame(
                              {team : [analyze_trade_value(player
                                                    , list(their_players)
                                                    , _H
                                                    , player_stats
                                                    , players_chosen) 
                                        for player in my_players]
                                for team, their_players in their_players_dict.items()
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

@st.cache_data()
def make_trade_target_display(_H
                  , player_stats : pd.DataFrame
                  , my_players : list[str]
                  , their_players : list[str]
                  , players_chosen : list[str]
                  , values_to_team : pd.Series
                  , format : str
                        ):
  """Make a dataframe showing which of your players would be good candidates to send to which other teams

  Args:
    _H: H-scoring agent, which can be used to calculate H-score 
    player_stats: DataFrame of player statistics 
    my_players: list of players on your team
    their_players_dict: list of players on the other team
    players_chosen: list of all chosen players
    values_to_team: value of your own players to the selected team
    format: Name of format. Included as input because it it an input to H
            and the cache should be re-calculated when format changes
)
  Returns:
      None
  """
  values_to_me = pd.Series([analyze_trade_value(player
                                      , list(my_players)
                                      , _H
                                      , player_stats
                                      , players_chosen) for player in their_players
                    ]
                    , index = their_players)
  values_to_me = np.clip(values_to_me,0, 1)

  #make this into a team-wise dict
  value_to_them = pd.Series([analyze_trade_value(player
                                      , [other_player for other_player in their_players if other_player != player]
                                      , _H
                                      , player_stats
                                      , players_chosen) for player in their_players
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
    st.dataframe(values_to_me_styled, use_container_width = True)  

  with c2: 
    values_to_team_styled = values_to_team.to_frame().style.format("{:.2%}") \
                            .map(stat_styler
                                , middle = 0
                                , multiplier = 15000
                            )
    st.dataframe(values_to_team_styled, use_container_width = True)  

  return values_to_me

@st.cache_data()
def make_trade_suggestion_display(_H
                  , player_stats : pd.DataFrame
                  , players_chosen : list[str]
                  , my_players : list[str]
                  , their_players : list[str]
                  , general_values : pd.Series
                  , replacement_value : float
                  , values_to_me : pd.Series
                  , values_to_them : pd.Series
                  , your_differential_threshold : float
                  , their_differential_threshold : float
                  , combo_params : list[tuple]
                  , format : str):
  """Shows automatic trade suggestions 

  Args:
    _H: H-scoring agent, which can be used to calculate H-score 
    player_stats: DataFrame of player statistics 
    players_chosen: list of all chosen players
    my_players: initial list of players on your team
    their_players: initial list of players on other team 
    general_values : series representing general values, for filtering purposes
    replacement_value : generic value of the top replacement player
    values_to_me : targetedness of counterparty players to you
    values_to_them : targetedness of your players to counterparty
    your_differential_threshold : for display, only include trades above this level of value for you
    their_differential_threshold : for display, only include trades above this level of value for counterparty
    combo_params : list of parameter sets for combos to try. See options page for details 
    format: Name of format. Included as input because it is an input to H
            and the cache should be re-calculated when format changes
  Returns:
      None
  """


  def get_combos(players_with_weight : list[tuple]
              , n : int) -> list[tuple]:
    #helper function just for getting all 1,2,3 combos etc. from a set of candidates
    player_combos_with_weight = list(itertools.combinations(players_with_weight,n))
    player_combos_with_total_weight = [(list(z[0] for z in m), sum(z[1] for z in m)) 
                                        for m in player_combos_with_weight]
    return player_combos_with_total_weight

  def get_cross_combos(n : int
                        , m : int
                        , heuristic_differential_threshold : float
                        , value_threshold : float) -> pd.DataFrame :
    #helper function for getting trades between combos. Creates a dataframe for vectorized filtering
    my_candidate_players = [p for p in my_players if values_to_them[p] > heuristic_differential_threshold ]
    their_candidate_players = [p for p in their_players if values_to_me[p] > heuristic_differential_threshold ]

    my_players_with_weight = [(p,general_values[p]) for p in my_candidate_players]
    their_players_with_weight = [(p,general_values[p])  for p in their_players]

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

  all_combos = pd.concat([get_cross_combos(n,m,hdt,vt) for n,m,hdt,vt in combo_params])

  full_dataframe = pd.DataFrame()

  for key, row in all_combos.iterrows():

    my_trade = row['My Trade']
    their_trade = row['Their Trade']

    my_general_value = row['My Value']
    their_general_value = row['Their Value']
    #check if the general value disparity is extreme. If it is, pass 

    my_others = [x for x in my_players if x not in my_trade]
    their_others = [x for x in their_players if x not in their_trade]

    trade_results = analyze_trade(my_others
                              , my_trade
                              , their_others
                              , their_trade
                              , _H
                              , player_stats
                              , players_chosen
                              , 1)
    your_score_pre_trade = trade_results[1]['pre']['H-score']
    your_score_post_trade = trade_results[1]['post']['H-score']
    their_score_pre_trade = trade_results[2]['pre']['H-score']
    their_score_post_trade = trade_results[2]['post']['H-score']

    your_differential = your_score_post_trade - your_score_pre_trade
    their_differential = their_score_post_trade - their_score_pre_trade

    if ( your_differential > your_differential_threshold ) & \
          (their_differential > their_differential_threshold ):
      new_row = pd.DataFrame({ 'Send' : [my_trade]
                                ,'Receive' : [their_trade]
                                ,'Your Differential' : [your_differential]
                                ,'Their Differential' : [their_differential]
                                })
      full_dataframe = pd.concat([full_dataframe, new_row])

  full_dataframe = full_dataframe.reset_index().drop(columns = 'index')
  if len(full_dataframe) > 0:
    goodness = full_dataframe['Your Differential']
    full_dataframe = full_dataframe.loc[list(goodness.sort_values(ascending = False).index)]

    full_dataframe_styled = full_dataframe.reset_index(drop = True).style.format("{:.2%}"
                                      , subset = ['Your Differential','Their Differential']) \
                            .map(stat_styler
                                , middle = 0
                                , multiplier = 15000
                                , subset = ['Your Differential','Their Differential']
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
    
@st.cache_data()
def make_trade_display(_H
                  , player_stats : pd.DataFrame
                  , players_chosen : list[str]
                  , n_iterations : int
                  , my_trade : list[str]
                  , their_trade : list[str]
                  , my_players : list[str]
                  , their_players : list[str]
                  , their_team_name : str
                  , format : str):
  """show the results of a potential trade

  Args:
    _H: H-scoring agent, which can be used to calculate H-score 
    player_stats: DataFrame of player statistics 
    players_chosen: list of all chosen players
    n_iterations: int, number of gradient descent steps
    my_trade: player(s) to be traded from your team
    their_trade: player(s) to be traded for
    my_players: initial list of players on your team
    their_players: initial list of players on other team 
    their_team_name: name of counterparty team
    format: Name of format. Included as input because it it an input to H
            and the cache should be re-calculated when format changes
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
          st.markdown('This trade does not benefit ' + their_team_name + ' :slightly_frowning_face:')
                  
      pre_to_post = pd.concat([their_team_pre_trade,their_team_post_trade], axis = 1).T
      pre_to_post.index = ['Pre-trade','Post-trade']
      pre_to_post_styled = h_percentage_styler(pre_to_post)
      st.dataframe(pre_to_post_styled, use_container_width = True, height = 108)

### Rank tabs 

@st.cache_data()
def make_rank_tab(scores : pd.DataFrame, player_multiplier : float):
  """Show rankings my general value

  Args:
      scores: Dataframe of floats, rows by player and columns by category
      player_multipliers: scaling factor to use for color-coded display of player stats
  
  Returns:
      None
  """
  scores.loc[:,'Rank'] = np.arange(scores.shape[0]) + 1
  scores.loc[:,'Player'] = scores.index
  scores = scores[['Rank','Player','Total'] + get_categories()]
  
  scores_styled = static_score_styler(scores,player_multiplier)
      
  rank_display = st.dataframe(scores_styled, hide_index = True)

@st.cache_data()
def make_h_rank_tab( info : dict
                  , omega : float
                  , gamma : float
                  , alpha : float
                  , beta : float
                  , n_picks : int
                  , n_iterations : int
                  , winner_take_all : bool
                  , punting : bool
                  , player_stats : pd.DataFrame):
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

  generator = H.get_h_scores([], [])
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

  h_res = h_percentage_styler(h_res)
  h_score_display = st.dataframe(h_res, hide_index = True)
