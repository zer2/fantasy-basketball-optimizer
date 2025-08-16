import streamlit as st
import pandas as pd 
import numpy as np
from src.helpers.helper_functions import h_percentage_styler, get_selected_categories, \
                                styler_a, styler_b, styler_c, stat_styler, \
                                get_your_differential_threshold, get_their_differential_threshold, get_combo_params
from src.math.position_optimization import check_team_eligibility
import itertools

@st.fragment
def make_trade_tab(H
                   ,selections_df : pd.DataFrame
                   , player_assignments : dict[list]
                   , z_scores_unselected: pd.DataFrame
                   , g_scores_unselected: pd.DataFrame):
  """Make the full trading tab- ideal destinations, targets, and suggestions

  Args:
    H: H-scoring agent, which can be used to calculate H-score 
    selections_df: The selections df from the rosters page- potentially modified by the user
    player_assignments: Dictionary form of the selections df
    z_scores_unselected: Z-score dataframe, filtered to only include unselected players
    g_scores_unselected: G-score dataframe, filtered to only include unselected players

)
  Returns:
      None
  """
  
  c1, c2 = st.columns([0.5,0.5])

  with c1: 
    trade_party_seat = st.selectbox(f'Which team do you want to trade from?'
        , selections_df.columns
        , index = 0)

  with c2: 
    trade_party_players = [x for x in selections_df[trade_party_seat] if x != 'RP']

    if len(trade_party_players) < st.session_state.n_picks:
        st.markdown("""This team is not full yet! Fill it and other teams out on the 
                    "Rosters" page then come back here""")

    else:
        
        counterparty_players_dict = { team : players for team, players in selections_df.items() 
                                if ((team != trade_party_seat) & (not  any(p == 'RP' for p in players)))
                                  }
        
        if len(counterparty_players_dict) >=1:

          trade_counterparty_seat = st.selectbox(
              f'Which team do you want to trade with?',
              [s for s in counterparty_players_dict.keys()],
              index = 0
            )
          
          trade_counterparty_players = counterparty_players_dict[trade_counterparty_seat].values

        else: 
          trade_counterparty_players = []

  st.divider()

  if len(trade_party_players) == st.session_state.n_picks:

    if len(trade_counterparty_players) < st.session_state.n_picks:
      st.markdown('The other team is not full yet!')
    else:

      c1, c2 = st.columns([0.4,0.6])

      with c1: 

        #load up the selected suggested trade if applicable
        if 'trade_suggestions_df' in st.session_state:
            selected_rows = st.session_state.trade_suggestions_df.selection.rows

            if len(selected_rows) > 0:
              selected_row = st.session_state.df.iloc[selected_rows[0]]

              trade_possible = all ([x in trade_party_players for x in selected_row['Send']]) & \
                              all ([x in trade_counterparty_players for x in selected_row['Receive']])
              if trade_possible:
                default_party_players = selected_row['Send']
                default_counterparty_players = selected_row['Receive']

              else:
                  default_party_players = None
                  default_counterparty_players = None
                  
        else:
            default_party_players = None
            default_counterparty_players = None

        players_sent = st.multiselect(
          'Which players are you trading?'
          ,trade_party_players
          ,default = default_party_players
          )

        players_received = st.multiselect(
              'Which players are you receiving?'
              ,trade_counterparty_players
              ,default = default_counterparty_players

          )

      with c2:

        h_tab, z_tab, g_tab = st.tabs(['H-score','Z-score','G-score'])

        if (len(players_sent) == 0) | (len(players_received) == 0):
          st.markdown('A trade must include at least one player from each team')

        else:

          with h_tab:
            make_trade_h_tab(H
                            , player_assignments 
                            , st.session_state.n_iterations 
                            , trade_party_seat
                            , players_sent
                            , trade_counterparty_seat
                            , players_received
                            , st.session_state.scoring_format
                            , st.session_state.info_key)

          with z_tab:
            make_trade_score_tab(st.session_state.z_scores 
                              , players_sent
                              , players_received 
                              , st.session_state.params['z-score-player-multiplier']
                              , st.session_state.params['z-score-team-multiplier']
                              , st.session_state.info_key
                              )
          with g_tab:
            make_trade_score_tab(st.session_state.g_scores 
                              , players_sent
                              , players_received 
                              , st.session_state.params['g-score-player-multiplier']
                              , st.session_state.params['g-score-team-multiplier']
                              , st.session_state.info_key
                              )


      if st.session_state.scoring_format == 'Rotisserie':
        general_value = st.session_state.z_scores.sum(axis = 1)
        replacement_value = z_scores_unselected.iloc[0].sum() 
      else:
        general_value = st.session_state.g_scores.sum(axis = 1)
        replacement_value = g_scores_unselected.iloc[0].sum()

      #slightly hacky way to make all of the multiselects blue
      st.markdown("""
          <style>
              span[data-baseweb="tag"][aria-label="1 for 1, close by backspace"]{
                  background-color: #3580BB; color:white;
              }
              span[data-baseweb="tag"][aria-label="2 for 2, close by backspace"]{
                  background-color: #3580BB; color:white;
              }
              span[data-baseweb="tag"][aria-label="3 for 3, close by backspace"]{
                  background-color: #3580BB; color:white;
              }
          </style>
          """, unsafe_allow_html=True)
      
      st.divider()

      c1, c2 = st.columns([0.5,0.5])

      with c1:
    
        st.markdown('Suggested trades. Check the box to analyze the trade')

      with c2:

        trade_filter = st.multiselect(''
                                    , [(x[0],x[1]) for x in get_combo_params()]
                                    , format_func = lambda x: str(x[0]) + ' for ' + str(x[1])
                                    , default = [(1,1)])
        
      filtered_combo_params = [x for x in get_combo_params() if (x[0],x[1]) in trade_filter]

      st.session_state.df = make_trade_suggestion_df(H
            , player_assignments
            , trade_party_seat
            , trade_counterparty_seat
            , general_value
            , replacement_value
            , get_your_differential_threshold()
            , get_their_differential_threshold()
            , filtered_combo_params
            , st.session_state.scoring_format
            , st.session_state.info_key) 
      
      if st.session_state.df is None: 
        st.markdown('No promising trades found')
      else:
        full_dataframe_styled = st.session_state.df.reset_index(drop = True).style.format("{:.2%}"
                                      , subset = ['Your Score'
                                                ,'Their Score']) \
                            .map(stat_styler
                                , middle = 0
                                , multiplier = 15000
                                , subset = ['Your Score'
                                          ,'Their Score']
                            ).set_properties(**{
                                  'font-size': '12pt',
                              })
        st.dataframe(full_dataframe_styled
                , key = 'trade_suggestions_df'
                , on_select = 'rerun'
                , selection_mode = 'single-row'
                , hide_index = True
                , column_config={
                            "Send": st.column_config.ListColumn("Send", width = 'large')
                            ,"Receive": st.column_config.ListColumn("Receive", width = 'large')
                            ,"Your Score": st.column_config.TextColumn("Your Score", width = 'small')
                            ,"Their Score": st.column_config.TextColumn("Their Score", width = 'small')

                }
                , use_container_width=True
)

@st.cache_data(show_spinner = False, ttl = 3600)
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
                                                                                , ['Total'] + get_selected_categories()]) \
                                              .map(stat_styler, subset = pd.IndexSlice[players_sent + players_received
                                                                                  , get_selected_categories()]
                                                                                  , multiplier = player_multiplier) \
                                              .map(stat_styler, subset = pd.IndexSlice[['Total Difference']
                                                                                  , get_selected_categories()]
                                                                                  , multiplier = player_multiplier)  
  st.dataframe(full_frame_styled
                , use_container_width = True
                , height = len(full_frame) * 35 + 38
                    )     

@st.cache_data(show_spinner = False, ttl = 3600)
def get_cross_combos(n : int
                      , m : int
                      , my_players : list[str]
                      , their_players : list[str]
                      , general_values : pd.Series
                      , replacement_value : float
                      , value_threshold : float
                      , info_key : int) -> pd.DataFrame :
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
    value_threshold : only consider trades with absolute value of G-score difference below the value threshold 

  Returns:
      Dataframe of viable trades according to the criteria
  """
  #helper function for getting trades between combos. Creates a dataframe for vectorized filtering


  my_players_with_weight = [(p,general_values[p]) for p in my_players]
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

def get_combos(players_with_weight : list[tuple]
            , n : int) -> list[tuple]:
  #helper function just for getting all 1,2,3 combos etc. from a set of candidates
  player_combos_with_weight = list(itertools.combinations(players_with_weight,n))
  player_combos_with_total_weight = [(list(z[0] for z in m), sum(z[1] for z in m)) 
                                      for m in player_combos_with_weight]
  return player_combos_with_total_weight

@st.cache_data(show_spinner = False, ttl = 3600)
def make_combo_df(all_combos : list
                  , my_team : str
                  , their_team : str
                  , _H
                  , player_assignments : dict[list[str]]
                  , scoring_format : str
                  , info_key) -> pd.DataFrame:
  """Makes a dataframe of all trade possibilities according to specifications

  Args:
    combos: list of trades to try. These are tuples where the first specifies players to send, and the second to receive 
    my_players: initial list of players on your team
    their_players: initial list of players on other team 
    _H: H-scoring agent, which can be used to calculate H-score 
    player_assignments: 
    scoring_format: Name of format. Included as input because it is an input to H
            and the cache should be re-calculated when format changes
  Returns:
      None
  """
  
  def process_row(row):
      
      my_trade = row['My Trade']
      their_trade = row['Their Trade']

      #check if the general value disparity is extreme. If it is, pass 
      trade_results = analyze_trade(my_team
                                , my_trade
                                , their_team
                                , their_trade
                                , _H
                                , player_assignments
                                , 1)
      
      if trade_results is not None:
        your_score_pre_trade = trade_results[1]['pre']['H-score']
        your_score_post_trade = trade_results[1]['post']['H-score']
        their_score_pre_trade = trade_results[2]['pre']['H-score']
        their_score_post_trade = trade_results[2]['post']['H-score']

        your_differential = your_score_post_trade - your_score_pre_trade
        their_differential = their_score_post_trade - their_score_pre_trade

        new_row = pd.DataFrame({ 'Send' : [my_trade]
                                  ,'Receive' : [their_trade]
                                  ,'Your Score' : [your_differential]
                                  ,'Their Score' : [their_differential]
                                  })
        return new_row
      else:
        return pd.DataFrame()
      
  full_dataframe = pd.concat([process_row(row) for key, row in all_combos.iterrows()])

  full_dataframe = full_dataframe.sort_values('Your Score', ascending = False)

  return full_dataframe

@st.cache_data(show_spinner = """Finding suggested trades. How long this will take depends on 
                                  the trade parameters"""
               , ttl = 3600)
def make_trade_suggestion_df(_H
                  , player_assignments : dict[list[str]]
                  , my_team : str
                  , their_team : str
                  , general_values : pd.Series
                  , replacement_value : float
                  , your_differential_threshold : float
                  , their_differential_threshold : float
                  , combo_params : list[tuple]
                  , scoring_format : str
                  , info_key : int):
  """Shows automatic trade suggestions 

  Args:
    _H: H-scoring agent, which can be used to calculate H-score 
    player_assignments: 
    my_players: initial list of players on your team
    their_players: initial list of players on other team 
    general_values : series representing general values, for filtering purposes
    replacement_value : generic value of the top replacement player
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

  my_candidates, their_candidates = identify_trade_candidates(_H, my_team, their_team, player_assignments)
  
  all_combos = pd.concat([get_cross_combos(n
                                , m
                                , my_candidates 
                                , their_candidates 
                                , general_values 
                                , replacement_value 
                                , vt
                                , st.session_state.info_key) for n,m,vt in combo_params])
  

  full_dataframe = make_combo_df(all_combos
                  , my_team
                  , their_team
                  , _H
                  , player_assignments 
                  , scoring_format
                  , st.session_state.info_key) 
  
  my_threshold_criteria = full_dataframe['Your Score'] > your_differential_threshold
  their_threshold_criteria = full_dataframe['Their Score'] > their_differential_threshold

  full_dataframe = full_dataframe[my_threshold_criteria & their_threshold_criteria]
  

  if len(full_dataframe) > 0:
    
    return full_dataframe
  else: 
    return None

@st.cache_data(show_spinner = False, ttl = 3600)
def make_trade_h_tab(_H
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
                                , player_assignments
                                ,n_iterations)
      your_team_pre_trade = trade_results[1]['pre']
      your_team_post_trade = trade_results[1]['post']
      their_team_pre_trade = trade_results[2]['pre']
      their_team_post_trade = trade_results[2]['post']

      if your_team_pre_trade['H-score'] < your_team_post_trade['H-score']:
        my_emoji = '👍'
      else: 
        my_emoji = '👎'

      if their_team_pre_trade['H-score'] < their_team_post_trade['H-score']:
        their_emoji = '👍'
      else:
        their_emoji = '👎'

      your_team_tab, their_team_tab = st.tabs(['Your Team ' + my_emoji
                                               ,'Their Team' + their_emoji])

      with your_team_tab:
        
        pre_to_post = pd.concat([your_team_pre_trade,your_team_post_trade], axis = 1).T
        pre_to_post.index = ['Pre-trade','Post-trade']
        pre_to_post_styled = h_percentage_styler(pre_to_post)
        st.dataframe(pre_to_post_styled, use_container_width = True, height = 108)

      with their_team_tab:
      

                    
        pre_to_post = pd.concat([their_team_pre_trade,their_team_post_trade], axis = 1).T
        pre_to_post.index = ['Pre-trade','Post-trade']
        pre_to_post_styled = h_percentage_styler(pre_to_post)
        st.dataframe(pre_to_post_styled, use_container_width = True, height = 108)

#ZR: This should be cachable!
def analyze_trade(team_1
                  ,team_1_trade : list[str]
                  ,team_2
                  ,team_2_trade : list[str]
                  ,H
                  ,player_assignments : dict[list[str]]
                  ,n_iterations : int
                  ) -> dict:    

    """Compute the results of a potential trade

    Args:
      team_1_other: remaining players, not to be traded from the first team
      team_1_trade: player(s) to be traded from the first team
      team_2_other: remaining players, not to be traded from the first team
      team_2_trade: player(s) to be traded from the second team
      H: H-scoring agent, which can be used to calculate H-score 
      players_chosen: list of all chosen players
      n_iterations: int, number of gradient descent steps

    Returns:
      Dictionary with results of the trade
    """


    post_trade_team_1 = [p for p in player_assignments[team_1] if p not in team_1_trade] + team_2_trade
    post_trade_team_2 = [p for p in player_assignments[team_2] if p not in team_2_trade] + team_1_trade

    post_trade_assignments = player_assignments.copy()

    post_trade_assignments[team_1] = post_trade_team_1

    #immediately return None if the first team is ineligible based on position
    team_1_positions = st.session_state.info['Positions'].loc[post_trade_team_1]
    team_1_eligible = check_team_eligibility(team_1_positions)
    if not team_1_eligible:
       return None

    post_trade_assignments[team_2] = post_trade_team_2

    #do the same for the second team
    team_2_positions = st.session_state.info['Positions'].loc[post_trade_team_1]
    team_2_eligible = check_team_eligibility(team_2_positions)
    if not team_2_eligible:
       return None

    res_1_1 = next(H.get_h_scores(player_assignments, team_1))
    res_2_2 = next(H.get_h_scores(player_assignments, team_2))
 
    n_player_diff = len(team_1_trade) - len(team_2_trade)

    if n_player_diff > 0:
        generator = H.get_h_scores(post_trade_assignments, team_1)
        for i in range(n_iterations):
            res_1_2  = next(generator)
        
        res_2_1 = next(H.get_h_scores(post_trade_assignments, team_2))

    elif n_player_diff == 0:
        res_1_2 = next(H.get_h_scores(post_trade_assignments, team_1))
        res_2_1 = next(H.get_h_scores(post_trade_assignments, team_2))

    else:
        res_1_2 = next(H.get_h_scores(post_trade_assignments, team_1))

        generator = H.get_h_scores(post_trade_assignments, team_2)
        for i in range(n_iterations):
            res_2_1= next(generator)
    
    #helper function just for this procedure
    def get_full_row(scores, rates):

        idxmax = scores.idxmax()
        score = pd.Series([scores[idxmax]], index = ['H-score'])
        rate = rates.loc[idxmax]

        return pd.concat([score, rate])

    team_1_info = {'pre' : get_full_row(res_1_1['Scores'], res_1_1['Rates'])
                        ,'post' : get_full_row(res_1_2['Scores'], res_1_2['Rates'])}
    team_2_info = {'pre' : get_full_row(res_2_2['Scores'], res_2_2['Rates'])
                        ,'post' : get_full_row(res_2_1['Scores'], res_2_1['Rates'])}
                      
    results_dict = {1 : team_1_info
                    ,2 : team_2_info
                   }

    return results_dict

def identify_trade_candidates(_H
                              , my_team : str
                              , their_team : str
                              , player_assignments : dict):
  """Identify players from within the two teams to be trade candidates with each other
  The methodology is to check how valuable players are to their own team vs the other team
  The players that are relatively value to the other team (normalizing for average player value)
  are the candidates

    Args:
      my_team: identifier for the trading team
      their_team: identifier for the counterparty team
      H: H-scoring agent, which can be used to calculate H-score 
      player_assignments: Dictionary form of the selections df


    Returns:
      Tuple of two player lists (my_players, their_players)
  """

  my_players = player_assignments[my_team]
  their_players = player_assignments[their_team]

  my_values_to_me = pd.Series([analyze_trade_value(player
                                      , my_team
                                      , _H
                                      , player_assignments) for player in my_players
                    ]
                    , index = my_players)
  my_values_to_me = my_values_to_me - my_values_to_me.mean() #normalize

  their_values_to_me = pd.Series([analyze_trade_value(player
                                      , my_team
                                      , _H
                                      , player_assignments) for player in their_players
                    ]
                    , index = their_players)
  their_values_to_me = their_values_to_me - their_values_to_me.mean() #normalize


  my_values_to_them = pd.Series([analyze_trade_value(player
                                      , their_team
                                      , _H
                                      , player_assignments) for player in my_players
                    ]
                    , index = my_players)
  my_values_to_them = my_values_to_them - my_values_to_them.mean() #normalize

  their_values_to_them = pd.Series([analyze_trade_value(player
                                      , their_team
                                      , _H
                                      , player_assignments) for player in their_players
                    ]
                    , index = their_players)
  their_values_to_them = their_values_to_them - their_values_to_them.mean() #normalize

  my_differences = my_values_to_me - my_values_to_them
  their_differences = their_values_to_them - their_values_to_me 

  my_candidates = [p for p in my_players if my_differences[p] < 0]
  their_candidates = [p for p in their_players if their_differences[p] < 0]

  return my_candidates, their_candidates

def analyze_trade_value(player : str
                  ,team : str
                  ,H
                  ,player_assignments : dict[list[str]]
                  ) -> float:    

    """Estimate how valuable a player would be to a particular team

    Args:
      player: player to evaluate
      rest_of_team: other player(s) on team
      H: H-scoring agent, which can be used to calculate H-score 
      players_chosen: list of all chosen players

    Returns:
      Float, relative H-score value
    """

    without_player = player_assignments.copy()
    without_player[team] = [p for p in without_player[team] if p != player]

    with_player = player_assignments.copy()
    if player not in with_player[team]:
        with_player[team] = with_player[team] + [player]

    res_without_player= next(H.get_h_scores(without_player,team, exclusion_list = [player]))
    res_with_player = next(H.get_h_scores(with_player, team))

    res = (res_with_player['Scores'].max() - res_without_player['Scores'].max())

    return res