import streamlit as st
import pandas as pd 
import numpy as np
from src.helpers.helper_functions import get_position_numbers_unwound, static_score_styler, h_percentage_styler, get_selected_categories, \
                                styler_a, stat_styler, get_position_structure
from src.math.algorithm_helpers import savor_calculation
from src.data_retrieval.get_data import get_htb_adp
from src.math.algorithm_helpers import combinatorial_calculation

@st.cache_data(show_spinner = True, ttl = 3600)
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

  scores_unselected = _scores[~_scores.index.isin(selection_list)]

  if remaining_cash:

    scores_unselected.loc[:,'$ Value'] = savor_calculation(scores_unselected['Total']
                                                          , total_players - len(selection_list)
                                                          , remaining_cash
                                                          , st.session_state['streaming_noise'])
    
  scores_unselected_styled = static_score_styler(scores_unselected, player_multiplier)

  scores_display = st.dataframe(scores_unselected_styled, use_container_width = True)

  return scores_unselected

def make_hashable(obj):
    """Recursively convert obj into a hashable form.
    - dict  -> frozenset of (key, value)
    - list  -> frozenset of items
    - set   -> frozenset of items
    - tuple -> tuple of items (preserve order)
    """
    if isinstance(obj, dict):
        return frozenset((k, make_hashable(v)) for k, v in obj.items())
    if isinstance(obj, list):
        return frozenset(make_hashable(v) for v in obj)
    if isinstance(obj, set):
        return frozenset(make_hashable(v) for v in obj)
    if isinstance(obj, tuple):
        return tuple(make_hashable(v) for v in obj)
    
#ZR: Change the name of this
def make_h_cand_tab(_H
                    ,_g_scores
                    ,_z_scores
                    ,player_assignments
                    ,draft_seat
                    ,n_iterations
                    ,v
                    ,display_period : int = 5
                    ,cash_remaining_per_team : dict[int] = None
                    ,generic_player_value : pd.Series = None
                    ,total_players : int = None
                    ,drop_player : str = None):
  

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

  #ZR: This cache should include format too- auction vs draft, and other things
  if (draft_seat, make_hashable(player_assignments), n_iterations) in st.session_state.res_cache:
    cached_info = st.session_state.res_cache[(draft_seat, make_hashable(player_assignments), n_iterations)]
    res = cached_info['res']
    iteration_range = range(cached_info['iteration'] - 1, n_iterations)
    cached_res = True

  else:
    _H = _H.clear_initial_weights()
            
    if drop_player is None:
      generator = _H.get_h_scores(player_assignments, draft_seat, cash_remaining_per_team)
    else:
      from copy import deepcopy 
      player_assignments = deepcopy(player_assignments)
      player_assignments[draft_seat] = [player for player in player_assignments[draft_seat] if player != drop_player]
      generator = _H.get_h_scores(player_assignments, draft_seat, cash_remaining_per_team)
    
    iteration_range = range(max(1,n_iterations))

    cached_res = False

  placeholder = st.empty()

  adps = get_htb_adp()

  #if n_iterations is 0 we run just once
  for i in iteration_range:

    if not cached_res:
      res = next(generator)
      
    rosters = res['Rosters']

    if rosters.shape[1] > 0:
      fits_roster = rosters.loc[:,0] >= 0
    else:
      fits_roster = pd.Series([True] * len(res['Scores']), index = res['Scores'].index)

    rosters = rosters[list(fits_roster.values)]
    score = res['Scores'][fits_roster]
    weights = res['Weights'][fits_roster]
    win_rates = res['Rates'][fits_roster]
    position_shares = res['Position-Shares']
    future_diffs = res['Future-Diff']

    win_rates.columns = get_selected_categories()

    #normalize weights by what we expect from other drafters
    weights = pd.DataFrame(weights
                  , index = score.index
                  , columns = get_selected_categories())/v
    
    with placeholder.container():

      score.name = 'H-score'
      score_df = pd.DataFrame(score)

      display = ((i+1) % display_period == 0) or (i == n_iterations - 1) or (n_iterations <= 1)

      h_tab, g_tab = st.tabs(['H-score','G-score'])

      with h_tab:

        if cash_remaining_per_team:
          
          if display:

            if sum(fits_roster) == 0:
              st.error('Illegal roster!')
              st.stop()

            rate_df = win_rates.dropna()
            rate_display = score_df.merge(rate_df, left_index = True, right_index = True)

            players_chosen = [x for v in player_assignments.values() for x in v if x == x]
            total_cash_remaining = np.sum([v for k, v in cash_remaining_per_team.items()])

            rate_display.loc[:,'$ Value'] = savor_calculation(score_df.sort_values(by = 'H-score',ascending = False)
                                                            , total_players - len(players_chosen)
                                                            , total_cash_remaining
                                                            , st.session_state['streaming_noise_h'])
            
            rate_display = rate_display[['$ Value','H-score'] + get_selected_categories()]

            rate_display = pd.DataFrame({'Your $ Value' : rate_display['$ Value']
                                          , '$ Value' : generic_player_value.loc[rate_display.index]})
            
            rate_display.loc[:,'Difference'] = rate_display['Your $ Value'] - rate_display['$ Value']

            rate_display = rate_display.sort_values('Your $ Value', ascending = False)
            score_df = score_df.loc[rate_display.index]

            rate_display = rate_display.join(rate_df)

            rate_display = rate_display[['Difference','Your $ Value','$ Value'] + list(rate_df.columns)]

            rate_display_styled = rate_display.style.format("{:.1f}"
                                                              , subset = ['Your $ Value', '$ Value','Difference']) \
                      .map(styler_a
                          , subset = ['Your $ Value', '$ Value']) \
                      .background_gradient(axis = None
                                          ,cmap = 'PiYG'
                                          ,subset = ['Difference']) \
                      .map(stat_styler, middle = 0.5, multiplier = 300, subset = rate_df.columns) \
                      .format('{:,.1%}', subset = rate_df.columns)
            
            st.dataframe(rate_display_styled)
        else:

            if display:

              score_df = score_df.sort_values('H-score',ascending = False)

              if sum(fits_roster) == 0:
                st.error('Illegal roster!')
                st.stop()

              if st.session_state.scoring_format == 'Rotisserie':
                e_points = res['Rates'][fits_roster] * (len(player_assignments) -1) + 1
                rate_df = e_points.loc[score_df.index].dropna()
                style_format = "{:.1f}"
                format_middle = (len(player_assignments) -1)/2 + 1
                format_multiplier = 15 

              else:
                rate_df = win_rates.loc[score_df.index].dropna()
                style_format = "{:.1%}"
                format_middle = 0.5
                format_multiplier = 300

              if st.session_state.data_option == 'Projection':
                rate_display = score_df.merge(adps, left_index = True, right_index = True, how = 'left') \
                                        .merge(rate_df, left_index = True, right_index = True)      
                adp_col = ['ADP']
              else:
                rate_display = score_df.merge(rate_df, left_index = True, right_index = True)    
                adp_col = []
                    
              if drop_player is not None:
                def color_blue(label):
                    return "background-color: blue; color:white" if label == drop_player else None
                
                rate_display_styled = rate_display.reset_index().style.format("{:.1%}"
                                ,subset = pd.IndexSlice[:,['H-score']]) \
                                                      .format("{:.1f}"
                                ,subset = pd.IndexSlice[:,adp_col]) \
                        .map(styler_a
                              , subset = pd.IndexSlice[:,['H-score'] + adp_col]) \
                        .map(stat_styler, middle = format_middle, multiplier = format_multiplier, subset = rate_df.columns) \
                        .format(style_format, subset = rate_df.columns) \
                        .map(color_blue, subset = pd.IndexSlice[:,['Player']])
                
                g_scores_unselected = _g_scores[_g_scores.index.isin(score_df.index)].sort_values('Total', ascending = False)
                
                rate_display = score_df.merge(win_rates.dropna()
                              , left_index = True
                              , right_index = True)
                
                st.dataframe(rate_display_styled
                            , key = 'rate_display_' + str(i)
                            , selection_mode = 'single-row'
                            , use_container_width = True
                            , hide_index = True)

              else:
                rate_display_styled = rate_display.style.format("{:.1%}"
                                ,subset = pd.IndexSlice[:,['H-score']]) \
                                                      .format("{:.1f}"
                                ,subset = pd.IndexSlice[:,adp_col]) \
                        .map(styler_a
                              , subset = pd.IndexSlice[:,['H-score'] + adp_col]) \
                        .map(stat_styler, middle = format_middle, multiplier = format_multiplier, subset = rate_df.columns) \
                        .format(style_format, subset = rate_df.columns)
                  
                st.dataframe(rate_display_styled
                                  , key = 'rate_display'
                                , use_container_width = True)            
                
              if st.session_state.scoring_format == 'Rotisserie':

                  st.caption('''Expected totals are based on standard fantasy point scoring. 
                            One point for last, two points for second last, etc. The baseline 
                            expected total for a category is ''' + str(format_middle))
      with g_tab:

        if display:

          g_df = _g_scores.loc[score.index]
          g_display = score_df.merge(g_df, left_index = True, right_index = True)

          if cash_remaining_per_team is not None:
              ''' #ZR: Fix this for auctions; make it show the dollar values 
              g_display.loc[:,'$ Value'] = savor_calculation(g_display['Total']
                                                          , total_players - len(selection_list)
                                                          , remaining_cash
                                                          , st.session_state['streaming_noise'])
              '''

          if drop_player is not None:

            scores_unselected_styled = static_score_styler(g_display.reset_index()
                                              , st.session_state.params['g-score-player-multiplier']
                                              , st.session_state.params['g-score-total-multiplier'])
            
            scores_unselected_styled = scores_unselected_styled.map(color_blue
                                                                    , subset = pd.IndexSlice[:,['Player']])
            st.dataframe(scores_unselected_styled, use_container_width = True, hide_index = True)
    
          else:
            scores_unselected_styled = static_score_styler(g_display
                                              , st.session_state.params['g-score-player-multiplier']
                                              , st.session_state.params['g-score-total-multiplier'])

            st.dataframe(scores_unselected_styled, use_container_width = True)         

  #save info for the detailed launcher
  hashable_player_assignments =  make_hashable(player_assignments)

  if (draft_seat, hashable_player_assignments, n_iterations) not in res:

    st.session_state.res_cache[(draft_seat, hashable_player_assignments, n_iterations)] = {'res' : res
                                                                                          ,'iteration' : i}

    g_scores_unselected = _g_scores[_g_scores.index.isin(score_df.index)].sort_values('Total', ascending = False)

    score_df.loc[:,'Rank'] = range(1, len(score_df) + 1)

    if 'ADP' in rate_display.columns:
      rate_display = rate_display.drop(columns = ['ADP'])

    st.session_state.info_for_detailed_view =  dict(player_assignments = player_assignments
            ,draft_seat = draft_seat
            ,score_df = score_df
            ,win_rates = win_rates
            ,_g_scores = _g_scores
            ,future_diffs = future_diffs
            ,weights = weights
            ,position_shares = position_shares
            ,res = res
            ,_H = _H
            ,rosters = rosters
            ,rate_display = rate_display
            ,g_scores_unselected = g_scores_unselected
            ,iteration = i)
    
  st.markdown(
        """
    <style>
    div[data-testid="stDialog"] div[role="dialog"]:has(.big-dialog) {
        width: 90vw;
        height: 90vh;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )      
  
  launch_button = st.button('Launch detailed analysis window', on_click = make_detailed_view)
                       

@st.dialog('Detailed View')
def make_detailed_view():
    st.html("<span class='big-dialog'></span>")
    st.session_state.run_h_score = True
    
    passed_info = st.session_state.info_for_detailed_view
    player_assignments = passed_info['player_assignments']
    draft_seat = passed_info['draft_seat']
    score_df = passed_info['score_df']
    rate_display = passed_info['rate_display']
    _g_scores = passed_info['_g_scores']
    g_scores_unselected = passed_info['g_scores_unselected']
    future_diffs = passed_info['future_diffs']
    weights = passed_info['weights']
    position_shares = passed_info['position_shares']
    res = passed_info['res']
    _H = passed_info['_H']
    rosters = passed_info['rosters']

    my_players = player_assignments[draft_seat]

    c1, c2 = st.columns([0.5,0.5])

    with c1:

      player_name = st.selectbox('Candidate player'
                                          ,score_df.index
                                          ,index = 0)
      
      player_last_name = player_name.split(' ')[1]
        
    n_per_position, roster_inverted_styled = get_roster_assignment_view(player_name = player_name
                                                                        ,player_last_name = player_last_name
                                                                        ,my_players = my_players
                                                                        ,rosters = rosters)
    
    main_df_styled = make_main_df_styled(_g_scores
                        , player_name
                        , player_last_name
                        , my_players
                        , res
                        , _H
                        , future_diffs)

    rate_df_limited_styled = make_rate_display_styled(rate_display
                                , player_name
                                , player_last_name)

    weights_styled = pd.DataFrame(weights.loc[player_name]).T.style.format("{:.0%}").background_gradient(axis = None)
    
    positions_styled = get_positions_styled(n_per_position
                             , position_shares
                             , player_name)

    g_scores_to_display_styled, h_scores_to_display_styled, player_location_g, player_location_h = \
      get_ranking_views(g_scores_unselected
                      , player_name
                      , score_df)

    with c1:

      st.markdown('Category weights for future picks')
      st.dataframe(weights_styled, hide_index = True)

      st.markdown('Flex position allocations for future flex spot picks')
      st.write(positions_styled)

      st.markdown('Roster assignments for chosen players')
      st.write(roster_inverted_styled, hide_index = True)

    with c2:

      st.dataframe(rate_df_limited_styled, hide_index = True)

      st.markdown('G-score expectations')
      st.dataframe(main_df_styled)

      c1_1, c1_2 = st.columns([0.5,0.5])
      
      with c1_1: 
        st.markdown('Rank **' + str(player_location_g + 1) + '** in Total G-score among available players')
        st.dataframe(g_scores_to_display_styled, height = 248)

      with c1_2: 
        st.markdown('Rank **' + str(player_location_h + 1) + '** in H-score among available players')
        st.dataframe(h_scores_to_display_styled, height = 248)

  
def get_positions_styled(n_per_position : dict
                          , position_shares : pd.DataFrame
                          , player_name : str):

  position_share_df = pd.DataFrame({p + '-' + str(n_per_position[p]): 
                                    position_shares[p].loc[player_name] * n_per_position[p] 
                                    for p in get_position_structure()['flex_list']}
                                    ).T.fillna(0)
  
  position_share_df = position_share_df[get_position_structure()['base_list']]
  position_share_df.loc['Total',:] = position_share_df.sum()

  return position_share_df.style.format("{:.2f}").background_gradient(axis = None)
  
def get_roster_assignment_view(player_name : str
                                ,player_last_name : str
                                ,my_players : list[str]
                                ,rosters : pd.DataFrame):
  #helper function for make_detailed_view. Creates s one-row dataframe with columns for position slot and 
  #players already on the team + candidate player in the positions given to them by the algorithm

  def get_player(row
               , i
               , player_list):
    #helper function to get the player indicated by the raw roster number

    if row[0] == -1:
        return None
    else:
      n = list(row).index(i)
      return player_list[n]
    
  def style_rosters(x):
    if len(x) ==0:
      return 'background-color:white'
    elif x in my_players:
      return 'background-color: lightgrey; color:black;'
    else:
      return 'background-color: lightblue; color:black;'

  my_players = [x.split(' ')[1] for x in my_players if x == x]

  player_list = my_players + [player_last_name] + [''] * (rosters.shape[1] - len(my_players) - 1)
      
  roster_row = rosters.loc[player_name]
  roster_inverted = [get_player(roster_row, i, player_list) for i in range(len(roster_row))] 

  roster_inverted = pd.DataFrame({'Players' : roster_inverted} 
                                  ,index = get_position_numbers_unwound()
                          )
  
  roster_unfilled = roster_inverted[roster_inverted['Players'] == '']
  position_slots = pd.Series(roster_unfilled.index)
  position_slots = position_slots.str.replace('\d+', '', regex = True)

  n_per_position = position_slots.value_counts()

  roster_inverted_styled = roster_inverted.T.style.map(style_rosters)

  return n_per_position, roster_inverted_styled

def make_rate_display_styled(rate_display : pd.DataFrame
                              , player_name : str
                              , player_last_name : str):
  rate_df_limited = pd.DataFrame({player_last_name : rate_display.loc[player_name]}).T

  #ZR: If auction, this should have "Your $ value" and "$ value" with no "H-score". Also it should be colored 
  if  '$ Value' in rate_display.columns:
      rate_df_limited_styled = rate_df_limited.style.format("{:.1f}"
                                              , subset = ['Your $ Value', '$ Value','Difference']) \
                                          .map(styler_a
                                              , subset = ['Your $ Value', '$ Value']) \
                                          .map(stat_styler, middle = 0.5, multiplier = 300, subset = get_selected_categories()) \
                                          .format('{:,.1%}', subset = get_selected_categories())
  else: 
    rate_df_limited_styled = h_percentage_styler(rate_df_limited)
  return rate_df_limited_styled

def make_main_df_styled(_g_scores
                        , player_name
                        , player_last_name
                        , my_players
                        , res
                        , _H
                        , future_diffs):

  team_so_far = _g_scores[_g_scores.index.isin(my_players)].sum()

  player = _g_scores.loc[player_name]

  if len([p for p in my_players if p == p]) > 0:
    other_teams_imputed = player + team_so_far - res['Diff'].loc[player_name] * _H.original_v #convert to G-score
  else:
    other_teams_imputed = player - res['Diff'].loc[player_name] * _H.original_v #convert to G-score

  remaining_value_imputed = future_diffs.loc[player_name] * _H.original_v - other_teams_imputed
  remaining_value_imputed['Total'] = remaining_value_imputed.sum()

  if len([player for player in my_players if player == player]) > 0:
    main_res_dict = {'Team so far' : team_so_far
                ,player_last_name : player
                ,'Future picks' : remaining_value_imputed}
  else:
    main_res_dict = {player_last_name : player
                ,'Future picks' : remaining_value_imputed}
  
  main_df = pd.DataFrame(main_res_dict).T
  main_df.loc['Total', : ] = main_df.sum()
  
  main_df_styled = static_score_styler(main_df, st.session_state.params['g-score-total-multiplier'])

  return main_df_styled

def get_ranking_views(g_scores_unselected
                      , player_name
                      ,score_df):
    def color_blue(label):
        return "background-color: lightblue; color:black" if label == player_name else None

    g_scores_unselected.loc[:,'Rank'] = range(1, len(g_scores_unselected) + 1)
    player_location_g = g_scores_unselected.index.get_loc(player_name)
    g_scores_to_display = pd.DataFrame({'Rank' : g_scores_unselected['Rank']
                                        ,'Player' : g_scores_unselected.index
                                        ,'Total' : g_scores_unselected['Total']
                                        }).set_index('Rank')
    g_scores_to_display_styled = g_scores_to_display.style.map(stat_styler
                                                                , middle = 0.5
                                                                , multiplier = 10
                                                                , subset = ['Total']
                                                                , mode = 'yellow') \
                                                                .format("{:.2f}", subset = ['Total']) \
                                                                .map(color_blue, subset = ['Player'])
    
    player_location_h = score_df.index.get_loc(player_name)
    h_scores_to_display = pd.DataFrame({'Rank' : score_df['Rank']
                                        ,'Player' : score_df.index
                                        ,'H-score' : score_df['H-score']
                                        }).set_index('Rank')
    h_scores_to_display_styled = h_scores_to_display.style.map(stat_styler
                                                                , middle = 0.5
                                                                , multiplier = 1000
                                                                , subset = ['H-score']
                                                                , mode = 'yellow') \
                                                                .format("{:.1%}", subset = ['H-score']) \
                                                                .map(color_blue, subset = ['Player'])
    
    return g_scores_to_display_styled, h_scores_to_display_styled, player_location_g, player_location_h
