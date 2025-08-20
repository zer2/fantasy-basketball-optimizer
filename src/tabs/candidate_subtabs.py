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

    #this represents a run that needs to optimize weights, and so should display weights
    dynamic_run = weights.iloc[0,0] == weights.iloc[0,0] 

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

            comparison_df = pd.DataFrame({'Your $ Value' : rate_display['$ Value']
                                          , '$ Value' : generic_player_value.loc[rate_display.index]})
            
            comparison_df.loc[:,'Difference'] = comparison_df['Your $ Value'] - comparison_df['$ Value']

            comparison_df = comparison_df.sort_values('Your $ Value', ascending = False)
            score_df = score_df.loc[comparison_df.index]

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
                          ,g_scores_unselected = g_scores_unselected)
                
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
                
                hashable_player_assignments =  make_hashable(player_assignments)

                if (draft_seat, hashable_player_assignments, n_iterations) not in res:

                  st.session_state.res_cache[(draft_seat, hashable_player_assignments, n_iterations)] = {'res' : res
                                                                                                        ,'iteration' : i}

                  g_scores_unselected = _g_scores[_g_scores.index.isin(score_df.index)].sort_values('Total', ascending = False)

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

  st.markdown(
        """
    <style>
    div[data-testid="stDialog"] div[role="dialog"]:has(.big-dialog) {
        width: 80vw;
        height: 80vh;
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
    win_rates = passed_info['win_rates']
    _g_scores = passed_info['_g_scores']
    g_scores_unselected = passed_info['g_scores_unselected']
    future_diffs = passed_info['future_diffs']
    weights = passed_info['weights']
    position_shares = passed_info['position_shares']
    res = passed_info['res']
    _H = passed_info['_H']
    rosters = passed_info['rosters']

    c1, c2 = st.columns([0.5,0.5])

    with c1:

      player_name = st.selectbox('Detailed View Player'
                                          ,score_df.index
                                          ,index = 0)

    player_last_name = player_name.split(' ')[1]

    my_players = [x.split(' ')[1] for x in player_assignments[draft_seat] if x == x]

    player_list = my_players + [player_last_name] + [''] * (rosters.shape[1] - len(my_players) - 1)

    def get_player(row, i):

      if row[0] == -1:
          return None
      else:
        n = list(row).index(i)
        return player_list[n]
      
    rosters = rosters.loc[score_df.index]
        
    roster_row = rosters.loc[player_name]
    roster_inverted = [get_player(roster_row, i) for i in range(len(roster_row))] 

    roster_inverted = pd.DataFrame({'Players' : roster_inverted} 
                                    ,index = get_position_numbers_unwound()
                            )
    
    roster_unfilled = roster_inverted[roster_inverted['Players'] == '']
    position_slots = pd.Series(roster_unfilled.index)
    position_slots = position_slots.str.replace('\d+', '', regex = True)

    n_per_position = position_slots.value_counts()

    def style_rosters(x):
        if len(x) ==0:
          return 'background-color:white'
        elif x in my_players:
          return 'background-color: lightgrey; color:black;'
        else:
          return 'background-color: lightblue; color:black;'
        
    roster_inverted_styled = roster_inverted.T.style.map(style_rosters)

    rate_display = score_df.merge(win_rates.dropna(), left_index = True, right_index = True)

    rate_df_limited = pd.DataFrame({player_last_name : rate_display.loc[player_name]}).T
    rate_df_limited_styled = h_percentage_styler(rate_df_limited)

    my_players = player_assignments[draft_seat]
    team_so_far = _g_scores[_g_scores.index.isin(my_players)].sum()

    player = _g_scores.loc[player_name]

    if len([player for player in my_players if player == player]) > 0:
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

    weights_styled = pd.DataFrame(weights.loc[player_name]).T.style.format("{:.0%}").background_gradient(axis = None)

    position_share_df = pd.DataFrame({p + '-' + str(n_per_position[p]): 
                                      position_shares[p].loc[player_name] * n_per_position[p] 
                                      for p in get_position_structure()['flex_list']}
                                      ).T.fillna(0)
    
    position_share_df = position_share_df[get_position_structure()['base_list']]
    position_share_df.loc['Total',:] = position_share_df.sum()

    positions_styled = position_share_df.style.format("{:.2f}").background_gradient(axis = None)

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
    score_df.loc[:,'Rank'] = range(1, len(score_df) + 1)
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


    with c1:

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

    with c2:

      st.markdown('Category weights for future picks')
      st.dataframe(weights_styled, hide_index = True)

      st.markdown('Flex position allocations for future flex spot picks')
      st.write(positions_styled)

      st.markdown('Roster assignments for chosen players')
      st.write(roster_inverted_styled, hide_index = True)


@st.fragment()
def make_cand_matchup_tab(cdfs
                          , players
                          , teams
                          , draft_seat
                          , drop_player
                          , i):

    opponents = [team for team in teams if team != draft_seat]
    tabs = st.tabs(opponents)

    for opponent_index, tab in zip(range(len(opponents)), tabs):
      with tab:

        c1, c2 = st.columns([0.2, 0.8])

        with c1: 
          st.caption('''If scoring type is Most Categories, overall score is probability of winning a matchup. Otherwise, 
                     it is the average percent of expected points won''')
          st.caption('''Also note that these scores assume that you will dynamically adapt with remaining picks
                      based on your algorithm parameter preferences, while your opponent will not. If you want to 
                     see results as they stand now without any dynamic adaptations, set punting level to "no dynamic adaptation"''')
        with c2: 

          cdfs_selected = cdfs[opponent_index].loc[players]

          if st.session_state.scoring_format == 'Head to Head: Most Categories':

            #We've already calculated this but it is not retained by the algorithm agent
            cdfs_expanded = np.expand_dims(cdfs_selected, axis = 2)
            cdfs_selected.loc[:,'Overall'] = combinatorial_calculation(cdfs_expanded
                                                                       , 1 - cdfs_expanded)

          else:

            cdfs_selected.loc[:,'Overall'] = cdfs_selected.mean(axis = 1)


          cdfs_selected = cdfs_selected[['Overall'] + get_selected_categories()]

          if drop_player is not None:

              cdfs_styled = h_percentage_styler(cdfs_selected.reset_index(), drop_player = drop_player)
              st.dataframe(cdfs_styled, hide_index = True)

          else:
              cdfs_styled = h_percentage_styler(cdfs_selected)

              st.dataframe(cdfs_styled)
