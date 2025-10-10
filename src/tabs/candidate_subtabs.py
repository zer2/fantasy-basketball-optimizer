import streamlit as st
import pandas as pd 
import numpy as np
from src.helpers.helper_functions import get_position_numbers_unwound, static_score_styler, h_percentage_styler, get_selected_categories, \
                                styler_a, stat_styler, get_position_structure, style_rosters
from src.math.algorithm_helpers import auction_value_adjuster
from src.helpers.helper_functions import get_n_drafters

def make_hashable(obj):
    """
    Recursively convert obj into a hashable, canonical form.
      - dict  -> tuple(sorted (key_h, value_h) pairs by key)
      - list  -> frozenset of items (order-insensitive, multiplicity ignored)
      - set   -> frozenset of items
      - tuple -> tuple of items (order preserved)
      - other -> returned as-is (must already be hashable)
    """
    if isinstance(obj, dict):
        # Sort by key for a canonical order. If keys are not mutually comparable,
        # fall back to sorting by repr(key).
        try:
            items = sorted(obj.items(), key=lambda kv: kv[0])
        except TypeError:
            items = sorted(obj.items(), key=lambda kv: repr(kv[0]))
        return tuple((make_hashable(k), make_hashable(v)) for k, v in items)

    if isinstance(obj, list):
        return frozenset(make_hashable(v) for v in obj)

    if isinstance(obj, set):
        return frozenset(make_hashable(v) for v in obj)

    if isinstance(obj, tuple):
        return tuple(make_hashable(v) for v in obj)

    # Base case: strings, numbers, bools, None, etc.
    # Base case: strings, numbers, bools, None, etc.
    if obj == obj:
       return obj
    else:
       return ''
        
def make_cand_tab(_H
                    ,_g_scores : pd.DataFrame
                    ,player_assignments : dict[list[str]]
                    ,draft_seat : str
                    ,n_iterations : int
                    ,display_period : int = 5
                    ,cash_remaining_per_team : dict[int] = None
                    ,generic_player_value : pd.Series = None
                    ,original_player_value : pd.Series = None
                    ,total_players : int = None
                    ,drop_player : str = None):
  

  """Make a tab showing H-scores for the current draft situation

  Args:
      H:
      player_assignments: dict of who has drafted what player
      draft_seat: seat from which to calculate H-score
      n_iterations:number of iterations to run H-scoring for
      display_period: periodicity of showing current results to the user. Higher period is quicker to execute fully, but the display won't be up-to-date the whole time
      cash_remaining_per_team: dictionary of team -> amount of cash they have remaining
      generic_player_value: Series of values calculated based on SAVOR and remaining cash/players 
      original_player_value: Series of values calculated for all players assuming full cash before any players have been taken
      total_players: number of players to be drafted
      drop_player: a player which is considered for being dropped, for a waiver selection

  Returns:
      None
  """

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
  
  if (cash_remaining_per_team is not None) and (st.session_state.data_source == 'Enter your own data'):
    cand_table_height = 505 #more room is needed for the auction string that goes at the bottom
  else: 
    cand_table_height = 535

  placeholder = st.empty()

  if cash_remaining_per_team:
    selection_list =  [p for t in player_assignments.values() for p in t if p ==p]
    remaining_cash = sum(cash for team, cash in cash_remaining_per_team.items())

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
                  , columns = get_selected_categories())/_H.v.reshape(1,len(_H.v))
    
    with placeholder.container():

      score.name = 'H-score'
      score_df = pd.DataFrame(score)

      display =  (i % display_period == 0) or (i == n_iterations - 1) or (n_iterations <= 1)

      if st.session_state.mode == 'Season Mode':
        h_tab, g_tab,  = st.tabs(['H-score','G-score'])
      else:
        h_tab, pbp_tab, g_tab,  = st.tabs(['H-score','H-score details','G-score'])

      with h_tab:

        if cash_remaining_per_team:
          
          if display:

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

            #rate_df = win_rates.dropna()
            rate_display = score_df.merge(rate_df, left_index = True, right_index = True)

            players_chosen = [x for v in player_assignments.values() for x in v if x == x]
            total_cash_remaining = np.sum([v for k, v in cash_remaining_per_team.items()])

            rate_display.loc[:,'Your $'] = auction_value_adjuster(score_df['H-score']
                                                            , total_players - len(players_chosen)
                                                            , total_cash_remaining
                                                            , st.session_state['streaming_noise'])
            
            rate_display = pd.DataFrame({'Your $' : rate_display['Your $']
                                          , 'Gnrc. $' : generic_player_value.loc[rate_display.index]
                                          , 'Orig. $' : original_player_value.loc[rate_display.index]}
                                        )
            
            rate_display.loc[:,'Difference'] = (rate_display['Your $'] - rate_display['Gnrc. $']).round(1)

            rate_display = rate_display.sort_values('Your $', ascending = False)
            score_df = score_df.loc[rate_display.index]

            rate_display = rate_display.join(rate_df)

            rate_display = rate_display[['Difference','Your $','Gnrc. $','Orig. $'] + list(rate_df.columns)]

            #ZR: Something about this is inefficient with the streamlit implementation. It slows things down a lot
            rate_display_styled = rate_display.reset_index().style.format("{:.1f}"
                                                              , subset = ['Your $', 'Gnrc. $','Difference','Orig. $']) \
                      .map(styler_a
                          , subset = ['Your $', 'Gnrc. $','Orig. $']) \
                      .map(stat_styler, middle = format_middle, multiplier = 6, subset = ['Difference'], mode = 'secondary') \
                      .map(stat_styler, middle = format_middle, multiplier = format_multiplier, subset = rate_df.columns) \
                      .format(style_format, subset = rate_df.columns)._compute()
            
            st.dataframe(rate_display_styled
                         , hide_index = True
                         , height = cand_table_height
                         , use_container_width = True)
            
            make_auction_string(original_player_value 
                    , score_df.index 
                    , rate_display 
                    , remaining_cash)

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
                        .map(color_blue, subset = pd.IndexSlice[:,['Player']])._compute()
                

                g_display = _g_scores[_g_scores.index.isin(score_df.index)].sort_values('Total', ascending = False)
                
                rate_display = score_df.merge(win_rates.dropna()
                              , left_index = True
                              , right_index = True)
                
                st.dataframe(rate_display_styled
                            , key = 'rate_display_' + str(i)
                            , selection_mode = 'single-row'
                            , use_container_width = True
                            , hide_index = True
                            , height = cand_table_height)

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
                                , use_container_width = True
                                , height = cand_table_height)            
                
              if st.session_state.scoring_format == 'Rotisserie':

                  st.caption('''Expected totals are based on standard fantasy point scoring. 
                            One point for last, two points for second last, etc. The baseline 
                            expected total for a category is ''' + str(format_middle))
      with g_tab:

        if display:

          g_display = _g_scores.loc[score.index]

          if cash_remaining_per_team is not None:

              g_display = g_display.sort_values('Total', ascending = False)

              g_display.loc[:,'Gnrc. $'] = auction_value_adjuster(g_display['Total']
                                                          , total_players - len(selection_list)
                                                          , remaining_cash
                                                          , st.session_state['streaming_noise']) 
              
              #ZR: maybe we don't need to run this every time
              g_score_savor = auction_value_adjuster(_g_scores['Total']
                                                          , total_players 
                                                          , get_n_drafters() * st.session_state.cash_per_team
                                                          , st.session_state['streaming_noise']) 
              
              g_display.loc[:,'Orig. $'] =  g_score_savor.loc[g_display.index]


          if drop_player is not None:

            scores_unselected_styled = static_score_styler(g_display.reset_index()
                                              , st.session_state.params['g-score-player-multiplier']
                                              , st.session_state.params['g-score-total-multiplier'])
            
            scores_unselected_styled = scores_unselected_styled.map(color_blue
                                                                    , subset = pd.IndexSlice[:,['Player']])
            st.dataframe(scores_unselected_styled
                         , use_container_width = True
                         , hide_index = True
                         , height = cand_table_height)
    
          else:
            scores_unselected_styled = static_score_styler(g_display
                                              , st.session_state.params['g-score-player-multiplier']
                                              , st.session_state.params['g-score-total-multiplier'])

            st.dataframe(scores_unselected_styled
                         , use_container_width = True
                         , height = cand_table_height)  
            
          if cash_remaining_per_team is not None:
              make_auction_string(original_player_value 
                    , score_df.index 
                    , rate_display 
                    , remaining_cash)

      if display and not st.session_state.mode == 'Season Mode':
          
        with pbp_tab:

          score_df.loc[:,'Rank'] = range(1, len(score_df) + 1)

          make_detailed_view(player_assignments
                       ,draft_seat
                       ,score_df
                       ,rate_display
                       ,_g_scores
                       ,g_display
                       ,future_diffs
                       ,weights
                       ,position_shares
                       ,res
                       ,_H
                       ,rosters
                       ,cash_remaining_per_team
                       ,original_player_value
                       ,i) 

            
@st.fragment
def make_detailed_view(player_assignments : dict[list[str]]
                       ,draft_seat : str
                       ,score_df : pd.DataFrame
                       ,rate_display : pd.DataFrame
                       ,_g_scores : pd.DataFrame
                       ,g_display : pd.DataFrame
                       ,future_diffs : pd.DataFrame
                       ,weights : pd.DataFrame
                       ,position_shares : dict[pd.DataFrame]
                       ,res : dict
                       ,_H
                       ,rosters : pd.DataFrame
                       ,cash_remaining_per_team: dict[int]
                       ,original_player_value : pd.Series
                       ,iteration : int):
    """   
    Load up information stored from the main candidate view, and create a view that shows details for individual players
    I think this is useful for research/understanding what the algorithm is thinking

    Args:
        Various arguments from the make_cand_tab function. Will change around depending on what is included here

    Returns:
        None
    """

    my_players = player_assignments[draft_seat]

    if st.session_state.data_source == 'Enter your own data':

      player_name = st.selectbox('Candidate player'
                                          ,score_df.index
                                          ,index = 0
                                          ,label_visibility = 'collapsed'
                                          ,key = 'candidate_player_detailed' + str(iteration))
      
      player_last_name = player_name.split(' ')[1]

      c2, c1 = st.tabs(['Expectations','Future Strategy'])

      display_rank_tables = False

    else:
      c1, c2 = st.columns([0.5,0.5])

      display_rank_tables = True

      with c1:

        player_name = st.selectbox('Candidate player'
                                            ,score_df.index
                                            ,index = 0
                                            ,label_visibility = 'collapsed'
                                            ,key = 'candidate_player' + str(iteration))
        
        player_last_name = player_name.split(' ')[1]

    if (len([x for x in my_players if x == x]) < st.session_state.n_picks - 1) and (rosters.shape[1] > 0):
      n_per_position, roster_inverted_styled = get_roster_assignment_view(player_name = player_name
                                                                          ,player_last_name = player_last_name
                                                                          ,my_players = my_players
                                                                          ,rosters = rosters)
      
      positions_styled = get_positions_styled(n_per_position
                              , position_shares
                              , player_name)
      
    
      main_df_styled = make_main_df_styled(_g_scores
                          , player_name
                          , player_last_name
                          , my_players
                          , res
                          , _H
                          , future_diffs)

    weights_styled = pd.DataFrame(weights.loc[player_name]).T.style.format("{:.0%}") \
                        .map(stat_styler
                             , middle = 0.9
                             , multiplier = 500
                             , mode = 'tertiary')

    g_scores_to_display_styled, h_scores_to_display_styled, player_location_g, player_location_h = \
      get_ranking_views(g_display
                      , player_name
                      , score_df)
    
    with c1:

      if (len([x for x in my_players if x == x]) < st.session_state.n_picks - 1):

        st.markdown('Category weights for future picks')
        st.dataframe(weights_styled, hide_index = True)

        if (rosters.shape[1] > 0):

          #positions_styled becomes None when there are no more flex allocations
          if positions_styled is not None:
            st.markdown('Flex position allocations for future flex spot picks')
            st.write(positions_styled)
            
          st.markdown('Roster assignments for chosen players')
          st.write(roster_inverted_styled, hide_index = True)



        else: 
          st.write('Roster assignments are not calculated when position is not available')

      else:
        st.write('Category weights and position allocations not calculated for last player')

      if cash_remaining_per_team and display_rank_tables:

        remaining_cash = sum(cash for team, cash in cash_remaining_per_team.items())
        
        make_auction_string(original_player_value
                        , score_df.index
                        , rate_display
                        , remaining_cash
                        , player_name
                        , player_last_name)

    with c2:

      rate_df_limited_styled = make_rate_display_styled(rate_display
                            , player_name
                            , player_last_name)
      st.dataframe(rate_df_limited_styled, hide_index = True, height = 73)

      if (len([x for x in my_players if x == x]) < st.session_state.n_picks - 1):

        st.markdown('G-score expectations (difference vs. other teams)')
        st.dataframe(main_df_styled)

      if cash_remaining_per_team:

        if display_rank_tables:
          make_auction_value_df(rate_display, g_display, player_name)
        else:
          make_auction_value_df(rate_display.loc[[player_name]], g_display.loc[[player_name]], player_name)

          remaining_cash = sum(cash for team, cash in cash_remaining_per_team.items())

          make_auction_string(original_player_value
                  , score_df.index
                  , rate_display
                  , remaining_cash
                  , player_name
                  , player_last_name)

      else:

        c1_1, c1_2 = st.columns([0.5,0.5])
        
        with c1_1: 
          st.markdown('Rank **' + str(player_location_h + 1) + '** in H-score among available players')
          if display_rank_tables:
            st.dataframe(h_scores_to_display_styled, height = 158)

        with c1_2: 
          st.markdown('Rank **' + str(player_location_g + 1) + '** in Total G-score among available players')
          if display_rank_tables:
            st.dataframe(g_scores_to_display_styled, height = 158)

def make_auction_string(original_player_value : pd.Series
                        , remaining_player_list : list
                        , rate_display : pd.DataFrame
                        , remaining_cash : int
                        , player_name : str = None
                        , player_last_name : str = None):
  """Create a string describing the current auction situation and if there is a candidate player, how much their value is inflated by the situation

  Args:
      H:
      original_player_value: Series of values calculated for all players assuming full cash before any players have been taken
      remaining_player_list : List of players not yet drafted
      rate_display: Dataframe with added info on player values for you 
      remaining_cash: total cash remaining for all drafters
      player_name: full name of candidate player
      player_last_name: just their last name

  Returns:
      String describing the auction situation
  """

  original_value_of_unchosen_players = original_player_value.loc[remaining_player_list].sum()
  inflation_factor_unchosen_players = original_value_of_unchosen_players/remaining_cash

  inflation_factor_formatted = str(np.round(inflation_factor_unchosen_players * 100,1)) + '%'

  if player_name is not None:
    player_value_inflated = str(int(np.round(rate_display['Your $'].loc[player_name] * inflation_factor_unchosen_players)))

  if inflation_factor_unchosen_players > 1:
    text_1 = 'Based on overspending so far, there is less money available than there is' + \
            ' original value remaining. The remaining players have original H-score \$ value of ' + \
            inflation_factor_formatted + ' of their generic cash-adjusted values'
            
    if player_name is not None:
      text_2 = '. Multiplying your H-score-based \$ value for ' + player_last_name + ' by that factor suggests that it would be' +\
               ' reasonable to pay up to $' + player_value_inflated + ' for them' 
    else: 
      text_2 = ''

  else:
    text_1 = 'Based on underspending so far, there is more money available than there is' + \
            ' original value remaining. The remaining players have original H-score \$ value of ' + \
            inflation_factor_formatted + ' of their generic cash-adjusted values'
    
    if player_name is not None:
      text_2 = '. Multiplying your H-score-based \$ value for ' + player_last_name + ' by that factor suggests that it would be reasonable to expect' +\
               'as little as \$' + player_value_inflated + ' for them' 
    else: 
      text_2 = ''

  st.caption(text_1 + text_2)                                                  
            
def make_auction_value_df(rate_display : pd.DataFrame
                          , g_display : pd.DataFrame
                          , player_name : str):
  #make a dataframe with $ according to your H-score, plus generically from H-score and G-score
  #ZR: Arguably this should have something about overpayments 

  big_value_df = pd.DataFrame({'Your H$' : rate_display['Your $']
                                ,'Gnrc. H$' : rate_display['Gnrc. $']
                                , 'Orig. H$' : rate_display['Orig. $']
                                ,'Gnrc. G$' : g_display['Gnrc. $']
                                ,'Orig. G$' : g_display['Orig. $']}).sort_values('Your H$', ascending = False)
  cols = ['Your H$','Gnrc. H$','Orig. H$','Gnrc. G$','Orig. G$']

  def color_blue(label):
    return "background-color: lightblue; color:black" if label == player_name else None
  
  big_value_df_styled = big_value_df.reset_index().style.format("{:.1f}", subset = cols) \
                                                                  .background_gradient(cmap = 'Oranges', subset = cols, axis = None) \
                                                                  .map(color_blue, subset = 'Player')
  
  #only set the size of the dataframe when it is not just a single player
  if len(big_value_df) > 1:
    height = 218
  else:
    height = None
  st.dataframe(big_value_df_styled, hide_index = True, height = height)

def get_positions_styled(n_per_position : dict
                          , position_shares : pd.DataFrame
                          , player_name : str):
  
  position_share_df = pd.DataFrame({p + '-' + str(n_per_position[p]): 
                                    position_shares[p].loc[player_name] * n_per_position[p] 
                                    for p in get_position_structure()['flex_list'] if p in n_per_position.keys()}
                                    ).T

  #this line is to make sure the columns are lined up in the normal way
  position_share_df = position_share_df[[p for p in get_position_structure()['base_list'] if p in position_share_df.columns]]

  if len(position_share_df) > 0:
    position_share_df.loc['Total',:] = position_share_df.sum()

    #-999 is a hack to encode 'missing info' for the stat_styler function. None doesn't work for some reason
    return position_share_df.fillna(-999).style.format("{:.2f}") \
                            .map(stat_styler, middle = 0, multiplier = 50, mode = 'tertiary') \

  
  else: 

    return None
  
def get_roster_assignment_view(player_name : str
                                ,player_last_name : str
                                ,my_players : list[str]
                                ,rosters : pd.DataFrame):
  """Creates a one-row dataframe with columns for position slot and players already on the team + candidate player in the positions given to them by the algorithm

  Args:
      player_name: full name of candidate player
      player_last_name: just their last name
      my_players : players already on the team
      rosters: output from the H-scoring model; dataframe of roster assignments 

  Returns:
      None
  """

  def get_player(row
               , i
               , player_list):
    #helper function to get the player indicated by the raw roster number

    if row[0] == -1:
        return None
    else:
      n = list(row).index(i)
      return player_list[n]
    
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

  roster_inverted_styled = roster_inverted.T.style.map(style_rosters, my_players = my_players)

  return n_per_position, roster_inverted_styled

def make_rate_display_styled(rate_display : pd.DataFrame
                              , player_name : str
                              , player_last_name : str):
  """Creates a one-row dataframe with category-level win rates

  Args:
      rate_display: Dataframe with all category-level expected win rates for all players
      player_name: full name of candidate player
      player_last_name: just their last name

  Returns:
      None
  """
  rate_df_limited = pd.DataFrame({player_last_name : rate_display.loc[player_name]}).T

  if  'Gnrc. $' in rate_display.columns:
      st.markdown('Expected win rates if taken at no cost')

      rate_df_limited = rate_df_limited.drop(columns = ['Difference','Your $', 'Gnrc. $', 'Orig. $'])
      
      rate_df_limited_styled = rate_df_limited.style \
                                          .map(stat_styler, middle = 0.5, multiplier = 300, subset = get_selected_categories()) \
                                          .format('{:,.1%}', subset = get_selected_categories())
  else: 
    st.markdown('Expected win rates if taken')
    rate_df_limited_styled = h_percentage_styler(rate_df_limited)
  return rate_df_limited_styled

def make_main_df_styled(_g_scores : pd.DataFrame
                        , player_name : str
                        , player_last_name : str
                        , my_players : list[str]
                        , res : dict
                        , _H
                        , future_diffs : pd.DataFrame):
  
  """Creates a dataframe for current and future expected G-score for the team

  Args:
      _g_scores: Table of category-level G-scores for all players
      player_name: full name of candidate player
      player_last_name: just their last name
      my_players : players already on the team
      res: all results from H-scoring
      H: H-scoring agent
      future_diffs: expected future G-scores, from res (ZR: might be redundant)

  Returns:
      None
  """

  player = _g_scores.loc[player_name].drop('Total')
  total_diff = res['Diff'].loc[player_name] * _H.original_v 
  future_diff = res['Future-Diff'].loc[player_name] * _H.original_v
  current_diff = total_diff - future_diff

  total_diff_plus_player = total_diff + player

  main_res_dict = {'Current diff' : current_diff
                ,player_last_name : player
                ,'Future player diff' : future_diff
                ,'Total diff' : total_diff_plus_player}
  
  main_df = pd.DataFrame(main_res_dict).T
  main_df.loc[:,'Total'] = main_df.sum(axis = 1)
  
  main_df_styled = static_score_styler(main_df, st.session_state.params['g-score-total-multiplier'])

  return main_df_styled

def get_ranking_views(g_display : pd.DataFrame
                      , player_name : str
                      ,score_df : pd.DataFrame):
    """Creates tables for the candidate player's ranking by both H-score and G-score

    Args:
        g_display: table of g-scores with extra info added for display
        player_last_name: just their last name
        score_df: Dataframe with H-scores per player 

    Returns:
        None
    """
    def color_blue(label):
        return "background-color: lightblue; color:black" if label == player_name else None

    g_display.loc[:,'Rank'] = range(1, len(g_display) + 1)
    player_location_g = g_display.index.get_loc(player_name)
    g_scores_to_display = pd.DataFrame({'Rank' : g_display['Rank']
                                        ,'Player' : g_display.index
                                        ,'Total' : g_display['Total']
                                        }).set_index('Rank')
    g_scores_to_display_styled = g_scores_to_display.style.map(stat_styler
                                                                , middle = 0.5
                                                                , multiplier = 10
                                                                , subset = ['Total']
                                                                , mode = 'secondary') \
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
                                                                , mode = 'secondary') \
                                                                .format("{:.1%}", subset = ['H-score']) \
                                                                .map(color_blue, subset = ['Player'])
    
    return g_scores_to_display_styled, h_scores_to_display_styled, player_location_g, player_location_h