import streamlit as st
import pandas as pd 
import numpy as np
from src.helpers.helper_functions import get_position_numbers_unwound, static_score_styler, h_percentage_styler, get_selected_categories, \
                                styler_a, stat_styler
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
  _H = _H.clear_initial_weights()
          
  if drop_player is None:
    generator = _H.get_h_scores(player_assignments, draft_seat, cash_remaining_per_team)
  else:
    from copy import deepcopy 
    player_assignments = deepcopy(player_assignments)
    player_assignments[draft_seat] = [player for player in player_assignments[draft_seat] if player != drop_player]
    generator = _H.get_h_scores(player_assignments, draft_seat, cash_remaining_per_team)

  placeholder = st.empty()

  adps = get_htb_adp()

  #if n_iterations is 0 we run just once
  for i in range(max(1,n_iterations)):

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

    win_rates.columns = get_selected_categories()

    #this represents a run that needs to optimize weights, and so should display weights
    dynamic_run = weights.iloc[0,0] == weights.iloc[0,0] 

    #normalize weights by what we expect from other drafters
    weights = pd.DataFrame(weights
                  , index = score.index
                  , columns = get_selected_categories())/v
    
    
    with placeholder.container():

      position_shares_list = [(k, v) for k, v in position_shares.items()]
      position_shares_tab_names = ['Flex- ' + x[0] for x in position_shares_list]
      position_shares_res = [x[1][list(fits_roster.values)] for x in position_shares_list]



      if dynamic_run:

        if cash_remaining_per_team:
          all_tabs = st.tabs(['Targets','Weights','Rosters'] + position_shares_tab_names + ['Z-scores','G-Scores','Matchups'])
          target_tab = all_tabs[0]
          weight_tab = all_tabs[1]
          roster_tab = all_tabs[2]


        else:

          if st.session_state.scoring_format == 'Rotisserie':
            first_tab = 'Expected Totals'
          else:
            first_tab = 'Expected Win Rates'

          all_tabs = st.tabs([first_tab, 'Weights','Rosters'] + position_shares_tab_names + ['Z-scores','G-Scores','Matchups'])
          rate_tab = all_tabs[0]
          weight_tab = all_tabs[1]       
          roster_tab = all_tabs[2]

        position_tabs = all_tabs[3:-3]
        raw_z_tab = all_tabs[-3]
        raw_g_tab = all_tabs[-2]
        matchup_tab = all_tabs[-1]

      else:
          all_tabs = st.tabs(['Expected Win Rates','Z-scores','G-Scores','Matchups'])
          rate_tab = all_tabs[0]
          raw_z_tab = all_tabs[1]       
          raw_g_tab = all_tabs[2]
          matchup_tab = all_tabs[3]

      score.name = 'H-score'
      score_df = pd.DataFrame(score)

      display = ((i+1) % display_period == 0) or (i == n_iterations - 1) or (n_iterations <= 1)

      if cash_remaining_per_team:
         
        with target_tab:

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
        with rate_tab:

          if display:

            score_df = score_df.sort_values('H-score',ascending = False)

            if sum(fits_roster) == 0:
              st.error('Illegal roster!')
              st.stop()

            if st.session_state.scoring_format == 'Rotisserie':
              diffs = res['Diff'][fits_roster]
              rate_df = diffs.loc[score_df.index].dropna()
              style_format = "{:.1f}"
              format_middle = 0
              format_multiplier = 20 

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
              
              rate_display_styled = rate_display.reset_index().style.format("{:.1f}"
                              ,subset = pd.IndexSlice[:,['H-score']]) \
                                                    .format("{:.1f}"
                              ,subset = pd.IndexSlice[:,adp_col]) \
                      .map(styler_a
                            , subset = pd.IndexSlice[:,['H-score'] + adp_col]) \
                      .map(stat_styler, middle = format_middle, multiplier = format_multiplier, subset = rate_df.columns) \
                      .format(style_format, subset = rate_df.columns) \
                      .map(color_blue, subset = pd.IndexSlice[:,['Player']])

              st.dataframe(rate_display_styled
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
              
              st.dataframe(rate_display_styled, use_container_width = True)

      
      if display and dynamic_run:

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

      if display and dynamic_run:
                    
          with roster_tab:

            my_players = [x.split(' ')[1] for x in player_assignments[draft_seat] if x == x]

            player_list = my_players + [None] + [''] * (rosters.shape[1] - len(my_players) - 1)

            def get_player(row, i):

              if row[0] == -1:
                 return None
              else:
                n = list(row).index(i)
                return player_list[n]
          
            filler = {x : x.split(' ')[1] for x in rosters.index}

            rosters = rosters.loc[score_df.index]
            
            rosters_inverted = [[get_player(row, i) for i in range(len(row))] for k,row in rosters.iterrows()]
            rosters_inverted = pd.DataFrame(rosters_inverted 
                                                    ,index = rosters.index
                                                    ,columns = get_position_numbers_unwound()
                                            )
                                    
            for col in rosters_inverted:
               rosters_inverted[col] = rosters_inverted[col].fillna(filler)


            def style_rosters(x):
               if len(x) ==0:
                  return 'background-color:white'
               elif x in my_players:
                  return 'background-color: lightgrey; color:black;'
               else:
                  return 'background-color: lightblue; color:black;'

            rosters_styled = rosters_inverted.style.map(style_rosters)

            st.dataframe(rosters_styled
                         , column_config = {col : st.column_config.TextColumn(width = 'small') for col in rosters_styled.columns}
                           )

      if display and dynamic_run:

        for tab, position_shares_df in zip(position_tabs, position_shares_res):
            with tab: 
                
            
                  share_display = score_df.merge(position_shares_df.loc[score_df.index].dropna()
                                        , left_index = True
                                        , right_index = True)
                  share_display_styled = share_display.style.format("{:.0%}"
                                                              , subset = position_shares_df.columns)\
                            .format("{:.1%}"
                                    ,subset = pd.IndexSlice[:,['H-score']]) \
                            .map(styler_a
                                  , subset = pd.IndexSlice[:,['H-score']]) \
                            .background_gradient(axis = None,subset = position_shares_df.columns) 
                  st.dataframe(share_display_styled, use_container_width = True)

      with raw_g_tab:

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

      with raw_z_tab:

        if display:

          z_df = _z_scores.loc[score.index]
          z_display = score_df.merge(z_df, left_index = True, right_index = True)
          
          if drop_player is not None:
            scores_unselected_styled = static_score_styler(z_display.reset_index()
                                                        , st.session_state.params['z-score-player-multiplier']
                                                        , st.session_state.params['z-score-total-multiplier'])
            
            scores_unselected_styled = scores_unselected_styled.map(color_blue, subset = pd.IndexSlice[:,['Player']])
            st.dataframe(scores_unselected_styled, use_container_width = True, hide_index = True)

          else:
            scores_unselected_styled = static_score_styler(z_display
                                              , st.session_state.params['z-score-player-multiplier']
                                              , st.session_state.params['z-score-total-multiplier'])

            st.dataframe(scores_unselected_styled, use_container_width = True)

      with matchup_tab:

        if i >= n_iterations-1: 
          make_cand_matchup_tab(res['CDFs']
                                , score_df.index
                                , list(player_assignments.keys())
                                , draft_seat
                                , drop_player
                                , i)

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
