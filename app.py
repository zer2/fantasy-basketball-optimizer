import streamlit as st
st.set_page_config(page_title='Fantasy BBall Optimization'
                   , page_icon=':basketball:'
                   , layout="wide"
                   , initial_sidebar_state="auto"
                   , menu_items=None)

import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import os 
import yaml

from src.helper_functions import listify, make_progress_chart, read_markdown_file, stat_styler, styler_a,styler_b, styler_c
from src.get_data import get_historical_data, get_current_season_data, get_darko_data, get_partial_data
from src.process_player_data import process_player_data
from src.run_algorithm import HAgent, analyze_trade

with open("parameters.yaml", "r") as stream:
    try:
       params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc) 

if 'run' not in st.session_state:
    st.session_state.run = False

def run():
    st.session_state.run = True

def stop_run():
    st.session_state.run = False
      
counting_statistics = params['counting-statistics'] 
percentage_statistics = params['percentage-statistics'] 
volume_statistics = params['percentage-statistics'] 

z_score_player_multiplier = params['z-score-player-multiplier']
z_score_team_multiplier = params['z-score-team-multiplier']
g_score_player_multiplier = params['g-score-player-multiplier']
g_score_team_multiplier = params['g-score-team-multiplier']

historical_df = get_historical_data(params)
current_data, expected_minutes = get_current_season_data(params)
darko_data = get_darko_data(expected_minutes, params)

### Make app

st.title('Optimization for Fantasy Basketball :basketball:')

coefficient_df = pd.read_csv('./coefficients.csv', index_col = 0)

about_tab, param_tab, stat_tab, draft_tab, rank_tab = st.tabs([":scroll: About",":control_knobs: Parameters", ":bar_chart: Player Stats", ":man-bouncing-ball: Draft", ":first_place_medal: Player Rankings"])

with about_tab:

  intro_tab, static_explanation_tab, dynamic_explanation_tab, data_tab,trading_tab = st.tabs(['Intro','G-scoring','H-scoring','Data Sources','Waivers & Trading'])

  with intro_tab:
      c2,c2,c3 = st.columns([0.1,0.8,0.1])
      with c2:
          intro_md = read_markdown_file('about/intro.md')
          st.markdown(intro_md, unsafe_allow_html=True)

  with static_explanation_tab:
      c2,c2,c3 = st.columns([0.1,0.8,0.1])
      with c2:
          static_md = read_markdown_file('about/static_explanation.md')
          st.markdown(static_md, unsafe_allow_html=True)
      
  with dynamic_explanation_tab:
      c2,c2,c3 = st.columns([0.1,0.8,0.1])
      with c2:
          dynamic_md = read_markdown_file('about/dynamic_explanation.md')
          st.markdown(dynamic_md, unsafe_allow_html=True)

  with data_tab:
      c2,c2,c3 = st.columns([0.1,0.8,0.1])
      with c2:
          dynamic_md = read_markdown_file('about/data_sources.md')
          st.markdown(dynamic_md, unsafe_allow_html=True)
        
  with trading_tab:
      c2,c2,c3 = st.columns([0.1,0.8,0.1])
      with c2:
          trading_md = read_markdown_file('about/trading.md')
          st.markdown(trading_md, unsafe_allow_html=True)    
        
with param_tab: 
  left, middle, right = st.columns([0.25,0.25,0.5])

  with left: 
    st.header('General')
    
    format = st.selectbox(
      'Which format are you playing?',
      ('Rotisserie', 'Head to Head: Each Category', 'Head to Head: Most Categories')
      , index = 1)
  
    if format == 'Rotisserie':
      st.caption('Note that it is recommended to use Z-scores rather than G-scores to evaluate players for Rotisserie. Also, Rotisserie H-scores are experimental and have not been tested')
    else:
      st.caption('Note that it is recommended to use G-scores rather than Z-scores to evaluate players for Head to Head')

    winner_take_all = format == 'Head to Head: Most Categories'

    unique_datasets_historical = [str(x) for x in pd.unique(historical_df.index.get_level_values('Season'))]
    unique_datasets_current = list(current_data.keys())
    unique_datasets_darko = list(darko_data.keys())

    all_datasets = unique_datasets_historical + unique_datasets_current + unique_datasets_darko
      
    dataset_name = st.selectbox(
      'Which dataset do you want to default to?'
      ,all_datasets
      ,index = len(all_datasets)-1
    )

    df = get_partial_data(historical_df, current_data, darko_data, dataset_name)

    df.index = df.index + ' (' + df['Position'] + ')'
    df.index.name = 'Player'

    n_drafters = st.number_input(r'How many drafters are in your league?'
                    , min_value = 2
                    , value = 12)

    n_picks = st.number_input(r'How many players will each drafter choose?'
                , min_value = 1
                , value = 13)

    #perhaps the dataframe should be uneditable, and users just get to enter the next players picked? With an undo button?
    selections = pd.DataFrame({'Drafter ' + str(n+1) : [None] * n_picks for n in range(n_drafters)})

    #make the selection df use a categorical variable for players, so that only players can be chosen, and it autofills
    player_category_type = CategoricalDtype(categories=list(df.index), ordered=True)
    selections = selections.astype(player_category_type)
  
  with middle: 
      st.header('Player Statistics')

      psi = st.number_input(r'Select a $\psi$ value'
                        , min_value = 0.0
                        , value = 0.85
                       , max_value = 1.0)
      psi_str = r'''$\psi$ controls how injury rates are dealt with. If $\psi$ is $X\%$, and a player is expected to miss $Y\%$ of weeks
                    entirely, their counting statistics will be multiplied by $(1-Y*X)$. So for example is if $\psi$ is $50\%$ and a 
                    player is expected to miss $20\%$ of weeks, their counting statistics will be multiplied by $(1-0.5*0.2) =  90\%$'''
    
      st.caption(psi_str)

      st.subheader(f"Coefficients")
      conversion_factors = st.data_editor(coefficient_df
                                   , column_config = {'Conversion Factor' :  '𝜏² / σ²'}
                                                      )

      st.caption('σ² and 𝜏² are defined in the paper. Player stats are input as averages rather than week-by-week numbers, so 𝜏² must be estimated. The default conversion factors from σ² to 𝜏² are based on historical values')


  
  with right:
    st.header('H-scoring Algorithm')

    left_col, right_col = st.columns(2)

    with left_col:
      omega = st.number_input(r'Select a $\omega$ value', value = 1.5)
      omega_str = r'''The higher $\omega$ is, the more aggressively the algorithm will try to punt. Slightly more technically, 
                      it quantifies how much better the optimal player choice will be compared to the player that would be 
                      chosen with baseline weights'''
      st.caption(omega_str)
    
      gamma = st.number_input(r'Select a $\gamma$ value', value = 0.1)
      gamma_str = r'''$\gamma$ also influences the level of punting, complementing omega. Tuning gamma is not suggested but you can 
              tune it if you want. Higher values imply that the algorithm will have to give up more general value to find the
               players that  work best for its strategy'''
      st.caption(gamma_str)
  
      nu = st.number_input(r'Select a $\nu$ value', value = 0.77, min_value = 0.0, max_value = 1.0)
      nu_str = r'''Covariance matrix is calculated with position averages multiplied by $\nu$ subtracted out. $\nu=0$ is appropriate 
                  if there are no position requirements, $\nu=1$ is appropriate if position requirements are fully strict '''
      st.caption(nu_str)

    with right_col:
      alpha = st.number_input(r'Select a $\alpha$ value', value = 0.01, min_value = 0.0)
      alpha_str = r'''$\alpha$ is the initial step size for gradient descent. Tuning $\alpha$ is not recommended'''
      st.caption(alpha_str)
  
      beta = st.number_input(r'Select a $\beta$ value', value = 0.25, min_value = 0.0)
      beta_str = r'''$\beta$ is the degree of step size decay. Tuning $\beta$ is not recommended'''
      st.caption(beta_str)
  
      n_iterations = st.number_input(r'Select a number of iterations for gradient descent to run', value = 30, min_value = 0, max_value = 10000)
      n_iterations_str = r'''More iterations take more computational power, but theoretically achieve better convergence'''
      st.caption(n_iterations_str)

      punting = n_iterations > 0
  
with stat_tab:
  st.header('Per-game stats')
  st.caption(f"Per-game player projections below, from default data source. feel free to edit as you see fit")

  player_stats_editable = st.data_editor(df) # 👈 An editable dataframe
  player_stats = player_stats_editable.copy()

  #re-adjust from user inputs
  player_stats[r'Free Throw %'] = player_stats[r'Free Throw %']/100
  player_stats[r'Field Goal %'] = player_stats[r'Field Goal %']/100
  player_stats[r'No Play %'] = player_stats[r'No Play %']/100
  player_stats[counting_statistics + volume_statistics] = player_stats[counting_statistics + volume_statistics] * 3
  
with draft_tab:

  rotisserie = format == 'Rotisserie'
  
  left, right = st.columns(2)

  with left:

    seat =  st.number_input(r'Which drafter are you?'
        , min_value = 1
        #, value = default_seat
       , max_value = n_drafters)

    draft_tab, injury_tab = st.tabs(['Draft Bpard','Injury List'])
    
    with draft_tab: 
        st.caption('P.S: The draft board is copy-pastable. You can save it in Excel after you are done')
        selections_editable = st.data_editor(selections, hide_index = True)  

    with injury_tab:
        st.caption(f"List of players that you think will be injured for the foreseeable future, and so should be ignored")
        injured_players = st.multiselect('Injured players', player_stats.index, default = params['injury-ignore-darko'])

    player_stats = player_stats.drop(injured_players)
    info = process_player_data(player_stats, conversion_factors, psi, nu, n_drafters, n_picks, rotisserie, params)

    z_scores = info['Z-scores']
    g_scores = info['G-scores']
    categories = [x for x in z_scores.columns if x != 'Total']

    players_chosen = [x for x in listify(selections_editable) if x ==x]
    my_players = [p for p in selections_editable['Drafter ' + str(seat)].dropna()]

    H = HAgent(info = info
         , omega = omega
         , gamma = gamma
         , alpha = alpha
         , beta = beta
         , n_picks = n_picks
         , winner_take_all = winner_take_all
         , punting = punting)
    _, base_h_score = next(H.get_h_scores(player_stats, my_players, players_chosen))

  with right:

    team_tab, cand_tab, waiver_tab, trade_tab = st.tabs(["Team", "Candidates","Waiver Moves","Trades"])
    #add two more tabs: "Waiver moves" and "Trades", only unlocked when n = n_picks

    with team_tab:
    
      team_selections = selections_editable['Drafter ' + str(seat)].dropna()

      z_tab, g_tab, h_tab = st.tabs(["Z-score", "G-score","H-score"])
        
      with z_tab:
        team_stats_z = z_scores[z_scores.index.isin(team_selections)]

        n_players_on_team = team_stats_z.shape[0]

        if n_players_on_team > 0:
            expected_z = z_scores[0:n_players_on_team*n_drafters].mean() * n_players_on_team
    
            team_stats_z.loc['Total', :] = team_stats_z.sum(axis = 0)
            team_stats_z.loc['Expected', :] = expected_z
            team_stats_z.loc['Difference', :] = team_stats_z.loc['Total',:] - team_stats_z.loc['Expected',:]
    
            team_stats_z_styled = team_stats_z.style.format("{:.2}").map(styler_a) \
                                                        .map(styler_b, subset = pd.IndexSlice[['Expected','Total'], counting_statistics + percentage_statistics]) \
                                                        .map(styler_c, subset = pd.IndexSlice[['Expected','Total'], ['Total']]) \
                                                        .map(stat_styler, subset = pd.IndexSlice[team_selections, counting_statistics + percentage_statistics], multiplier = z_score_player_multiplier) \
                                                        .applymap(stat_styler, subset = pd.IndexSlice['Difference', counting_statistics + percentage_statistics], multiplier = z_score_team_multiplier)
        else:
            team_stats_z_styled = pd.DataFrame()


        z_display = st.dataframe(team_stats_z_styled, use_container_width = True)        
        
      with g_tab:
        team_stats_g = g_scores[g_scores.index.isin(team_selections)]

        n_players_on_team = team_stats_g.shape[0]

        if n_players_on_team > 0:

            expected_g = g_scores[0:n_players_on_team*n_drafters].mean() * n_players_on_team
            team_stats_g.loc['Total', :] = team_stats_g.sum(axis = 0)
            team_stats_g.loc['Expected', :] = expected_g
            team_stats_g.loc['Difference', :] = team_stats_g.loc['Total',:] - team_stats_g.loc['Expected',:]
            
            team_stats_g_styled = team_stats_g.style.format("{:.2}").map(styler_a) \
                                                        .map(styler_b, subset = pd.IndexSlice[['Expected','Total'], counting_statistics + percentage_statistics]) \
                                                        .map(styler_c, subset = pd.IndexSlice[['Expected','Total'], ['Total']]) \
                                                        .map(stat_styler, subset = pd.IndexSlice[team_selections, counting_statistics + percentage_statistics], multiplier = g_score_player_multiplier) \
                                                        .applymap(stat_styler, subset = pd.IndexSlice['Difference', counting_statistics + percentage_statistics], multiplier = g_score_team_multiplier)
        else:
            team_stats_g_styled = pd.DataFrame()
    
        g_display = st.dataframe(team_stats_g_styled, use_container_width = True)
    
      with h_tab:
        if len(my_players) < n_picks:
            st.markdown('Your team is not full yet! Come back here when you have a full team')
        else:
            st.markdown('The H-score of team ' + str(seat) + ' is ' + str(base_h_score.round(3).values[0]))
          
    with cand_tab:

      subtab1, subtab2, subtab3 = st.tabs(["Z-score", "G-score", "H-score"])
    
      with subtab1:
        z_scores_unselected = z_scores[~z_scores.index.isin(listify(selections_editable))]
        z_scores_unselected_styled = z_scores_unselected.style.format("{:.2}").map(styler_a).map(stat_styler, subset = pd.IndexSlice[:,counting_statistics + percentage_statistics], multiplier = z_score_player_multiplier)
        z_scores_display = st.dataframe(z_scores_unselected_styled)
        
      with subtab2:
        g_scores_unselected = g_scores[~g_scores.index.isin(listify(selections_editable))]
        g_scores_unselected_styled = g_scores_unselected.style.format("{:.2}").map(styler_a).map(stat_styler, subset = pd.IndexSlice[:,counting_statistics + percentage_statistics], multiplier = g_score_player_multiplier)
        g_scores_display = st.dataframe(g_scores_unselected_styled)
    
      with subtab3:

        if not st.session_state.run:
          with st.form(key='my_form_to_submit'):
            h_score_button = st.form_submit_button(label='Run H-score algorithm', on_click = run) 
      
        else:

          #record the fact that the run has already been invoked, no need to invoke it again
          stop_run()

          n_players = n_drafters * n_picks
      
          generator = H.get_h_scores(player_stats, my_players, players_chosen)
    
          placeholder = st.empty()
          all_res = []

          #if n_iterations is 0 we run just once with punting set to False
          for i in range(max(1,n_iterations)):
      
            c, res = next(generator)
            all_res = all_res + [res]
            #normalize weights by what we expect from other drafters
            c = pd.DataFrame(c, index = res.index, columns = categories)/info['v'].T
            c = (c * 100).round()
              
            with placeholder.container():
  
              score_tab, weight_tab = st.tabs(['Scores','Weights'])
  
              with score_tab:
                c1, c2 = st.columns([0.3,0.7])
      
                with c1:
                  res = res.sort_values(ascending = False).round(3)
                  res.name = 'H-score'
      
      
                  st.dataframe(pd.DataFrame(res))
      
                with c2:
                  st.plotly_chart(make_progress_chart(all_res), use_container_width = True)
    
              with weight_tab:
                c_df = c.loc[res.index].dropna().round().astype(int)
                c_df = c_df.style.background_gradient(axis = None)
                st.dataframe(c_df)
                

    with waiver_tab:
        if len(my_players) < n_picks:
            st.markdown('Your team is not full yet! Come back here when you have a full team')
        else:
                  
            drop_player = st.selectbox(
              'Which player are you considering dropping?'
              ,my_players
              ,index = len(my_players)-1
            )

            mod_my_players = [x for x in my_players if x != drop_player]

            z_tab, g_tab, h_tab = st.tabs(['Z-score','G-score','H-score'])

            with z_tab:
                st.markdown('Projected team stats with chosen player:')
                no_drop = team_stats_z.loc[['Total'],:]
                no_drop.index = [drop_player]
                
                drop_player_stats_z = z_scores.loc[drop_player]
                new_z =  team_stats_z.loc['Total',:] + z_scores_unselected - drop_player_stats_z

                new_z = pd.concat([no_drop,new_z])
                new_z_styled = new_z.style.format("{:.2}").map(styler_a).map(stat_styler, subset = pd.IndexSlice[:,counting_statistics + percentage_statistics], multiplier = z_score_team_multiplier)

                st.dataframe(new_z_styled) 

            with g_tab:
                st.markdown('Projected team stats with chosen player:')
                no_drop = team_stats_g.loc[['Total'],:]
                no_drop.index = [drop_player]
                
                drop_player_stats_g = g_scores.loc[drop_player]
                new_g =  team_stats_g.loc['Total',:] + g_scores_unselected - drop_player_stats_g

                new_g = pd.concat([no_drop,new_g])
                new_g_styled = new_g.style.format("{:.2}").map(styler_a).map(stat_styler, subset = pd.IndexSlice[:,counting_statistics + percentage_statistics], multiplier = g_score_team_multiplier)

                st.dataframe(new_g_styled) 

            with h_tab:
                _, res= next(H.get_h_scores(player_stats, mod_my_players, players_chosen))
    
                res = res - base_h_score.values[0]
                res = res.sort_values(ascending = False).round(3)
                res.name = 'H-score differential'
    
                st.dataframe(res)

            #make a dropdown of each player on the team 
            #for each player, try removing that player, then run the H-scoring generator once to generate a recommended replacement and whether they would be better for the team
        
    with trade_tab:
        if len(my_players) < n_picks:
            st.markdown('Your team is not full yet! Come back here when you have a full team')
        else:

            second_seat =  st.number_input(r'Which drafter are you trading with?'
                , min_value = 1
               , max_value = n_drafters)

            second_team_selections = selections_editable['Drafter ' + str(second_seat)].dropna()

            if len(second_team_selections) < n_picks:
                st.markdown('This team is not full yet! Come back here when it is')
            elif second_seat == seat:
                st.markdown('You cannot trade with yourself!')
            else:

                my_trade = st.multiselect(
                  'Which players are you trading?'
                  ,team_selections
                )
    
                second_trade = st.multiselect(
                      'Which players are you receiving?'
                      ,second_team_selections
                    )
                my_trade_len = len(my_trade)
                second_trade_len = len(second_trade)
                if (my_trade_len == 0) | (second_trade_len == 0):
                    st.markdown('Need to trade at least one player')
                elif abs(my_trade_len - second_trade_len) > 6:
                    st.markdown("Too lopsided of a trade! The computer can't handle it :frowning:")
                else:
                    my_others = [x for x in team_selections if x not in my_trade]
                    second_others = [x for x in second_team_selections if x not in second_trade]
        
                    trade_results = analyze_trade(my_others, my_trade, second_others, second_trade,H, player_stats, players_chosen,n_iterations)
                    your_team_pre_trade = trade_results[0].max()
                    your_team_post_trade = trade_results[1].max()
                    their_team_pre_trade = trade_results[3].max()
                    their_team_post_trade = trade_results[2].max()

                    if your_team_pre_trade < your_team_post_trade:
                        st.markdown('This trade benefits your team. H-score goes from ' + str(np.round(your_team_pre_trade,2)) + ' to ' + str(np.round(your_team_post_trade,2)))
                    else:
                        st.markdown('This trade does not benefit your team. H-score goes from ' + str(np.round(your_team_pre_trade,2)) + ' to ' + str(np.round(your_team_post_trade,2)))

                    if their_team_pre_trade < their_team_post_trade:
                        st.markdown('This trade benefits their team. H-score goes from ' + str(np.round(their_team_pre_trade,2)) + ' to ' + str(np.round(their_team_post_trade,2)))
                    else:
                        st.markdown('This trade does not benefit their team. H-score goes from ' + str(np.round(their_team_pre_trade,2)) + ' to ' + str(np.round(their_team_post_trade,2)))
                      
with rank_tab:
  z_rank_tab, g_rank_tab, h_rank_tab = st.tabs(['Z-score','G-score','H-score'])

  with z_rank_tab:
    
      z_score_columns_original = z_scores.columns
      z_scores.loc[:,'Rank'] = np.arange(z_scores.shape[0]) + 1
      z_scores.loc[:,'Player'] = z_scores.index
      z_scores = z_scores[['Rank','Player'] + counting_statistics + percentage_statistics + ['Total']]

      z_scores_styled = z_scores.style.format("{:.2f}"
                                             ,subset = pd.IndexSlice[:,counting_statistics + percentage_statistics + ['Total']]) \
                                        .map(styler_a
                                            , subset = pd.IndexSlice[:,['Total']]) \
                                        .map(stat_styler
                                           , subset = pd.IndexSlice[:,counting_statistics + percentage_statistics]
                                           , multiplier = z_score_player_multiplier)
    
      z_scores_display = st.dataframe(z_scores_styled, hide_index = True)
  with g_rank_tab:
      st.markdown('Placeholder')
  with h_rank_tab:
      st.markdown('Placeholder')
    
