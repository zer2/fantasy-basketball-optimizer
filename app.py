import streamlit as st
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import os 
from typing import Callable
import yaml
from yfpy.models import League

from src.helper_functions import listify, make_progress_chart, make_about_tab, stat_styler, styler_a,styler_b, styler_c, get_categories
from src.get_data import get_historical_data, get_current_season_data, get_darko_data, get_specified_stats, get_player_metadata
from src.process_player_data import process_player_data
from src.run_algorithm import HAgent, analyze_trade
from src import yahoo_connect
from src.tabs import *
from src.data_editor import make_data_editor

#from streamlit_profiler import Profiler

#with Profiler():

### SETUP
st.set_page_config(page_title='Fantasy BBall Optimization'
          , page_icon=':basketball:'
          , layout="wide"
          , initial_sidebar_state="auto"
          , menu_items=None)

if 'player_stats_key' not in st.session_state:
    st.session_state.player_stats_key = 0

if 'run_h_score' not in st.session_state:
    st.session_state.run_h_score = False

if 'intro_button_disabled' not in st.session_state:
    st.session_state.intro_button_disabled = False

def run_h_score():
    st.session_state.run_h_score = True

def stop_run_h_score():
    st.session_state.run_h_score = False

def finish_intro():
  st.session_state.intro_complete = True

  st.session_state.n_drafters = n_drafters
  st.session_state.n_picks = n_picks
  st.session_state.selections = selections
  st.session_state.mode = mode
        
if 'params' not in st.session_state:
  with open("parameters.yaml", "r") as stream:
      try:
        st.session_state.params = yaml.safe_load(stream)
      except yaml.YAMLError as exc:
          print(exc) 

counting_statistics = st.session_state.params['counting-statistics'] 
volume_statistics = st.session_state.params['percentage-statistics'] 

historical_df = get_historical_data()
current_data, expected_minutes = get_current_season_data()
darko_data = get_darko_data(expected_minutes)

coefficient_df = pd.read_csv('./coefficients.csv', index_col = 0)

st.title('Optimization for Fantasy Basketball :basketball:')


### Get intro information

if 'intro_complete' not in st.session_state:
    st.session_state.intro_complete = False

if not st.session_state.intro_complete:

  mode = st.selectbox(
    'Which mode do you want to use?',
    ('Draft Mode', 'Season Mode')
    , index = 0)

  # Setting default values
  n_drafters = 12
  n_picks = 13

  if mode == 'Season Mode':

    data_source = st.selectbox(
      'How would you like to set draft player info? You can either enter your own data or fetch from a Yahoo league',
      ('Enter your own data', 'Retrieve from Yahoo Fantasy')
      , index = 0)

  else:
    data_source = 'Enter your own data'
    
  if data_source == 'Enter your own data':
    n_drafters = st.number_input(r'How many drafters are in your league?'
                                  , min_value = 2
                                  , value = 12)

    n_picks = st.number_input(r'How many players will each drafter choose?'
                  , min_value = 1
                  , value =13)
        
    # perhaps the dataframe should be uneditable, and users just get to enter the next players picked? With an undo button?
    
    selections = pd.DataFrame({'Drafter ' + str(n+1) : [None] * n_picks for n in range(n_drafters)})

  else:

    st.session_state['intro_button_disabled'] = True

    selections = None

    access_token_dir = yahoo_connect.get_yahoo_access_token()

    if access_token_dir is not None:

      user_leagues = yahoo_connect.get_user_leagues(access_token_dir)

      get_league_labels: Callable[[League], str] = lambda league: f"{league.name.decode('UTF-8')} ({league.season}-{league.season + 1} Season)"

      yahoo_league = st.selectbox(
        label='Which league should player data be pulled from?',
        options=user_leagues,
        format_func=get_league_labels,
        index=None
      )
      
      if yahoo_league is not None:

        yahoo_league_id = yahoo_league.league_id

        player_metadata = get_player_metadata()

        team_players_df = yahoo_connect.get_yahoo_players_df(access_token_dir, yahoo_league_id, player_metadata)
        n_drafters = team_players_df.shape[1]
        n_picks = team_players_df.shape[0]

        #make the selection df use a categorical variable for players, so that only players can be chosen, and it autofills
        
        selections = team_players_df

        yahoo_connect.clean_up_access_token(access_token_dir)

        st.session_state['intro_button_disabled'] = False

        st.write('Player info successfully retrieved from yahoo fantasy! :partying_face:')

    if selections is None:
      selections = pd.DataFrame({'Drafter ' + str(n+1) : [None] * n_picks for n in range(n_drafters)})

  intro_complete = st.button("Go!"
                                    , use_container_width = True
                                    , disabled = st.session_state['intro_button_disabled']
                                    , on_click = finish_intro)

### Build app 

else:

  n_drafters = st.session_state.n_drafters
  n_picks = st.session_state.n_picks
  selections = st.session_state.selections
  mode = st.session_state.mode

  if mode == 'Draft Mode':
    main_tabs = st.tabs([":control_knobs: Parameters"
                ,":bar_chart: Player Info"
                ,":first_place_medal: Player Scores & Rankings"
                ,":man-bouncing-ball: Drafting"
                ,":scroll: About"])

    param_tab = main_tabs[0]
    info_tab = main_tabs[1]
    rank_tab = main_tabs[2]
    draft_tab = main_tabs[3]
    about_tab = main_tabs[4]
                  
  elif mode == 'Season Mode':
    main_tabs = st.tabs([":control_knobs: Parameters"
                    ,":bar_chart: Player Info"
                    ,":first_place_medal: Player Scores & Rankings"
                    ,":stadium: Rosters"
                    ,":crossed_swords: Matchups"
                    ,":man_standing: Waiver Wire & Free Agents"
                    ,":clipboard: Trading"
                    ,":scroll: About"])

    param_tab = main_tabs[0]
    info_tab = main_tabs[1]
    rank_tab = main_tabs[2]
    rosters_tab = main_tabs[3]
    matchup_tab = main_tabs[4]
    waiver_tab = main_tabs[5]
    trade_tab = main_tabs[6]
    about_tab = main_tabs[7]
                  
  with param_tab: 

    general_params, advanced_params = st.tabs(['General','Advanced'])

    with general_params:
      
      scoring_format = st.selectbox(
        'Which format are you playing?',
        ('Rotisserie', 'Head to Head: Each Category', 'Head to Head: Most Categories')
        , index = 1)
    
      if scoring_format == 'Rotisserie':
        st.caption('Note that it is recommended to use Z-scores rather than G-scores to evaluate players for Rotisserie. Also, Rotisserie H-scores are experimental and have not been tested')
      else:
        st.caption('Note that it is recommended to use G-scores rather than Z-scores to evaluate players for Head to Head')

      winner_take_all = scoring_format == 'Head to Head: Most Categories'
      rotisserie = scoring_format == 'Rotisserie'

      unique_datasets_historical = [str(x) for x in pd.unique(historical_df.index.get_level_values('Season'))]
      unique_datasets_current = list(current_data.keys())
      unique_datasets_darko = list(darko_data.keys())

      all_datasets = unique_datasets_historical + unique_datasets_current + unique_datasets_darko
      all_datasets.reverse()
        
      dataset_name = st.selectbox(
        'Which dataset do you want to default to?'
        ,all_datasets
        ,index = 0
      )

      raw_stats_df = get_specified_stats(historical_df, current_data, darko_data, dataset_name)

      player_category_type = CategoricalDtype(categories=list(raw_stats_df.index), ordered=True)
      selections = selections.astype(player_category_type)

    with advanced_params:

      player_param_column, algorithm_param_column, trade_param_column = st.columns([0.25,0.5,0.25])

      with player_param_column:
        st.header('Player Statistics')

        psi = st.number_input(r'Select a $\psi$ value'
                          , min_value = 0.0
                          , value = 0.85
                        , max_value = 1.0)
        psi_str = r'''$\psi$ controls how injury rates are dealt with. For example is if $\psi$ is $50\%$ and a 
                      player is expected to miss $20\%$ of weeks, their counting statistics will be multiplied by $(1-0.5*0.2) =  90\%$'''
      
        st.caption(psi_str)

        mult_col, coef_col = st.columns(2)
      
        st.subheader(f"Multipliers")

        multiplier_df = pd.DataFrame({'Multiplier' : [1,1,1,1,1,1,1,1,1]}
                                    , index = coefficient_df.index).T
        multipliers = st.data_editor(multiplier_df, hide_index = True)
        multipliers = multipliers.T
    
        st.caption('Manual multipliers for Z-scores and G-scores. E.g. to ignore turnovers completely, set the turnovers multiplier to 0. Note that H-scores will assume other drafters are using this weighting as well')
          
        st.subheader(f"Coefficients")
        conversion_factors = st.data_editor(coefficient_df.T, hide_index = True)
        conversion_factors = conversion_factors.T                                                      

        st.caption('Conversion factor for translating from σ² to 𝜏² as defined in the paper. Player stats are input as averages rather than week-by-week numbers, so 𝜏² must be estimated. The default conversion factors from σ² to 𝜏² are based on historical values')

      with algorithm_param_column:
        st.header('H-scoring Parameters')

        left_algo_param_col, right_algo_param_col = st.columns(2)

        with left_algo_param_col:
          omega = st.number_input(r'Select a $\omega$ value', value = 1.0)
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

        with right_algo_param_col:
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

      with trade_param_column:
          st.header('Trading Parameters')

          your_differential_threshold = st.number_input(
                r'Your differential threshold for the automatic trade suggester'
                , value = 0)
          ydt_str = r'''Only trades which improve your H-score 
                        by the threshold (as a percentage) will be shown'''
          st.caption(ydt_str)
          your_differential_threshold = your_differential_threshold /100

          their_differential_threshold = st.number_input(
                r'Counterparty differential threshold for the automatic trade suggester'
                , value = 0)
          tdt_str = r'''Only trades which improve the counterparty's H-score 
                      by the threshold (as a percentage) will be shown'''
          st.caption(tdt_str)
          their_differential_threshold = their_differential_threshold/100

          with st.form("Combo params"):
            combo_params_default = pd.DataFrame({'N-traded' : [1,2,3]
                                          ,'N-received' : [1,2,3]
                                          ,'T1' : [0,0,0]
                                          ,'T2' : [1,0.25,0.1]}
                                          )

            combo_params_df = st.data_editor(combo_params_default
                                              , hide_index = True
                                              , num_rows = "dynamic"
                                              , column_config={
                                "N-traded": st.column_config.NumberColumn("N-traded", default=1)
                                ,"N-received": st.column_config.NumberColumn("N-received", default=1)
                                ,"T1": st.column_config.NumberColumn("T1", default=0)
                                ,"T2": st.column_config.NumberColumn("T2", default=0)

                                                        }
                                                ) 
            combo_params_df[['N-traded','N-received']] = \
                  combo_params_df[['N-traded','N-received']].astype(int)

            combo_params_df['T1'] = combo_params_df['T1']/100
            
            submitted = st.form_submit_button("Submit", use_container_width = True)

          combo_params_str =  \
            """Each row is a specification for a kind of trade that will be automatically evaluated. 
            N-traded is the number of players traded from your team, and N-received is the number of 
            players to receive in the trade. T1 is a threshold of 'targetedness' as shown in the Target
            column. Only players with the specified targetedness or above will be considered. T2 is a 
            threshold of G-score difference (or Z-score for Rotisseries); trades that have general value 
            differences larger than T2 will not be evaluated"""
          st.caption(combo_params_str)

          combo_params = tuple(combo_params_df.itertuples(name = None, index = None))

  with info_tab:

    stat_tab, injury_tab = st.tabs([
                    "Player Stats"
                    ,"Injury Status"])

    with stat_tab:
      st.header('Per-game stats')
      st.caption(f"Per-game player projections below, from default data source. feel free to edit as you see fit")

      #player_stats_editable = st.data_editor(raw_stats_df, key = 'player_stats') # 👈 An editable dataframe
      player_stats_editable = make_data_editor(raw_stats_df)
      player_stats = player_stats_editable.copy()

      #re-adjust from user inputs
      player_stats[r'Free Throw %'] = player_stats[r'Free Throw %']/100
      player_stats[r'Field Goal %'] = player_stats[r'Field Goal %']/100
      player_stats[r'No Play %'] = player_stats[r'No Play %']/100
      player_stats[counting_statistics + volume_statistics] = player_stats[counting_statistics + volume_statistics] * 3

    with injury_tab:
        st.caption(f"List of players that you think will be injured for the foreseeable future, and so should be ignored")
        injury_list = st.session_state.params['injury-ignore-darko'] if 'DARKO' in dataset_name else None
        injured_players = st.multiselect('Injured players', player_stats.index, default = injury_list)

        player_stats = player_stats.drop(injured_players)
        info = process_player_data(player_stats
                                ,conversion_factors
                                ,multipliers
                                ,psi
                                ,nu
                                ,n_drafters
                                ,n_picks
                                ,rotisserie)

        z_scores = info['Z-scores']
        g_scores = info['G-scores']

  with rank_tab:
      z_rank_tab, g_rank_tab, h_rank_tab = st.tabs(['Z-score','G-score','H-score'])
        
      with z_rank_tab:
        make_rank_tab(z_scores
                      , st.session_state.params['z-score-player-multiplier'])  

      with g_rank_tab:
        make_rank_tab(g_scores
                      , st.session_state.params['g-score-player-multiplier'])  

      with h_rank_tab:
        rel_score_string = 'Z-scores' if rotisserie else 'G-scores'
        st.caption('Note that these scores are unique to the ' + scoring_format + ' format and all the H-scoring parameters defined on the parameter tab')
        st.caption('Category scores are expected weekly win rates given approximate punt-adjusted future picks')
        make_h_rank_tab(info
                      ,omega
                      ,gamma
                      ,alpha
                      ,beta
                      ,n_picks
                      ,n_iterations
                      ,winner_take_all
                      ,punting
                      ,player_stats)

  H = HAgent(info = info
      , omega = omega
      , gamma = gamma
      , alpha = alpha
      , beta = beta
      , n_picks = n_picks
      , winner_take_all = winner_take_all
      , punting = punting)   

  if mode == 'Draft Mode':
    with draft_tab:
      
      left, right = st.columns(2)

      with left:

        draft_seat = st.selectbox(f'Which team are you?'
            , selections.columns
            , index = 0)
        
        st.caption("""Enter which players have been drafted by which teams below""")
        selections_editable = st.data_editor(selections, hide_index = True)  
        selection_list = listify(selections_editable)

        players_chosen = [x for x in listify(selections_editable) if x ==x]
        my_players = selections_editable[draft_seat].dropna()

        z_scores_unselected = z_scores[~z_scores.index.isin(selection_list)]
        g_scores_unselected = g_scores[~g_scores.index.isin(selection_list)]

      with right:

        cand_tab, team_tab = st.tabs(["Candidates","Team"])
              
        with cand_tab:

          z_cand_tab, g_cand_tab, h_cand_tab = st.tabs(["Z-score", "G-score", "H-score"])
                    
          with z_cand_tab:
            
            z_scores_unselected = make_cand_tab(z_scores_unselected
                                          , st.session_state.params['z-score-player-multiplier'])

          with g_cand_tab:

            g_scores_unselected = make_cand_tab(g_scores_unselected
                                          , st.session_state.params['g-score-player-multiplier'])

          with h_cand_tab:

            if len(my_players) == n_picks:
              st.markdown('Team is complete!')
                      
            elif not st.session_state.run_h_score:
              with st.form(key='my_form_to_submit'):
                h_score_button = st.form_submit_button(label='Run H-score algorithm'
                                                    , on_click = run_h_score) 
                        
            else:

              #record the fact that the run has already been invoked, no need to invoke it again
              stop_run_h_score()

              n_players = n_drafters * n_picks
          
              generator = H.get_h_scores(my_players, players_chosen)
        
              placeholder = st.empty()
              all_res = []

              #if n_iterations is 0 we run just once with punting set to False
              for i in range(max(1,n_iterations)):

                res = next(generator)
                score = res['Scores']
                weights = res['Weights']
                win_rates = res['Rates']

                all_res = all_res + [score]
                #normalize weights by what we expect from other drafters
                
                weights = pd.DataFrame(weights
                              , index = score.index
                              , columns = get_categories())/info['v'].T
                
                win_rates.columns = get_categories()
                
                with placeholder.container():
      
                  rate_tab, weight_tab = st.tabs(['Expected Win Rates', 'Weights'])
                      
                  score = score.sort_values(ascending = False).round(3)
                  score.name = 'H-score'
                  score = pd.DataFrame(score)

                  with rate_tab:
                    rate_df = win_rates.loc[score.index].dropna()
                    rate_display = score.merge(rate_df, left_index = True, right_index = True)
                    rate_display = rate_display.style.format("{:.1%}"
                                      ,subset = pd.IndexSlice[:,['H-score']]) \
                              .map(styler_a
                                    , subset = pd.IndexSlice[:,['H-score']]) \
                              .map(stat_styler, middle = 0.5, multiplier = 300, subset = rate_df.columns) \
                              .format('{:,.1%}', subset = rate_df.columns)
                    st.dataframe(rate_display, use_container_width = True)
                  with weight_tab:
                    weight_df = weights.loc[score.index].dropna()
                    weight_display = score.merge(weight_df
                                          , left_index = True
                                          , right_index = True)
                    weight_display = weight_display.style.format("{:.0%}"
                                                                , subset = weight_df.columns)\
                              .format("{:.1%}"
                                      ,subset = pd.IndexSlice[:,['H-score']]) \
                              .map(styler_a
                                    , subset = pd.IndexSlice[:,['H-score']]) \
                              .background_gradient(axis = None,subset = weight_df.columns) 
                    st.dataframe(weight_display, use_container_width = True)

        with team_tab:

          if len(my_players) == n_picks:
            base_h_res = get_base_h_score(info
                                          ,omega
                                          ,gamma
                                          ,alpha
                                          ,beta
                                          ,n_picks
                                          ,winner_take_all
                                          ,punting
                                          ,player_stats
                                          ,my_players
                                          ,players_chosen)

            base_h_score = base_h_res['Scores']
            base_win_rates = base_h_res['Rates']

          else:
            base_h_score = None
            base_win_rates = None

          make_full_team_tab(z_scores
                            ,g_scores
                            ,my_players
                            ,n_drafters
                            ,n_picks
                            ,base_h_score
                            ,base_win_rates
                            )
  elif mode == 'Season Mode':

    with rosters_tab:

      left, right = st.columns([0.5,0.5])
        
      with left:

        roster_inspection_seat = st.selectbox(f'Which team do you want to get aggregated statistics for?'
            , selections.columns
            , index = 0)

        st.caption("""Enter which player is on which team below""")
        selections_editable = st.data_editor(selections, hide_index = True)  
        selection_list = listify(selections_editable)

        players_chosen = [x for x in listify(selections_editable) if x ==x]
        inspection_players = selections_editable[roster_inspection_seat].dropna()

        z_scores_unselected = z_scores[~z_scores.index.isin(selection_list)]
        g_scores_unselected = g_scores[~g_scores.index.isin(selection_list)]

        with right: 

          base_h_res = get_base_h_score(info
                                        ,omega
                                        ,gamma
                                        ,alpha
                                        ,beta
                                        ,n_picks
                                        ,winner_take_all
                                        ,punting
                                        ,player_stats
                                        ,inspection_players
                                        ,players_chosen)

          base_h_score = base_h_res['Scores']
          base_win_rates = base_h_res['Rates']

          make_full_team_tab(z_scores
                              ,g_scores
                              ,inspection_players
                              ,n_drafters
                              ,n_picks
                              ,base_h_score
                              ,base_win_rates
                              )

    with matchup_tab:
      make_matchup_tab(info['X-scores']
                      ,selections_editable
                      ,scoring_format
                      )

    with waiver_tab:

        c1, c2 = st.columns([0.5,0.5])

        with c1: 
          waiver_inspection_seat = st.selectbox(f'Which team so you want to drop a player from?'
              , selections.columns
              , index = 0)

        with c2: 
            waiver_players = selections_editable[waiver_inspection_seat].dropna()

            if len(waiver_players) < n_picks:
                st.markdown("""This team is not full yet!""")

            else:

              waiver_team_stats_z = z_scores[z_scores.index.isin(waiver_players)]
              waiver_team_stats_z.loc['Total', :] = waiver_team_stats_z.sum(axis = 0)

              waiver_team_stats_g = g_scores[g_scores.index.isin(waiver_players)]
              waiver_team_stats_g.loc['Total', :] = waiver_team_stats_g.sum(axis = 0)

              if rotisserie:
                worst_player = list(z_scores.index[z_scores.index.isin(waiver_players)])[-1]
              else:
                worst_player = list(g_scores.index[g_scores.index.isin(waiver_players)])[-1]

              default_index = list(waiver_players).index(worst_player)

              drop_player = st.selectbox(
                'Which player are you considering dropping?'
                ,waiver_players
                ,index = default_index
              )

        if len(waiver_players) == n_picks:

          mod_waiver_players = [x for x in waiver_players if x != drop_player]

          z_waiver_tab, g_waiver_tab, h_waiver_tab = st.tabs(['Z-score','G-score','H-score'])

          with z_waiver_tab:

              st.markdown('Projected team stats with chosen player:')
              make_waiver_tab(z_scores
                            , z_scores_unselected
                            , waiver_team_stats_z
                            , drop_player
                            , st.session_state.params['z-score-team-multiplier'])

          with g_waiver_tab:

              st.markdown('Projected team stats with chosen player:')
              make_waiver_tab(g_scores
                            , g_scores_unselected
                            , waiver_team_stats_g
                            , drop_player
                            , st.session_state.params['g-score-team-multiplier'])

          with h_waiver_tab:

              base_h_res = get_base_h_score(info
                              ,omega
                              ,gamma
                              ,alpha
                              ,beta
                              ,n_picks
                              ,winner_take_all
                              ,punting
                              ,player_stats
                              ,waiver_players
                              ,players_chosen)

              waiver_base_h_score = base_h_res['Scores']
              waiver_base_win_rates = base_h_res['Rates']

              make_h_waiver_df(H
                          , player_stats
                          , mod_waiver_players
                          , drop_player
                          , players_chosen
                          , waiver_base_h_score
                          , waiver_base_win_rates)

    with trade_tab:

      c1, c2 = st.columns([0.5,0.5])

      with c1: 
        trade_party_seat = st.selectbox(f'Which team do you want to trade from?'
            , selections.columns
            , index = 0)

      with c2: 
        trade_party_players = selections_editable[trade_party_seat].dropna()

        if len(trade_party_players) < n_picks:
            st.markdown("""This team is not full yet! Fill it and other teams out on the 
                        "Teams" page then come back here""")

        else:
          
          counterparty_players_dict = { team : players for team, players in selections_editable.items() 
                                  if ((team != trade_party_seat) & (not  any(p!=p for p in players)))
                                    }

          trade_counterparty_seat = st.selectbox(
              f'Which team do you want to trade with?',
              [s for s in counterparty_players_dict.keys()],
              index = 0
            )
          
          trade_counterparty_players = counterparty_players_dict[trade_counterparty_seat]

      if len(trade_party_players) == n_picks:

        if len(trade_counterparty_players) < n_picks:
          st.markdown('The other team is not full yet!')
        else:

          inspection_tab, destinations_tab, target_tab, suggestions_tab = st.tabs(['Trade Inspection'
                                                              ,'Destinations'
                                                              ,'Targets'
                                                              ,'Trade Suggestions'
                                                              ])
          with inspection_tab:

            c1, c2 = st.columns(2)

            with c1: 

              with st.form("trade inspection form"):

                players_sent = st.multiselect(
                  'Which players are you trading?'
                  ,trade_party_players
                  )
                  
                players_received = st.multiselect(
                      'Which players are you receiving?'
                      ,trade_counterparty_players
                  )

                submitted = st.form_submit_button("Submit", use_container_width = True)

            with c2: 

              h_tab, z_tab, g_tab = st.tabs(['H-score','Z-score','G-score'])

              if (len(players_sent) == 0) | (len(players_received) == 0):
                st.markdown('A trade must include at least one player from each team')

              else:

                with h_tab:
                  make_trade_display(H
                                  , player_stats 
                                  , players_chosen 
                                  , n_iterations 
                                  , players_sent
                                  , players_received
                                  , trade_party_players
                                  , trade_counterparty_players
                                  , trade_counterparty_seat
                                  , scoring_format)
                with z_tab:
                  make_trade_score_tab(z_scores 
                                    , players_sent
                                    , players_received 
                                    , st.session_state.params['z-score-player-multiplier']
                                    , st.session_state.params['z-score-team-multiplier']
                                    )
                with g_tab:
                  make_trade_score_tab(g_scores 
                                    , players_sent
                                    , players_received 
                                    , st.session_state.params['g-score-player-multiplier']
                                    , st.session_state.params['g-score-team-multiplier']
                                    )



          with destinations_tab:

            values_to_team = make_trade_destination_display(H
                                  , player_stats
                                  , trade_party_players 
                                  , counterparty_players_dict 
                                  , players_chosen 
                                  , scoring_format
                                        )

          with target_tab:

            values_to_me = make_trade_target_display(H
                  , player_stats
                  , trade_party_players 
                  , trade_counterparty_players
                  , players_chosen 
                  , values_to_team[trade_counterparty_seat]
                  , scoring_format
                        )

            with suggestions_tab:

              if rotisserie:
                general_value = z_scores.sum(axis = 1)
                replacement_value = z_scores_unselected.iloc[0].sum() 
              else:
                general_value = g_scores.sum(axis = 1)
                replacement_value = g_scores_unselected.iloc[0].sum()

              #slightly hacky way to make all of the multiselects blue
              st.markdown("""
                  <style>
                      span[data-baseweb="tag"][aria-label="1 for 1, close by backspace"]{
                          background-color: lightblue; color:black;
                      }
                      span[data-baseweb="tag"][aria-label="2 for 2, close by backspace"]{
                          background-color: lightblue; color:black;
                      }
                      span[data-baseweb="tag"][aria-label="3 for 3, close by backspace"]{
                          background-color: lightblue; color:black;
                      }
                  </style>
                  """, unsafe_allow_html=True)

              trade_filter = st.multiselect('Which kinds of trades do you want get suggestions for?'
                                        , [(1,1),(2,2),(3,3)]
                                        , format_func = lambda x: str(x[0]) + ' for ' + str(x[1])
                                        , default = [(1,1)])

              make_trade_suggestion_display(H
                  , player_stats 
                  , players_chosen 
                  , trade_party_players 
                  , trade_counterparty_players
                  , general_value
                  , replacement_value
                  , values_to_me
                  , values_to_team[trade_counterparty_seat]
                  , your_differential_threshold
                  , their_differential_threshold
                  , combo_params
                  , trade_filter
                  , scoring_format )               

  with about_tab:

    tabs = st.tabs(['Intro'
                    ,'G-scoring'
                    ,'H-scoring'
                    ,'Turnovers'
                    ,'Waivers & Trading'
                    ,'Data Sources'])

    about_paths = ['intro.md'
                  ,'static_explanation.md'
                  ,'dynamic_explanation.md'
                  ,'turnovers.md'
                  ,'data_sources.md'
                  ,'trading.md']

    for tab, path in zip(tabs, about_paths):
      with tab:
        make_about_tab(path)   

