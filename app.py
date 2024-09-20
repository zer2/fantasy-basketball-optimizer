import streamlit as st
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import os 
from typing import Callable
import yaml
from yfpy.models import League
import time
from schedule import every, repeat, run_pending
from src.helper_functions import  get_position_numbers, listify \
                                  ,increment_player_stats_version, increment_info_key, increment_default_key \
                                  ,get_games_per_week, get_categories, get_ratio_statistics, clear_draft_board
from src.get_data import get_historical_data, get_current_season_data, get_darko_data, get_specified_stats, \
                        get_player_metadata, get_data_from_snowflake, combine_nba_projections
from src.process_player_data import process_player_data
from src.algorithm_agents import HAgent
from src import yahoo_connect
from src.tabs import *
from src.data_editor import make_data_editor
from src.drafting import make_drafting_tab_own_data, make_drafting_tab_live_data, run_autodraft, make_auction_tab_live_data

#from streamlit_profiler import Profiler

#with Profiler():

### SETUP
st.set_page_config(page_title='Fantasy BBall Optimization'
          , page_icon=':basketball:'
          , layout="wide"
          , initial_sidebar_state="auto"
          , menu_items=None)

if 'player_stats_editable' not in st.session_state:
    st.session_state.player_stats_editable = 0

if 'player_stats_editable_version' not in st.session_state:
    st.session_state.player_stats_editable_version = 0

if 'player_stats_default_key' not in st.session_state:
    st.session_state.player_stats_default_key = 0

if 'info_key' not in st.session_state:
    st.session_state.info_key = 100000

if 'injured_players' not in st.session_state:
    st.session_state['injured_players'] = set()

if 'schedule' not in st.session_state:
    st.session_state['schedule'] = {}

if 'mode' not in st.session_state:
    st.session_state['mode'] = 'Draft Mode'

if 'selections_df_original' not in st.session_state:
    st.session_state['selections_df_original'] = pd.DataFrame()

if 'live_draft_active' not in st.session_state:
    st.session_state.live_draft_active = False

if 'yahoo_league_id' not in st.session_state:
    st.session_state.yahoo_league_id = None

if 'draft_results' not in st.session_state:
    st.session_state.draft_results = None
    st.session_state.live_draft_active = False

if 'run_h_score' not in st.session_state:
    st.session_state.run_h_score = False

def run_h_score():
    st.session_state.run_h_score = True

def stop_run_h_score():
    st.session_state.run_h_score = False

with open("parameters.yaml", "r") as stream:
  st.session_state.all_params = yaml.safe_load(stream)

def load_params(league):
  st.session_state.params = st.session_state.all_params[league]

st.title('Optimization for Fantasy Basketball :basketball:')

### Build app 

if st.session_state['mode'] == 'Draft Mode':
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

elif st.session_state['mode'] == 'Auction Mode':
  main_tabs = st.tabs([":control_knobs: Parameters"
              ,":bar_chart: Player Info"
              ,":first_place_medal: Player Scores & Rankings"
              ,":moneybag: Auction"
              ,":scroll: About"])

  param_tab = main_tabs[0]
  info_tab = main_tabs[1]
  rank_tab = main_tabs[2]
  auction_tab = main_tabs[3]
  about_tab = main_tabs[4]
                
elif st.session_state['mode'] == 'Season Mode':
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

  general_params, data_params, advanced_params = st.tabs(['General','Data','Advanced'])

  with general_params:

    c1, c2 = st.columns(2)

    with c1: 

      league = st.selectbox(
        'Which fantasy sport are you playing?',
        ('NBA', 'WNBA') #MLB excluded for now
        , index = 0
        , key = 'league'
        , on_change = increment_default_key
        )
      
      load_params(st.session_state.league)

      mode = st.selectbox(
        'Which mode do you want to use?',
        ('Draft Mode', 'Auction Mode','Season Mode')
        , index = 0
        , key = 'mode')
      
      # Setting default values
      #st.session_state.n_drafters = st.session_state.params['options']['n_drafters']['default']
      #st.session_state.n_picks = st.session_state.params['options']['n_picks']['default']

      #These are based on 2023-2024 excluding injury
      #might need to modify these at some point? 


      data_source = st.selectbox(
        'How would you like to set draft player info? You can either enter your own data or fetch from a Yahoo league',
        ('Enter your own data', 'Retrieve from Yahoo Fantasy')
        , on_change = clear_draft_board
        , index = 0)

        
      if data_source == 'Enter your own data':
        n_drafters = st.number_input(r'How many drafters are in your league?'
                                      , key = 'n_drafters'
                                      , min_value = st.session_state.params['options']['n_drafters']['min']
                                      , value = st.session_state.params['options']['n_drafters']['default']
                                      )

        n_picks = st.number_input(r'How many players will each drafter choose?'
                      , key = 'n_picks'
                      , min_value = st.session_state.params['options']['n_picks']['min']
                      , value = st.session_state.params['options']['n_picks']['default'])
                      
        # perhaps the dataframe should be uneditable, and users just get to enter the next players picked? With an undo button?
        
        st.session_state.selections_default = pd.DataFrame(
          {'Drafter ' + str(n+1) : [np.nan] * st.session_state.n_picks for n in range(st.session_state.n_drafters)}
          )

      else:

        st.session_state.selections_default = None

        auth_dir = yahoo_connect.get_yahoo_access_token()
        st.session_state.auth_dir = auth_dir

        if auth_dir is not None:

          user_leagues = yahoo_connect.get_user_leagues(auth_dir)
          
          get_league_labels: Callable[[League], str] = lambda league: f"{league.name.decode('UTF-8')} ({league.season}-{league.season + 1} Season)"

          yahoo_league = st.selectbox(
            label='Which league should player data be pulled from?',
            options=user_leagues,
            format_func=get_league_labels,
            index=None,
            on_change = clear_draft_board

          )


          if yahoo_league is not None:
              st.session_state.yahoo_league_id = yahoo_league.league_id

              #ZR: Ideally we could fix this for mock drafts with dummies
              st.session_state.team_names = list(yahoo_connect.get_teams_dict(st.session_state.yahoo_league_id, auth_dir).values())
              st.session_state.n_drafters = len(yahoo_connect.get_teams_dict(st.session_state.yahoo_league_id, auth_dir))
          else:
               yahoo_league = st.number_input(label =  "For a mock draft, manually write in league ID (from URL, after mlid = )"
                               ,min_value = 0
                               ,value = None
                               ,key = 'yahoo_league_id'
                               , on_change = clear_draft_board)
               
               if st.session_state.yahoo_league_id is not None:
                st.session_state.team_names = list(yahoo_connect.get_teams_dict(st.session_state.yahoo_league_id, auth_dir).values())                
                st.session_state.n_drafters = len(yahoo_connect.get_teams_dict(st.session_state.yahoo_league_id, auth_dir))
               else:
                st.session_state.n_drafters = 12 #Kind of a hack
                

          if (st.session_state.mode == 'Season Mode'):

            if st.session_state.yahoo_league_id is not None:

              player_metadata = get_player_metadata()

              team_players_df = yahoo_connect.get_yahoo_players_df(auth_dir, st.session_state.yahoo_league_id, player_metadata)
              st.session_state.n_drafters = team_players_df.shape[1]
              st.session_state.n_picks = team_players_df.shape[0]

              #make the selection df use a categorical variable for players, so that only players can be chosen, and it autofills
              
              st.session_state.selections_default = team_players_df

              #Just trying for now!
              player_statuses = yahoo_connect.get_player_statuses(st.session_state.yahoo_league_id, auth_dir, player_metadata)

              st.session_state['injured_players'].update(set(list(player_statuses['Player'][ \
                                                                      (player_statuses['Status'] == 'INJ')
                                                                      ]
                                                                      )
                                                              )
                                                        )

              st.session_state['schedule'] = yahoo_connect.get_yahoo_schedule(st.session_state.yahoo_league_id
                                                                          , auth_dir
                                                                          , player_metadata)

              st.session_state['matchups'] = yahoo_connect.get_yahoo_matchups(st.session_state.yahoo_league_id
                                      , auth_dir)

              yahoo_connect.clean_up_access_token(auth_dir)
              
              st.write('Player info successfully retrieved from yahoo fantasy! :partying_face:')

          else:

            if st.session_state.yahoo_league_id is not None:

              st.session_state.selections_default = None
              st.session_state.n_drafters = len(yahoo_connect.get_teams_dict(st.session_state.yahoo_league_id, auth_dir))
              st.session_state.team_names = list(yahoo_connect.get_teams_dict(st.session_state.yahoo_league_id, auth_dir).values())
              st.session_state.n_picks = 13 #ZR: fix this

            else: 
              st.session_state.n_picks = 13
              st.session_state.n_drafters = 12

        if st.session_state.selections_default is None:
          st.session_state.selections_default = pd.DataFrame(
            {'Drafter ' + str(n+1) : [None] * st.session_state.n_picks for n in range(st.session_state.n_drafters)}
            )

          
      #set default position numbers, based on n_picks
      all_position_defaults = st.session_state.params['options']['positions']
      
      if st.session_state.n_picks in all_position_defaults:
        position_defaults = all_position_defaults[st.session_state.n_picks]
      else:
        position_defaults = all_position_defaults[st.session_state.params['options']['n_picks']['default']]
        st.error('''There is no default position structure for a league with this number of picks. 
                 Position structure must be met manually on the Advanced tab.''')

    with c2: 
    

        kind_of_dataset = st.selectbox(
                                  'Which kind of dataset do you want to use? (specify further on the data tab)'
                                  ,['Projection','Historical']
                                  , index = 0
                                  , on_change = increment_default_key
        )

        def run_autodraft_and_increment():
          increment_player_stats_version()
          run_autodraft()


        scoring_format = st.selectbox(
          'Which format are you playing?',
          ('Rotisserie', 'Head to Head: Each Category', 'Head to Head: Most Categories')
          , key = 'scoring_format'
          , index = 1)
      
        if scoring_format == 'Rotisserie':
          st.caption('Note that H-scores for Rotisserie are experimental and have not been tested')

        rotisserie = scoring_format == 'Rotisserie'

        punting_levels = st.session_state.params['punting_defaults']

        default_punting = st.session_state.params['punting_default_index'][scoring_format]

        punting_level = st.selectbox(
          'What level of punting do you want H-scores to apply when modeling your future draft picks?'
          ,list(punting_levels.keys())
          ,index = default_punting
        )

        st.caption('''For more granular control, use the Advanced tab which is next to this one''')

        with st.form("more options"):

          categories = st.multiselect('Which categories does you league use?'
                      , key = 'selected_categories'
                      , options = st.session_state.params['counting-statistics'] + \
                                list(st.session_state.params['ratio-statistics'].keys())
                      , default = st.session_state.params['default-categories']
                            )
        

          if st.session_state['mode'] == 'Draft Mode':
              autodrafters = st.multiselect('''Which drafter(s) should be automated with an auto-drafter?'''
                    ,options = st.session_state.selections_default.columns
                    ,key = 'autodrafters'
                    ,default = None)
          
          submit = st.form_submit_button("Lock in",on_click = run_autodraft_and_increment)

  with data_params:
    if st.session_state.league == 'NBA':

        current_data, expected_minutes = get_current_season_data()
        #darko_data = get_darko_data(expected_minutes)

        unique_datasets_current = list(current_data.keys())
        #unique_datasets_darko = list(darko_data.keys())
        if kind_of_dataset == 'Historical':
        
          historical_df = get_historical_data()

          unique_datasets_historical = reversed([str(x) for x in pd.unique(historical_df.index.get_level_values('Season'))])

          dataset_name = st.selectbox(
            'Which dataset do you want to default to?'
            ,unique_datasets_historical
            ,index = 0
            ,on_change = increment_default_key
          )
          raw_stats_df = get_specified_stats(dataset_name
                                    , st.session_state.player_stats_default_key)
  
        else: 

          hashtag_c, roto_c, bbm_c, = st.columns(3)

          with hashtag_c:
            hashtag_upload = get_data_from_snowflake('HTB_PROJECTION_TABLE')
            hashtag_slider = st.slider('Hashtag Basketball Weight'
                                      , min_value = 0.0
                                      , value = 1.0
                                      , max_value = 1.0
                                      , on_change = increment_default_key)

          with roto_c:

            rotowire_slider = st.slider('RotoWire Weight'
                            , min_value = 0.0
                            , max_value = 1.0
                            , on_change = increment_default_key)

            
            rotowire_file = st.file_uploader("Upload RotoWire data, as a csv"
                                            , type=['csv']
                                            , on_change = increment_default_key)
            if rotowire_file is not None:
              rotowire_upload  = pd.read_csv(rotowire_file, skiprows = 1)
            else:
              rotowire_upload = None

            if (rotowire_slider > 0) & (rotowire_upload is None):
              st.error('Upload RotoWire projection file')
              st.stop()

          with bbm_c:

            bbm_slider = st.slider('BBM Weight'
                      , min_value = 0.0
                      , max_value = 1.0
                      , on_change = increment_default_key)

            bbm_file = st.file_uploader('''Upload Basketball Monster Per Game Stats, as a csv. To get all required columns for 
                                            projections, you may have to export to excel then save as CSV utf-8.'''
                                            , type=['csv']
                                            , on_change = increment_default_key)
            if bbm_file is not None:
              # Adding a 
              bbm_upload  = pd.read_csv(bbm_file)
            else:
              bbm_upload = None


            if (bbm_slider > 0) & (bbm_upload is None):
              st.error('Upload Basketball Monster projection file')
              st.stop()

          raw_stats_df = combine_nba_projections(hashtag_upload
                            , rotowire_upload
                            , bbm_upload
                            , hashtag_slider
                            , rotowire_slider
                            , bbm_slider)
                    
    else:
          all_datasets = ['RotoWire (req. upload)'] 
          raw_stats_df = get_specified_stats('RotoWire (req. upload)')

  with advanced_params:

    player_param_column, position_param_column, algorithm_param_column, trade_param_column = st.columns(4)

    with player_param_column:
      st.subheader('Player Statistics')

      upsilon = st.number_input(r'Select a $\upsilon$ value'
                        , key = 'upsilon'
                        , min_value = float(st.session_state.params['options']['upsilon']['min'])
                        , value = float(st.session_state.params['options']['upsilon']['default'])
                      , max_value = float(st.session_state.params['options']['upsilon']['max']))
      upsilon_str = r'''Injury rates are scaled down by $\upsilon$. For example, if a player is expected to 
                    miss $20\%$ of games and $\upsilon$ is $75\%$, then it will be assumed that they miss 
                    $15\%$ of games instead'''
      st.caption(upsilon_str)


      psi = st.number_input(r'Select a $\psi$ value'
                        , key = 'psi'
                        , min_value = float(st.session_state.params['options']['psi']['min'])
                        , value = float(st.session_state.params['options']['psi']['default'])
                      , max_value = float(st.session_state.params['options']['psi']['max']))
      psi_str = r'''It it assumed that of the games a player will miss, 
                    they are replaced by a replacement-level player for $\psi \%$ of them'''
    
      st.caption(psi_str)

      mult_col, coef_col = st.columns(2)
    
      st.subheader(f"Coefficients")
      coefficient_series = pd.Series(st.session_state.params['coefficients'])
      conversion_factors = st.data_editor(coefficient_series, hide_index = True)
      conversion_factors = conversion_factors.T                                                      

      st.caption('Conversion factor for translating from σ² to 𝜏² as defined in the paper. Player stats are input as averages rather than week-by-week numbers, so 𝜏² must be estimated. The default conversion factors from σ² to 𝜏² are based on historical values')

    with position_param_column: 

      st.subheader('Position Requirements')

      st.caption('The H-scoring algorithm will choose players assuming that its team ultimately need to fit this structure')

      left_position_col, right_position_col = st.columns(2)

      with left_position_col:

        st.write('Base positions')

        for position_code, position_info in st.session_state.params['position_structure']['base'].items():

          st.number_input(position_info['full_str']
                    , key = 'n_' + position_code
                    , value = position_defaults['base'][position_code]
                    , min_value = 0
                        )
        
      with right_position_col:

        st.write('Flex positions')

        for position_code, position_info in st.session_state.params['position_structure']['flex'].items():

          st.number_input(position_info['full_str']
                    , key = 'n_' + position_code
                    , value = position_defaults['flex'][position_code]
                    , min_value = 0
                        )
          
      implied_n_picks = sum(n for n in get_position_numbers().values())
      
      if implied_n_picks > st.session_state.n_picks:
        st.error('There are more position slots than picks in your league. Change your configuration before proceeding')
        st.stop()

      elif implied_n_picks < st.session_state.n_picks:
        st.error('There are fewer position slots than picks in your league. Change your configuration before proceeding')
        st.stop()

    with algorithm_param_column:
        
        st.subheader('Algorithm Parameters')

        omega = st.number_input(r'Select a $\omega$ value'
                              , key = 'omega'
                              , value = punting_levels[punting_level]['omega']
                              , min_value = float(st.session_state.params['options']['omega']['min'])
                              , max_value = float(st.session_state.params['options']['omega']['max']))
        omega_str = r'''The higher $\omega$ is, the more aggressively the algorithm will try to punt. Slightly more technically, 
                        it quantifies how much better the optimal player choice will be compared to the player that would be 
                        chosen with baseline weights'''
        st.caption(omega_str)

        if omega > 0:
      
          gamma = st.number_input(r'Select a $\gamma$ value'
                                , key = 'gamma'
                                , value = punting_levels[punting_level]['gamma']
                                , min_value = float(st.session_state.params['options']['gamma']['min'])
                                , max_value = float(st.session_state.params['options']['gamma']['max']))
          gamma_str = r'''$\gamma$ also influences the level of punting, complementing omega. Tuning gamma is not suggested but you can 
                  tune it if you want. Higher values imply that the algorithm will have to give up more general value to find the
                  players that  work best for its strategy'''
          st.caption(gamma_str)

        else: 
          gamma = None

        if omega > 0:
    
          n_iterations = st.number_input(r'Select a number of iterations for gradient descent to run'
                                    , key = 'n_iterations'
                                    , value = st.session_state.params['options']['n_iterations']['default']
                                    , min_value = st.session_state.params['options']['n_iterations']['min']
                                    , max_value = st.session_state.params['options']['n_iterations']['max'])
          n_iterations_str = r'''More iterations take more computational power, but theoretically achieve better convergence'''
          st.caption(n_iterations_str)
        else:
          n_iterations = 0 


        chi = st.number_input(r'Select a $\chi$ value'
                , key = 'chi'
                , value = float(st.session_state.params['options']['chi']['default'])
                , min_value = float(st.session_state.params['options']['chi']['min'])
                , max_value = float(st.session_state.params['options']['chi']['max']))
        
        chi_str = r'''The relative variance compared to week-to-week variance to use for Rotisserie. 
                      If performance means were known exactly beforehand, chi would be 1/M where M 
                      is the number weeks in the season. However, in practice, season-long means are 
                      not known before the season begins, so it is recommended to set chi to be higher 
                      '''
        st.caption(chi_str)
        
        if st.session_state['mode'] == 'Auction Mode':

          streaming_noise = st.number_input(r'Select an $S_{\sigma}$ value'
                                    , key = 'streaming_noise'
                                    , value = 1.0
                                    , min_value = 0.0
                                    , max_value = None)
          stream_noise_str = r'''$S_{\sigma}$ controls the SAVOR algorithm. When it is high, 
                                more long-term performance noise is expected, making low-value 
                                players more heavily down-weighted due to the possibility that
                                they drop below  streaming-level value'''
          st.caption(stream_noise_str)         

          streaming_noise_h = st.number_input(r'Select an $H_{\sigma}$ value'
                        , key = 'streaming_noise_h'
                        , value = 0.1
                        , min_value = 0.0
                        , max_value = None)

          stream_noise_str_h = r'''$H_{\sigma}$ controls the SAVOR algorithm for H-scores''' 
          st.caption(stream_noise_str_h)         

    with trade_param_column:
        st.subheader('Trading Parameters')

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

  if st.session_state['schedule']: 
    stat_tab, injury_tab, games_played_tab = st.tabs([
                "Player Stats"
                ,"Injury Status"
                ,"Game Volume"
                ])
  else:
    stat_tab, injury_tab = st.tabs([
                    "Player Stats"
                    ,"Injury Status"])


  with stat_tab:
    st.caption(f"Per-game player projections below, from default data source. Edit as you see fit")

    player_stats_editable = make_data_editor(raw_stats_df
                                        , key_name = 'player_stats_editable'
                                        , lock_in_button_str = "Lock in Player Stats")
    player_stats = player_stats_editable.copy()

    #re-adjust from user inputs
    counting_statistics = st.session_state.params['counting-statistics'] 
    volume_statistics = [ratio_stat_info['volume-statistic'] for ratio_stat_info in st.session_state.params['ratio-statistics'].values()]
    ratio_statistics = get_ratio_statistics()

    player_stats[ratio_statistics + ['Games Played %']] = player_stats[ratio_statistics + ['Games Played %']]/100
    #make the upsilon adjustment
    player_stats['Games Played %'] = 100 - ( 100 - player_stats['Games Played %']) * upsilon 

    for col in counting_statistics + volume_statistics:
      player_stats[col] = player_stats_editable[col] * player_stats['Games Played %']/100 * get_games_per_week()


  with injury_tab:
      st.caption(f"List of players that you think will be injured for the foreseeable future, and so should be ignored")
      default_injury_list = [p for p in st.session_state['injured_players'] \
                              if (p in player_stats.index) and (not (p in listify(st.session_state.selections_default))) 
                              ]
      
      injured_players = st.multiselect('Injured players'
                              , player_stats.index
                              , default = default_injury_list
                              , on_change = increment_player_stats_version)

      player_stats = player_stats.drop(injured_players)

      st.session_state.player_stats = player_stats

      info = process_player_data(None
                              ,player_stats
                              ,conversion_factors
                              ,upsilon
                              ,psi
                              ,st.session_state.n_drafters
                              ,st.session_state.n_picks
                              ,st.session_state.params
                              ,st.session_state.player_stats_editable_version + st.session_state.player_stats_default_key)
      st.session_state.info = info #useful for testing

      mov = info['Mov']
      vom = info['Vom']
      
      v = np.sqrt(mov/vom)  if scoring_format == 'Rotisserie' else  np.sqrt(mov/(mov + vom))

      v = np.array(v/v.sum()).reshape(1,len(v))
      
      st.session_state.v = v
      st.session_state.z_scores = info['Z-scores']
      st.session_state.g_scores = info['G-scores']

  if st.session_state['schedule']:
    with games_played_tab: 

      #get schedule: 
      #get game weeks: with yfpy query.get_game_weeks_by_game_id
      schedule = st.session_state['schedule'] 
      week_chosen = st.selectbox('Select a particular fantasy week'
                                ,list(schedule.keys()))

      default_potential_games = schedule[week_chosen].reindex(player_stats.index).fillna(3)
      game_stats = pd.DataFrame({ 'Potential Games' : default_potential_games
                                  }
                                  ,index = player_stats.index
                                    )

      st.caption(f"""Projections for games played below, broken down by number of potential games.
                    Just for reference, these do not effect projections""")
                    
      st.dataframe(game_stats)

if 'selections_df' not in st.session_state:
  st.session_state.selections_df = st.session_state.selections_default

with rank_tab:
    z_rank_tab, g_rank_tab, h_rank_tab = st.tabs(['Z-score','G-score','H-score'])
      
    with z_rank_tab:
        
        z_scores = st.session_state.z_scores.copy()

        if st.session_state.mode == 'Auction Mode':

          z_scores.loc[:,'$ Value'] = savor_calculation(z_scores['Total']
                                                                , st.session_state.n_picks * st.session_state.n_drafters
                                                                , 200 * st.session_state.n_drafters
                                                                , st.session_state['streaming_noise'])
          
        make_rank_tab(z_scores
                      , st.session_state.params['z-score-player-multiplier']
                      , st.session_state.info_key)  

    with g_rank_tab:

        g_scores = st.session_state.g_scores.copy()

        if st.session_state.mode == 'Auction Mode':

          g_scores.loc[:,'$ Value'] = savor_calculation(g_scores['Total']
                                                                , st.session_state.n_picks * st.session_state.n_drafters
                                                                , 200 * st.session_state.n_drafters
                                                                , st.session_state['streaming_noise'])
        make_rank_tab(g_scores
                      , st.session_state.params['g-score-player-multiplier']
                      , st.session_state.info_key)  

    with h_rank_tab:
      rel_score_string = 'Z-scores' if rotisserie else 'G-scores'

      if st.session_state['mode'] == 'Auction Mode':
        taken_str = 'for free'
      else:
        taken_str = 'with the first overall pick'

      if scoring_format == 'Rotisserie':
        first_str = """Rankings are based on estimates of win probability against a field of 
                  eleven opposing teams given the candidate player is taken """ + taken_str + """ and 
                  future picks are adjusted accordingly. Corresponding category scores, based on the
                  probability of scoring a point against in arbitrary opponent, are calculated with 
                  H-scoring adjustments incorporated."""
      elif scoring_format == 'Head to Head: Most Categories': 
        first_str = """Rankings are based on estimates of overall win probability for an arbitrary head to head 
                matchup, given the candidate player is taken """ + taken_str + """ and future picks are adjusted 
                accordingly. Corresponding category scores are calculated with H-scoring adjustments incorporated."""
      elif scoring_format == 'Head to Head: Each Category': 
        first_str = """Rankings are based on estimates of mean category win probability for an arbitrary head to head 
                matchup, given the candidate player is taken """ + taken_str + """ and future picks are adjusted 
                accordingly. Corresponding category scores are calculated with H-scoring adjustments incorporated."""

      second_str = 'Note that these scores are unique to the ' + scoring_format + \
                ' format and all the H-scoring parameters defined on the parameter tab'
      
      if st.session_state['mode'] == 'Auction Mode':
        third_str = r'. $ values assume 200 per drafter'
      else: 
        third_str = ''

      st.caption(first_str + ' ' + second_str + third_str)

      h_ranks = make_h_rank_tab(info
                    ,omega
                    ,gamma
                    ,st.session_state.n_picks
                    ,st.session_state.n_drafters
                    ,n_iterations
                    ,scoring_format
                    ,st.session_state['mode']
                    ,psi
                    ,upsilon
                    ,chi
                    ,st.session_state.info_key)
      
      st.session_state.h_ranks = h_ranks

H = HAgent(info = info
    , omega = omega
    , gamma = gamma
    , n_picks = st.session_state.n_picks
    , n_drafters = st.session_state.n_drafters
    , dynamic = n_iterations > 0
    , scoring_format = scoring_format
    , chi = chi )

if st.session_state['mode'] == 'Draft Mode':

  if 'row' not in st.session_state:
    st.session_state.row = 0

  if 'drafter' not in st.session_state:
    st.session_state.drafter = 0

  with draft_tab:

    if data_source == 'Enter your own data':
      make_drafting_tab_own_data(H)
    else:
      make_drafting_tab_live_data(H)
    
if st.session_state['mode'] == 'Auction Mode':

  with auction_tab:

    if data_source == 'Enter your own data':

      left, right = st.columns([0.4,0.6])

      with left:

        cash_per_team = st.number_input(r'How much cash does each team have to pick players?'
                  , key = 'cash_per_team'
                  , min_value = 1
                  , value = 200)

        teams = ['Team ' + str(n) for n in range(st.session_state.n_drafters)]

        auction_selections_default = pd.DataFrame([[None] * 3] * st.session_state.n_picks * st.session_state.n_drafters
                                          ,columns = ['Player','Team','Cost'])

        player_category_type = CategoricalDtype(categories=list(raw_stats_df.index), ordered=True)

        auction_selections_default.loc[:'Player'] = \
            auction_selections_default.loc[:'Player'].astype(player_category_type)

        st.caption("""Enter which players have been selected by which teams, and for how much, below""")

        auction_selections = st.data_editor(auction_selections_default
                      ,column_config = 
                      {"Player" : st.column_config.SelectboxColumn(options = list(raw_stats_df.index))
                      ,"Team" : st.column_config.SelectboxColumn(options = teams)
                      ,'Cost' : st.column_config.NumberColumn(min_value = 0
                                                            , step = 1)}
                      , hide_index = True
                      , use_container_width = True
                      )

        selection_list = auction_selections['Player'].dropna()

        total_cash = cash_per_team * st.session_state.n_drafters

        amount_spent = auction_selections['Cost'].dropna().sum()

        remaining_cash = total_cash - amount_spent
        
        st.caption(r'\$' + str(remaining_cash) + r' remains out of \$' + str(total_cash) + ' originally available' )

      with right: 
        auction_seat = st.selectbox(f'Which team are you?'
            , teams
            , index = 0)
        
        cash_spent_per_team = auction_selections.dropna().groupby('Team', observed = False)['Cost'].sum()
        cash_remaining_per_team = cash_per_team - cash_spent_per_team
        player_assignments = auction_selections.dropna().groupby('Team', observed = False)['Player'].apply(list)

        for team in teams:
          if not team in cash_remaining_per_team.index:
            cash_remaining_per_team.loc[team] = cash_per_team

          if not team in player_assignments.index:
            player_assignments.loc[team] = []

        my_players = player_assignments[auction_seat]
        n_my_players = len(my_players)

        my_remaining_cash = cash_remaining_per_team[auction_seat]

        st.caption(r'You have \$' + str(my_remaining_cash) + r' remaining out of \$' + str(cash_per_team) \
                + ' to select ' + str(st.session_state.n_picks - n_my_players) + ' of ' + str(st.session_state.n_picks) + ' players')

        cand_tab, team_tab = st.tabs(["Candidates","Team"])
              
        with cand_tab:

          z_cand_tab, g_cand_tab, h_cand_tab = st.tabs(["Z-score", "G-score", "H-score"])
                    
          with z_cand_tab:
            
            z_scores_unselected = make_cand_tab(st.session_state.z_scores
                          ,selection_list
                          , st.session_state.params['z-score-player-multiplier']
                          ,remaining_cash
                          ,st.session_state.n_drafters * st.session_state.n_picks
                          ,st.session_state.info_key)

          with g_cand_tab:

            g_scores_unselected = make_cand_tab(st.session_state.g_scores
                          , selection_list
                          , st.session_state.params['g-score-player-multiplier']
                          ,remaining_cash
                          ,st.session_state.n_drafters * st.session_state.n_picks
                          ,st.session_state.info_key)

          with h_cand_tab:

            if len(my_players) == st.session_state.n_picks:
              st.markdown('Team is complete!')
                      
            elif not st.session_state.run_h_score:
              with st.form(key='my_form_to_submit'):
                h_score_button = st.form_submit_button(label='Run H-score algorithm'
                                                    , on_click = run_h_score) 
                        
            else:

              #record the fact that the run has already been invoked, no need to invoke it again
              stop_run_h_score()

              h_ranks_unselected = h_ranks[~h_ranks.index.isin(selection_list)]
              h_defaults_savor = savor_calculation(h_ranks_unselected['H-score']
                                                            , st.session_state.n_picks * st.session_state.n_drafters - len(selection_list)
                                                            , remaining_cash
                                                            , st.session_state['streaming_noise_h'])
              
              h_defaults_savor = pd.Series(h_defaults_savor.values, index = h_ranks_unselected['Player'])

              make_h_cand_tab(H
                    ,st.session_state.g_scores
                    ,st.session_state.z_scores
                    ,player_assignments.to_dict()
                    ,auction_seat
                    ,n_iterations
                    ,v
                    ,5 #display frequency
                    ,cash_remaining_per_team.to_dict()
                    ,h_defaults_savor
                    ,st.session_state.n_drafters * st.session_state.n_picks)

        with team_tab:

          base_h_score = None
          base_win_rates = None

          make_full_team_tab(st.session_state.z_scores
                            ,st.session_state.g_scores
                            ,my_players
                            ,st.session_state.n_drafters
                            ,st.session_state.n_picks
                            ,base_h_score
                            ,base_win_rates
                            ,st.session_state.info_key
                            )
    else:
      make_auction_tab_live_data(H) 

elif st.session_state['mode'] == 'Season Mode':

  with rosters_tab:

    left, right = st.columns([0.5,0.5])
      
    with left:

      st.caption("""Enter which player is on which team below""")
      selections_df = st.data_editor(st.session_state.selections
                                        , hide_index = True
                                        , height = st.session_state.n_picks * 35 + 50)  
      selection_list = listify(selections_df)

      player_assignments = selections_df.to_dict('list')

      z_scores_unselected = st.session_state.z_scores[~st.session_state.z_scores.index.isin(selection_list)]
      g_scores_unselected = st.session_state.g_scores[~st.session_state.g_scores.index.isin(selection_list)]

      with right: 

        roster_inspection_seat = st.selectbox(f'Which team do you want to get aggregated statistics for?'
        , selections_df.columns
        , index = 0)

        inspection_players = selections_df[roster_inspection_seat].dropna()

        if len(inspection_players) == st.session_state.n_picks:

          base_h_res = get_base_h_score(info
                                        ,omega
                                        ,gamma
                                        ,st.session_state.n_picks
                                        ,st.session_state.n_drafters
                                        ,scoring_format
                                        ,chi
                                        ,player_assignments
                                        ,roster_inspection_seat
                                        ,st.session_state.info_key)

          base_h_score = base_h_res['Scores']
          base_win_rates = base_h_res['Rates']

        else:
          base_h_score = None
          base_win_rates = None

        make_full_team_tab(st.session_state.z_scores
                            ,st.session_state.g_scores
                            ,inspection_players
                            ,st.session_state.n_drafters
                            ,st.session_state.n_picks
                            ,base_h_score
                            ,base_win_rates
                            ,st.session_state.info_key
                            )

  with matchup_tab:

    if (mode == 'Draft Mode') or not st.session_state.schedule:

      if scoring_format == 'roto':
        st.write('No matchups for Rotisserie')
      else:
        make_matchup_matrix(info['X-scores']
                        ,selections_df
                        ,scoring_format
                        ,st.session_state.info_key
                        )

    else:

        c1, c2, c3 = st.columns(3)

        with c1:
          matchup_seat = st.selectbox(f'Which team do you want to get expected matchup results for?'
                                            , selections.columns
                                            , index = 0)
        
        with c2:
          matchup_week = st.selectbox(f'For which week?'
                                    , st.session_state['schedule'].keys()
                                    , index = 0)

          week_number = int(matchup_week.split(':')[0].split(' ')[1])

          relevant_matchups = st.session_state['matchups'][matchup_seat]

        if week_number in relevant_matchups.keys():

          opponent_seat = relevant_matchups[int(week_number)].teams[1].name.decode('UTF-8')

          st.write(matchup_seat + "'s opponent for week " + str(week_number) + " is " + \
                  opponent_seat)

        else:
          with c3: 
            opponent_seat = st.selectbox(f'Against which team?'
                                              , [s for s in st.session_state.selections.columns if s != matchup_seat]
                                              , index = 0)
        st.write("""Predicted win likelihoods for """ + matchup_seat + """ below. Note that these reflect the 
                  expected game volume for each player based on the NBA's schedule""")

        make_matchup_tab(player_stats
                        , st.session_state.selections
                        , matchup_seat
                        , opponent_seat
                        , matchup_week
                        , st.session_state.n_picks
                        , st.session_state.n_drafters
                        , conversion_factors
                        , psi
                        , st.session_state.nu
                        , scoring_format )
        ######## END TAB
  with waiver_tab:

      c1, c2 = st.columns([0.5,0.5])

      with c1: 
        waiver_inspection_seat = st.selectbox(f'Which team so you want to drop a player from?'
            , st.session_state.selections.columns
            , index = 0)

      with c2: 
          waiver_players = selections_df[waiver_inspection_seat].dropna()

          if len(waiver_players) < st.session_state.n_picks:
              st.markdown("""This team is not full yet!""")

          else:

            waiver_team_stats_z = st.session_state.z_scores[st.session_state.z_scores.index.isin(waiver_players)]
            waiver_team_stats_z.loc['Total', :] = waiver_team_stats_z.sum(axis = 0)

            waiver_team_stats_g = st.session_state.g_scores[st.session_state.g_scores.index.isin(waiver_players)]
            waiver_team_stats_g.loc['Total', :] = waiver_team_stats_g.sum(axis = 0)

            if rotisserie:
              worst_player = list(st.session_state.z_scores.index[st.session_state.z_scores.index.isin(waiver_players)])[-1]
            else:
              worst_player = list(st.session_state.g_scores.index[st.session_state.g_scores.index.isin(waiver_players)])[-1]

            default_index = list(waiver_players).index(worst_player)

            drop_player = st.selectbox(
              'Which player are you considering dropping?'
              ,waiver_players
              ,index = default_index
            )

      if len(waiver_players) == st.session_state.n_picks:

        mod_waiver_players = [x for x in waiver_players if x != drop_player]

        z_waiver_tab, g_waiver_tab, h_waiver_tab = st.tabs(['Z-score','G-score','H-score'])

        with z_waiver_tab:

            st.markdown('Projected team stats with chosen player:')
            make_waiver_tab(z_scores
                          , selection_list
                          , waiver_team_stats_z
                          , drop_player
                          , st.session_state.params['z-score-team-multiplier']
                          , st.session_state.info_key)

        with g_waiver_tab:

            st.markdown('Projected team stats with chosen player:')
            make_waiver_tab(st.session_state.g_scores
                          , selection_list
                          , waiver_team_stats_g
                          , drop_player
                          , st.session_state.params['g-score-team-multiplier']
                          , st.session_state.info_key)

        with h_waiver_tab:

            base_h_res = get_base_h_score(info
                            ,omega
                            ,gamma
                            ,st.session_state.n_picks
                            ,st.session_state.n_drafters
                            ,scoring_format
                            ,chi
                            ,player_assignments
                            ,waiver_inspection_seat
                            ,st.session_state.info_key)

            waiver_base_h_score = base_h_res['Scores']
            waiver_base_win_rates = base_h_res['Rates']

            make_h_waiver_df(H
                        , player_stats
                        , mod_waiver_players
                        , drop_player
                        , player_assignments
                        , waiver_inspection_seat
                        , waiver_base_h_score
                        , waiver_base_win_rates
                        , st.session_state.info_key)

  with trade_tab:

    c1, c2 = st.columns([0.5,0.5])

    with c1: 
      trade_party_seat = st.selectbox(f'Which team do you want to trade from?'
          , selections_df.columns
          , index = 0)

    with c2: 
      trade_party_players = selections_df[trade_party_seat].dropna()

      if len(trade_party_players) < st.session_state.n_picks:
          st.markdown("""This team is not full yet! Fill it and other teams out on the 
                      "Teams" page then come back here""")

      else:
        
        counterparty_players_dict = { team : players for team, players in selections_df.items() 
                                if ((team != trade_party_seat) & (not  any(p!=p for p in players)))
                                  }
        
        if len(counterparty_players_dict) >=1:

          trade_counterparty_seat = st.selectbox(
              f'Which team do you want to trade with?',
              [s for s in counterparty_players_dict.keys()],
              index = 0
            )
          
          trade_counterparty_players = counterparty_players_dict[trade_counterparty_seat]

        else: 
          trade_counterparty_players = []

    if len(trade_party_players) == st.session_state.n_picks:

      if len(trade_counterparty_players) < st.session_state.n_picks:
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

            z_tab, g_tab, h_tab = st.tabs(['Z-score','G-score','H-score'])

            if (len(players_sent) == 0) | (len(players_received) == 0):
              st.markdown('A trade must include at least one player from each team')

            else:

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
              with h_tab:
                make_trade_h_tab(H
                                , player_stats 
                                , player_assignments 
                                , n_iterations 
                                , trade_party_seat
                                , players_sent
                                , trade_counterparty_seat
                                , players_received
                                , scoring_format
                                , st.session_state.info_key)


        with destinations_tab:

          values_to_team = make_trade_destination_display(H
                                , player_stats
                                , player_assignments 
                                , trade_party_seat 
                                , scoring_format
                                , st.session_state.info_key
                                      )

        with target_tab:

          values_to_me = make_trade_target_display(H
                , player_stats
                , trade_party_seat
                , trade_counterparty_seat
                , player_assignments
                , values_to_team[trade_counterparty_seat]
                , scoring_format
                , st.session_state.info_key
                      )

          with suggestions_tab:

            if rotisserie:
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

            trade_filter = st.multiselect('Which kinds of trades do you want get suggestions for?'
                                      , [(1,1),(2,2),(3,3)]
                                      , format_func = lambda x: str(x[0]) + ' for ' + str(x[1])
                                      , default = [(1,1)])

            make_trade_suggestion_display(H
                , player_stats 
                , player_assignments
                , trade_party_seat
                , trade_counterparty_seat
                , general_value
                , replacement_value
                , values_to_me
                , values_to_team[trade_counterparty_seat]
                , your_differential_threshold
                , their_differential_threshold
                , combo_params
                , trade_filter
                , scoring_format
                , st.session_state.info_key)               

with about_tab:

  tabs = st.tabs(['Intro'
                  ,'G-scoring'
                  ,'H-scoring'
                  ,'Rotisserie'
                  ,'Turnovers'
                  ,'Injuries'
                  ,'Auctions'
                  ,'Waivers & Trading'
                  ,'Data Sources'])

  about_paths = ['intro.md'
                ,'static_explanation.md'
                ,'dynamic_explanation.md'
                ,'rotisserie.md'
                ,'turnovers.md'
                ,'injury.md'
                ,'auctions.md'
                ,'trading.md'
                ,'data_sources.md']

  for tab, path in zip(tabs, about_paths):
    with tab:
      make_about_tab(path)   

