import streamlit as st
import pandas as pd
import numpy as np
import os 
from typing import Callable
import yaml
from yfpy.models import League
import time
from schedule import every, repeat, run_pending
from src.helpers.helper_functions import  get_position_numbers, listify \
                                  ,increment_player_stats_version, increment_info_key, increment_default_key \
                                  ,get_games_per_week, get_categories, get_ratio_statistics, get_selections_default
from src.data_retrieval.get_data import get_historical_data, get_current_season_data, get_darko_data, get_specified_stats, \
                        get_player_metadata, get_data_from_snowflake, combine_nba_projections
from src.math.process_player_data import process_player_data
from src.math.algorithm_agents import HAgent
from src.tabs.ranks import make_rank_tab, make_h_rank_tab
from src.tabs.trading import make_trade_tab
from src.helpers.data_editor import make_data_editor
from src.tabs.drafting import make_drafting_tab_own_data, make_drafting_tab_live_data, make_auction_tab_live_data \
                          ,make_auction_tab_own_data, increment_and_reset_draft, clear_draft_board
from src.tabs.matchups import make_matchup_tab, make_matchup_matrix
from src.tabs.team_subtabs import make_full_team_tab, roster_inspection
from src.tabs.candidate_subtabs import make_h_waiver_df, make_waiver_tab
from src.tabs.other_tabs import make_about_tab
from src.math.algorithm_helpers import savor_calculation
from src.math.algorithm_agents import get_base_h_score
from src.platform_integration.fantrax_integration import FantraxIntegration
from src.platform_integration.yahoo_integration import YahooIntegration
from pandas.api.types import CategoricalDtype

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
    st.session_state.player_stats_default_key = np.random.randint(100000)

if 'info_key' not in st.session_state:
    st.session_state.info_key = 100000

if 'injured_players' not in st.session_state:
    st.session_state['injured_players'] = set()

if 'schedule' not in st.session_state:
    st.session_state['schedule'] = {}

if 'mode' not in st.session_state:
    st.session_state['mode'] = 'Draft Mode'

if 'have_locked_in' not in st.session_state:
  st.session_state.have_locked_in = False

if 'live_draft_active' not in st.session_state:
    st.session_state.live_draft_active = False

if 'yahoo_league_id' not in st.session_state:
    st.session_state.yahoo_league_id = None

if 'draft_results' not in st.session_state:
    st.session_state.draft_results = None

if 'run_h_score' not in st.session_state:
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
        , on_change = increment_and_reset_draft
        )
      
      load_params(st.session_state.league)

      data_source = st.selectbox(
        'Do you want to integrate with a fantasy platform?'
        , ['Enter your own data', 'Retrieve from Yahoo Fantasy','Retrieve from Fantrax']
        , key = 'data_source'
        , on_change = increment_and_reset_draft
        , index = 0)

      # Setting default values
      #st.session_state.n_drafters = st.session_state.params['options']['n_drafters']['default']
      #st.session_state.n_picks = st.session_state.params['options']['n_picks']['default']

      #These are based on 2023-2024 excluding injury
      #might need to modify these at some point? 

      
      if st.session_state.data_source == 'Enter your own data':
        mode_options = ('Draft Mode', 'Auction Mode','Season Mode')      

      elif st.session_state.data_source == 'Retrieve from Yahoo Fantasy':
        mode_options = YahooIntegration().get_available_modes()

      elif st.session_state.data_source == 'Retrieve from Fantrax':
        mode_options = FantraxIntegration().get_available_modes()

      mode = st.selectbox(
        'Which mode do you want to use?'
        , mode_options
        , index = 0
        , key = 'mode'
        , on_change = increment_and_reset_draft)
        
      if data_source == 'Enter your own data':
        n_drafters = st.number_input(r'How many drafters are in your league?'
                                      , key = 'n_drafters'
                                      , min_value = st.session_state.params['options']['n_drafters']['min']
                                      , value = st.session_state.params['options']['n_drafters']['default']
                                      , on_change = clear_draft_board
                                      )

        n_picks = st.number_input(r'How many players will each drafter choose?'
                      , key = 'n_picks'
                      , min_value = st.session_state.params['options']['n_picks']['min']
                      , value = st.session_state.params['options']['n_picks']['default']
                      , on_change = clear_draft_board)
        
        st.write('Enter team names here:')
        
        team_df = st.data_editor(pd.DataFrame({'Team ' + str(i) : ['Drafter ' + str(i)] for i in range(n_drafters)})
                       , hide_index = True
                       , on_change = increment_and_reset_draft)
        
        st.session_state.team_names = list(team_df.iloc[0])
                             
        # perhaps the dataframe should be uneditable, and users just get to enter the next players picked? With an undo button?
        #Should this just be called if selections_df not in session state?
        st.session_state.selections_default = pd.DataFrame(
          {team : [np.nan] * st.session_state.n_picks for team in st.session_state.team_names}
          )
        
        if 'selections_df' not in st.session_state:
          st.session_state.selections_df = st.session_state.selections_default 


      else:
        
        if data_source == 'Retrieve from Yahoo Fantasy':

          yahoo_integration = YahooIntegration()
          yahoo_integration.setup()

          st.session_state.integration = yahoo_integration
                
        elif data_source == 'Retrieve from Fantrax':     

          fantrax_integration = FantraxIntegration()     

          fantrax_integration.setup()
          st.session_state.integration = fantrax_integration

        st.session_state.team_names = st.session_state.integration.get_team_names(st.session_state.integration.league_id
                                                                              ,st.session_state.integration.division_id) 
        st.session_state.n_drafters = len(st.session_state.team_names)
        st.session_state.n_picks = st.session_state.integration.get_n_picks(st.session_state.integration.league_id)

        st.session_state.selections_default = st.session_state.integration.selections_default

        st.session_state.selections_df = st.session_state.selections_default

      #set default position numbers, based on n_picks
      all_position_defaults = st.session_state.params['options']['positions']
      
      if st.session_state.n_picks in all_position_defaults:
        position_defaults = all_position_defaults[st.session_state.n_picks]
      else:
        position_defaults = all_position_defaults[st.session_state.params['options']['n_picks']['default']]

        if st.session_state.mode != 'Season Mode':
          st.error('''There is no default position structure for a league with''' + str(st.session_state.n_picks) + \
                  ''' picks. Position structure must be met manually on the Advanced tab.''')

    with c2: 
          
          scoring_format = st.selectbox(
              'Which format are you playing?',
              ('Rotisserie', 'Head to Head: Each Category', 'Head to Head: Most Categories')
              , key = 'scoring_format'
              , index = 1
              , on_change = increment_default_key)
          
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

            data_options = ['Projection','Historical'] if data_source == 'Enter your own data' else ['Projection']
      

            kind_of_dataset = st.selectbox(
                                      'Which kind of dataset do you want to use? (specify further on the data tab)'
                                      , data_options
                                      ,key = 'data_option'
                                      , index = 0
            )

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
                
            c1, c2 = st.columns([0.2,0.8])
            
            with c1: 
              submit = st.form_submit_button("Lock in",on_click = increment_and_reset_draft)
            with c2:
              st.warning('Make sure to lock in after making changes')

  with data_params:

    if st.session_state.league == 'NBA':

        #current_data, expected_minutes = get_current_season_data()
        #darko_data = get_darko_data(expected_minutes)

        #unique_datasets_current = list(current_data.keys())
        #unique_datasets_darko = list(darko_data.keys())
        if kind_of_dataset == 'Historical':
        
          historical_df = get_historical_data()

          unique_datasets_historical = reversed([str(x) for x in pd.unique(historical_df.index.get_level_values('Season'))])

          dataset_name = st.selectbox(
            'Which dataset do you want to default to?'
            ,unique_datasets_historical
            ,index = 0
            ,on_change = increment_and_reset_draft
          )
          raw_stats_df = get_specified_stats(dataset_name
                                    , st.session_state.player_stats_default_key)
            
        else: 

          with st.form('basketball_data_sources'):

            hashtag_c, roto_c, bbm_c, = st.columns(3)

            with hashtag_c:
              hashtag_upload = get_data_from_snowflake('HTB_PROJECTION_TABLE')
              hashtag_slider = st.slider('Hashtag Basketball Weight'
                                        , min_value = 0.0
                                        , value = 1.0
                                        , max_value = 1.0)

            with roto_c:

                rotowire_slider = st.slider('RotoWire Weight'
                                , min_value = 0.0
                                , max_value = 1.0)

                
                rotowire_file = st.file_uploader("Upload RotoWire data, as a csv"
                                                , type=['csv'])
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
                          , max_value = 1.0)

                bbm_file = st.file_uploader('''Upload Basketball Monster Per Game Stats, as a csv. To get all required columns for 
                                                projections, you may have to export to excel then save as CSV utf-8.'''
                                                , type=['csv'])
                if bbm_file is not None:
                  # Adding a 
                  bbm_upload  = pd.read_csv(bbm_file)
                else:
                  bbm_upload = None

            c1, c2 = st.columns([0.2,0.8])
            
            with c1: 
              submit = st.form_submit_button("Lock in",on_click = increment_default_key)
            with c2:
              st.warning('Make sure to lock in after making changes')

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
          
          rotowire_file = st.file_uploader("Upload RotoWire data, as a csv"
                                          , type=['csv'])
          if rotowire_file is not None:
            rotowire_upload  = pd.read_csv(rotowire_file, skiprows = 1)
          else:
            rotowire_upload = None
            st.error('Upload RotoWire projection file')
            st.stop()

          raw_stats_df = combine_nba_projections(None
                            , rotowire_upload
                            , None
                            , 0
                            , 1
                            , 0)

  with advanced_params:

      player_param_column, position_param_column, algorithm_param_column, trade_param_column = st.columns(4)

      with player_param_column:

        with st.form("player_stat_params"):

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

          chi = st.number_input(r'Select a $\chi$ value'
              , key = 'chi'
              , value = float(st.session_state.params['options']['chi']['default'])
              , min_value = float(st.session_state.params['options']['chi']['min'])
              , max_value = float(st.session_state.params['options']['chi']['max']))
      
          chi_str = r'''The relative variance compared to week-to-week variance to use for Rotisserie. 
                          Uncertainty in season-long means is higher than uncertainty week-over-week 
                          '''
          st.caption(chi_str)
        
          #I don't think we need people to be able to modify the coefficients
          coefficient_series = pd.Series(st.session_state.params['coefficients'])
          conversion_factors = coefficient_series.T                                                      

          #st.caption('Conversion factor for translating from σ² to 𝜏² as defined in the paper. Player stats are input as averages rather than week-by-week numbers, so 𝜏² must be estimated. The default conversion factors from σ² to 𝜏² are based on historical values')

          c1, c2 = st.columns([0.4,0.6])

          with c1:
            st.form_submit_button("Lock in")
          with c2:
            st.warning('Make sure to lock in after making changes')


      with position_param_column: 

        with st.form('position_params'):

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
            st.number_input('Bench- these players will be ignored for drafting. Not supported for other modes'
                            , key = 'n_bench'
                            , value = 0
                            , min_value = 0)  
                          
          c1, c2 = st.columns([0.4,0.6])

          with c1:
            st.form_submit_button("Lock in")
          with c2:
            st.warning('Make sure to lock in after making changes')
              
          implied_n_picks = sum(n for n in get_position_numbers().values()) + st.session_state.n_bench
          
          if (implied_n_picks != st.session_state.n_picks) & (st.session_state.mode != 'Season Mode'):
            st.error('This structure has ' + str(implied_n_picks) + ' position slots, but your league has ' + str(st.session_state.n_picks) + \
                    ' picks per manager. Adjust the position slots before proceeding')
            st.stop()

          st.session_state.n_starters = st.session_state.n_picks - st.session_state.n_bench



      with algorithm_param_column:
          
          with st.form('algo_params'):
          
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
          
            gamma = st.number_input(r'Select a $\gamma$ value'
                                  , key = 'gamma'
                                  , value = punting_levels[punting_level]['gamma']
                                  , min_value = float(st.session_state.params['options']['gamma']['min'])
                                  , max_value = float(st.session_state.params['options']['gamma']['max']))
            gamma_str = r'''$\gamma$ also influences the level of punting, complementing omega. Tuning gamma is not suggested but you can 
                    tune it if you want. Higher values imply that the algorithm will have to give up more general value to find the
                    players that  work best for its strategy'''
            st.caption(gamma_str)
        
            n_iterations = st.number_input(r'Select a number of iterations for gradient descent to run'
                                      , key = 'n_iterations'
                                      , value = punting_levels[punting_level]['n_iterations']
                                      , min_value = st.session_state.params['options']['n_iterations']['min']
                                      , max_value = st.session_state.params['options']['n_iterations']['max'])
            n_iterations_str = r'''More iterations take more computational power, but theoretically achieve better convergence'''
            st.caption(n_iterations_str)

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

            c1, c2 = st.columns([0.4,0.6])

            with c1:
              st.form_submit_button("Lock in")
            with c2:
              st.warning('Make sure to lock in after making changes')   

      with trade_param_column:
          
          with st.form('trading_form'):
            st.subheader('Trading Parameters')

            your_differential_threshold = st.number_input(
                  r'Your differential threshold for the automatic trade suggester'
                  , value = 0)
            ydt_str = r'''Only trades which improve your H-score 
                          by this percent will be shown'''
            st.caption(ydt_str)
            your_differential_threshold = your_differential_threshold /100

            their_differential_threshold = st.number_input(
                  r'Counterparty differential threshold for the automatic trade suggester'
                  , value = 0)
            tdt_str = r'''Only trades which improve their H-score 
                        by this percent will be shown'''
            st.caption(tdt_str)
            their_differential_threshold = their_differential_threshold/100

            combo_params_default = pd.DataFrame({'N-traded' : [1,2,3]
                                          ,'N-received' : [1,2,3]
                                          ,'T1' : [0.0,0.0,0.0]
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
              
            combo_params_str =  \
              """Each row is a specification for a kind of trade that will be automatically evaluated. 
              N-traded is the number of players traded from your team, and N-received is the number of 
              players to receive in the trade. T1 is a threshold of 'targetedness' as shown in the Target
              column. Only players with the specified targetedness or above will be considered- as a decimal 
              not a percentage, e.g. 0.02 instead of 2%. T2 is a 
              threshold of G-score difference (or Z-score for Rotisseries); trades that have general value 
              differences larger than T2 will not be evaluated"""
            st.caption(combo_params_str)

            combo_params = tuple(combo_params_df.itertuples(name = None, index = None))

            c1, c2 = st.columns([0.4,0.6])

            with c1:
              st.form_submit_button("Lock in")
            with c2:
              st.warning('Make sure to lock in after making changes')

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
      player_stats[col] = player_stats_editable[col].astype(float) * player_stats['Games Played %']/100 * get_games_per_week()


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
      
      #ZR: Idk why we need to drop duplicates      
      default_potential_games = schedule[week_chosen].drop_duplicates().reindex(player_stats.index).fillna(3)
      game_stats = pd.DataFrame({ 'Potential Games' : default_potential_games
                                  }
                                  ,index = player_stats.index
                                    )

      st.caption(f"""Projections for games played below, broken down by number of potential games.
                    Just for reference, these do not effect projections""")
                    
      st.dataframe(game_stats)

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
    , n_picks = st.session_state.n_starters
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

      make_auction_tab_own_data(H)
    else:
      make_auction_tab_live_data(H) 

elif st.session_state['mode'] == 'Season Mode':

  with rosters_tab:

    left, right = st.columns([0.5,0.5])
      
    with left:

      st.caption("""Enter which player is on which team below""")
      player_category_type = CategoricalDtype(categories=list(st.session_state.player_stats.index) + ['RP']
                                              , ordered=True)

      with st.form('manual_rosters'):

        selections_df = st.data_editor(st.session_state.selections_df.astype(player_category_type)
                                          , hide_index = True
                                          , height = st.session_state.n_picks * 35 + 50).fillna('RP')
        
        c1, c2 = st.columns([0.2,0.8])
            
        with c1: 
          submit = st.form_submit_button("Lock in")
        with c2:
          st.warning('Lock in to update rosters')

      selection_list = listify(selections_df)

      player_assignments = selections_df.to_dict('list')

      z_scores_unselected = st.session_state.z_scores[~st.session_state.z_scores.index.isin(selection_list)]
      g_scores_unselected = st.session_state.g_scores[~st.session_state.g_scores.index.isin(selection_list)]

      with right: 

        roster_inspection(selections_df.fillna('RP')
                        , info
                        , omega
                        , gamma
                        , scoring_format
                        , chi
                        , player_assignments)  

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
                                            , st.session_state.selections_df.columns
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
                                              , [s for s in st.session_state.selections_df.columns if s != matchup_seat]
                                              , index = 0)
        st.write("""Predicted win likelihoods for """ + matchup_seat + """ below. Note that these reflect the 
                  expected game volume for each player based on the NBA's schedule""")

        make_matchup_tab(player_stats
                        , selections_df
                        , matchup_seat
                        , opponent_seat
                        , matchup_week
                        , st.session_state.n_picks
                        , st.session_state.n_drafters
                        , conversion_factors
                        , psi
                        , scoring_format )
        ######## END TAB
  with waiver_tab:

      c1, c2 = st.columns([0.5,0.5])

      with c1: 
        waiver_inspection_seat = st.selectbox(f'Which team so you want to drop a player from?'
            , st.session_state.selections_df.columns
            , index = 0)

      with c2: 
          waiver_players = [x for x in selections_df[waiver_inspection_seat] if x != 'RP']

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

    make_trade_tab(H
                   , selections_df
                   , player_stats
                   , player_assignments
                   , z_scores_unselected
                   , g_scores_unselected
                   , combo_params
                   , rotisserie
                   , your_differential_threshold
                   , their_differential_threshold)              

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