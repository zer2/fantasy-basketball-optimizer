import streamlit as st
import pandas as pd
import numpy as np
import yaml
from src.helpers.helper_functions import  get_position_numbers, listify \
                                  ,increment_player_stats_version \
                                  ,get_games_per_week, get_ratio_statistics
from src.data_retrieval.get_data import get_historical_data, get_specified_stats, \
                        get_data_from_snowflake, combine_nba_projections, get_player_metadata, get_yahoo_key_to_name_mapper
from src.math.process_player_data import process_player_data
from src.math.algorithm_agents import HAgent
from src.tabs.ranks import make_full_rank_tab
from src.tabs.trading import make_trade_tab
from src.helpers.data_editor import make_data_editor
from src.tabs.drafting import make_drafting_tab_own_data, make_drafting_tab_live_data, make_auction_tab_live_data \
                          ,make_auction_tab_own_data, increment_and_reset_draft, clear_draft_board
from src.tabs.matchups import make_matchup_tab, make_matchup_matrix
from src.tabs.team_subtabs import roster_inspection
from src.tabs.waivers import make_full_waiver_tab
from src.tabs.other_tabs import make_about_tab
from src.parameter_collection.league_settings import league_settings_popover
from src.parameter_collection.player_stats import player_stats_popover
from src.parameter_collection.parameters import player_stat_param_popover, algorithm_param_popover, trade_param_popover
from src.parameter_collection.position_requirement import position_requirement_popover
from src.parameter_collection.format import format_popover
from pandas.api.types import CategoricalDtype

st.write('<style>div.block-container{padding-top:3rem;}</style>', unsafe_allow_html=True)

### SETUP
st.set_page_config(
          layout="wide"
          , page_icon=':basketball:'
          , page_title = 'Fantasy Sports Optimization'
          , initial_sidebar_state="auto"
          , menu_items=None)

if 'player_stats_version' not in st.session_state:
    st.session_state.player_stats_version = 0

if 'info_version' not in st.session_state:
    st.session_state.info_key = 100000

if 'data_source' not in st.session_state:
    st.session_state.data_source = 'Enter Your Own Data'


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

with st.sidebar:

  st.title('🏀 Fantasy Sports Optimizer')

  st.write("---")

  with st.popover(':small[League Settings]'):

    league_settings_popover()
  
  with st.popover(':small[Player Stats]').container(height = 500):

    player_stats = player_stats_popover()

  with st.popover(':small[Format & Categories]'):

    format_popover()
    
  with st.popover(':small[Player Stat Parameters]'):

    player_stat_param_popover()                                          

  with st.popover(':small[Algorithm Parameters]').container(height = 300):
                    
    algorithm_param_popover()

  with st.popover(':small[Trade Parameters]').container(height = 300):
          
    trade_param_popover()

  with st.popover(':small[Position Requirements]'):

    position_requirement_popover()

  st.write("---")

  st.caption('Algorithm sources:')

  st.link_button(":small[Paper 1: G-scores]", 'https://arxiv.org/abs/2307.02188')
  st.link_button(':small[Paper 2: H-scores]', 'https://arxiv.org/abs/2409.09884')
  st.link_button(':small[Paper 3: Roto]', 'https://arxiv.org/abs/2501.00933')

### Build app 

                
if st.session_state['mode'] == 'Season Mode':
  main_tabs = st.tabs(["🏟️ Rosters"
                  ,"⚔️ Matchups"
                  ,"⛹️‍♂️ Waiver Wire & Free Agents"
                  ,"📋 Trading"])

  rosters_tab = main_tabs[1]
  matchup_tab = main_tabs[2]
  waiver_tab = main_tabs[3]
  trade_tab = main_tabs[4]

mov = st.session_state.info['Mov']
vom = st.session_state.info['Vom']

v = np.sqrt(mov/vom)  if st.session_state.scoring_format == 'Rotisserie' else  np.sqrt(mov/(mov + vom))

v = np.array(v/v.sum()).reshape(1,len(v))

st.session_state.v = v
st.session_state.z_scores = st.session_state.info['Z-scores']
st.session_state.g_scores = st.session_state.info['G-scores']

H = HAgent(info = st.session_state.info
    , omega = st.session_state.omega
    , gamma = st.session_state.gamma
    , n_picks = st.session_state.n_starters
    , n_drafters = st.session_state.n_drafters
    , dynamic = st.session_state.n_iterations > 0
    , scoring_format = st.session_state.scoring_format
    , chi = st.session_state.chi )

if st.session_state['mode'] == 'Draft Mode':

  if 'row' not in st.session_state:
    st.session_state.row = 0

  if 'drafter' not in st.session_state:
    st.session_state.drafter = 0

  if st.session_state.data_source == 'Enter your own data':
    make_drafting_tab_own_data(H)
  else:
    make_drafting_tab_live_data(H)
    
if st.session_state['mode'] == 'Auction Mode':

  if st.session_state.data_source == 'Enter your own data':

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
                        , st.session_state.info
                        , st.session_state.omega
                        , st.session_state.gamma
                        , st.session_state.scoring_format
                        , st.session_state.chi
                        , player_assignments)  

  with matchup_tab:

    if (st.session_state.mode == 'Draft Mode') or not st.session_state.schedule:

      if st.session_state.scoring_format == 'Rotisserie':
        st.write('No matchups for Rotisserie')
      else:
        make_matchup_matrix(st.session_state.info['X-scores']
                        ,selections_df
                        ,st.session_state.scoring_format
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

        make_matchup_tab(st.session_state.player_stats
                        , selections_df
                        , matchup_seat
                        , opponent_seat
                        , matchup_week
                        , st.session_state.n_picks
                        , st.session_state.n_drafters
                        , st.session_state.conversion_factors
                        , st.session_state.psi
                        , st.session_state.scoring_format )
        ######## END TAB
  with waiver_tab:

      make_full_waiver_tab(H
                           ,selections_df
                           ,player_assignments
                           ,selection_list)

  with trade_tab:

    make_trade_tab(H
                   , selections_df
                   , player_assignments
                   , z_scores_unselected
                   , g_scores_unselected)              
