import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import streamlit as st
import numpy as np
import yaml
from src.helpers.helper_functions import gen_key, get_data_from_session_state, get_data_key, get_n_drafters, get_scoring_format, initialize_selections_df, store_dataset_in_session_state, using_manual_entry
from src.helpers.stylers import DarkStyler, LightStyler
from src.math.algorithm_agents import build_h_agent
from src.tabs.drafting import make_drafting_tab_own_data, make_drafting_tab_live_data \
                           ,make_auction_tab_live_data ,make_auction_tab_own_data
from src.tabs.season_mode import make_season_mode_tabs
from src.parameter_collection.league_settings import league_settings_popover
from src.parameter_collection.player_stats import player_stats_popover
from src.parameter_collection.parameters import player_stat_param_popover, algorithm_param_popover, trade_param_popover
from src.parameter_collection.position_requirement import position_requirement_popover
from src.parameter_collection.format import format_popover
#from wfork_streamlit_profiler import Profiler
import streamlit.components.v1 as components
from streamlit_theme import st_theme

#this reduces the padding at the top of the website, which is excessive otherwise 
st.write('<style>div.block-container{padding-top:3rem;}</style>', unsafe_allow_html=True)

### SETUP
st.set_page_config(
          layout="wide"
          , page_icon=':basketball:'
          , page_title = 'Fantasy Sports Optimization'
          , initial_sidebar_state="auto"
          , menu_items=None)

if 'data_source' not in st.session_state:
    st.session_state.data_source = 'Enter your own data'

if 'stat_options_key' not in st.session_state:
    st.session_state.stat_options_key = gen_key()

if 'injured_players' not in st.session_state:
    st.session_state['injured_players'] = set()

if 'have_locked_in' not in st.session_state:
  st.session_state.have_locked_in = False

if 'live_draft_active' not in st.session_state:
    st.session_state.live_draft_active = False

if 'draft_results' not in st.session_state:
    st.session_state.draft_results = None

if 'run_h_score' not in st.session_state:
    st.session_state.run_h_score = False

if 'data_dictionary' not in st.session_state:
  st.session_state.data_dictionary = {}

if 'all_params' not in st.session_state:
  with open("parameters.yaml", "r") as stream:
    st.session_state.all_params = yaml.safe_load(stream)

#Load up the theme and make a styler based on the theme
#st_theme sometimes fails right after the app loads, which necessitates the try-except clause 
try: 
  st.session_state.base = st_theme()['base']
  st.session_state.styler = DarkStyler() if st.session_state.base == 'dark' else LightStyler()
except: 
  st.session_state.base = 'light'
  st.session_state.styler = LightStyler()

with st.sidebar:

  st.title('🏀 Fantasy Sports Optimizer')

  st.write("---")

  with st.popover(':small[League Settings]'):

    league_settings_popover()
  
  #explicitly setting the heights helps to avoid a Streamlit bug
  #https://github.com/streamlit/streamlit/issues/8934
  with st.popover(':small[Player Stats]').container(height = 300):

    player_stats_popover()

  with st.popover(':small[Format & Categories]'):

    format_popover()
    
  with st.popover(':small[Player Stat Parameters]').container(height = 300):

    player_stat_param_popover()      

  with st.popover(':small[H-score Parameters]').container(height = 300):
                    
    algorithm_param_popover()

  with st.popover(':small[Trade Parameters]').container(height = 300):
          
    trade_param_popover()

  with st.popover(':small[Position Parameters]').container(height = 400):

    position_requirement_popover()

  st.write("---")

  st.link_button("Documentation", 'https://zer2.github.io/fantasy-basketball-optimizer/')

H, key = build_h_agent(get_data_key('info')
                  ,st.session_state.omega
                  ,st.session_state.gamma
                  ,st.session_state.n_starters
                  ,get_n_drafters()
                  ,st.session_state.beth
                  ,get_scoring_format()
                  ,st.session_state.n_iterations > 0)
store_dataset_in_session_state(H, 'H',key)

if using_manual_entry():
  initialize_selections_df()

if st.session_state['mode'] == 'Draft Mode':

  if 'row' not in st.session_state:
    st.session_state.row = 0

  if 'drafter' not in st.session_state:
    st.session_state.drafter = 0

  if using_manual_entry():
    make_drafting_tab_own_data()
  else:
    make_drafting_tab_live_data()
    
if st.session_state['mode'] == 'Auction Mode':

  if using_manual_entry():
    make_auction_tab_own_data()
  else:
    make_auction_tab_live_data()      

if st.session_state['mode'] == 'Season Mode':
  make_season_mode_tabs()