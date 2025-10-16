import streamlit as st
import numpy as np
import yaml
from src.helpers.helper_functions import get_n_drafters
from src.helpers.stylers import DarkStyler, LightStyler
from src.math.algorithm_agents import HAgent
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
import numpy

#this reduces the padding at the top of the website, which is excessive otherwise 
st.write('<style>div.block-container{padding-top:3rem;}</style>', unsafe_allow_html=True)

### SETUP
st.set_page_config(
          layout="wide"
          , page_icon=':basketball:'
          , page_title = 'Fantasy Sports Optimization'
          , initial_sidebar_state="auto"
          , menu_items=None)

#the randints are a hack, to address a known issue with streamlit. The keys and the caches can get out of synch
#see here: https://discuss.streamlit.io/t/st-session-state-values-are-reset-on-page-reload-but-st-cache-values-are-not/18059
if 'player_stats_version' not in st.session_state:
    st.session_state.player_stats_version = 0 + np.random.randint(0,10000, size = 1) * 1000

if 'info_key' not in st.session_state:
    st.session_state.info_key = 100000 + np.random.randint(0,10000, size = 1) * 1000

if 'data_source' not in st.session_state:
    st.session_state.data_source = 'Enter your own data'

if 'datasets' not in st.session_state:
   st.session_state.datasets = {}

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

if 'res_cache' not in st.session_state:
  st.session_state.res_cache = {}

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

    player_stats = player_stats_popover()

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


### Build app 

H = HAgent(info = st.session_state.info
    , omega = st.session_state.omega
    , gamma = st.session_state.gamma
    , n_picks = st.session_state.n_starters
    , n_drafters = get_n_drafters()
    , dynamic = st.session_state.n_iterations > 0
    , scoring_format = st.session_state.scoring_format
    , chi = st.session_state.chi
    , team_names = st.session_state.team_names)

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

if st.session_state['mode'] == 'Season Mode':
  make_season_mode_tabs(H)