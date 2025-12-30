import streamlit as st
import yaml

from src.helpers.helper_functions import gen_key, get_mode, initialize_selections_df \
                                      , set_draft_position, using_manual_entry
from src.helpers.cookie_control import store_options_as_cookies, reset_all_parameters
from src.helpers.stylers import DarkStyler, LightStyler
from src.tabs.drafting import make_drafting_tab_own_data, make_drafting_tab_live_data \
                           ,make_auction_tab_live_data ,make_auction_tab_own_data, update_data_and_info
from src.tabs.season_mode import make_season_mode_tabs
from src.parameter_collection.league_settings import league_settings_popover
from src.parameter_collection.player_stats import player_stats_popover
from src.parameter_collection.parameters import player_stat_param_popover, algorithm_param_popover, trade_param_popover
from src.parameter_collection.position_requirement import position_requirement_popover
from src.parameter_collection.format import format_popover
#from wfork_streamlit_profiler import Profiler
from streamlit_theme import st_theme
import extra_streamlit_components as stx

#this reduces the padding at the top of the website, which is excessive otherwise 
st.write('<style>div.block-container{padding-top:3rem;}</style>', unsafe_allow_html=True)

### SETUP
st.set_page_config(
          layout="wide"
          , page_icon=':basketball:'
          , page_title = 'Fantasy Sports Optimization'
          , initial_sidebar_state="auto"
          , menu_items=None)

if "allow_cookie_bootstrap" not in st.session_state:
    st.session_state.allow_cookie_bootstrap = True

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

  cookies = stx.CookieManager() 
  st.session_state.saved_cookies = cookies.get_all()
  reset_param_button = st.button('Reset parameters')

  if reset_param_button:

      reset_all_parameters(cookies)

  print('Post re-run')

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

with st.empty(): #we need this in st.empty because of a streamlit bug. It will allocate space otherwise
  #store all of the user preferences as cookies. This keeps them persistent across sessions 
  store_options_as_cookies(cookies) 

initialize_selections_df()

if using_manual_entry():
  update_data_and_info()

if get_mode() == 'Draft Mode':

  if 'draft_position' not in st.session_state:
    set_draft_position(0,0)

  if using_manual_entry():
    make_drafting_tab_own_data()
  else:
    make_drafting_tab_live_data()
    
elif get_mode() == 'Auction Mode':

  if using_manual_entry():
    make_auction_tab_own_data()
  else:
    make_auction_tab_live_data()      

elif get_mode() == 'Season Mode':

  make_season_mode_tabs()