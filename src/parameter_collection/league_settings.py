import streamlit as st

from src.platform_integration.fantrax_integration import FantraxIntegration
from src.platform_integration.yahoo_integration import YahooIntegration
from src.platform_integration.espn_integration import ESPNIntegration

from src.tabs.drafting import increment_and_reset_draft

from src.data_retrieval.get_data import get_player_metadata, get_yahoo_key_to_name_mapper

def league_settings_popover():
    #collect settings for the league and set up a platform integration if necessary

    league = st.selectbox(
            'Which fantasy sport are you playing?',
            ('NBA','MLB') #WNBA excluded for now
            , index = 0
            , key = 'league'
            , on_change = increment_and_reset_draft
            )
          
    st.session_state.params = st.session_state.all_params[league]

    integrations = {integration.get_description_string() : integration for integration in \
                    [YahooIntegration(), FantraxIntegration(), ESPNIntegration()]}

    data_source = st.selectbox(
      'Do you want to integrate with a fantasy platform?'
      , ['Enter your own data'] + list(integrations.keys())
      , key = 'data_source'
      , on_change = increment_and_reset_draft
      , index = 0)

    if st.session_state.data_source == 'Enter your own data':
      mode_options = ('Draft Mode', 'Auction Mode','Season Mode')      

    else:
      st.session_state.integration = integrations[data_source]
      mode_options = st.session_state.integration.get_available_modes()

    mode = st.selectbox(
      'Which mode do you want to use?'
      , mode_options
      , index = 0
      , key = 'mode'
      , on_change = increment_and_reset_draft)
      
    if not data_source == 'Enter your own data':
      
      st.session_state.integration.setup()

      st.session_state.team_names = st.session_state.integration.get_team_names(st.session_state.integration.league_id
                                                                            ,st.session_state.integration.division_id) 
      st.session_state.n_drafters = len(st.session_state.team_names)
      st.session_state.n_picks = st.session_state.integration.get_n_picks(st.session_state.integration.league_id)

      st.session_state.selections_default = st.session_state.integration.selections_default

      st.session_state.selections_df = st.session_state.selections_default

      st.session_state.player_metadata = get_player_metadata(st.session_state.data_source)

      st.session_state.yahoo_key_to_name_mapper = get_yahoo_key_to_name_mapper()