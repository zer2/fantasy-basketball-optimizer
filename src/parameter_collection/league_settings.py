import streamlit as st

from src.platform_integration.fantrax_integration import FantraxIntegration
from src.platform_integration.yahoo_integration import YahooIntegration
from src.platform_integration.espn_integration import ESPNIntegration

from src.tabs.drafting import increment_and_reset_draft, clear_draft_board
import pandas as pd
import numpy as np

from src.data_retrieval.get_data import get_yahoo_key_to_name_mapper

from src.helpers.helper_functions import get_team_names

def league_settings_popover():
    """Collect settings for the league and set up a platform integration if necessary

    First column:
    Adds four objects to session_state: 'league', 'params', 'data_source', and 'mode'
    If the user wants to integrate with a platform, also add 'integration' to session state

    Second column:
    Allows the user to set the total number of teams, name them, and assign autodrafters to them if necessary
    This function also creates a default dataframe for team selections 
    Adds four main objects to session state: 'n_drafters','n_picks', 'autodrafters', and 'selections_df' (the
    aforementioned default dataframe)
    For convenience, 'team_names' is also saved so that the selections_df doesn't need to be loaded every time 
    team names are checked

    Args:
        None

    Returns:
      None 
    """

    #no need for second column if there is no integration
    if st.session_state.data_source == 'Enter your own data':
        c1, c2 = st.columns([0.5,0.5])
    else:
        c1 = st.container()

    with c1:

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

            st.session_state.n_picks = st.session_state.integration.get_n_picks(st.session_state.integration.league_id)

            st.session_state.selections_default = st.session_state.integration.selections_default

            st.session_state.selections_df = st.session_state.selections_default

            st.session_state.yahoo_key_to_name_mapper = get_yahoo_key_to_name_mapper()

    if st.session_state.data_source == 'Enter your own data':
        with c2: 

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
            
            #ZR: It would probably be better to access this through a function that accesses team_df
            st.session_state.team_names = list(team_df.iloc[0])

            # perhaps the dataframe should be uneditable, and users just get to enter the next players picked? With an undo button?
            #Should this just be called if selections_df not in session state?
            st.session_state.selections_default = pd.DataFrame(
                {team : [np.nan] * st.session_state.n_picks for team in get_team_names()}
                )
            
            if 'selections_df' not in st.session_state:
                st.session_state.selections_df = st.session_state.selections_default 

            if (st.session_state['mode'] == 'Draft Mode'):
                autodrafters = st.multiselect('''Which drafter(s) should be automated with an auto-drafter?'''
                    ,options = get_team_names()
                    ,key = 'autodrafters'
                    ,default = None)
            