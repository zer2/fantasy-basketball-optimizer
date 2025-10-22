import streamlit as st

from src.platform_integration.fantrax_integration import FantraxIntegration
from src.platform_integration.yahoo_integration import YahooIntegration
from src.platform_integration.espn_integration import ESPNIntegration

from src.tabs.drafting import increment_and_reset_draft, clear_draft_board
import pandas as pd
import numpy as np

from src.helpers.helper_functions import get_mode, get_n_picks, get_params, get_selections_default, get_team_names, set_params, using_manual_entry

def league_settings_popover():
    """Collect settings for the league and set up a platform integration if necessary

    First column:
    Adds four objects to session_state: 'league', 'params', 'data_source', and 'mode'
    If the user wants to integrate with a platform, also add 'integration' to session state

    Second column:
    Allows the user to set the total number of teams and name them
    This function also creates a default dataframe for team selections 
    Adds three main objects to session state: 'n_drafters','n_picks', and 'selections_df' (the
    aforementioned default dataframe)
    For convenience, 'team_names' is also saved so that the selections_df doesn't need to be loaded every time 
    team names are checked

    Args:
        None

    Returns:
      None 
    """

    #no need for second column if there is no integration
    if using_manual_entry():
        c1, c2 = st.columns([0.5,0.5])
    else:
        c1 = st.container()

    with c1:

        league = st.selectbox(
                'Which fantasy sport are you playing?',
                ('NBA') #WNBA and MLB excluded for now
                , index = 0
                , key = 'league'
                , on_change = increment_and_reset_draft
                )
            
        set_params(league)
        params = get_params()

        integrations = {integration.get_description_string() : integration for integration in \
                        [YahooIntegration(), FantraxIntegration(), ESPNIntegration()]}

        data_source = st.selectbox(
        'Do you want to integrate with a fantasy platform?'
        , ['Enter your own data'] + list(integrations.keys())
        , key = 'data_source'
        , on_change = increment_and_reset_draft
        , index = 0)

        if using_manual_entry():
            mode_options = ('Draft Mode', 'Auction Mode','Season Mode')      

        else:
            st.session_state.integration = integrations[data_source]
            mode_options = st.session_state.integration.get_available_modes()

        mode = st.selectbox(
            'Which mode do you want to use?'
            , mode_options
            , index = 0
            , key = 'mode'
            , on_change = clear_draft_board)
        
        if not data_source == 'Enter your own data':
        
            st.session_state.integration.setup()




    if using_manual_entry():

        with c1: 

            if get_mode() == 'Draft Mode':
                st.toggle('Toggle third round reversal'
                          , key = 'third_round_reversal'
                          , on_change= clear_draft_board)
            else:
                st.session_state.third_round_reversal = False
                
        with c2: 

            n_drafters = st.number_input(r'How many drafters are in your league?'
                                        , key = 'n_drafters'
                                        , min_value = params['options']['n_drafters']['min']
                                        , value = params['options']['n_drafters']['default']
                                        , on_change = clear_draft_board
                                        )

            n_picks = st.number_input(r'How many players will each drafter choose?'
                            , key = 'n_picks'
                            , min_value = params['options']['n_picks']['min']
                            , value = params['options']['n_picks']['default']
                            , on_change = clear_draft_board)
                                    
            st.write('Enter team names here:')

            team_df = st.data_editor(pd.DataFrame({'Team ' + str(i) : ['Drafter ' + str(i)] for i in range(n_drafters)})
                            , hide_index = True
                            , key = 'team_name_df'
                            , on_change = increment_and_reset_draft)
            st.session_state.team_names = list(team_df.iloc[0])

            st.session_state.selections_default = pd.DataFrame(
                {team : [np.nan] * get_n_picks() for team in get_team_names()}
                )
            
    else: 
        st.session_state.third_round_reversal = False
