import streamlit as st
import pandas as pd
import numpy as np
from src.tabs.drafting import increment_and_reset_draft, clear_draft_board

def teams_and_autodraft_popover():
    """Allows the user to set the total number of teams, name them, and assign autodrafters to them if necessary
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

    if (st.session_state['mode'] == 'Draft Mode'):
        autodrafters = st.multiselect('''Which drafter(s) should be automated with an auto-drafter?'''
            ,options = st.session_state.team_names
            ,key = 'autodrafters'
            ,default = None)
          
