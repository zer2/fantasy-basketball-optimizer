import streamlit as st

from src.helpers.helper_functions import get_data_from_session_state, get_params, get_default, set_cookie


def format_popover():
    """Collect information from the user on the desired fantasy sport format.
    Adds two objects to session_state: 'scoring_format' and 'selected_categories'

    Args:
        None

    Returns:
      None 
    """
    with st.form('param form'):

        #ZR: need to figure out a way to save this to cookies ##
        scoring_format = st.selectbox(
                'Which format are you playing?',
                ('Rotisserie', 'Head to Head: Each Category', 'Head to Head: Most Categories')
                , key = 'scoring_format'
                , index = 1)
                    
        if scoring_format == 'Rotisserie':
            st.caption('Note that H-scores for Rotisserie are experimental and have not been tested')

        params = get_params()

        default_options = params['counting-statistics'] + \
                                list(params['ratio-statistics'].keys())
                                     
        #only include categories as options if they are available in the data 
        actual_options = [option for option in default_options if option in get_data_from_session_state('player_stats_v0').columns]

        categories = st.multiselect('Which categories does your league use?'
                        , key = 'selected_categories'
                        , options = actual_options
                        , default = get_default('categories')
                            )

        c1, c2 = st.columns([0.2,0.8])

        with c1: 
            st.form_submit_button('Lock in & process')
        with c2: 
            st.warning('Changes will not be reflected until this button is pressed')

    if len(categories) <= 1:
        st.error('Select at least two categories')
        st.stop()