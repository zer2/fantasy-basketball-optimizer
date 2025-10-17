import streamlit as st

from src.helpers.helper_functions import increment_player_stats_version

def format_popover():
    """Collect information from the user on the desired fantasy sport format.
    Adds two objects to session_state: 'scoring_format' and 'selected_categories'

    Args:
        None

    Returns:
      None 
    """
    with st.form('param form'):
        scoring_format = st.selectbox(
                'Which format are you playing?',
                ('Rotisserie', 'Head to Head: Each Category', 'Head to Head: Most Categories')
                , key = 'scoring_format'
                , index = 1)
            
        if scoring_format == 'Rotisserie':
            st.caption('Note that H-scores for Rotisserie are experimental and have not been tested')

        default_options = st.session_state.params['counting-statistics'] + \
                                list(st.session_state.params['ratio-statistics'].keys())
                                     
        #only include categories as options if they are available in the data 
        actual_options = [option for option in default_options if option in st.session_state.raw_stat_df.columns]

        categories = st.multiselect('Which categories does your league use?'
                        , key = 'selected_categories'
                        , options = actual_options
                        , default = st.session_state.params['default-categories']
                            )
        
        c1, c2 = st.columns([0.2,0.8])

        with c1: 
            #this is not directly changing player stats, but it changes which categories are relevant for them
            st.form_submit_button('Lock in & process', on_click=increment_player_stats_version)
        with c2: 
            st.warning('Changes will not be reflected until this button is pressed')

    if len(categories) <= 1:
        st.error('Select at least two categories')
        st.stop()