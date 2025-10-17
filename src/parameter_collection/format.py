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

        categories = st.multiselect('Which categories does your league use?'
                        , key = 'selected_categories'
                        , options = st.session_state.params['counting-statistics'] + \
                                list(st.session_state.params['ratio-statistics'].keys())
                        , default = st.session_state.params['default-categories']
                            )
        
        #this is not directly changing player stats, but it changes which categories are relevant for them
        st.form_submit_button('Lock in & process', on_click=increment_player_stats_version)
        
    if len(categories) <= 1:
        st.error('Select at least two categories')
        st.stop()