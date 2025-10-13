import streamlit as st
from src.helpers.helper_functions import  get_position_numbers, increment_info_key

def position_requirement_popover():
    """Allows the user to prescribe the position structure for teams to follow
    Position requirements are stored in session state as 'n_x', where x is the position code. For example 
    if 2 centers are required, then st.session_state.n_C is set to 2. 

    Args:
        None

    Returns:
      None 
    """

    #set default position numbers, based on n_picks

    with st.form('position_form'):

      all_position_defaults = st.session_state.params['options']['positions']
      
      if st.session_state.n_picks in all_position_defaults:
        position_defaults = all_position_defaults[st.session_state.n_picks]
      else:
        position_defaults = all_position_defaults[st.session_state.params['options']['n_picks']['default']]

        if st.session_state.mode != 'Season Mode':
          st.warning('''There is no default position structure for a league with ''' + str(st.session_state.n_picks) + \
                  ''' picks. Position structure must be met manually on the Advanced tab.''')


      caption_col, button_col = st.columns([0.8,0.2])

      with caption_col:
        st.warning('''H-scoring assumes team will need to fit this structure. 
                   Changes will not be reflected until 'Lock in & process' is pressed''')

      with button_col:
        st.form_submit_button('Lock in & process'
                              , on_click= increment_info_key)

      left_position_col, right_position_col = st.columns(2)

      with left_position_col:

        for position_code, position_info in st.session_state.params['position_structure']['base'].items():

          st.number_input(position_info['full_str']
                    , key = 'n_' + position_code
                    , value = position_defaults['base'][position_code]
                    , min_value = 0
                        )
        
      with right_position_col:

        for position_code, position_info in st.session_state.params['position_structure']['flex'].items():

          st.number_input(position_info['full_str']
                    , key = 'n_' + position_code
                    , value = position_defaults['flex'][position_code]
                    , min_value = 0
                        )
        st.number_input('Bench (these players are ignored for drafts)'
                        , key = 'n_bench'
                        , value = 0
                        , min_value = 0
                        )  
                        


      implied_n_picks = sum(n for n in get_position_numbers().values()) + st.session_state.n_bench
      
      if (implied_n_picks != st.session_state.n_picks) & (st.session_state.mode != 'Season Mode'):
        st.error('This structure has ' + str(implied_n_picks) + ' position slots, but your league has ' + str(st.session_state.n_picks) + \
                ' picks per manager. Adjust the position slots before proceeding')
        st.stop()

      st.session_state.n_starters = st.session_state.n_picks - st.session_state.n_bench