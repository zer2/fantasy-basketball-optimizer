import streamlit as st
import pandas as pd 
import numpy as np
from src.helpers.helper_functions import get_position_numbers_unwound, static_score_styler, h_percentage_styler, get_selected_categories, \
                                styler_a, styler_b, styler_c, stat_styler, \
                                get_selected_counting_statistics, get_selected_volume_statistics
from src.math.algorithm_agents import HAgent
from src.math.algorithm_helpers import savor_calculation
from src.math.process_player_data import process_player_data
from src.data_retrieval.get_data import get_htb_adp
import os
import itertools
from pathlib import Path
import gc
from src.math.algorithm_helpers import combinatorial_calculation

@st.cache_data(show_spinner = False, ttl = 3600)
def make_about_tab(md_path : str):
    """Make one of the tabs on the about page

    Args:
      md_path : string representing the path to the relevant markdown file for display
    Returns:
      None
    """
    c2,c2,c3 = st.columns([0.1,0.8,0.1])
    with c2:
        intro_md = Path('about/' + md_path).read_text()
        st.markdown(intro_md, unsafe_allow_html=True)

### Team tabs 





