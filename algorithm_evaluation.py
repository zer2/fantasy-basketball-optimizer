from src.simulation import *
#from streamlit_profiler import Profiler

import yaml
import streamlit as st

#with Profiler():

if 'info_key' not in st.session_state:
    st.session_state.info_key = 100000

if 'params' not in st.session_state:
  with open("parameters.yaml", "r") as stream:
      try:
        st.session_state.params = yaml.safe_load(stream)
      except yaml.YAMLError as exc:
          print(exc) 


st.set_page_config(page_title='Fantasy BBall Optimization- Testing Page'
          , page_icon=':basketball:'
          , layout="wide"
          , initial_sidebar_state="auto"
          , menu_items=None)

st.title('Testing algorithms for Fantasy Basketball :basketball:')

#add a test here for doing a linear regression to find gamma and omega 
#we can just use that gamma and that omega in the future 

validate()
