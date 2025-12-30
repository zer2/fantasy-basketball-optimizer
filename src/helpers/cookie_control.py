from src.helpers.helper_functions import get_params

import streamlit as st
import streamlit.components.v1 as components
import json
import time
'''
Cookie handlers
'''

all_params_list = ['n_drafters','n_picks','upsilon','psi','chi'
             ,'aleph','beth','omega','gamma','n_iterations','your_differential_threshold'
             ,'their_differential_threshold','streaming_noise','selected_categories'
             ,'n_Util','n_C' ,'n_G','n_PG','n_SG','n_F','n_PF','n_SF','n_bench']

def get_default(key, parameter_default = None):
    saved_cookies = st.session_state.saved_cookies

    if saved_cookies and key in saved_cookies:
        return saved_cookies[key]
    
    else:
        if parameter_default is not None: #default to the provided parameter value, if provided
            return parameter_default 
        else: #otherwise get the default straight from parameters
            return get_params()['options'][key]['default']

def store_options_as_cookies(cookies):
   #save the options selected by users as cookies, so they can persist across session states

   #need to add some other params

   for param in all_params_list: 
      if param in st.session_state:
         cookies.set(param, json.dumps(st.session_state[param]), key = 'cookie_' + param)

def reset_all_parameters(cookies):
   
    for param in all_params_list:
        st.session_state.pop(param, None)

    js_code = """
    <script>
    var cookies = document.cookie.split("; ");
    for (var i = 0; i < cookies.length; i++) {
        var cookie = cookies[i];
        var eqPos = cookie.indexOf("=");
        var name = eqPos > -1 ? cookie.substring(0, eqPos) : cookie;
        document.cookie = name + "=;expires=Thu, 01 Jan 1970 00:00:00 GMT;path=/";
    }
    window.parent.location.reload(); // Optional: Forces a full browser refresh
    </script>
    """
    components.html(js_code, height=0)
#this is after the function declarations, because the functions need to be properly loaded. 
#then we check if cookies are ready and potentially stop if not



