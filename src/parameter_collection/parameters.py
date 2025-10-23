import streamlit as st
import pandas as pd

from src.helpers.helper_functions import get_data_from_session_state, get_data_key, get_mode, get_n_picks, get_params, get_scoring_format, get_selected_categories, store_dataset_in_session_state
from src.math.process_player_data import make_upsilon_adjustment, process_player_data
from src.helpers.helper_functions import get_n_drafters

def player_stat_param_popover():
    """Collect information from the user on desired parameters for handling player injuries/uncertainty 
    Specifically three objects, all floats, are added to session_state: 'upsilon','psi', and 'chi'

    Args:
        None

    Returns:
      None 
    """
    params = get_params()
    
    upsilon = st.number_input(r'Select a $\upsilon$ value'
                      , key = 'upsilon'
                      , min_value = float(params['options']['upsilon']['min'])
                      , value = float(params['options']['upsilon']['default'])
                    , max_value = float(params['options']['upsilon']['max']))
    upsilon_str = r'''Injury rates are scaled down by $\upsilon$. For example, if a player is expected to 
                  miss $20\%$ of games and $\upsilon$ is $75\%$, then it will be assumed that they miss 
                  $15\%$ of games instead'''
    st.caption(upsilon_str)


    psi = st.number_input(r'Select a $\psi$ value'
                      , key = 'psi'
                      , min_value = float(params['options']['psi']['min'])
                      , value = float(params['options']['psi']['default'])
                    , max_value = float(params['options']['psi']['max']))
    psi_str = r'''It it assumed that of the games a player will miss, 
                  they are replaced by a replacement-level player for $\psi \%$ of them'''
  
    st.caption(psi_str)

    chi = st.number_input(r'Select a $\chi$ value'
        , key = 'chi'
        , value = float(params['options']['chi']['default'])
        , min_value = float(params['options']['chi']['min'])
        , max_value = float(params['options']['chi']['max']))

    chi_str = r'''The estimated variance in season-long projections relative to empirical week-to-week variance. 
                    for Rotisserie. E.g. if $\chi$ is 0.6, variance is effectively 60% of the week-to-week variance
                    observed value in previous seasons. 
                    '''
    st.caption(chi_str)

    aleph = st.number_input(r'Select a $\alef$ value'
        , key = 'aleph'
        , value = float(params['options']['aleph']['default'])
        , min_value = float(params['options']['aleph']['min'])
        , max_value = float(params['options']['aleph']['max']))

    aleph_str = r'''Extra correlation between volume-based categories for Rotisserie, to account for the fact that some managers 
                  will be more or less active. E.g. if $\alef$ is 0.1 and the correlation between blocks and points is 30%, 
                  the correlation will be considered 40% instead. 
                    '''
    st.caption(aleph_str)

    beth = st.number_input(r'Select a $\beth$ value'
        , key = 'beth'
        , value = float(params['options']['beth']['default'])
        , min_value = float(params['options']['beth']['min'])
        , max_value = None)

    beth_str = r'''Controls the degree to which projections of your team are adjusted downwards to the average, 
                    based on the Bayesian thinking that projections should be shrunk towards a prior. Higher values 
                    apply more distortion. See the documentation for details.
                    '''
    st.caption(beth_str)


    if get_mode() == 'Auction Mode':

      streaming_noise = st.number_input(r'Select an $S_{\sigma}$ value'
                                , key = 'streaming_noise'
                                , value = float(params['options']['S']['default'])
                                , min_value = float(params['options']['S']['min'])
                                , max_value = float(params['options']['S']['max'])
                              )
      stream_noise_str = r'''$S_{\sigma}$ controls the SAVOR algorithm. It roughly represents the 
                            standard deviation of dollar values expected for players during the 
                            season. When it is high, more long-term performance noise is expected, 
                            making low-value players more heavily down-weighted due to the possibility 
                            that they drop below streaming-level value'''
      st.caption(stream_noise_str)         
    
    df, key = make_upsilon_adjustment(get_data_key('player_stats_v1'), upsilon)
    store_dataset_in_session_state(df, 'player_stats_v2', key)

    info, key = process_player_data(None
                          ,get_data_key('player_stats_v2')
                          ,st.session_state.psi
                          ,st.session_state.chi
                          ,get_scoring_format()
                          ,get_n_drafters()
                          ,get_n_picks()
                          ,params
                          ,get_selected_categories())
    store_dataset_in_session_state(info, 'info', key)
    
def algorithm_param_popover():
    """Collect information from the user on desired parameters for H-scoring
    Adds three objects, all floats, to session_state: 'omega', 'gamma', and 'n_iterations'. 
    Also collects 'streaming_noise' and 'streaming_noise_h' if in auction mode

    Args:
        None

    Returns:
      None 
    """
    params = get_params()

    punting_default = params['punting_default']

    punting_levels = params['punting_defaults']

    omega = st.number_input(r'Select a $\omega$ value'
                          , key = 'omega'
                          , value = punting_levels[punting_default]['omega']
                          , min_value = float(params['options']['omega']['min'])
                          , max_value = float(params['options']['omega']['max']))
    omega_str = r'''The higher $\omega$ is, the more aggressively the algorithm will try to punt. Slightly more technically, 
                    it quantifies how much better the optimal player choice will be compared to the player that would be 
                    chosen with baseline weights'''
    st.caption(omega_str)
  
    gamma = st.number_input(r'Select a $\gamma$ value'
                          , key = 'gamma'
                          , value = punting_levels[punting_default]['gamma']
                          , min_value = float(params['options']['gamma']['min'])
                          , max_value = float(params['options']['gamma']['max']))
    gamma_str = r'''$\gamma$ also influences the level of punting, complementing omega. Tuning gamma is not suggested but you can 
            tune it if you want. Higher values imply that the algorithm will have to give up more general value to find the
            players that  work best for its strategy'''
    st.caption(gamma_str)

    n_iterations = st.number_input(r'Select a number of iterations for gradient descent to run'
                              , key = 'n_iterations'
                              , value = punting_levels[punting_default]['n_iterations']
                              , min_value = params['options']['n_iterations']['min']
                              , max_value = params['options']['n_iterations']['max'])
    n_iterations_str = r'''More iterations take more computational power, but theoretically achieve better convergence'''
    st.caption(n_iterations_str)
    
def trade_param_popover():
    """Collect information from the user on desired parameters for trade evaluation
    
    ZR: I think this could be moved to the trade tab itself. Might noe be necessary to have here

    Args:
        None

    Returns:
      None 
    """
    your_differential_threshold = st.number_input(
          r'Your differential threshold for the automatic trade suggester'
          , key = 'your_differential_threshold'
          , value = 0)
    ydt_str = r'''Only trades which improve your H-score 
                  by this percent will be shown'''
    st.caption(ydt_str)
    your_differential_threshold = your_differential_threshold /100

    their_differential_threshold = st.number_input(
          r'Counterparty differential threshold for the automatic trade suggester'
          , key = 'their_differential_threshold'
          , value = -0.2)
    tdt_str = r'''Only trades which improve their H-score 
                by this percent will be shown'''
    st.caption(tdt_str)
    their_differential_threshold = their_differential_threshold/100

    combo_params_default = pd.DataFrame({'N-traded' : [1,2,3]
                                  ,'N-received' : [1,2,3]
                                  ,'T' : [3,3,3]}
                                  )

    st.session_state['combo_params_df'] = st.data_editor(combo_params_default
                                                        , hide_index = True
                                                        , num_rows = "dynamic"
                                                        , column_config={
                                          "N-traded": st.column_config.NumberColumn("N-traded", default=1)
                                          ,"N-received": st.column_config.NumberColumn("N-received", default=1)
                                          ,"T": st.column_config.NumberColumn("T", default=0)

                                                                  }
                                                          ) 
                    
      
    combo_params_str =  \
      """Each row is a specification for a kind of trade that will be automatically evaluated. 
      N-traded is the number of players traded from your team, and N-received is the number of 
      players to receive in the trade. T is a threshold of H-score percent difference; trades that have 
      general value differences larger than T will not be evaluated"""
    st.caption(combo_params_str)