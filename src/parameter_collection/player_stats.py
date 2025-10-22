

import streamlit as st 
import pandas as pd
import numpy as np

from src.tabs.drafting import increment_and_reset_draft
from src.helpers.helper_functions import gen_key, get_data_from_session_state, get_data_key, get_league_type, get_raw_dataset, get_selections_default, listify \
                                            , drop_injured_players, get_player_name_column, store_dataset_in_session_state
from src.data_retrieval.get_data import get_historical_data, get_specified_historical_stats, combine_nba_projections
from src.data_retrieval.get_data_baseball import get_baseball_historical_data, combine_baseball_projections

def player_stats_popover(): 
    """Figures out where to get player stats from, and loads them into a dataframe

    Args:
        None

    Returns:
      None
    """

    if get_league_type() == 'NBA':

        get_nba_stats()
            
    elif get_league_type() == 'MLB':
            
        get_mlb_stats()

    default_injury_list = [p for p in st.session_state['injured_players'] \
                                if (p in get_data_from_session_state('player_stats_v0').index) and (not (p in listify(get_selections_default()))) 
                                ]
    
    player_stats_v0_key = get_data_key('player_stats_v0')
    df, key = drop_injured_players(player_stats_v0_key, default_injury_list)
    store_dataset_in_session_state(df ,'player_stats_v1', key)
    
def get_nba_stats():
    """Figures out where to get player stats from, and loads them into a dataframe, specifically for the NBA

    Args:
        None

    Returns:
      None
    """

    data_options = ['Projection','Historical'] if st.session_state.data_source == 'Enter your own data' else ['Projection']

    kind_of_dataset = st.selectbox(
                            'Which kind of dataset do you want to use?'
                            , data_options
                            ,key = 'data_option'
                            , index = 0
                            , on_change= increment_and_reset_draft
    )

    if kind_of_dataset == 'Historical':
            
        historical_df = get_historical_data()

        unique_datasets_historical = reversed([str(x) for x in pd.unique(historical_df.index.get_level_values('Season'))])

        dataset_name = st.selectbox(
            'Which season of data do you want to use?'
            ,unique_datasets_historical
            ,index = 0
            ,on_change = increment_and_reset_draft
        )
        df, key = get_specified_historical_stats(dataset_name, get_league_type())
        store_dataset_in_session_state(df, 'player_stats_v0', key)

    else:
        with st.form('Player stat form'):

            c1, c2, c3, c4, c5 = st.columns(5)

            with c1:

                if 'espn_slider_default_value' not in st.session_state:
                    st.session_state.espn_slider_default_value = 0.5

                espn_slider = st.slider('ESPN Weight'
                                        , min_value = 0.0
                                        , max_value = 1.0
                                        , value = st.session_state.espn_slider_default_value
                                        , key = 'espn_widget')
                
                st.session_state.espn_slider_default_value = espn_slider

            with c2:

                if 'htb_slider_default_value' not in st.session_state:
                    st.session_state.htb_slider_default_value = 0.0

                hashtag_slider = st.slider('Hashtag Weight'
                                        , min_value = 0.0
                                        , max_value = 1.0
                                        , value = st.session_state.htb_slider_default_value
                                        , key = 'hashtag_widget')
                
                st.session_state.htb_slider_default_value = hashtag_slider

                st.html(
                """
                <style>

                [data-testid='stFileUploaderDropzoneInstructions'] > div > span {
                display: none;
                }

                [data-testid='stFileUploaderDropzoneInstructions'] > div::before {
                content: '';
                }
                </style>
                """
                )
                
                hashtag_file = st.file_uploader('Copy average stats into a csv w/ utf-8 encoding and upload.'
                                                , type=['csv'])
                if hashtag_file is not None:
                    store_dataset_in_session_state(pd.read_csv(hashtag_file),'HTB','')

            with c3:

                if 'darko_slider_default_value' not in st.session_state:
                    st.session_state.darko_slider_default_value = 0.5

                darko_slider = st.slider('DARKO Weight'
                                        , min_value = 0.0
                                        , max_value = 1.0
                                        , value = st.session_state.darko_slider_default_value
                                        , key = 'darko_widget')
                
                st.session_state.darko_slider_default_value = darko_slider

            with c4:

                if 'bbm_slider_default_value' not in st.session_state:
                    st.session_state.bbm_slider_default_value = 0.0

                bbm_slider = st.slider('BBM Weight'
                        , min_value = 0.0
                        , max_value = 1.0
                        , value = st.session_state.bbm_slider_default_value
                        , key = 'bbm_widget')
                
                st.session_state.bbm_slider_default_value = bbm_slider

                bbm_file = st.file_uploader('''Export per-game stats to excel then save as CSV utf-8.'''
                                                , type=['csv'])
                
                if bbm_file is not None:
                    #there is no need for a key to the BBM file, since combine_nba_projections is always run when this 
                    #form is submitted
                    store_dataset_in_session_state(pd.read_csv(bbm_file),'BBM','')
                
            #no rotowire or darko for now- could revisit in future
            #DARKO isn't great becuase it doesnt have GP or Position by itself
            #and I dont think rotowire is really used at all
            rotowire_slider = 0   
            rotowire_upload = None 
            
            with c5: 

                def process_stat_options():
                    increment_and_reset_draft()
                    st.session_state.stat_options_key = gen_key()
            
                #ZR: this should also run the combine_nba_projections function on click. 
                submit_button = st.form_submit_button('Lock in & process'
                                                        , on_click = process_stat_options)

                st.warning('Changes will not be reflected until this button is pressed')

                if (hashtag_slider > 0) & ('HTB' not in st.session_state.data_dictionary):
                    st.error('Upload HTB projection file')
                    st.stop()
                    
                elif (bbm_slider > 0) & ('BBM' not in st.session_state.data_dictionary):
                     st.error('Upload Basketball Monster projection file')
                     st.stop()
                
                elif hashtag_slider + bbm_slider + darko_slider + espn_slider == 0:
                     st.error('Weights are all 0')
                     st.stop()

        df, key = combine_nba_projections(
                        hashtag_slider
                        , bbm_slider
                        , darko_slider
                        , espn_slider
                        , get_player_name_column()
                        , st.session_state.stat_options_key)
        store_dataset_in_session_state(df, 'player_stats_v0',key)
            

def get_mlb_stats():
    """Figures out where to get player stats from, and loads them into a dataframe, specifically for the MLB

    Args:
        None

    Returns:
      None
    """

    data_options = ['Projection']

    kind_of_dataset = st.selectbox(
                            'Which kind of dataset do you want to use? (specify further on the data tab)'
                            , data_options
                            ,key = 'data_option'
                            , index = 0
    )

    if kind_of_dataset == 'Historical':
            
        historical_df = get_baseball_historical_data()

        unique_datasets_historical = reversed([str(x) for x in pd.unique(historical_df.index.get_level_values('Season'))])

        dataset_name = st.selectbox(
            'Which dataset do you want to default to?'
            ,unique_datasets_historical
            ,index = 0
            ,on_change = increment_and_reset_draft
        )
        raw_stats_df = get_specified_historical_stats(dataset_name, get_league_type())
                
    else: 

        c1, c2 = st.columns(2)

        with c1:

            hashtag_slider = st.slider('Hashtag Basketball Weight'
                                    , min_value = 0.0
                                    , value = 1.0
                                    , max_value = 1.0)

        with c2:
            
            rotowire_slider = st.slider('RotoWire Weight'
                            , min_value = 0.0
                            , max_value = 1.0)

            
            rotowire_file_batters = st.file_uploader("Upload RotoWire data for batters, as a csv"
                                            , type=['csv'])
            rotowire_file_pitchers = st.file_uploader("Upload RotoWire data for pitchers, as a csv"
                            , type=['csv'])
            
            if (rotowire_file_batters is not None):
                rotowire_upload_batters  = pd.read_csv(rotowire_file_batters).rename(columns = {'K' :  'K.0'})
            else:
                rotowire_upload_batters = pd.DataFrame()

            if (rotowire_file_pitchers is not None):
                rotowire_upload_pitchers  = pd.read_csv(rotowire_file_pitchers).rename(columns = {'K' :  'K.1'
                                                                                                ,'BB' : 'BB.1'})
                rotowire_upload_pitchers.loc[:,'Pos'] = np.where(rotowire_upload_pitchers['SV'] > 0
                                                                ,'RP'
                                                                ,'SP')
            else:
                rotowire_upload_pitchers = pd.DataFrame()

            rotowire_upload_baseball = pd.concat([rotowire_upload_batters, rotowire_upload_pitchers]).reset_index()

            if (rotowire_slider > 0) & (len(rotowire_upload_baseball) == 0):
                    st.error('Upload at least one RotoWire projection file')            


        if (rotowire_slider > 0) & (len(rotowire_upload_baseball) == 0):
                st.stop()

        combine_baseball_projections(rotowire_upload_baseball
                            , hashtag_slider
                            , rotowire_slider
                            , st.session_state.data_source)
    return raw_stats_df