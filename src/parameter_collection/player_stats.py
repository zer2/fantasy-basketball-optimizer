

import streamlit as st 
import pandas as pd
import numpy as np

from src.tabs.drafting import increment_and_reset_draft
from src.helpers.helper_functions import get_games_per_week, listify, increment_player_stats_version, drop_injured_players

from src.data_retrieval.get_data import get_historical_data, get_specified_stats, combine_nba_projections
from src.data_retrieval.get_data_baseball import get_baseball_historical_data, combine_baseball_projections

def player_stats_popover(): 
    """Figures out where to get player stats from, and loads them into a dataframe

    Args:
        None

    Returns:
      raw_stat_df, dataframe 
    """

    if st.session_state.league == 'NBA':

        stats = get_nba_stats()
            
    elif st.session_state.league == 'MLB':
            
        stats = get_mlb_stats()

    raw_stat_df, player_metadata = stats

    default_injury_list = [p for p in st.session_state['injured_players'] \
                                if (p in raw_stat_df.index) and (not (p in listify(st.session_state.selections_default))) 
                                ]
    
    st.session_state.raw_stat_df = drop_injured_players(raw_stat_df
                                                        , default_injury_list
                                                        , st.session_state.player_stats_version)  


    st.session_state.raw_stat_df = raw_stat_df

    st.session_state.player_metadata = player_metadata   



    

def get_nba_stats():
    """Figures out where to get player stats from, and loads them into a dataframe, specifically for the NBA

    Args:
        None

    Returns:
      either a tuple of (raw_stat_df, player_metadata) or a string representing an error 
    """


    with st.form('Player stat form'):

        data_options = ['Projection','Historical'] if st.session_state.data_source == 'Enter your own data' else ['Projection']

        kind_of_dataset = st.selectbox(
                                'Which kind of dataset do you want to use?'
                                , data_options
                                ,key = 'data_option'
                                , index = 0
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
            raw_stats_df, player_metadata = get_specified_stats(dataset_name, st.session_state.league)
                    
        else: 

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
                
                hashtag_file = st.file_uploader('Copy HTB projections into a csv and upload it'
                                                , type=['csv'])
                if hashtag_file is not None:
                    st.session_state.datasets['htb']  = pd.read_csv(hashtag_file)

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
                    st.session_state.datasets['bbm']  = pd.read_csv(bbm_file)
                
            #no rotowire or darko for now- could revisit in future
            #DARKO isn't great becuase it doesnt have GP or Position by itself
            #and I dont think rotowire is really used at all
            rotowire_slider = 0   
            rotowire_upload = None 
            
            with c5: 
            
                submit_button = st.form_submit_button('Lock in & process'
                                                        , on_click = increment_player_stats_version)

                st.warning('Changes will not be reflected until this button is pressed')

                if (hashtag_slider > 0) & ('htb' not in st.session_state.datasets):
                    st.error('Upload HTB projection file')
                    st.stop()
                    
                elif (bbm_slider > 0) & ('bbm' not in st.session_state.datasets):
                     st.error('Upload Basketball Monster projection file')
                     st.stop()
                
                elif hashtag_slider + bbm_slider + darko_slider + espn_slider == 0:
                     st.error('Weights are all 0')
                     st.stop()

        
        raw_stats_df, player_metadata = combine_nba_projections(rotowire_upload
                        , st.session_state.datasets.get('bbm')
                        , st.session_state.datasets.get('htb')
                        , hashtag_slider
                        , bbm_slider
                        , darko_slider
                        , espn_slider
                        , st.session_state.data_source)
            
        return (raw_stats_df, player_metadata)

def get_mlb_stats():
    """Figures out where to get player stats from, and loads them into a dataframe, specifically for the MLB

    Args:
        None

    Returns:
      either a tuple of (raw_stat_df, player_metadata) or a string representing an error 
    """

    data_options = ['Projection']

    kind_of_dataset = st.selectbox(
                            'Which kind of dataset do you want to use? (specify further on the data tab)'
                            , data_options
                            ,key = 'data_option'
                            , index = 0
                            , on_change= increment_player_stats_version
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
        raw_stats_df = get_specified_stats(dataset_name, st.session_state.league)
                
    else: 

        c1, c2 = st.columns(2)

        with c1:

            hashtag_slider = st.slider('Hashtag Basketball Weight'
                                    , min_value = 0.0
                                    , value = 1.0
                                    , max_value = 1.0
                                    , on_change = increment_player_stats_version)

        with c2:
            
            rotowire_slider = st.slider('RotoWire Weight'
                            , min_value = 0.0
                            , max_value = 1.0
                            , on_change = increment_player_stats_version)

            
            rotowire_file_batters = st.file_uploader("Upload RotoWire data for batters, as a csv"
                                            , type=['csv']
                                            , on_change = increment_player_stats_version)
            rotowire_file_pitchers = st.file_uploader("Upload RotoWire data for pitchers, as a csv"
                            , type=['csv']
                            , on_change = increment_player_stats_version)
            
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

        raw_stats_df = combine_baseball_projections(rotowire_upload_baseball
                            , hashtag_slider
                            , rotowire_slider
                            , st.session_state.data_source)
    return raw_stats_df