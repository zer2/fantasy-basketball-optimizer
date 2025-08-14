

import streamlit as st 
import pandas as pd
import numpy as np

from src.tabs.drafting import increment_and_reset_draft

from src.data_retrieval.get_data import get_historical_data, get_specified_stats, combine_nba_projections
from src.data_retrieval.get_data_baseball import get_baseball_historical_data, combine_baseball_projections

def player_stats_popover(): 
    """Figures out where to get player stats from, and loads them into a dataframe

    Args:
        None

    Returns:
      raw_stat_df, dataframe 
    """

    #ZR: These should probably be classes, like the integrations

    if st.session_state.league == 'NBA':

        raw_stat_df = get_nba_stats()
               
    elif st.session_state.league == 'MLB':
            
        raw_stat_df = get_mlb_stats()

    return raw_stat_df
        
    

def get_nba_stats():
    """Figures out where to get player stats from, and loads them into a dataframe, specifically for the NBA

    Args:
        None

    Returns:
      raw_stat_df, dataframe 
    """

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
            'Which dataset do you want to default to?'
            ,unique_datasets_historical
            ,index = 0
            ,on_change = increment_and_reset_draft
        )
        raw_stats_df = get_specified_stats(dataset_name
                                            , st.session_state.player_stats_default_key)
                
    else: 

        c1, c2 = st.columns(2)

        with c1:

            hashtag_slider = st.slider('Hashtag Baseball Weight'
                                    , min_value = 0.0
                                    , value = 1.0
                                    , max_value = 1.0)

            bbm_slider = st.slider('BBM Weight'
                    , min_value = 0.0
                    , max_value = 1.0)

            bbm_file = st.file_uploader('''Upload Basketball Monster Per Game Stats, as a csv. To get all required columns for 
                                            projections, you may have to export to excel then save as CSV utf-8.'''
                                            , type=['csv'])
            if bbm_file is not None:
                bbm_upload  = pd.read_csv(bbm_file)
            else:
                bbm_upload = None

        with c2:
            
            darko_slider = st.slider('Darko Weight'
                                        , min_value = 0.0
                                        , max_value = 1.0)

            rotowire_slider = st.slider('RotoWire Weight'
                            , min_value = 0.0
                            , max_value = 1.0)

            
            rotowire_file = st.file_uploader("Upload RotoWire data, as a csv"
                                            , type=['csv'])
            if rotowire_file is not None:
                rotowire_upload  = pd.read_csv(rotowire_file, skiprows = 1)
            else:
                rotowire_upload = None

            if (rotowire_slider > 0) & (rotowire_upload is None):
                st.error('Upload RotoWire projection file')
                st.stop()

        if (bbm_slider > 0) & (bbm_upload is None):
            st.error('Upload Basketball Monster projection file')
            st.stop()

        raw_stats_df = combine_nba_projections(rotowire_upload
                            , bbm_upload
                            , hashtag_slider
                            , bbm_slider
                            , darko_slider
                            , rotowire_slider
                            , st.session_state.data_source)
        
    return raw_stats_df

def get_mlb_stats():
    """Figures out where to get player stats from, and loads them into a dataframe, specifically for the MLB

    Args:
        None

    Returns:
      raw_stat_df, dataframe 
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
        raw_stats_df = get_specified_stats(dataset_name
                                            , st.session_state.player_stats_default_key)
                
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

        raw_stats_df = combine_baseball_projections(rotowire_upload_baseball
                            , hashtag_slider
                            , rotowire_slider
                            , st.session_state.data_source)
    return raw_stats_df