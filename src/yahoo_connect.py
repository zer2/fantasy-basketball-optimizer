from requests.auth import HTTPBasicAuth
from src import yahoo_helper
from streamlit.logger import get_logger
from tempfile import mkdtemp
from typing import List, Optional
from yfpy.models import League, Team, Roster
from yfpy.query import YahooFantasySportsQuery

import json
import os
import pandas as pd
import requests
import shutil
import streamlit as st
import time
import yahoo_fantasy_api as yfa

LOGGER = get_logger(__name__)

def get_yahoo_access_token() -> Optional[str]:
    # Client_ID and Secret from https://developer.yahoo.com/apps/
    cid = st.secrets["YAHOO_CLIENT_ID"]
    cse = st.secrets["YAHOO_CLIENT_SECRET"]
    
    # Ensure that the Client ID and Secret are set
    if cid is None or cse is None:
        st.error("Client ID or Client Secret is not set. Please set the YAHOO_CLIENT_ID and YAHOO_CLIENT_SECRET environment variables.")
        st.stop()
    
    # URL for st button with Client ID in query string
    redirect_uri = "oob" #"oob"  # Out of band # "https://yahoo-ff-test.streamlit.app/" for dev version
    auth_page = f'https://api.login.yahoo.com/oauth2/request_auth?client_id={cid}&redirect_uri={redirect_uri}&response_type=code'
    
    # Show ST Button to open Yahoo OAuth2 Page
    if 'auth_code' not in st.session_state:
        st.session_state['auth_code'] = ''
    
    if 'access_token' not in st.session_state:
        st.session_state['access_token'] = ''
    
    if 'refresh_token' not in st.session_state:
        st.session_state['refresh_token'] = ''
    
    temp_dir = None
    
    st.write("1. Click the link below to authenticate with Yahoo and get the authorization code.")
    st.write(f"[Authenticate with Yahoo]({auth_page})")
    
    # Get Auth Code pasted by user
    st.write("2. Paste the authorization code here:")
    auth_code = st.text_input("Authorization Code")
    
    if auth_code:
        st.session_state['auth_code'] = auth_code
        st.success('Authorization code received!')
        #st.write(f'Your authorization code is: {auth_code}')
    
    # Get the token
    if st.session_state['auth_code'] and not st.session_state['access_token']:
        basic = HTTPBasicAuth(cid, cse)
        _data = {
            'redirect_uri': redirect_uri,
            'code': st.session_state['auth_code'],
            'grant_type': 'authorization_code'
        }
    
        try:
            r = requests.post('https://api.login.yahoo.com/oauth2/get_token', data=_data, auth=basic)
            r.raise_for_status()  # Will raise an exception for HTTP errors
            token_data = r.json()
            st.session_state['access_token'] = token_data.get('access_token', '')
            st.session_state['refresh_token'] = token_data.get('refresh_token', '')
            st.session_state['token_time'] = time.time()
            st.success('Access token received!')
        except requests.exceptions.HTTPError as err:
            st.error(f"HTTP error occurred: {err}")
        except Exception as err:
            st.error(f"An error occurred: {err}")
    
    # Store token data in file
    if st.session_state['access_token']:
        st.write("Now you can use the access token to interact with Yahoo's API.")

        temp_dir = mkdtemp()

        # Define the paths to the token and private files
        token_file_path = os.path.join(temp_dir, "token.json")
        private_file_path = os.path.join(temp_dir, "private.json")

        # Create the token file with all necessary details
        token_data = {
            "access_token": st.session_state['access_token'],
            "consumer_key": cid,
            "consumer_secret": cse,
            "guid": None,
            "refresh_token": st.session_state['refresh_token'],
            "expires_in": 3600, 
            "token_time": st.session_state['token_time'],
            "token_type": "bearer"
            }
        with open(token_file_path, 'w') as f:
            json.dump(token_data, f)

        # Create the private file with consumer key and secret
        private_data = {
            "consumer_key": cid,
            "consumer_secret": cse,
        }
        with open(private_file_path, 'w') as f:
            json.dump(private_data, f)

        return temp_dir

def clean_up_access_token(access_token_dir: str):
    shutil.rmtree(access_token_dir)

@st.cache_data(ttl=3600)
def get_yahoo_players_df(_access_token_dir: str, league_id: str, player_metadata: pd.Series) -> pd.DataFrame:
    teams_dict = get_teams_dict(league_id, _access_token_dir)
    team_ids = list(teams_dict.keys())
    rosters_dict = get_rosters_dict(league_id, _access_token_dir, team_ids)
    players_df = _get_players_df(rosters_dict, teams_dict, player_metadata)

    return players_df

@st.cache_data(ttl=3600)
def get_teams_dict(league_id: str, auth_path: str) -> dict[int, str]:
    LOGGER.info(f"League id: {league_id}")
    sc = YahooFantasySportsQuery(
        auth_dir=auth_path,
        league_id=league_id,
        game_code="nba"
    )
    LOGGER.info(f"sc: {sc}")
    teams = yahoo_helper.get_teams(sc)
    teams_dict = {team.team_id: team.name.decode('UTF-8') for team in teams}
    return teams_dict

@st.cache_data(ttl=3600)
def get_rosters_dict(league_id: str, auth_path: str, team_ids: list[int]) -> dict[int, Roster]:    

    league_id = league_id
    LOGGER.info(f"League id: {league_id}")
    sc = YahooFantasySportsQuery(
        auth_dir=auth_path,
        league_id=league_id,
        game_code="nba"
    )
    LOGGER.info(f"sc: {sc}")

    rosters_dict: dict[int, Roster] = {}

    for team_id in team_ids:
        roster = yahoo_helper.get_team_roster(sc, team_id)
        rosters_dict[team_id] = roster
    
    return rosters_dict

@st.cache_data(ttl=3600)
def get_user_leagues(auth_path: str) -> List[League]:
    sc = YahooFantasySportsQuery(
        auth_dir=auth_path,
        league_id="", # Shouldn't actually matter what this is since we are retrieving the user's leagues
        game_code="nba"
    )
    LOGGER.info(f"sc: {sc}")
    leagues = yahoo_helper.get_user_leagues(sc)
    
    # Sort in reverse chronological order
    sorted_leagues = sorted(leagues, key = lambda league: league.season, reverse=True)
    return sorted_leagues

def _get_players_df(rosters_dict: dict[int, Roster], teams_dict: dict[int, str], player_metadata: pd.Series):
    players_df = pd.DataFrame()

    team_players_dict = {}

    max_team_size = 0

    for team_id, roster in rosters_dict.items():
        team_name = teams_dict[team_id]
        relevant_player_names = [
            f'{player.name.full} ({player_metadata.loc[player.name.full]})' #Appending position after player name
            for player in roster.players 
            if 
                player.selected_position.position != 'IL'
                and player.selected_position.position is not None
        ]

        if len(relevant_player_names) > max_team_size:
            max_team_size = len(relevant_player_names)

        team_players_dict[team_name] = relevant_player_names

    for team_name, player_names in team_players_dict.items():
        if len(player_names) < max_team_size :
            player_names.extend([None] * (max_team_size - len(player_names)))
        
        players_df.loc[:,team_name] = player_names

    return players_df