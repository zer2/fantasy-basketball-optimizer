from requests.auth import HTTPBasicAuth
from src import yahoo_helper
from streamlit.logger import get_logger
from tempfile import mkdtemp
from typing import List, Optional
from yfpy.models import League, Team, Roster
from yfpy.query import YahooFantasySportsQuery
from src.get_data import get_nba_schedule, get_yahoo_key_to_name_mapper
from src.helper_functions import move_forward_one_pick
from collections import Counter

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

def clean_up_access_token(auth_dir: str):
    shutil.rmtree(auth_dir)

@st.cache_data(ttl=3600, show_spinner = False)
def get_yahoo_players_df(_auth_dir: str, league_id: str, player_metadata: pd.Series) -> pd.DataFrame:
    teams_dict = get_teams_dict(league_id, _auth_dir)
    team_ids = list(teams_dict.keys())
    rosters_dict = get_rosters_dict(league_id, _auth_dir, team_ids)
    players_df = _get_players_df(rosters_dict, teams_dict, player_metadata)

    return players_df

@st.cache_data(ttl=3600, show_spinner = False)
def get_teams_dict(league_id: str, _auth_path: str) -> dict[int, str]:
    LOGGER.info(f"League id: {league_id}")
    sc = YahooFantasySportsQuery(
        auth_dir=_auth_path,
        league_id=league_id,
        game_code="nba"
    )
    LOGGER.info(f"sc: {sc}")
    teams = yahoo_helper.get_teams(sc)
    teams_dict = {team.team_id: team.name.decode('UTF-8') for team in teams}
    return teams_dict

@st.cache_data(ttl=3600
                , show_spinner = "Fetching rosters from your Yahoo league. This should take about ten seconds")
def get_rosters_dict(league_id: str, _auth_path: str, team_ids: list[int]) -> dict[int, Roster]:    

    league_id = league_id
    LOGGER.info(f"League id: {league_id}")
    sc = YahooFantasySportsQuery(
        auth_dir=_auth_path,
        league_id=league_id,
        game_code="nba"
    )
    LOGGER.info(f"sc: {sc}")

    rosters_dict: dict[int, Roster] = {}

    for team_id in team_ids:
        roster = yahoo_helper.get_team_roster(sc, team_id)
        rosters_dict[team_id] = roster
    
    return rosters_dict

@st.cache_data(ttl=3600, show_spinner = False)
def get_user_leagues(_auth_path: str) -> List[League]:
    sc = YahooFantasySportsQuery(
        auth_dir=_auth_path,
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

        for player in roster.players:
            if player.selected_position.position == 'IL':
                st.session_state['injured_players'].add( \
                            f'{player.name.full} ({player_metadata.loc[player.name.full]})' 
                                                        )

        if len(relevant_player_names) > max_team_size:
            max_team_size = len(relevant_player_names)

        team_players_dict[team_name] = relevant_player_names

    for team_name, player_names in team_players_dict.items():
        if len(player_names) < max_team_size :
            player_names.extend([None] * (max_team_size - len(player_names)))
        
        players_df.loc[:,team_name] = player_names

    return players_df

@st.cache_data(ttl=3600
                , show_spinner = "Fetching player status information from Yahoo. This should take about thirty seconds")
def get_player_statuses(league_id: str, _auth_path: str, player_metadata: pd.Series) -> dict[int, str]:
    LOGGER.info(f"League id: {league_id}")
    sc = YahooFantasySportsQuery(
        auth_dir=_auth_path,
        league_id=league_id,
        game_code="nba"
    )
    LOGGER.info(f"sc: {sc}")
    player_status_records = yahoo_helper.get_league_players(sc, player_metadata)

    ##Fill out below
    player_status_series = pd.DataFrame.from_records(player_status_records)

    return player_status_series

@st.cache_resource(ttl=3600)
def get_yahoo_weeks(league_id: str, _auth_path: str) -> dict[int, str]:
    LOGGER.info(f"League id: {league_id}")
    sc = YahooFantasySportsQuery(
        auth_dir=_auth_path,
        league_id=league_id,
        game_code="nba"
    )
    
    LOGGER.info(f"sc: {sc}")
    weeks = sc.get_game_weeks_by_game_id("nba")

    return weeks

@st.cache_resource(ttl=3600
            , show_spinner = "Fetching matchup details from Yahoo. This should take about twenty seconds")
def get_yahoo_matchups(league_id: str, _auth_path: str) -> dict[int, str]:
    LOGGER.info(f"League id: {league_id}")

    teams = get_teams_dict(league_id, _auth_path)

    sc = YahooFantasySportsQuery(
        auth_dir=_auth_path,
        league_id=league_id,
        game_code="nba"
    )
    
    LOGGER.info(f"sc: {sc}")
    matchups = {team_name : {matchup.week : matchup for matchup in sc.get_team_matchups(team_id)} \
                        for team_id, team_name in teams.items()}
    return matchups

@st.cache_resource(ttl=3600)
def get_yahoo_schedule(league_id: str, _auth_path: str, player_metadata: pd.Series) -> dict[int, str]:
    yahoo_weeks = get_yahoo_weeks(league_id, _auth_path)
    nba_schedule = get_nba_schedule()

    league_players = get_player_statuses(league_id
                                            , _auth_path
                                            , player_metadata)

    league_players = league_players.set_index('Player')

    week_dict = {}
    for game_week in yahoo_weeks: 
        teams_playing_list = []
        for date in pd.date_range(game_week.start, game_week.end, freq = '1D'): 
            date_formatted = date.strftime("%m/%d/%Y %H:%M:%S")
            if date_formatted in nba_schedule:  
                teams_playing = nba_schedule[date_formatted]
                teams_playing_list = teams_playing_list + teams_playing

        game_counts = Counter(teams_playing_list)
        week_str = 'Week ' + str(game_week.display_name) + ': ' + \
                                    str(game_week.start) + ' to ' + \
                                    str(game_week.end)
        week_dict[week_str] = league_players['Team'].map(game_counts)

    return week_dict

def get_draft_results(league_id: str,_auth_path: str, player_metadata) -> List[League]:
    sc = YahooFantasySportsQuery(
        auth_dir=_auth_path,
        league_id=league_id, # Shouldn't actually matter what this is since we are retrieving the user's leagues
        game_code="nba"
    )
    LOGGER.info(f"sc: {sc}")

    mapper_table = get_yahoo_key_to_name_mapper().set_index('YAHOO_PLAYER_ID')

    try:
        draft_results = sc.get_league_draft_results()
    except:
        return None
            
    max_round = max([item.round for item in draft_results])
    n_picks = len(draft_results)
    n_drafters = int(n_picks/max_round)

    team_names = list(range(n_drafters))

    teams_dict = get_teams_dict(league_id, _auth_path)

    team_names = [teams_dict[int(draft_obj.team_key.split('.')[-1])] for draft_obj in draft_results[0:len(teams_dict)]]

    df = pd.DataFrame(index = list(range(max_round))
                      , columns = team_names)

    row = 0
    drafter = 0

    for draft_obj in draft_results:

        if len(draft_obj.player_key) > 0:
            drafted_player = mapper_table.loc[int(draft_obj.player_key.split('.')[-1])]

            team_name = teams_dict[int(draft_obj.team_key.split('.')[-1])]

            drafted_player_mod = ' '.join(drafted_player.values[0].split(' ')[0:2])

            drafted_player_mod = drafted_player_mod + ' (' + player_metadata[drafted_player_mod] + ')' 

            df.loc[row, team_name] = drafted_player_mod
            row, drafter = move_forward_one_pick(row, drafter, n_drafters)

    return df