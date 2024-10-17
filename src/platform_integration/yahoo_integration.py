import streamlit as st
from src.helpers.helper_functions import adjust_teams_dict_for_duplicate_names, get_selections_default
import pandas as pd
from src.platform_integration.platform_integration import PlatformIntegration
from src.tabs.drafting import clear_draft_board
from typing import Callable, List, Optional
from yfpy.models import League, Team, Roster

from requests.auth import HTTPBasicAuth
from src.platform_integration import yahoo_helper
from streamlit.logger import get_logger
from tempfile import mkdtemp
from yfpy.query import YahooFantasySportsQuery
from src.data_retrieval.get_data import get_nba_schedule, get_yahoo_key_to_name_mapper, get_player_metadata
from src.helpers.helper_functions import move_forward_one_pick, adjust_teams_dict_for_duplicate_names
from collections import Counter
from src.helpers.helper_functions import standardize_name

import json
import os
import pandas as pd
import requests
import shutil
import streamlit as st
import time

import numpy as np

LOGGER = get_logger(__name__)

class YahooIntegration(PlatformIntegration):

    @property
    def available_modes(self) -> list:
        return ['Draft Mode', 'Season Mode', 'Auction Mode']
    
    def get_available_modes(self) -> list:
        return self.available_modes

    def __init__(self):
        self.teams_dict = None
        
    def setup(self):
        """Collect information from the user, and use it to set up the integration.
        This function is not cached, so it is run every time the app re=runs

        Args:
            None

        Returns:
            None
        """
        self.auth_dir = self.get_yahoo_access_token()

        if self.auth_dir is None:
          st.stop()
        else:

          user_leagues = self.get_user_leagues()
          
          get_league_labels: Callable[[League], str] = lambda league: f"{league.name.decode('UTF-8')} ({league.season}-{league.season + 1} Season)"
          
          yahoo_league = st.selectbox(
            label='Which league should player data be pulled from?',
            options=user_leagues,
            format_func=get_league_labels,
            index=None,
            on_change = clear_draft_board
          )

          if yahoo_league is not None:
              self.league_id = yahoo_league.league_id

              #ZR: Ideally we could fix this for mock drafts with dummies
              self.n_drafters = len(self.get_teams_dict(self.league_id))
              self.team_names = self.get_team_names(self.league_id)

          else:
               self.league_id = st.number_input(label =  "For a mock draft, manually write in league ID (from URL, after mlid = )"
                               ,min_value = 0
                               ,value = None
                               ,key = 'yahoo_league_id'
                               , on_change = clear_draft_board)
               
               if self.league_id is not None:
                self.team_names = self.get_team_names(self.league_id)
                self.n_drafters = len(self.get_teams_dict(self.league_id)) 
               else:
                st.stop()


        st.write('Player info successfully retrieved from yahoo fantasy! :partying_face:')

    def set_defaults(self):

          if (st.session_state.mode == 'Season Mode'):

              player_metadata = get_player_metadata()

              team_players_df = self.get_rosters_df(self.league_id, player_metadata)
              self.n_drafters = team_players_df.shape[1]
              self.n_picks = team_players_df.shape[0]
              self.selections_default = team_players_df

              #make the selection df use a categorical variable for players, so that only players can be chosen, and it autofills
              
              #Just trying for now!
              #This is running into rate limits from Yahoo. Commenting it out for now
              #player_statuses = self.get_player_statuses(st.session_state.yahoo_league_id, player_metadata)
              #  
              #st.session_state['injured_players'].update(set(list(player_statuses['Player'][ \
              #                                                        (player_statuses['Status'] == 'INJ')
              #                                                        ]
              #                                                        )
              #                                                )
              #                                          )
              #st.session_state['schedule'] = self.get_yahoo_schedule(st.session_state.yahoo_league_id
              #                                                            , player_metadata)
              #  
              #st.session_state['matchups'] = self.get_yahoo_matchups(st.session_state.yahoo_league_id)

          else:

            self.n_drafters = len(self.get_teams_dict(self.league_id))
            self.team_names = list(self.get_teams_dict(self.league_id).values())
            self.n_picks = 13 #ZR: fix this
            self.selections_default = get_selections_default(self.n_picks,self.n_drafters)

              
    def get_yahoo_access_token(_self) -> Optional[str]:
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

    def clean_up_access_token(_self):
        shutil.rmtree(_self.auth_dir)

    @st.cache_data(ttl=3600, show_spinner = False)
    def get_rosters_df(_self
                       , league_id: str
                       , player_metadata: pd.Series) -> pd.DataFrame:
        """Get a dataframe with a column per team and cell per player chosen by that team

        Args:
            player_metadata

        Returns:
            DataFrame with roster info
        """
    
        teams_dict = _self.get_teams_dict(league_id)
        team_ids = list(teams_dict.keys())
        rosters_dict = _self.get_rosters_dict(league_id, team_ids)
        players_df = _self._get_players_df(rosters_dict, teams_dict, player_metadata).fillna(np.nan)

        return players_df

    @st.cache_data(ttl=300, show_spinner = False)
    def get_teams_dict(_self
                       , league_id: str) -> dict[int, str]:
        """Get a dictionary relating the names of teams to their associated IDs
        Sometimes yahoo shuts off its API access after a mock draft is over. For that reason, when the API can't be accessed, 
        this function returns the old value

        Args:
            league_id: A yahoo league ID

        Returns:
            DataFrame with roster info
        """

        LOGGER.info(f"League id: {league_id}")
        sc = YahooFantasySportsQuery(
            auth_dir= _self.auth_dir,
            league_id=league_id,
            game_code="nba"
        )
        LOGGER.info(f"sc: {sc}")
        try:
            teams = yahoo_helper.get_teams(sc)
        except: 
            #If yahoo isn't returning anything, just keep the same dict
            return _self.teams_dict

        teams_dict = {team.team_id: team.name.decode('UTF-8') for team in teams}

        teams_dict = adjust_teams_dict_for_duplicate_names(teams_dict)

        _self.teams_dict = teams_dict

        return teams_dict

    @st.cache_data(ttl=3600
                    , show_spinner = "Fetching rosters from your Yahoo league. This should take about ten seconds")
    def get_rosters_dict(_self
                         , league_id: str
                         , team_ids: list[int]) -> dict[int, Roster]:    
        """Get a dictionary relating team IDs to their rosters

        Args:
            league_Id: a yahoo league ID
            team_ids: a list of Yahoo team IDs

        Returns:
            Dictionary with roster info
        """

        league_id = league_id
        LOGGER.info(f"League id: {league_id}")
        sc = YahooFantasySportsQuery(
            auth_dir=_self.auth_dir,
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
    def get_user_leagues(_self) -> List[League]:
        """Get a list of leagues that the user is a part of 

        Args:
            None

        Returns:
            List of leagues
        """
        sc = YahooFantasySportsQuery(
            auth_dir=_self.auth_dir,
            league_id="", # Shouldn't actually matter what this is since we are retrieving the user's leagues
            game_code="nba"
        )
        LOGGER.info(f"sc: {sc}")
        leagues = yahoo_helper.get_user_leagues(sc)        
        
        # Sort in reverse chronological order
        sorted_leagues = sorted(leagues, key = lambda league: league.season, reverse=True)
        return sorted_leagues

    def _get_players_df(_self
                        , rosters_dict: dict[int, Roster]
                        , teams_dict: dict[int, str]
                        , player_metadata: pd.Series):
        """Get a dataframe with a column per team and cell per player chosen by that team, based on roster_dict

        Args:
            rosters_dict: dictionary with roster info
            teams_dict: dictionary relating team IDs to name
            player_metadata

        Returns:
            DataFrame with roster info
        """
        players_df = pd.DataFrame()

        team_players_dict = {}
        player_metadata.index = [' '.join(player.split('(')[0].split(' ')[0:2]) for player in player_metadata.index]

        max_team_size = 0

        for team_id, roster in rosters_dict.items():
            team_name = teams_dict[team_id]
            relevant_player_names = [
                f'{standardize_name(player.name.full)} ({player_metadata.loc[standardize_name(player.name.full)]})' #Appending position after player name
                for player in roster.players 
                if 
                    player.selected_position.position != 'IL'
                    and player.selected_position.position is not None
            ]

            for player in roster.players:
                if player.selected_position.position == 'IL':
                    st.session_state['injured_players'].add( \
                                f'{standardize_name(player.name.full)} ({player_metadata.loc[standardize_name(player.name.full)]})' 
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
    def get_player_statuses(_self
                            , league_id: str
                            , player_metadata: pd.Series) -> dict[int, str]:
        #get player statuses, such as whether or not they are injured. Currently not working
        
        LOGGER.info(f"League id: {league_id}")
        sc = YahooFantasySportsQuery(
            auth_dir= _self.auth_dir,
            league_id=league_id,
            game_code="nba"
        )
        LOGGER.info(f"sc: {sc}")
        #ignore
        player_status_records = yahoo_helper.get_league_players(sc, player_metadata)

        ##Fill out below
        player_status_series = pd.DataFrame.from_records(player_status_records)

        return player_status_series

    @st.cache_resource(ttl=3600)
    def get_yahoo_weeks(_self, league_id: str) -> dict[int, str]:
        #get dictionary of weeks in the fantasy season. Currently not being used
        LOGGER.info(f"League id: {league_id}")
        sc = YahooFantasySportsQuery(
            auth_dir= _self.auth_dir,
            league_id=league_id,
            game_code="nba"
        )
        
        LOGGER.info(f"sc: {sc}")
        weeks = sc.get_game_weeks_by_game_id("nba")

        return weeks

    @st.cache_resource(ttl=3600
                , show_spinner = "Fetching matchup details from Yahoo. This should take about twenty seconds")
    def get_yahoo_matchups(_self, league_id: str, _auth_path: str) -> dict[int, str]:
        #get which fantasy teams are against which others. Currently not being used 
        LOGGER.info(f"League id: {league_id}")

        teams = _self.get_teams_dict(_self.league_id)

        sc = YahooFantasySportsQuery(
            auth_dir= _self.auth_dir,
            league_id=league_id,
            game_code="nba"
        )
        
        LOGGER.info(f"sc: {sc}")
        matchups = {team_name : {matchup.week : matchup for matchup in sc.get_team_matchups(team_id)} \
                            for team_id, team_name in teams.items()}
        return matchups

    @st.cache_resource(ttl=3600)
    def get_yahoo_schedule(_self
                           ,league_id: str
                           , _auth_path: str
                           , player_metadata: pd.Series) -> dict[int, str]:
        #get numbers of games played per week by each team. Currently not being used

        yahoo_weeks = _self.get_yahoo_weeks(league_id, _auth_path)
        nba_schedule = get_nba_schedule()

        league_players = _self.get_player_statuses(league_id
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

    def get_draft_results(_self
                          , player_metadata):
        
        """Get a tuple with
        1) a dataframe reflecting the state of the draft, with np.nan in place of undrafted players
               structure is one column per team, one row per pick 
        2) a string representing the status of the draft 

        Args:
            player_metadata

        Returns:
            tuple
        """
            
        sc = YahooFantasySportsQuery(
            auth_dir=_self.auth_dir,
            league_id=_self.league_id, 
            game_code="nba"
        )
        LOGGER.info(f"sc: {sc}")

        mapper_table = get_yahoo_key_to_name_mapper().set_index('YAHOO_PLAYER_ID')

        try:
            draft_results = sc.get_league_draft_results()
        except Exception as e:
            if st.session_state.live_draft_active:
                return st.session_state.draft_results, 'Success'
            else:
                return st.session_state.draft_results, 'Draft has not started yet'
                
        max_round = max([item.round for item in draft_results])
        n_picks = len(draft_results)
        n_drafters = int(n_picks/max_round)

        _self.n_drafters = n_drafters #ZR: hack, this is bad 

        teams_dict = _self.get_teams_dict(_self.league_id)

        all_team_ids = [draft_obj.team_key.split('.')[-1] for draft_obj in draft_results[0:n_drafters]]

        team_names = ['Drafter ' + team_id if int(team_id) not in teams_dict else teams_dict[int(team_id)] for team_id in all_team_ids]

        df = pd.DataFrame(index = list(range(max_round))
                        , columns = team_names)

        row = 0
        drafter = 0

        if len(draft_results) > 0:
            try:
                if hasattr(draft_results[0], 'cost'):
                    if draft_results[0].cost is not None:
                        error_string = 'This is an auction, not a draft! Change the game mode'
                        return None, error_string
            except:     
                error_string = 'Something has gone wrong- refresh analysis in a few moments'
                return None, error_string
            
        draft_result_raw_df = pd.DataFrame([(draft_obj.player_key, draft_obj.team_key) for draft_obj in draft_results \
                                            if len(draft_obj.player_key) > 0]
                                        , columns = ['Player','Team'] )
        
        #next_team = teams_dict[int(draft_results[len(draft_result_raw_df)].team_key.split('.')[-1])]
        #if next_team != st.session_state.draft_seat:
        #    return None, True

        player_codes = draft_result_raw_df['Player'].str.split('.').str[-1].astype(int).values
        draft_result_raw_df['Player'] = ['RP' if x not in mapper_table.index else mapper_table.loc[x, 'NBA_PLAYER_NAME'] for x in player_codes]
        draft_result_raw_df['PlayerMod'] = draft_result_raw_df['Player'].apply(lambda x : ' '.join(x.split(' ')[0:2]))

        draft_result_raw_df['PlayerMod'] = ['RP' if x not in player_metadata.index else x + ' (' + player_metadata[x] + ')' 
                                            for x in draft_result_raw_df['PlayerMod'].astype(str)]
        draft_result_raw_df['Team'] = draft_result_raw_df['Team'].str.split('.').str[-1].astype(int)
        draft_result_raw_df['Team'] = ['Drafter ' + team_id if int(team_id) not in teams_dict else teams_dict[int(team_id)]
                                    for team_id in draft_result_raw_df['Team']]
                
        #ZR: I am pretty sure we don't need a for loop to do this
        for k, v in draft_result_raw_df.iterrows():
            df.loc[row, v['Team']] = v['PlayerMod']
            row, drafter = move_forward_one_pick(row, drafter, n_drafters)

        return df, 'Success'

    def get_auction_results(_self
                            , player_metadata):
        """Get a tuple with
        1) a dataframe reflecting the state of the auction. Structure is three columns; player/team/cost
        2) a string representing the status of the draft 

        Args:
            player_metadata

        Returns:
            tuple
        """
            
        sc = YahooFantasySportsQuery(
            auth_dir= _self.auth_dir
            ,league_id= _self.league_id
            ,game_code="nba"
        )
        LOGGER.info(f"sc: {sc}")

        mapper_table = get_yahoo_key_to_name_mapper().set_index('YAHOO_PLAYER_ID')

        try:
            draft_results = sc.get_league_draft_results()
        except Exception as e:
            if st.session_state.live_draft_active:
                return st.session_state.draft_results, 'Success'
            else:
                return st.session_state.draft_results, 'Auction has not started'

        if len(draft_results) > 0:
            try:
                if not hasattr(draft_results[0], 'cost'):
                    error_string = 'This is a draft, not an auction! Change the game mode'
                    return None, error_string
            except: 
                error_string = 'Something has gone wrong- refresh analysis in a few moments'
                return None, error_string

        teams_dict = _self.get_teams_dict(_self.league_id)

        def parse_draft_obj(draft_obj):

            drafted_player = mapper_table.loc[int(draft_obj.player_key.split('.')[-1])]

            team_id = draft_obj.team_key.split('.')[-1]
            team_name = 'Drafter ' + team_id if int(team_id) not in teams_dict else teams_dict[int(team_id)]

            drafted_player_mod = ' '.join(drafted_player.values[0].split(' ')[0:2])

            if drafted_player_mod in player_metadata.index:
                drafted_player_mod = drafted_player_mod + ' (' + player_metadata[drafted_player_mod] + ')' 
            else:
                drafted_player_mod = 'RP'

            row = pd.Series({'Player' : drafted_player_mod
                                ,'Cost' : draft_obj.cost
                                ,'Team' : team_name})
            return row
        
        
        df = pd.concat([parse_draft_obj(draft_obj) for draft_obj in draft_results], axis = 1).T

        return df, 'Success'

    def get_n_picks(_self, league_id):
        return _self.n_picks
    
    def get_team_names(_self, league_id):
        return list(_self.get_teams_dict(league_id).values())