import streamlit as st
from src.helpers.helper_functions import adjust_teams_dict_for_duplicate_names, get_data_key
import pandas as pd
from src.platform_integration.platform_integration import PlatformIntegration
from src.tabs.drafting import clear_draft_board, increment_and_reset_draft
from typing import Callable, List, Optional
from espn_api.basketball import League
import re 

from requests.auth import HTTPBasicAuth
from src.platform_integration import yahoo_helper
from streamlit.logger import get_logger
from tempfile import mkdtemp
from yfpy.query import YahooFantasySportsQuery
from src.helpers.helper_functions import move_forward_one_pick, adjust_teams_dict_for_duplicate_names
from collections import Counter
from src.helpers.helper_functions import get_fixed_player_name

import json
import os
import pandas as pd
import requests
import shutil
import streamlit as st
import time

import numpy as np

LOGGER = get_logger(__name__)

class ESPNIntegration(PlatformIntegration):

    def __init__(self):
        self.division_id = None #as far as I know, ESPN doesnt have divisions

    @property
    def description_string(self) -> str:
        return 'Retrieve from ESPN'
    
    def get_description_string(self) -> str:
        return self.description_string
    
    @property
    def player_name_column(self) -> str:
        return 'ESPN_NAME'
    
    def get_player_name_column(self) -> str:
        return self.player_name_column
    
    @property
    def available_modes(self) -> list:
        return ['Season Mode']
    
    def get_available_modes(self) -> list:
        return self.available_modes

    def setup(self):
        """Collect information from the user, and use it to set up the integration.
        This function is not cached, so it is run every time the app re=runs

        Args:
            None

        Returns:
            None
        """
        
        if ('espn_s2' not in st.session_state) or ('espn_swid' not in st.session_state):
            self.get_espn_credentials()
        else:
            user_leagues = self.get_user_leagues()

            def get_league_labels(league):
                season = league['metaData']['entry']['seasonId']
                return f"{league['metaData']['entry']['groups'][0]['groupName']} ({str(season - 1)}-{str(season)} Season)"
   
            espn_league = st.selectbox(
                label='Which league should player data be pulled from?',
                options=user_leagues,
                format_func=get_league_labels,
                index=None,
                on_change = increment_and_reset_draft,
               )
                        
            if espn_league is not None:
              self.league_id = espn_league['id']
              self.year = espn_league['metaData']['entry']['seasonId']

              #ZR: Ideally we could fix this for mock drafts with dummies
              self.team_names = self.get_team_names(self.league_id)
              self.n_drafters = len(self.get_teams_dict(self.league_id)) 

              team_players_df = self.get_rosters_df(self.league_id)
              self.selections_default = team_players_df

              self.n_picks = team_players_df.shape[0]
            else: 
              st.stop()

    def get_teams_dict(self
                       , league_id
                       , division_id = None):

        league = League(league_id = league_id.split(':')[1]
                , year = self.year
                , espn_s2= st.session_state.espn_s2
                , swid=st.session_state.espn_swid)
        
        return {team.team_id : team.team_name for team in league.teams}
    
    def get_team_names(_self, league_id, division_id = None):
        return list(_self.get_teams_dict(league_id).values())
        
    def get_user_leagues(_self):

        query = "https://fan.api.espn.com/apis/v2/fans/%7B" + st.session_state.espn_swid + \
                "%7D?displayHiddenPrefs=true&context=fantasy&useCookieAuth=true&source=fantasyapp-ios&featureFlags=challengeEntries"
        
        response = requests.get(query)

        leagues = response.json()['preferences']

        sorted_leagues = sorted(leagues, key = lambda league: league['metaData']['entry']['seasonId'], reverse=True)
        return sorted_leagues

    def get_rosters_df(self, league_id):
        """Get a dataframe with a column per team and cell per player chosen by that team

        Args:
            league_id: ESPN league id

        Returns:
            DataFrame with roster info
        """
        league = League(league_id = league_id.split(':')[1]
                        , year = self.year
                        , espn_s2= st.session_state.espn_s2
                        , swid=st.session_state.espn_swid)
                
        teams = league.teams

        team_players_dict = {team.team_name :
                [get_fixed_player_name(player.name, get_data_key('info')) for player in team.roster] 
                              for team in teams}
                
        max_team_size = max([len(x) for x in team_players_dict.values()])

        players_df = pd.DataFrame()

        for team_name, player_names in team_players_dict.items():
            if len(player_names) < max_team_size :
                player_names.extend([None] * (max_team_size - len(player_names)))
            
            players_df.loc[:,team_name] = player_names

        return players_df
        


    @st.dialog("Authenticate with ESPN")
    def get_espn_credentials(self):

        """Prompt user for S2 and SWID
        """

        st.caption('''Find your ESPN s2 and SWID by opening a tab with ESPN, logging into your account, and using 
                   [this web plug-in](https://chromewebstore.google.com/detail/espn-cookie-finder/oapfffhnckhffnpiophbcmjnpomjkfcj?hl=en&pli=1)
                   . SWID can be copy-pasted with or without brackets''')
        espn_s2 = st.text_input('Enter your ESPN s2')
        espn_swid = st.text_input('Enter your ESPN SWID')

        if (len(espn_s2) ==0) or (len(espn_swid) ==0):
            st.stop()
        else:
            st.session_state.espn_s2 = espn_s2
            st.session_state.espn_swid = re.sub('{|}','',espn_swid)
            st.rerun()

    def get_draft_results(_self):
        pass

    def get_auction_results():
        pass

    def get_n_picks(_self, league_id) -> int:
        return _self.n_picks
