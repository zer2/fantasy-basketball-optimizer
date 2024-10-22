from fantraxapi import FantraxAPI
import streamlit as st
from src.helpers.helper_functions import adjust_teams_dict_for_duplicate_names, get_selections_default
import pandas as pd
from src.platform_integration.platform_integration import PlatformIntegration
from src.tabs.drafting import clear_draft_board
from src.data_retrieval.get_data import get_player_metadata

class FantraxIntegration(PlatformIntegration):

    @property
    def available_modes(self) -> list:
        return ['Draft Mode', 'Season Mode']
    
    def get_available_modes(self) -> list:
        return self.available_modes

    @st.cache_data()
    def get_api(_self
                , league_id : str):
        """Get an API object from the FantraxAPI package. It is associated with a league

        Args:
            league_id: A fantrax league ID

        Returns:
            API object
        """

        return FantraxAPI(league_id)
    
    def setup(self):
        """Collect information from the user, and use it to set up the integration.
        This function is not cached, so it is run every time the app re=runs

        Args:
            None

        Returns:
            None
        """

        self.league_id = st.text_input(
            label='Which league ID should player data be pulled from? (Find league ID after /league/ in your draft room URL)'
            ,value = None
            ,on_change = clear_draft_board
          )    
        
        if self.league_id is None:
            st.stop()
        else:
                
            division_dict = self.get_division_dict(self.league_id)

            if len(division_dict) == 0:
                self.division_id = None
            else:
                division = st.selectbox(label = 'Which division are you in?'
                                        ,options = list(division_dict.keys())
                                        , on_change = clear_draft_board)
                
                self.division_id = division_dict[division]
                
            self.teams_dict = self.get_teams_dict_by_division(self.league_id, self.division_id)

        self.team_names = list(self.teams_dict.keys())  
        self.n_drafters = len(self.team_names)
        self.n_picks = self.get_n_picks(self.league_id)

        if (st.session_state.mode == 'Draft Mode'):
            self.selections_default = get_selections_default(self.n_picks
                                                            ,self.n_drafters)
        else:
            #this is all messed up lol

            st.session_state.player_metadata = get_player_metadata()
            player_metadata = st.session_state.player_metadata.copy()
            player_metadata.index = [' '.join(player.split('(')[0].split(' ')[0:2]) for player in player_metadata.index]

            self.selections_default = self.get_rosters_df(player_metadata)
            self.n_drafters = st.session_state.selections_default.shape[1]
            self.n_picks = st.session_state.selections_default.shape[0]

            st.session_state['schedule'] = None
            st.session_state['matchups'] = None

        st.write('Team info successfully retrieved from fantrax! :partying_face:')
        st.write('Note that for fantrax, only active players are pulled in and counted')

    @st.cache_data()
    def get_division_dict(_self
                          , league_id: str) -> dict:
        """Get a dictionary relating the names of divisions to their associated IDs

        Args:
            league_id: A fantrax league ID

        Returns:
            dict of structure {name : id}
        """
        api = _self.get_api(league_id)
        api_response = api._request("getStandings", view="SCHEDULE")['displayedLists']['tabs'] 
        divisions = {x['name'] : x['id'] for x in api_response
                if x['name'] not in ['All','Combined','Results','Season Stats','Playoffs']}
        return divisions

    @st.cache_data()
    def get_teams_dict_by_division(_self
                                   , league_id : str
                                   , division_id : str) -> dict:
        """Get a dictionary relating the names of teams to their associated IDs, filtered by division ID

        Args:
            league_id: A fantrax league ID
            division_id: A fantrax division ID

        Returns:
            dict with structure {team id : team name}
        """
        api = _self.get_api(league_id)
        if division_id is None:
            teams_dict = {t['name'] : t['id'] for t in api._request("getFantasyTeams")['fantasyTeams']}
            return adjust_teams_dict_for_duplicate_names(teams_dict)
        else:
            standings_rows = api._request("getStandings", view= division_id)['tableList'][0]['rows']
            if len(standings_rows) == 0:
                standings_rows = api._request("getStandings", view= division_id)['tableList'][1]['rows']

            res = {x['fixedCells'][1]['content']  : x['fixedCells'][1]['teamId'] for x in standings_rows }
            return res

    @st.cache_data(ttl = 3600)
    def get_n_picks(_self
                    , league_id : str):
        """Get the number of picks that will be made in a fantrax draft

        Args:
            league_id: A fantrax league ID

        Returns:
            Integer, number of picks
        """
        api = _self.get_api(league_id)
        api_response = api._request("getTeamRosterInfo", teamId=list(_self.teams_dict.values())[0])['miscData']['statusTotals']
        return min( sum([x['max'] for x in api_response \
                    if x['name'] != "Inj Res"]), 16
                    )
    
    def get_team_info(_self
                      , team_id : str) -> dict:
        """Get a dictionary of team info,

        Args:
            team_id: A fantrax division ID

        Returns:
            dictionary with team info
        """
        api = _self.get_api(_self.league_id)
        res = api._request("getTeamRosterInfo", teamId=team_id)['tables'][0]['rows']
        return res
    
    def get_rosters_df(_self
                       , player_metadata):
        """Get a dataframe with a column per team and cell per player chosen by that team

        Args:
            player_metadata

        Returns:
            DataFrame with roster info
        """
    
        exclusions = ('3') if st.session_state.mode == 'Season Mode' else ()

        def get_rosters(team_id):
            roster = []
            for z in _self.get_team_info(team_id):
                if 'scorer' in z: 
                    player = get_fixed_player_name(z['scorer']['name'], player_metadata)
                    if z['statusId'] in exclusions:
                        st.session_state['injured_players'].add(player)
                    else:
                        roster = roster + [player]
            return roster
        
        rosters = { name : get_rosters(team_id) 
                            for name, team_id in _self.teams_dict.items() 
            }

        []
        
        rosters_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in rosters.items() ]))
                
        return rosters_df
    
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
            
        rosters_df = _self.get_rosters_df(player_metadata)
        
        return rosters_df, 'Success'
    
    def get_auction_results(_self, league_id, player_metadata):
        #not implemented
        return None
    
    @st.cache_data()
    def get_team_names(_self
                       , league_id
                       , division_id) -> list:
        #get list of team names. Cached by league id and division id; when they change the function will refresh
        return list(_self.teams_dict.keys())
    
    
@st.cache_data()
def get_fixed_player_name(player_name : str
                          , _player_metadata) -> str:
    
    """Fix player name string to adhere to common standard

    Args:
        player_name: string
        player_metadata

    Returns:
        fixed name string
     """

    def name_renamer(name):
      if name == 'Nicolas Claxton':
         name = 'Nic Claxton'
      if name == 'C.J. McCollum':
         name = 'CJ McCollum'
      if name == 'R.J. Barrett':
         name = 'RJ Barrett'
      if name == 'Herb Jones':
         name = 'Herbert Jones'
      if name == 'Cam Johnson':
         name = 'Cameron Johnson'
      if name == 'O.G. Anunoby':
         name = 'OG Anunoby'
      if name == 'Alexandre Sarr':
         name = 'Alex Sarr'
      if name == 'Cameron Thomas':
         name = 'Cam Thomas'
      if name == 'Kelly Oubre Jr.':
          name = 'Kelly Oubre'
      return name
    
    player_name = name_renamer(player_name)

    if player_name in _player_metadata.index:

        return player_name + ' (' + _player_metadata[player_name] + ')'
    else:
        return 'RP'