from fantraxapi import FantraxAPI
import streamlit as st
from src.helper_functions import adjust_teams_dict_for_duplicate_names, get_selections_default
import pandas as pd
from src.platform_integration.platform_integration import PlatformIntegration
from src.drafting import clear_draft_board

class FantraxIntegration(PlatformIntegration):

    @property
    def available_modes(self) -> list:
        return ['Draft Mode', 'Season Mode']
    
    def get_available_modes(self) -> list:
        return self.available_modes

    @st.cache_data()
    def get_api(_self, league_id):
        return FantraxAPI(league_id)
    
    def setup(self):

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

                st.session_state.player_metadata = st.session_state.player_stats['Position']
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
    def get_division_dict(_self, league_id) -> dict:
        api = _self.get_api(league_id)
        api_response = api._request("getStandings", view="SCHEDULE")['displayedLists']['tabs'] 
        divisions = {x['name'] : x['id'] for x in api_response
                if x['name'] not in ['All','Combined','Results','Season Stats','Playoffs']}
        return divisions

    @st.cache_data()
    def get_teams_dict_by_division(_self, league_id, division_id) -> dict:
        #return a dictionary with structure {team id : team name}
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
    def get_n_picks(_self, league_id):
        api = _self.get_api(league_id)
        api_response = api._request("getTeamRosterInfo", teamId=list(_self.teams_dict.values())[0])['miscData']['statusTotals']
        return min( sum([x['max'] for x in api_response \
                    if x['name'] != "Inj Res"]), 16
                    )
    
    def get_team_info(_self,api, team_id):
        res = api._request("getTeamRosterInfo", teamId=team_id)['tables'][0]['rows']
        return res
    
    def get_rosters_df(_self, player_metadata):
    
        api = _self.get_api(_self.league_id)

        exclusions = ('2','3') if st.session_state.mode == 'Season Mode' else ()
        
        rosters = { name : [ get_fixed_player_name(z['scorer']['name'], player_metadata) \
                            for z in _self.get_team_info(api, team_id) if 'scorer' in z and z['statusId'] not in exclusions] 
                            for name, team_id in _self.teams_dict.items() 
            }
        
        rosters_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in rosters.items() ]))
                
        return rosters_df
    
    def get_draft_results(_self, player_metadata):
            
        rosters_df = _self.get_rosters_df(player_metadata)
        
        return rosters_df, 'Success'
    
    def get_auction_results(_self, league_id, player_metadata):
        #not implemented
        return None
    
    def get_team_names(_self, league_id):
        return list(_self.teams_dict.keys())
    
    
def get_fixed_player_name(player_name, player_metadata):

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

    if player_name in player_metadata.index:

        return player_name + ' (' + player_metadata[player_name] + ')'
    else:
        return 'RP'