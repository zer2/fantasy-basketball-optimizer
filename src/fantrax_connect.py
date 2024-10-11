from fantraxapi import FantraxAPI
import streamlit as st
from src.helper_functions import adjust_teams_dict_for_duplicate_names
import pandas as pd

def get_api(league_id):
    return FantraxAPI(league_id)

@st.cache_data(ttl = 3600)
def get_n_picks(league_id):
    api = get_api(league_id)
    teams = get_teams_dict(league_id)
    return min( sum([x['max'] for x in api._request("getTeamRosterInfo", teamId=list(teams.values())[0])['miscData']['statusTotals'] \
                if x['name'] != "Inj Res"]), 16
                )

@st.cache_data(ttl = 3600)
def get_division_dict(league_id):
    api = get_api(league_id)
    divisions = {x['name'] : x['id'] for x in api._request("getStandings", view="SCHEDULE")['displayedLists']['tabs'] 
             if x['name'] not in ['All','Combined','Results','Season Stats','Playoffs']}
    return divisions

@st.cache_data(ttl = 3600)
def get_teams_dict(league_id):
    teams_dict = {t['name'] : t['id'] for t in get_api(league_id)._request("getFantasyTeams")['fantasyTeams']}
    return adjust_teams_dict_for_duplicate_names(teams_dict)

@st.cache_data(ttl = 3600)
def get_teams_dict_by_division(league_id, division_id):
    api = get_api(league_id)

    #ZR: Sometimes the first entry in tablelist has the information. sometimes it ie empty and we need to check the second
    standings_rows = api._request("getStandings", view= division_id)['tableList'][0]['rows']
    if len(standings_rows) == 0:
        standings_rows = api._request("getStandings", view= division_id)['tableList'][1]['rows']

    res = {x['fixedCells'][1]['content']  : x['fixedCells'][1]['teamId'] 
     for x in standings_rows }
    
    return res

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
    
def get_team_info(api, team_id):
    return api._request("getTeamRosterInfo", teamId=team_id)['tables'][0]['rows']

def get_fantrax_roster(league_id
                        , player_metadata):
    
    api = get_api(league_id)
    
    rosters = { name : [ get_fixed_player_name(z['scorer']['name'], player_metadata) for z in get_team_info(api, team_id) if 'scorer' in z] 
           for name, team_id in st.session_state.teams_dict.items() 
          }
            
    return rosters
    
def get_player_statuses(league_id, player_metadata):
    return None

def get_draft_results(league_id
                      ,player_metadata):
    
    rosters = get_fantrax_roster(league_id
                , player_metadata)
    
    rosters_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in rosters.items() ]))
    
    return rosters_df, 'Success'
    
#def get_fantrax_matchups(league_id: str, _auth_path: str) -> dict[int, str]:
#
#    teams = get_teams_dict(league_id, _auth_path)
#
#    matchups = {team_name : {matchup.week : matchup.opponent for matchup in get_team_matchups(league_id, team_id)} \
#                        for team_id, team_name in teams.items()}
#    return matchups