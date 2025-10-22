from typing import List
from yfpy.models import League, Team, Roster, Player
from yfpy.query import YahooFantasySportsQuery
import streamlit as st
from streamlit.logger import get_logger
import pandas as pd
LOGGER = get_logger(__name__)

def get_user_leagues(sc: YahooFantasySportsQuery) -> List[League]:
    """
    Generates a list of leagues that the user has been a part of
    
    Parameters:
    - sc (object): The YahooFantasySportsQuery object.

    Returns:
    - List[League]: A list of the leagues.
    """
    leagues: List[dict[str, League]] = sc.get_user_leagues_by_game_key(game_key="nba") # type: ignore

    if isinstance(leagues[0], dict):
        # The yfpy method for some reason sometimes returns a list of dicts rather than the leagues directly
        leagues = [league_dict["league"] for league_dict in leagues] 

    try:
        leagues: List[dict[str, League]] = sc.get_user_leagues_by_game_key(game_key="nba") # type: ignore

        if isinstance(leagues[0], dict):
            # The yfpy method for some reason sometimes returns a list of dicts rather than the leagues directly
            leagues = [league_dict["league"] for league_dict in leagues] 
    except: #if there are no user leagues, yfpy will error out 
        leagues = [] 
    return leagues

def get_teams(sc: YahooFantasySportsQuery) -> List[Team]:
    """
    Generates a list of teams for the given league id
    
    Parameters:
    - sc (object): The YahooFantasySportsQuery object.

    Returns:
    - List[Team]: A list of the teams in the league.
    """

    teams = sc.get_league_teams()
    return teams

def get_team_roster(sc: YahooFantasySportsQuery, team_id: int) -> Roster:
    """
    Retrieves the roster for a given team id for the current week
    
    Parameters:
    - sc (object): The YahooFantasySportsQuery object.

    Returns:
    - Roster: The roster for the given team id.
    """
    roster = sc.get_team_roster_by_week(team_id=team_id)
    return roster

def get_league_players(sc: YahooFantasySportsQuery) -> List[League]:
    """
    
    Parameters:
    - sc (object): The YahooFantasySportsQuery object.

    Returns:
    - List[League]: A list of the leagues.
    """
    player_dicts: List[dict[str, Player]] = sc.get_league_players() # type: ignore

    #ZR: AXY I think this code never gets called and is irrelevant. So I am not fixing it for now

    player_status_records = [
            {'Status': player.status
            , 'Eligible Positions' : player.display_position
            , 'Team' : player.editorial_team_abbr
            , 'ID' : player.player_id
            } #Add team
            for player in player_dicts if player.player_id != 10401 #This is a hack, to deal with the second Jaylin Williams
                    ]
    return player_status_records

def get_draft_results(sc: YahooFantasySportsQuery) -> List[League]:
    """
    
    Parameters:
    - sc (object): The YahooFantasySportsQuery object.

    Returns:
    - unsure?
    """
    draft_results = sc.get_league_draft_results() 
    return draft_results
