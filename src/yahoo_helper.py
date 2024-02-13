from yfpy.models import Team, Roster
from yfpy.query import YahooFantasySportsQuery
from streamlit.logger import get_logger
LOGGER = get_logger(__name__)

def _get_teams_dict(teams):
    """
    Extracts team ids and names from the provided teams data.
    
    Parameters:
    - teams (list): A list of Team instances.
    
    Returns:
    - dict: A dictionary mapping team ids to team names.
    """
    try:
        return {team.team_id: team.name.decode('UTF-8') for team in teams}
    except Exception as e:
        LOGGER.exception("Failed to get team ids")
        raise e  # Reraise the exception after logging it

def get_teams(sc: YahooFantasySportsQuery):
    """
    Generates a dict of team ids mapped to team names
    
    Parameters:
    - sc (object): The YahooFantasySportsQuery object.

    Returns:
    - dict: A dictionary mapping the team ids to team names.
    """

    teams = sc.get_league_teams()
    teams_dict = _get_teams_dict(teams)
    return teams_dict

def get_team_roster(sc: YahooFantasySportsQuery, team_id: int) -> Roster:
    """
    Retrieves the roster for a given team id for the current week
    
    Parameters:
    - sc (object): The YahooFantasySportsQuery object.

    Returns:
    - Roster: The roster for the given team id.
    """
    roster = sc.get_team_roster_by_week(team_id)
    return roster