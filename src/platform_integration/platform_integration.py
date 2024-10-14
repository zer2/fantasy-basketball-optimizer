import abc
import pandas as pd

class PlatformIntegration(abc.ABC):

    @property
    @abc.abstractmethod
    def available_modes() -> list:
        #return list of modes available with this integration
        pass

    @abc.abstractmethod
    def setup():
        #get info from user about the league or division, and save results back as fields of the class 
        #if necessary, run any authentication steps
        pass

    @abc.abstractmethod
    def get_rosters_df() -> pd.DataFrame:
        pass

    #
    # 
    # Drafting methods 
    #
    #

    @abc.abstractmethod
    def get_n_picks() -> int:
        #return the number of picks that each time will make during the draft 
        pass

    @abc.abstractmethod
    def get_team_names() -> list:
        #get a list of team names
        pass

    @abc.abstractmethod
    def get_draft_results() -> tuple[pd.DataFrame, str]:
        #get a tuple with
        # 1) a dataframe reflecting the state of the draft, with np.nan in place of undrafted players
        #       structure is one column per team, one row per pick 
        # 2) a string representing the status of the draft 
        pass

    @abc.abstractmethod
    def get_auction_results() -> tuple[pd.DataFrame, str]:
        #get a tuple with
        # 1) a dataframe reflecting the state of the draft, with np.nan in place of undrafted players
        #       structure is one column per team, one row per pick 
        # 2) a string representing the status of the draft 
        pass


