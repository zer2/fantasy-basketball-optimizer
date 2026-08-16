import os

import pytest

from streamlit.testing.v1 import AppTest

from espn_api.basketball import League as NBALeague
from espn_api.wbasketball import League as WNBALeague

from src.platform_integration.espn_integration import ESPNIntegration \
                                                    , ESPN_GAME_ABBREVS \
                                                    , KNOWN_GAME_ABBREVS \
                                                    , get_espn_league_class \
                                                    , entry_game_abbrev \
                                                    , filter_entries_for_league
from src.platform_integration.fantrax_integration import FantraxIntegration
from src.platform_integration.yahoo_integration import YahooIntegration

def test_get_espn_league_class():
    """Make sure the right espn_api League class is selected for each league type"""

    assert get_espn_league_class('WNBA') is WNBALeague
    assert get_espn_league_class('NBA') is NBALeague

    #anything that isn't WNBA should fall back to the NBA class
    assert get_espn_league_class('MLB') is NBALeague

def test_entry_game_abbrev():
    """Make sure game abbreviations are extracted from fan API league ids without raising"""

    assert entry_game_abbrev({'id' : 'wfba:123'}) == 'wfba'
    assert entry_game_abbrev({'id' : 'fba:9'}) == 'fba'

    #ids without a colon or that aren't strings should produce None rather than raising
    assert entry_game_abbrev({'id' : 'nocolon'}) is None
    assert entry_game_abbrev({'id' : 123}) is None

def test_filter_entries_for_league():
    """Make sure fan API entries are filtered for WNBA but left untouched for other leagues"""

    wnba_entry = {'id' : 'wfba:123'}
    nba_entry = {'id' : 'fba:9'}
    nfl_entry = {'id' : 'ffl:55'}
    nhl_entry = {'id' : 'fhl:7'}
    mlb_entry = {'id' : 'flb:8'}
    unknown_prefix_entry = {'id' : 'xyz:42'}
    no_colon_entry = {'id' : 'nocolon'}
    non_string_id_entry = {'id' : 123}

    entries = [wnba_entry, nba_entry, nfl_entry, nhl_entry, mlb_entry
               , unknown_prefix_entry, no_colon_entry, non_string_id_entry]

    #WNBA keeps wfba entries plus anything that can't be recognized, and drops other known games
    wnba_result = filter_entries_for_league(entries, 'WNBA')
    assert wnba_result == [wnba_entry, unknown_prefix_entry, no_colon_entry, non_string_id_entry]

    #NBA gets the input back unchanged: same objects, same order
    nba_result = filter_entries_for_league(entries, 'NBA')
    assert len(nba_result) == len(entries)
    assert all(result_entry is original_entry for result_entry, original_entry
               in zip(nba_result, entries))

def test_espn_contract_constants():
    """Make sure the module-level abbreviation contract is in place"""

    assert ESPN_GAME_ABBREVS == {'NBA' : 'fba', 'WNBA' : 'wfba'}
    assert set(KNOWN_GAME_ABBREVS) == {'ffl', 'fba', 'fhl', 'flb', 'wfba'}

def test_supported_leagues():
    """Make sure ESPN supports both basketball leagues while others default to NBA-only"""

    assert ESPNIntegration().supported_leagues == ['NBA', 'WNBA']

    #Yahoo and Fantrax don't declare their own list, so they get the base class default
    assert YahooIntegration().supported_leagues == ['NBA']
    assert FantraxIntegration().supported_leagues == ['NBA']

@pytest.mark.skipif(os.environ.get('WNBA_LIVE') != '1', reason = 'set WNBA_LIVE=1 to run live API tests')
def test_wnba_league_offers_espn_only():
    """Make sure switching the app to WNBA offers only manual entry and the ESPN integration"""

    at = AppTest.from_file('app.py', default_timeout = 600)
    at.run()

    at.selectbox('league').select('WNBA')
    at.run()

    assert at.selectbox('data_source').options == ['Enter your own data', 'Retrieve from ESPN']
    assert not at.exception

def test_nba_league_offers_all_integrations():
    """Make sure the NBA default still offers Yahoo, Fantrax, and ESPN in the original order"""

    at = AppTest.from_file('app.py', default_timeout = 600)
    at.run()

    assert at.selectbox('data_source').options == ['Enter your own data', 'Retrieve from Yahoo'
                                                   , 'Retrieve from Fantrax', 'Retrieve from ESPN']

    #the offline NBA run fails later at the Snowflake connection, which is expected;
    #just make sure nothing blew up inside the integration/league-settings code itself
    for exception in at.exception:
        stack_trace = '\n'.join(exception.stack_trace)
        assert 'platform_integration' not in stack_trace
        assert 'league_settings' not in stack_trace
