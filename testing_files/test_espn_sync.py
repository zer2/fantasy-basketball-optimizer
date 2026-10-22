# testing_files/test_espn_sync.py
# Offline unit tests for ESPN platform syncing, the platform player lookup's name fallback, and team lookup resilience.

import pandas as pd
import pytest

from backend.data_retrieval import strip_accents
from backend.player_identity import RP_PLAYER_ID, make_player_identity
from backend.platform_integration.base import PlatformConfig
from backend.platform_integration.helpers import build_platform_player_id_lookup
from backend.platform_integration.integrations.espn import ESPNIntegration


_DONCIC_ID = 1629029
_JOKIC_ID = 203999
_WASHINGTON_ID = 1629023
_MURPHY_ID = 1630530
_CLAXTON_ID = 1629651
_HUNTER_ID = 1629631
_HARDAWAY_ID = 203501
_SYNTHETIC_ID = -7          # a projection name that resolved to no NBA id


def _sample_registry() -> dict:
    return {
        _DONCIC_ID: make_player_identity(_DONCIC_ID, 'Luka Doncic', 'PG,SG'),
        _JOKIC_ID: make_player_identity(_JOKIC_ID, 'Nikola Jokic', 'C'),
        _WASHINGTON_ID: make_player_identity(_WASHINGTON_ID, 'PJ Washington', 'PF'),
        _MURPHY_ID: make_player_identity(_MURPHY_ID, 'Trey Murphy', 'SF,SG'),
        _CLAXTON_ID: make_player_identity(_CLAXTON_ID, 'Nicolas Claxton', 'C'),
        _HUNTER_ID: make_player_identity(_HUNTER_ID, 'DeAndre Hunter', 'SF'),
        _HARDAWAY_ID: make_player_identity(_HARDAWAY_ID, 'Tim Hardaway Jr.', 'SG'),
        _SYNTHETIC_ID: make_player_identity(_SYNTHETIC_ID, 'Rookie Nobody Jr.', 'PG'),
    }


def _sample_unified_table() -> pd.DataFrame:
    return pd.DataFrame({
        'NBA_PLAYER_ID': [_DONCIC_ID, _JOKIC_ID, _WASHINGTON_ID, _MURPHY_ID, _CLAXTON_ID, _HUNTER_ID, _HARDAWAY_ID],
        'MASTER_PLAYER_NAME': ['Luka Doncic', 'Nikola Jokic', 'PJ Washington', 'Trey Murphy III', 'Nicolas Claxton',
                               "De'Andre Hunter", 'Tim Hardaway Jr.'],
        'ESPN_NAME': ['Luka Doncic', None, 'P.J. Washington', 'Trey Murphy', None, 'DeAndre Hunter', None],
        'DARKO_NAME': ['Luka Doncic', 'Nikola Jokic', 'PJ Washington', 'Trey Murphy III', 'Nic Claxton',
                       'DeAndre Hunter', 'Tim Hardaway Jr.'],
        'ROTOWIRE_NAME': ['Luka Doncic', 'Nikola Jokic', 'P.J. Washington', 'Trey Murphy', 'Nicolas Claxton',
                          "De'Andre Hunter", 'Tim Hardaway Jr.'],
        'BBM_NAME': ['Luka Doncic', 'Nikola Jokic', 'PJ Washington', 'Trey Murphy III', 'Nic Claxton',
                     'DeAndre Hunter', 'Tim Hardaway Jr.'],
        'HTB_NAME': ['Luka Doncic', 'Nikola Jokic', 'PJ Washington', 'Trey Murphy', 'Nicolas Claxton',
                     'DeAndre Hunter', 'Tim Hardaway Jr.'],
    })


def _sample_name_resolvers() -> tuple[dict[str, int], dict[str, int]]:
    """What get_player_name_resolvers gives for the sample table: every spelling in a name column -> its NBA id."""
    table = _sample_unified_table()
    exact = {name: int(nba_player_id) for column in table.columns if column.endswith('_NAME')
             for name, nba_player_id in zip(table[column], table['NBA_PLAYER_ID']) if isinstance(name, str)}
    return exact, {strip_accents(name): nba_player_id for name, nba_player_id in exact.items()}


def _sample_lookup():
    return build_platform_player_id_lookup(_sample_registry(), 'ESPN_NAME', _sample_unified_table(),
                                           _sample_name_resolvers())


class _MockPlayer:
    def __init__(self, name: str, player_id: int = 1234, injured: bool = False, lineup_slot: str = 'BE'):
        self.name = name
        self.playerId = player_id
        self.injured = injured
        self.lineupSlot = lineup_slot


class _MockTeam:
    def __init__(self, team_id, team_name: str, roster_players: list[_MockPlayer]):
        self.team_id = team_id
        self.team_name = team_name
        self.roster = roster_players


class _MockLeague:
    def __init__(self, teams: list[_MockTeam], lineup_slot_counts: dict = None, draft_type: str = 'SNAKE'):
        self.teams = teams
        settings = {'rosterSettings': {'lineupSlotCounts': lineup_slot_counts or {}},
                    'draftSettings': {'type': draft_type}}
        self.espn_request = type('EspnRequest', (), {'get_league': lambda request: {'settings': settings}})()


def test_the_lookup_tries_the_platform_key_then_every_spelling_the_table_knows():
    lookup = _sample_lookup()
    assert lookup.resolve_player_id('P.J. Washington', 'P.J. Washington') == _WASHINGTON_ID   # ESPN_NAME
    assert lookup.resolve_player_id('Nic Claxton', 'Nic Claxton') == _CLAXTON_ID              # DARKO_NAME
    assert lookup.resolve_player_id('Trey Murphy III', 'Trey Murphy III') == _MURPHY_ID       # MASTER / BBM
    assert lookup.resolve_player_id('Luka Dončić', 'Luka Dončić') == _DONCIC_ID               # without accents
    assert lookup.resolve_player_id('Nikola Jokić', 'Nikola Jokić') == _JOKIC_ID


def test_the_lookup_keeps_generational_suffixes():
    """Jr., Sr. and II are part of a name: Tim Hardaway Jr. is not Tim Hardaway."""
    lookup = _sample_lookup()
    assert lookup.resolve_player_id('Tim Hardaway Jr.', 'Tim Hardaway Jr.') == _HARDAWAY_ID
    assert lookup.resolve_player_id('Tim Hardaway', 'Tim Hardaway') == RP_PLAYER_ID


def test_the_lookup_falls_back_to_the_sessions_projection_names():
    """A player only a projection names (no unified row, so a synthetic id) is found by that name."""
    lookup = _sample_lookup()
    assert lookup.resolve_player_id(55555, 'Rookie Nobody Jr.') == _SYNTHETIC_ID
    assert lookup.resolve_player_id(55555, None) == RP_PLAYER_ID, 'no name, no fallback'
    assert lookup.resolve_player_id('Random Street Baller', 'Random Street Baller') == RP_PLAYER_ID


def test_the_lookup_leaves_out_a_projection_name_two_players_share():
    registry = _sample_registry() | {-8: make_player_identity(-8, 'Rookie Nobody Jr.', 'SF')}
    lookup = build_platform_player_id_lookup(registry, 'ESPN_NAME', _sample_unified_table(), _sample_name_resolvers())
    assert lookup.resolve_player_id(55555, 'Rookie Nobody Jr.') == RP_PLAYER_ID


def test_espn_get_draft_results_robust_team_lookup(monkeypatch):
    integration = ESPNIntegration(s2='cookie_s2', swid='{ABC-123}')
    t1_players = [
        _MockPlayer('Luka Dončić'),
        _MockPlayer('P.J. Washington', injured=True),
    ]
    t2_players = [
        _MockPlayer('Nikola Jokić'),
        _MockPlayer('Nic Claxton', lineup_slot='IR'),
    ]

    mock_league = _MockLeague([
        _MockTeam(1, 'Dallas Mavericks', t1_players),
        _MockTeam(2, 'Denver Nuggets', t2_players),
    ])
    monkeypatch.setattr(integration, '_make_league', lambda league_id: mock_league)

    lookup = _sample_lookup()

    # Test with string team IDs, int team IDs, and renamed teams
    config = PlatformConfig(
        platform='Retrieve from ESPN',
        league_id='fba:99999::2024',
        division_id=None,
        teams_dict={'Dallas Mavericks': '1', 'Denver Nuggets': 2},
        player_name_column='ESPN_NAME',
        seat_names=['Dallas Mavericks', 'Denver Nuggets'],
    )

    state = integration.get_draft_results(config, 'Season Mode', lookup)

    # An injury status alone keeps a player on the roster; the IR slot moves him to the injured list (as Yahoo's IL).
    assert state.player_assignments['Dallas Mavericks'] == [_DONCIC_ID, _WASHINGTON_ID]
    assert state.player_assignments['Denver Nuggets'] == [_JOKIC_ID]
    assert state.injured_players == [_CLAXTON_ID]


def test_espn_swid_formatting():
    integration_with_braces = ESPNIntegration(swid='{12345-ABCD}')
    assert integration_with_braces._format_swid() == '{12345-ABCD}'

    integration_no_braces = ESPNIntegration(swid='12345-ABCD')
    assert integration_no_braces._format_swid() == '{12345-ABCD}'

    integration_whitespace = ESPNIntegration(swid='  {12345-ABCD}  ')
    assert integration_whitespace._format_swid() == '{12345-ABCD}'


def test_espn_league_shape_reads_the_roster_from_the_settings(monkeypatch):
    """The picks are the league's lineup slots and bench, IR left out -- not the longest current roster (which counts
    IR, and is 0 before the draft). Slot counts of league 1801165221 (2026-10-20)."""
    integration = ESPNIntegration()
    mock_league = _MockLeague([_MockTeam(1, 'Team A', []), _MockTeam(2, 'Team B', [])],
                              {'0': 1, '1': 1, '2': 1, '3': 1, '4': 2, '5': 1, '6': 1, '7': 0, '8': 0, '9': 0, '10': 0,
                               '11': 2, '12': 4, '13': 3, '14': 0})
    monkeypatch.setattr(integration, '_make_league', lambda league_id: mock_league)

    shape = integration.fetch_league_shape('fba:123::2024', None)
    assert shape.n_drafters == 2
    assert shape.team_names == ['Team A', 'Team B']
    assert shape.slot_counts == {'PG': 1, 'SG': 1, 'SF': 1, 'PF': 1, 'C': 2, 'G': 1, 'F': 1, 'Util': 6}
    assert shape.n_picks == 14
    assert shape.is_auction_draft is False


def test_espn_league_shape_refuses_a_slot_the_app_does_not_have(monkeypatch):
    integration = ESPNIntegration()
    mock_league = _MockLeague([_MockTeam(1, 'Team A', [])], {'0': 1, '9': 1, '12': 3})   # 9 = PF/C
    monkeypatch.setattr(integration, '_make_league', lambda league_id: mock_league)
    with pytest.raises(RuntimeError, match="no slot for: 'PF/C' x1"):
        integration.fetch_league_shape('fba:123::2024', None)


def test_espn_split_league_id_composite_fan_api():
    # NBA live Fan API format: <entryId>:<groupId>:<sportId>:<seasonId>::<seasonId>
    prefix, num_id, year = ESPNIntegration._split_league_id('1:1946378079:3:2027::2027')
    assert prefix == 'fba'
    assert num_id == '1946378079'
    assert year == 2027

    # WNBA live Fan API format: sportId 5
    prefix_w, num_id_w, year_w = ESPNIntegration._split_league_id('1:562402811:5:2026::2026')
    assert prefix_w == 'wfba'
    assert num_id_w == '562402811'
    assert year_w == 2026

    # Legacy / mock prefixed formats
    prefix_m, num_id_m, year_m = ESPNIntegration._split_league_id('fba:999::2025')
    assert prefix_m == 'fba'
    assert num_id_m == '999'
    assert year_m == 2025



# ── Public leagues: no cookies ───────────────────────────────────────────────

def test_a_typed_league_id_without_a_season_is_put_in_the_current_season(monkeypatch):
    from backend.platform_integration.integrations import espn
    monkeypatch.setattr(espn, '_CURRENT_SEASON_BY_GAME', {'fba': 2027, 'wfba': 2026})
    assert ESPNIntegration.complete_league_id('1801165221') == '1801165221::2027'
    assert ESPNIntegration.complete_league_id('wfba:98765') == 'wfba:98765::2026'
    assert ESPNIntegration.complete_league_id('1801165221::2026') == '1801165221::2026', 'a pasted season is kept'


def test_no_saved_cookies_builds_the_integration_without_them(monkeypatch):
    from backend.api import helpers
    monkeypatch.setattr(helpers, 'has_espn_credentials', lambda client_id: False)
    assert helpers.build_credentials_for('Retrieve from ESPN', 'someone') is None
    integration = ESPNIntegration(**(helpers.build_credentials_for('Retrieve from ESPN', 'someone') or {}))
    assert integration._s2 is None and integration._format_swid() is None


def test_a_private_league_without_cookies_says_to_save_them(monkeypatch):
    from espn_api.requests.espn_requests import ESPNAccessDenied, ESPNInvalidLeague
    from backend.platform_integration.integrations import espn

    def refuse(**arguments):
        raise ESPNAccessDenied('denied')

    def missing(**arguments):
        raise ESPNInvalidLeague('missing')

    monkeypatch.setattr(espn, 'NBALeague', refuse)
    with pytest.raises(PermissionError, match='is private: save the espn_s2 and SWID cookies'):
        ESPNIntegration().fetch_league_shape('1801165221::2026', None)
    with pytest.raises(PermissionError, match='saved cookies do not give access'):
        ESPNIntegration(s2='s2', swid='swid').fetch_league_shape('1801165221::2026', None)
    monkeypatch.setattr(espn, 'NBALeague', missing)
    with pytest.raises(LookupError, match='no league 1801165221 in the 2025-26 season'):
        ESPNIntegration().fetch_league_shape('1801165221::2026', None)


def test_an_unmatched_espn_player_keeps_his_place_and_his_name(monkeypatch):
    """A rostered player the session cannot match stays in the roster as the replacement player, and the selections
    name him at his position there, so the roster can show who he is."""
    integration = ESPNIntegration()
    mock_league = _MockLeague([_MockTeam(1, 'Dallas Mavericks', [
        _MockPlayer('Luka Dončić'), _MockPlayer('Nobody Anyone Knows'), _MockPlayer('Nic Claxton', lineup_slot='IR'),
        _MockPlayer('Another Stranger', lineup_slot='IR'), _MockPlayer('P.J. Washington')])])
    monkeypatch.setattr(integration, '_make_league', lambda league_id: mock_league)
    config = PlatformConfig(platform='Retrieve from ESPN', league_id='fba:1::2024', division_id=None,
                            teams_dict={'Dallas Mavericks': '1'}, player_name_column='ESPN_NAME',
                            seat_names=['Dallas Mavericks'])
    state = integration.get_draft_results(config, 'Season Mode', _sample_lookup())
    assert state.player_assignments['Dallas Mavericks'] == [_DONCIC_ID, RP_PLAYER_ID, _WASHINGTON_ID]
    assert state.unmatched_player_names == {'Dallas Mavericks': {1: 'Nobody Anyone Knows'}}
    assert state.injured_players == [_CLAXTON_ID], 'an unmatched player on IR is nobody to put on the injured list'


def test_a_leagues_seasons_are_the_current_one_and_every_earlier_one(monkeypatch):
    """An ESPN league keeps its id across seasons; the connector lists its seasons for the user to choose from."""
    from backend.platform_integration.integrations import espn
    monkeypatch.setattr(espn, '_CURRENT_SEASON_BY_GAME', {'fba': 2027})
    integration = ESPNIntegration()
    league = _MockLeague([])
    league.espn_request = type('EspnRequest', (), {'get_league': lambda request: {'status': {'previousSeasons': [2025, 2026]}}})()
    asked = []
    monkeypatch.setattr(integration, '_make_league', lambda league_id: asked.append(league_id) or league)
    assert integration.list_league_seasons('1801165221::2025') == [2027, 2026, 2025]
    assert asked == ['1801165221::2027'], 'read in the current season, whatever season the id named'
