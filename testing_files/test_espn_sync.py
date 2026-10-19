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
    def __init__(self, teams: list[_MockTeam]):
        self.teams = teams


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

    assert state.player_assignments['Dallas Mavericks'] == [_DONCIC_ID, _WASHINGTON_ID]
    assert state.player_assignments['Denver Nuggets'] == [_JOKIC_ID, _CLAXTON_ID]

    # Both injured player and IR slot player captured
    assert _WASHINGTON_ID in state.injured_players
    assert _CLAXTON_ID in state.injured_players


def test_espn_swid_formatting():
    integration_with_braces = ESPNIntegration(swid='{12345-ABCD}')
    assert integration_with_braces._format_swid() == '{12345-ABCD}'

    integration_no_braces = ESPNIntegration(swid='12345-ABCD')
    assert integration_no_braces._format_swid() == '{12345-ABCD}'

    integration_whitespace = ESPNIntegration(swid='  {12345-ABCD}  ')
    assert integration_whitespace._format_swid() == '{12345-ABCD}'


def test_espn_fetch_league_shape_empty_rosters_fallback(monkeypatch):
    integration = ESPNIntegration()
    mock_league = _MockLeague([
        _MockTeam(1, 'Team A', []),
        _MockTeam(2, 'Team B', []),
    ])
    monkeypatch.setattr(integration, '_make_league', lambda league_id: mock_league)

    shape = integration.fetch_league_shape('fba:123::2024', None)
    assert shape.n_drafters == 2
    assert shape.n_picks == 13  # Defaulted to 13 rather than 0
    assert shape.team_names == ['Team A', 'Team B']


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

