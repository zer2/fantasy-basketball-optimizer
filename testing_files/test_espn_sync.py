# testing_files/test_espn_sync.py
# Offline unit tests for ESPN platform syncing, name normalization, and team lookup resilience.

import pandas as pd
import pytest

from backend.player_identity import RP_PLAYER_ID, make_player_identity
from backend.platform_integration.base import PlatformConfig
from backend.platform_integration.helpers import (
    normalize_player_name, build_platform_player_id_lookup,
)
from backend.platform_integration.integrations.espn import (
    ESPNIntegration, resolve_espn_player,
)


_DONCIC_ID = 1629029
_JOKIC_ID = 203999
_WASHINGTON_ID = 1629023
_MURPHY_ID = 1630530
_CLAXTON_ID = 1629651
_HUNTER_ID = 1629631


def _sample_registry() -> dict:
    return {
        _DONCIC_ID: make_player_identity(_DONCIC_ID, 'Luka Doncic', 'PG,SG'),
        _JOKIC_ID: make_player_identity(_JOKIC_ID, 'Nikola Jokic', 'C'),
        _WASHINGTON_ID: make_player_identity(_WASHINGTON_ID, 'PJ Washington', 'PF'),
        _MURPHY_ID: make_player_identity(_MURPHY_ID, 'Trey Murphy', 'SF,SG'),
        _CLAXTON_ID: make_player_identity(_CLAXTON_ID, 'Nicolas Claxton', 'C'),
        _HUNTER_ID: make_player_identity(_HUNTER_ID, 'DeAndre Hunter', 'SF'),
    }


def _sample_unified_table() -> pd.DataFrame:
    return pd.DataFrame({
        'NBA_PLAYER_ID': [_DONCIC_ID, _JOKIC_ID, _WASHINGTON_ID, _MURPHY_ID, _CLAXTON_ID, _HUNTER_ID],
        'MASTER_PLAYER_NAME': ['Luka Doncic', 'Nikola Jokic', 'PJ Washington', 'Trey Murphy III', 'Nicolas Claxton', "De'Andre Hunter"],
        'ESPN_NAME': ['Luka Doncic', None, 'P.J. Washington', 'Trey Murphy', None, 'DeAndre Hunter'],
        'DARKO_NAME': ['Luka Doncic', 'Nikola Jokic', 'PJ Washington', 'Trey Murphy III', 'Nic Claxton', 'DeAndre Hunter'],
        'ROTOWIRE_NAME': ['Luka Doncic', 'Nikola Jokic', 'P.J. Washington', 'Trey Murphy', 'Nicolas Claxton', "De'Andre Hunter"],
        'BBM_NAME': ['Luka Doncic', 'Nikola Jokic', 'PJ Washington', 'Trey Murphy III', 'Nic Claxton', 'DeAndre Hunter'],
        'HTB_NAME': ['Luka Doncic', 'Nikola Jokic', 'PJ Washington', 'Trey Murphy', 'Nicolas Claxton', 'DeAndre Hunter'],
    })


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


def test_normalize_player_name_accents():
    assert normalize_player_name('Luka Dončić') == 'luka doncic'
    assert normalize_player_name('Nikola Jokić') == 'nikola jokic'
    assert normalize_player_name('Dennis Schröder') == 'dennis schroder'
    assert normalize_player_name('Alperen Şengün') == 'alperen sengun'
    assert normalize_player_name('Jonas Valančiūnas') == 'jonas valanciunas'


def test_normalize_player_name_suffixes():
    assert normalize_player_name('Trey Murphy III') == 'trey murphy'
    assert normalize_player_name('Tim Hardaway Jr.') == 'tim hardaway'
    assert normalize_player_name('Marcus Morris Sr.') == 'marcus morris'
    assert normalize_player_name('Dereck Lively II') == 'dereck lively'


def test_normalize_player_name_punctuation():
    assert normalize_player_name('P.J. Washington') == 'pj washington'
    assert normalize_player_name('C.J. McCollum') == 'cj mccollum'
    assert normalize_player_name("De'Andre Hunter") == 'deandre hunter'
    assert normalize_player_name('O.G. Anunoby') == 'og anunoby'


def test_build_platform_player_id_lookup_for_espn():
    registry = _sample_registry()
    unified = _sample_unified_table()
    lookup = build_platform_player_id_lookup(registry, 'ESPN_NAME', unified)

    # Direct match
    assert lookup.get('Luka Doncic') == _DONCIC_ID
    # Normalized match
    assert lookup.get('luka doncic') == _DONCIC_ID
    # Match through MASTER_PLAYER_NAME when ESPN_NAME was None
    assert lookup.get('Nikola Jokic') == _JOKIC_ID
    assert lookup.get('nikola jokic') == _JOKIC_ID
    # P.J. Washington and PJ Washington both resolved
    assert lookup.get('P.J. Washington') == _WASHINGTON_ID
    assert lookup.get('pj washington') == _WASHINGTON_ID
    # Nicolas Claxton and alias Nic Claxton both resolved
    assert lookup.get('Nicolas Claxton') == _CLAXTON_ID
    assert lookup.get('nic claxton') == _CLAXTON_ID


def test_resolve_espn_player():
    registry = _sample_registry()
    unified = _sample_unified_table()
    lookup = build_platform_player_id_lookup(registry, 'ESPN_NAME', unified)

    # Accented name from ESPN API -> resolved to canonical
    p_doncic = _MockPlayer('Luka Dončić')
    assert resolve_espn_player(p_doncic, lookup) == _DONCIC_ID

    p_jokic = _MockPlayer('Nikola Jokić')
    assert resolve_espn_player(p_jokic, lookup) == _JOKIC_ID

    # Punctuation / suffix
    p_pj = _MockPlayer('P.J. Washington Jr.')
    assert resolve_espn_player(p_pj, lookup) == _WASHINGTON_ID

    p_murphy = _MockPlayer('Trey Murphy III')
    assert resolve_espn_player(p_murphy, lookup) == _MURPHY_ID

    # Nickname / alias
    p_claxton = _MockPlayer('Nic Claxton')
    assert resolve_espn_player(p_claxton, lookup) == _CLAXTON_ID

    # Unknown player
    p_unknown = _MockPlayer('Random Street Baller')
    assert resolve_espn_player(p_unknown, lookup) == RP_PLAYER_ID


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

    registry = _sample_registry()
    unified = _sample_unified_table()
    lookup = build_platform_player_id_lookup(registry, 'ESPN_NAME', unified)

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
