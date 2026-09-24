# testing_files/test_platform_integration.py
# Unit tests for the platform_integration package (Phase 0: Fantrax).

import pandas as pd

from backend.platform_integration.helpers import (
    deduplicate_team_names, build_platform_player_id_lookup,
)
from backend.platform_integration.integrations.fantrax import (
    FantraxIntegration, to_unified_fantrax_id,
)
from backend.platform_integration.integrations.yahoo import YahooIntegration, _pad_with_open_seats
from backend.platform_integration.integrations.espn import ESPNIntegration
from backend.platform_integration.base import PlatformConfig
from backend.player_identity import RP_PLAYER_ID, make_player_identity


_JOKIC_ID, _ADEBAYO_ID, _HARDEN_ID = 203999, 1628389, 201935


def _player_registry() -> dict:
    return {
        _JOKIC_ID:   make_player_identity(_JOKIC_ID,   'Nikola Jokic', 'C'),
        _ADEBAYO_ID: make_player_identity(_ADEBAYO_ID, 'Bam Adebayo',  'C,PF'),
        _HARDEN_ID:  make_player_identity(_HARDEN_ID,  'James Harden', 'PG,SG'),
    }


def _fantrax_unified_table() -> pd.DataFrame:
    # UNIFIED_PLAYER_TABLE: Fantrax players bridge by FANTRAX_ID -> the row's NBA id.
    return pd.DataFrame({
        'FANTRAX_ID':    ['j01', 'b02', 'h03', 'x04'],
        'NBA_PLAYER_ID': [_JOKIC_ID, _ADEBAYO_ID, _HARDEN_ID, 999999],
    })


# ── Player-id lookup (UNIFIED_PLAYER_TABLE-backed) ────────────────────────────

def test_build_platform_player_id_lookup_maps_platform_key_to_session_id():
    lookup = build_platform_player_id_lookup(
        _player_registry(), 'FANTRAX_ID', _fantrax_unified_table(),
    )
    assert lookup['j01'] == _JOKIC_ID
    assert lookup['b02'] == _ADEBAYO_ID
    assert lookup['h03'] == _HARDEN_ID


def test_build_platform_player_id_lookup_filters_to_registry_and_yields_rp():
    lookup = build_platform_player_id_lookup(
        _player_registry(), 'FANTRAX_ID', _fantrax_unified_table(),
    )
    # 'x04' bridges to an id the session's registry lacks -> omitted, RP fallback applies.
    assert 'x04' not in lookup
    assert lookup.get('nobody', RP_PLAYER_ID) == RP_PLAYER_ID


# ── Team-name dedup (the Fantrax bug fix) ─────────────────────────────────────

def test_deduplicate_team_names_disambiguates():
    pairs = [('Team A', '1'), ('Team B', '2'), ('Team A', '3')]
    assert deduplicate_team_names(pairs) == {'Team A': '1', 'Team B': '2', 'Team A 2': '3'}


def test_deduplicate_team_names_preserves_every_team():
    # Three teams share a display name — none should be lost (the bug being fixed).
    pairs = [('Dup', 'a'), ('Dup', 'b'), ('Dup', 'c')]
    result = deduplicate_team_names(pairs)
    assert list(result.values()) == ['a', 'b', 'c']
    assert set(result.keys()) == {'Dup', 'Dup 2', 'Dup 3'}


# ── Fantrax draft/roster fetch (mocked API + unified table) ───────────────────

def _fake_beta_api(rosters_by_team: dict):
    """Stands in for the Beta API's getTeamRosters, whose shape is
    {'rosters': {team_id: {'teamName', 'rosterItems'}}}."""
    def fetch(endpoint: str, league_id: str, **params):
        if endpoint == 'getTeamRosters':
            return {'rosters': {team_id: {'teamName': team_id, 'rosterItems': items}
                                for team_id, items in rosters_by_team.items()}}
        raise AssertionError(f'unexpected Beta API endpoint {endpoint!r}')
    return fetch


# Prebuilt platform-key -> player-id lookup (the integration consumes this; the
# builder that produces it is exercised separately above).
# Keyed the way UNIFIED_PLAYER_TABLE.FANTRAX_ID actually spells a Fantrax id -- wrapped in
# asterisks. The Beta API sends the bare id, and the integration wraps it to match; keying this
# on bare ids would have passed while production matched nothing.
_PLAYER_ID_LOOKUP = {
    '*j01*': _JOKIC_ID,
    '*b02*': _ADEBAYO_ID,
    '*h03*': _HARDEN_ID,
}


def test_to_unified_fantrax_id_wraps_bare_ids_only():
    assert to_unified_fantrax_id('03e75') == '*03e75*'
    assert to_unified_fantrax_id('*03e75*') == '*03e75*'   # never double-wrapped


def _fantrax_with_fake_api(monkeypatch, roster_rows_by_team) -> FantraxIntegration:
    integration = FantraxIntegration()
    monkeypatch.setattr(integration, 'fetch_from_beta_api', _fake_beta_api(roster_rows_by_team))
    return integration


def test_get_draft_results_maps_ids_and_excludes_injured_in_season(monkeypatch):
    roster_rows = {
        't1': [
            {'scorerId': 'j01', 'status': 'ACTIVE'},
            {'scorerId': 'b02', 'status': 'IR'},                # injured reserve
            {'emptySlot': True},                                # skipped (no player id)
        ],
        't2': [
            {'scorerId': 'h03', 'status': 'ACTIVE'},
        ],
    }
    integration = _fantrax_with_fake_api(monkeypatch, roster_rows)
    config = PlatformConfig(
        platform='Retrieve from Fantrax', league_id='LID', division_id=None,
        teams_dict={'Team One': 't1', 'Team Two': 't2'},
        player_name_column='FANTRAX_ID',
        seat_names=['Team One', 'Team Two'],
    )
    state = integration.get_draft_results(config, 'Season Mode', _PLAYER_ID_LOOKUP)

    assert state.injured_players == [_ADEBAYO_ID]
    assert state.player_assignments == {
        'Team One': [_JOKIC_ID],
        'Team Two': [_HARDEN_ID],
    }


def test_get_draft_results_keeps_injured_in_draft_mode(monkeypatch):
    roster_rows = {'t1': [{'scorerId': 'b02', 'status': 'IR'}]}
    integration = _fantrax_with_fake_api(monkeypatch, roster_rows)
    config = PlatformConfig(
        platform='Retrieve from Fantrax', league_id='LID', division_id=None,
        teams_dict={'T': 't1'}, player_name_column='FANTRAX_ID',
        seat_names=['T'],
    )
    state = integration.get_draft_results(config, 'Draft Mode', _PLAYER_ID_LOOKUP)

    assert state.injured_players == []
    assert state.player_assignments == {'T': [_ADEBAYO_ID]}


# ── Yahoo draft/auction parsing (pure logic, no yfpy) ─────────────────────────

class _FakeDraftObj:
    def __init__(self, player_key, team_key, cost=None):
        self.player_key = player_key
        self.team_key = team_key
        self.cost = cost


def test_yahoo_assignments_from_draft_groups_by_team_with_costs():
    config = PlatformConfig(
        platform='Retrieve from Yahoo', league_id='123', division_id=None,
        teams_dict={'Team One': '1', 'Team Two': '2'},
        player_name_column='YAHOO_PLAYER_ID',
        seat_names=['Team One', 'Team Two'],
    )
    player_id_lookup = {100: _JOKIC_ID, 200: _HARDEN_ID}
    draft = [
        _FakeDraftObj('nba.p.100', 'nba.l.123.t.1', cost=50),
        _FakeDraftObj('nba.p.200', 'nba.l.123.t.2', cost=30),
        _FakeDraftObj('nba.p.999', 'nba.l.123.t.1', cost=5),   # unknown id -> RP
    ]
    assignments, costs = YahooIntegration()._assignments_from_draft(draft, config, player_id_lookup)

    assert assignments == {'Team One': [_JOKIC_ID, RP_PLAYER_ID], 'Team Two': [_HARDEN_ID]}
    assert costs == {'Team One': [50.0, 5.0], 'Team Two': [30.0]}


def test_yahoo_open_seats_pad_the_joined_teams():
    """A room that is still filling still has every one of its seats."""
    assert _pad_with_open_seats(['Zach', 'Burak'], 5) == [
        'Zach', 'Burak', 'Open seat 3', 'Open seat 4', 'Open seat 5']
    # Nothing to pad once the room is full, and never fewer seats than teams.
    assert _pad_with_open_seats(['Zach', 'Burak'], 2) == ['Zach', 'Burak']
    assert _pad_with_open_seats(['Zach', 'Burak'], 1) == ['Zach', 'Burak']


def test_yahoo_open_seat_label_never_collides_with_a_real_team_name():
    """Someone calling their team 'Open seat 3' must not merge two seats into one."""
    seats = _pad_with_open_seats(['Open seat 3'], 3)
    assert seats == ['Open seat 3', 'Open seat 2', 'Open seat 3 (2)']
    assert len(set(seats)) == 3


def test_yahoo_unfilled_seats_appear_on_the_board_with_empty_rosters():
    """An empty seat is a drafter with no picks, not a team missing from the league.

    A board that omits seats is what produced the KeyError-turned-500 when the seat being
    evaluated for was not among its keys.
    """
    config = PlatformConfig(
        platform='Retrieve from Yahoo', league_id='123', division_id=None,
        teams_dict={'Team One': '1'},
        player_name_column='YAHOO_PLAYER_ID',
        seat_names=['Team One', 'Open seat 2', 'Open seat 3'],
    )
    draft = [_FakeDraftObj('nba.p.100', 'nba.l.123.t.1')]
    assignments, _ = YahooIntegration()._assignments_from_draft(draft, config, {100: _JOKIC_ID})

    assert assignments == {'Team One': [_JOKIC_ID], 'Open seat 2': [], 'Open seat 3': []}


def test_yahoo_build_auth_url():
    url = YahooIntegration.build_auth_url('myclient')
    assert 'client_id=myclient' in url
    assert 'response_type=code' in url


# ── ESPN (composite league id + roster mapping, no espn_api network) ───────────

class _FakeEspnPlayer:
    def __init__(self, name): self.name = name


class _FakeEspnTeam:
    def __init__(self, team_id, team_name, roster_names):
        self.team_id = team_id
        self.team_name = team_name
        self.roster = [_FakeEspnPlayer(n) for n in roster_names]


class _FakeEspnLeague:
    def __init__(self, teams): self.teams = teams


def test_espn_split_league_id_carries_season():
    assert ESPNIntegration._split_league_id('abc:12345::2024') == ('12345', 2024)


def test_espn_get_draft_results_maps_rosters(monkeypatch):
    integration = ESPNIntegration(s2='x', swid='y')
    fake = _FakeEspnLeague([
        _FakeEspnTeam(1, 'Team One', ['Nikola Jokic']),
        _FakeEspnTeam(2, 'Team Two', ['James Harden', 'Unknown Guy']),   # unknown -> RP
    ])
    monkeypatch.setattr(integration, '_make_league', lambda league_id: fake)
    config = PlatformConfig(
        platform='Retrieve from ESPN', league_id='abc:1::2024', division_id=None,
        teams_dict={'Team One': '1', 'Team Two': '2'},
        player_name_column='ESPN_NAME',
        seat_names=['Team One', 'Team Two'],
    )
    player_id_lookup = {'Nikola Jokic': _JOKIC_ID, 'James Harden': _HARDEN_ID}
    state = integration.get_draft_results(config, 'Season Mode', player_id_lookup)

    assert state.player_assignments == {
        'Team One': [_JOKIC_ID],
        'Team Two': [_HARDEN_ID, RP_PLAYER_ID],
    }
