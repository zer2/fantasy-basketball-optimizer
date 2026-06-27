# testing_files/test_platform_integration.py
# Unit tests for the platform_integration package (Phase 0: Fantrax).

import pandas as pd

from backend.platform_integration.helpers import (
    deduplicate_team_names, build_platform_name_lookup,
)
from backend.platform_integration.integrations.fantrax import FantraxIntegration
from backend.platform_integration.integrations.yahoo import YahooIntegration
from backend.platform_integration.integrations.espn import ESPNIntegration
from backend.platform_integration.base import PlatformConfig


def _info_with_positions() -> dict:
    # info['Positions'] is indexed by the canonical 'Name (POS)' identifier with
    # list-of-position-codes values, mirroring process_player_data.
    positions = pd.Series({
        'Nikola Jokic (C)':     ['C'],
        'Bam Adebayo (C,PF)':   ['C', 'PF'],
        'James Harden (PG,SG)': ['PG', 'SG'],
    })
    return {'Positions': positions}


def _fantrax_mapping_view() -> pd.DataFrame:
    # PLAYER_MAPPING_VIEW: platform spellings differ from the canonical PLAYER_NAME.
    return pd.DataFrame({
        'FANTRAX_PLAYER_NAME': ['Nik Jokic', 'Bam A.', 'Jim Harden'],
        'PLAYER_NAME':         ['Nikola Jokic', 'Bam Adebayo', 'James Harden'],
    })


# ── Name lookup (PLAYER_MAPPING_VIEW-backed) ──────────────────────────────────

def test_build_platform_name_lookup_maps_platform_spelling_to_canonical():
    lookup = build_platform_name_lookup(
        _info_with_positions(), 'FANTRAX_PLAYER_NAME', _fantrax_mapping_view(),
    )
    assert lookup['Nik Jokic']  == 'Nikola Jokic (C)'
    assert lookup['Bam A.']     == 'Bam Adebayo (C,PF)'
    assert lookup['Jim Harden'] == 'James Harden (PG,SG)'


def test_build_platform_name_lookup_omits_unknown_so_get_yields_rp():
    lookup = build_platform_name_lookup(
        _info_with_positions(), 'FANTRAX_PLAYER_NAME', _fantrax_mapping_view(),
    )
    assert lookup.get('Nobody Atall', 'RP') == 'RP'


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


# ── Fantrax draft/roster fetch (mocked API + mapping view) ────────────────────

class _FakeFantraxAPI:
    def __init__(self, roster_rows_by_team: dict):
        self._roster_rows_by_team = roster_rows_by_team

    def _request(self, method: str, **kwargs):
        if method == 'getTeamRosterInfo':
            return {'tables': [{'rows': self._roster_rows_by_team[kwargs['teamId']]}]}
        raise AssertionError(f'unexpected _request method {method!r}')


# Prebuilt platform-name -> canonical lookup (the integration consumes this; the
# builder that produces it is exercised separately above).
_NAME_LOOKUP = {
    'Nik Jokic':  'Nikola Jokic (C)',
    'Bam A.':     'Bam Adebayo (C,PF)',
    'Jim Harden': 'James Harden (PG,SG)',
}


def _fantrax_with_fake_api(monkeypatch, roster_rows_by_team) -> FantraxIntegration:
    integration = FantraxIntegration()
    fake = _FakeFantraxAPI(roster_rows_by_team)
    monkeypatch.setattr(integration, '_make_api', lambda league_id: fake)
    return integration


def test_get_draft_results_maps_names_and_excludes_injured_in_season(monkeypatch):
    roster_rows = {
        't1': [
            {'scorer': {'name': 'Nik Jokic'}, 'statusId': '1'},
            {'scorer': {'name': 'Bam A.'}, 'statusId': '3'},   # injured reserve
            {'no_scorer': True},                                # skipped (no 'scorer')
        ],
        't2': [
            {'scorer': {'name': 'Jim Harden'}, 'statusId': '1'},
        ],
    }
    integration = _fantrax_with_fake_api(monkeypatch, roster_rows)
    config = PlatformConfig(
        platform='Retrieve from Fantrax', league_id='LID', division_id=None,
        teams_dict={'Team One': 't1', 'Team Two': 't2'},
        player_name_column='FANTRAX_PLAYER_NAME',
    )
    state = integration.get_draft_results(config, 'Season Mode', _NAME_LOOKUP)

    assert state.injured_players == ['Bam Adebayo (C,PF)']
    assert state.player_assignments == {
        'Team One': ['Nikola Jokic (C)'],
        'Team Two': ['James Harden (PG,SG)'],
    }


def test_get_draft_results_keeps_injured_in_draft_mode(monkeypatch):
    roster_rows = {'t1': [{'scorer': {'name': 'Bam A.'}, 'statusId': '3'}]}
    integration = _fantrax_with_fake_api(monkeypatch, roster_rows)
    config = PlatformConfig(
        platform='Retrieve from Fantrax', league_id='LID', division_id=None,
        teams_dict={'T': 't1'}, player_name_column='FANTRAX_PLAYER_NAME',
    )
    state = integration.get_draft_results(config, 'Draft Mode', _NAME_LOOKUP)

    assert state.injured_players == []
    assert state.player_assignments == {'T': ['Bam Adebayo (C,PF)']}


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
    )
    name_lookup = {100: 'Nikola Jokic (C)', 200: 'James Harden (PG,SG)'}
    draft = [
        _FakeDraftObj('nba.p.100', 'nba.l.123.t.1', cost=50),
        _FakeDraftObj('nba.p.200', 'nba.l.123.t.2', cost=30),
        _FakeDraftObj('nba.p.999', 'nba.l.123.t.1', cost=5),   # unknown id -> RP
    ]
    assignments, costs = YahooIntegration()._assignments_from_draft(draft, config, name_lookup)

    assert assignments == {'Team One': ['Nikola Jokic (C)', 'RP'], 'Team Two': ['James Harden (PG,SG)']}
    assert costs == {'Team One': [50.0, 5.0], 'Team Two': [30.0]}


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
    )
    name_lookup = {'Nikola Jokic': 'Nikola Jokic (C)', 'James Harden': 'James Harden (PG,SG)'}
    state = integration.get_draft_results(config, 'Season Mode', name_lookup)

    assert state.player_assignments == {
        'Team One': ['Nikola Jokic (C)'],
        'Team Two': ['James Harden (PG,SG)', 'RP'],
    }
