# testing_files/test_platform_integration.py
# Unit tests for the platform_integration package (Phase 0: Fantrax).

import pandas as pd

from backend.platform_integration.helpers import (
    deduplicate_team_names, build_platform_name_lookup,
)
from backend.platform_integration.integrations.fantrax import FantraxIntegration
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
