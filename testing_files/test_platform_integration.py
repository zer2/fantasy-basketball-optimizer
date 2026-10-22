# testing_files/test_platform_integration.py
# Unit tests for the platform_integration package (Phase 0: Fantrax).

from types import SimpleNamespace

import pandas as pd
import pytest
from yfpy.exceptions import YahooFantasySportsDataNotFound
from yfpy.models import League
from yfpy.query import YahooFantasySportsQuery

from backend.platform_integration.helpers import (
    PlatformPlayerLookup, deduplicate_team_names, build_platform_player_id_lookup, fill_empty_slots_with_replacement,
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


def _lookup_by_key(player_ids_by_platform_key: dict) -> PlatformPlayerLookup:
    """A lookup with platform keys only: the integration tests pin the key mapping, and the name fallback is
    exercised in test_espn_sync.py."""
    return PlatformPlayerLookup(player_ids_by_platform_key, {}, {})


def _fantrax_unified_table() -> pd.DataFrame:
    # UNIFIED_PLAYER_TABLE: Fantrax players bridge by FANTRAX_ID -> the row's NBA id.
    return pd.DataFrame({
        'FANTRAX_ID':    ['j01', 'b02', 'h03', 'x04'],
        'NBA_PLAYER_ID': [_JOKIC_ID, _ADEBAYO_ID, _HARDEN_ID, 999999],
    })


# ── Player-id lookup (UNIFIED_PLAYER_TABLE-backed) ────────────────────────────

def test_build_platform_player_id_lookup_maps_platform_key_to_session_id():
    lookup = build_platform_player_id_lookup(
        _player_registry(), 'FANTRAX_ID', _fantrax_unified_table(), ({}, {}),
    )
    assert lookup.resolve_player_id('j01', None) == _JOKIC_ID
    assert lookup.resolve_player_id('b02', None) == _ADEBAYO_ID
    assert lookup.resolve_player_id('h03', None) == _HARDEN_ID


def test_build_platform_player_id_lookup_filters_to_registry_and_yields_rp():
    lookup = build_platform_player_id_lookup(
        _player_registry(), 'FANTRAX_ID', _fantrax_unified_table(), ({}, {}),
    )
    # 'x04' bridges to an id the session's registry lacks -> omitted, RP fallback applies.
    assert 'x04' not in lookup.player_ids_by_platform_key
    assert lookup.resolve_player_id('nobody', None) == RP_PLAYER_ID


def test_connect_patch_sets_the_lookup_before_the_rebuild(monkeypatch):
    # The draft-state poll runs without the session lock, so it can land during the rebuild a connect's
    # patch runs. The lookup used to be built only after that rebuild: a poll in between found the config
    # and no lookup, and crashed (502, 2026-10-09 -- three polls during a twenty-second self-play).
    from backend.services import session_management
    from backend.state.session import Session

    monkeypatch.setattr(session_management, 'get_unified_player_table', _fantrax_unified_table)
    monkeypatch.setattr(session_management, 'get_player_name_resolvers', lambda: ({}, {}))
    lookup_when_the_rebuild_ran = []
    monkeypatch.setattr(session_management, 'build_agent',
                        lambda session, **_: lookup_when_the_rebuild_ran.append(session.platform_player_id_lookup))
    session = Session(
        id='s1',
        current_settings={'scoring_format': 'Rotisserie', 'categories': ['Points'],
                          'most_categories_weight': None, 'tiebreaker_category': None,
                          'n_picks': 13, 'slot_counts': {'Util': 13}},
        player_registry=_player_registry(),
    )
    config = PlatformConfig(platform='Retrieve from Fantrax', league_id='L', division_id=None,
                            teams_dict={'Ann': '1'}, player_name_column='FANTRAX_ID', seat_names=['Ann'])

    session_management.apply_patch(session, patch={}, from_step=4, platform_config=config,
                                   csv_bytes=None, uploaded_dfs=None)

    assert [lookup.player_ids_by_platform_key for lookup in lookup_when_the_rebuild_ran] \
        == [{'j01': _JOKIC_ID, 'b02': _ADEBAYO_ID, 'h03': _HARDEN_ID}], \
        'the lookup must be in place beside the config while the rebuild runs'
    assert session.platform_player_id_lookup.resolve_player_id('j01', None) == _JOKIC_ID


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
_PLAYER_ID_LOOKUP = _lookup_by_key({
    '*j01*': _JOKIC_ID,
    '*b02*': _ADEBAYO_ID,
    '*h03*': _HARDEN_ID,
})


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
    player_id_lookup = _lookup_by_key({100: _JOKIC_ID, 200: _HARDEN_ID})
    draft = [
        _FakeDraftObj('nba.p.100', 'nba.l.123.t.1', cost=50),
        _FakeDraftObj('nba.p.200', 'nba.l.123.t.2', cost=30),
        _FakeDraftObj('nba.p.999', 'nba.l.123.t.1', cost=5),   # unknown id -> RP
    ]
    assignments, costs = YahooIntegration()._assignments_from_draft(draft, config, player_id_lookup)

    assert assignments == {'Team One': [_JOKIC_ID, RP_PLAYER_ID], 'Team Two': [_HARDEN_ID]}
    assert costs == {'Team One': [50.0, 5.0], 'Team Two': [30.0]}


class _FakeDraftResultsQuery:
    """A yfpy query whose draft results come back in the given shape."""
    def __init__(self, draft_results):
        self.draft_results = draft_results

    def get_league_draft_results(self):
        return self.draft_results


def _two_team_config():
    return PlatformConfig(
        platform='Retrieve from Yahoo', league_id='123', division_id=None,
        teams_dict={'Team One': '1', 'Team Two': '2'},
        player_name_column='YAHOO_PLAYER_ID',
        seat_names=['Team One', 'Team Two'],
    )


@pytest.mark.parametrize('mode', ['Auction Mode', 'Draft Mode'])
def test_yahoo_reads_a_single_draft_result(monkeypatch, mode):
    # yfpy returns exactly one draft result as a bare {'draft_result': ...} dict, not a one-element list. Indexed
    # with [0] that raised KeyError: 0 on every poll between an auction's first sale and its second (production,
    # 2026-10-11), each a 502.
    integration = YahooIntegration()
    cost = 50 if mode == 'Auction Mode' else None
    single = {'draft_result': _FakeDraftObj('nba.p.100', 'nba.l.123.t.1', cost=cost)}
    monkeypatch.setattr(integration, '_make_query', lambda league_id: _FakeDraftResultsQuery(single))
    lookup = _lookup_by_key({100: _JOKIC_ID})
    if mode == 'Auction Mode':
        selections = integration.get_auction_results(_two_team_config(), mode, lookup)
        assert selections.costs == {'Team One': [50.0], 'Team Two': []}
    else:
        selections = integration.get_draft_results(_two_team_config(), mode, lookup)
    assert selections.status == 'Success'
    assert selections.player_assignments == {'Team One': [_JOKIC_ID], 'Team Two': []}


def test_yahoo_reads_several_draft_results_as_before(monkeypatch):
    integration = YahooIntegration()
    several = [_FakeDraftObj('nba.p.100', 'nba.l.123.t.1', cost=50),
               _FakeDraftObj('nba.p.200', 'nba.l.123.t.2', cost=30)]
    monkeypatch.setattr(integration, '_make_query', lambda league_id: _FakeDraftResultsQuery(several))
    selections = integration.get_auction_results(_two_team_config(), 'Auction Mode', _lookup_by_key({100: _JOKIC_ID, 200: _HARDEN_ID}))
    assert selections.player_assignments == {'Team One': [_JOKIC_ID], 'Team Two': [_HARDEN_ID]}


class _FakeAllRostersQuery:
    """A yfpy query answering league/<key>/teams/roster with the given teams as Yahoo's raw JSON, counting requests."""
    def __init__(self, teams):
        self.teams, self.requests = teams, []

    def get_league_key(self):
        return '478.l.123'

    def get_response(self, url):
        self.requests.append(url)
        return SimpleNamespace(json=lambda: {'fantasy_content': {'league': [{}, {'teams': self.teams}]}})


def _fake_yahoo_teams(teams):
    """Yahoo's teams/roster JSON: count-keyed mappings, each record a list of one-field entries (with an empty list
    between some, as Yahoo sends them)."""
    def fake_team(team_id, players):
        roster_players = {str(index): {'player': [[{'player_key': f'478.p.{player_id}'}, [], {'player_id': str(player_id)},
                                                   {'name': {'full': f'Player {player_id}', 'first': 'Player'}}],
                                                  {'selected_position': [{'coverage_type': 'date'}, {'position': position}]}]}
                          for index, (player_id, position) in enumerate(players)}
        roster_players['count'] = len(players)
        return {'team': [[{'team_key': f'478.l.123.t.{team_id}'}, {'team_id': str(team_id)}, {'name': f'Team {team_id}'}],
                         {'roster': {'coverage_type': 'date', '0': {'players': roster_players if players else []}}}]}
    fake = {str(index): fake_team(team_id, players) for index, (team_id, players) in enumerate(teams)}
    fake['count'] = len(teams)
    return fake


def test_yahoo_season_rosters_come_from_one_request(monkeypatch):
    # One request per team, one after another, took 4.5 s and sixteen Yahoo calls for a sixteen-team league before
    # Season Mode showed a roster (2026-10-13); every roster now comes from a single request, read as raw JSON.
    integration = YahooIntegration()
    fake = _FakeAllRostersQuery(_fake_yahoo_teams([
        (1, [(100, 'PG'), (999, 'BN'), (300, 'IL')]),   # 999 unknown -> RP; 300 on the injured list
        (2, [(200, 'C')]),
    ]))
    monkeypatch.setattr(integration, '_make_query', lambda league_id: fake)
    lookup = _lookup_by_key({100: _JOKIC_ID, 200: _HARDEN_ID, 300: _ADEBAYO_ID})
    state = integration.get_draft_results(_two_team_config(), 'Season Mode', lookup)
    assert state.player_assignments == {'Team One': [_JOKIC_ID, RP_PLAYER_ID], 'Team Two': [_HARDEN_ID]}
    assert state.injured_players == [_ADEBAYO_ID]
    assert len(fake.requests) == 1 and fake.requests[0].endswith('/league/478.l.123/teams/roster')


def test_yahoo_season_roster_of_a_team_with_no_players(monkeypatch):
    # Yahoo sends an empty team's players as an empty list, not a count-keyed mapping.
    integration = YahooIntegration()
    monkeypatch.setattr(integration, '_make_query',
                        lambda league_id: _FakeAllRostersQuery(_fake_yahoo_teams([(1, [(100, 'PG')]), (2, [])])))
    assert integration.get_draft_results(_two_team_config(), 'Season Mode', _lookup_by_key({100: _JOKIC_ID})).player_assignments \
        == {'Team One': [_JOKIC_ID], 'Team Two': []}


def _yahoo_settings_with_positions(positions):
    return SimpleNamespace(roster_positions=[SimpleNamespace(position=name, count=count) for name, count in positions])


def test_yahoo_league_positions_become_the_slot_structure():
    # The league's own positions, in place of a default that may not even fit its picks (an 11-pick league kept the
    # 13-slot default, 2026-10-12). BN is a swappable bench, counted as Util (docs/hscores.md); IL, IL+, NA are left out.
    settings = _yahoo_settings_with_positions([('PG', 1), ('SG', 1), ('G', 1), ('SF', 1), ('PF', 1), ('F', 1), ('C', 1),
                                               ('Util', 1), ('BN', 3), ('IL', 2), ('NA', 1)])
    slots = YahooIntegration._read_slot_counts(settings, '227961')
    assert slots == {'PG': 1, 'SG': 1, 'G': 1, 'SF': 1, 'PF': 1, 'F': 1, 'C': 1, 'Util': 4}
    assert sum(slots.values()) == YahooIntegration._read_roster_size(settings, '227961'), 'the slots fit the picks'


def test_yahoo_position_the_app_cannot_seat_fails_the_connection():
    with pytest.raises(RuntimeError, match='no slot for'):
        YahooIntegration._read_slot_counts(_yahoo_settings_with_positions([('PG', 1), ('SF/PF', 1)]), '1')


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
    assignments, _ = YahooIntegration()._assignments_from_draft(draft, config, _lookup_by_key({100: _JOKIC_ID}))

    assert assignments == {'Team One': [_JOKIC_ID], 'Open seat 2': [], 'Open seat 3': []}


def test_yahoo_build_auth_url():
    url = YahooIntegration.build_auth_url('myclient')
    assert 'client_id=myclient' in url
    assert 'response_type=code' in url


class _QueryRaising:
    """A stand-in for the account-level yfpy query whose games call raises `error`."""
    def __init__(self, error): self.error = error
    def get_user_games(self): raise self.error


def test_yahoo_list_leagues_is_empty_for_an_account_with_no_nba_history(monkeypatch):
    # yfpy reports an account that has never played Yahoo fantasy basketball as data-not-found; that is an empty list,
    # not the 502 such an account got straight after authenticating
    integration = YahooIntegration(auth_dir='unused')
    monkeypatch.setattr(integration, '_bare_query', lambda: _QueryRaising(YahooFantasySportsDataNotFound('no games')))
    assert integration.list_leagues() == []


class _CannedResponse:
    def __init__(self, payload): self.payload, self.url, self.status_code = payload, 'canned', 200
    def json(self): return self.payload


def _make_query_answering_games(seasons: list[str]) -> YahooFantasySportsQuery:
    """A REAL yfpy query whose HTTP call returns Yahoo's games payload for `seasons`, so the shape list_leagues sees is
    the one yfpy's own unpacking produces -- not a guess at it. A hand-made stand-in returning a bare Game is exactly
    how the first one-season fix passed its test and still failed in production."""
    games = {str(index): {'game': [{'game_key': str(400 + index), 'game_id': str(400 + index), 'season': season,
                                    'code': 'nba'}]}
             for index, season in enumerate(seasons)}
    games['count'] = len(seasons)
    payload = {'fantasy_content': {'users': {'0': {'user': [{'guid': 'someone'}, {'games': games}]}, 'count': 1}}}
    query = object.__new__(YahooFantasySportsQuery)   # skips the constructor's OAuth handshake
    query.offline = False
    query.all_output_as_json_str = False
    query.fantasy_content_data_field = 'fantasy_content'
    query.game_code = 'nba'
    query.executed_queries = []
    query.get_response = lambda url: _CannedResponse(payload)
    return query


@pytest.mark.parametrize('seasons', [['2026'], ['2024', '2026', '2025']])
def test_yahoo_list_leagues_handles_any_number_of_seasons(monkeypatch, seasons):
    # production 2026-10-01: for one season yfpy returns {'game': Game}, not a list; iterating it yielded 'game' and
    # list_leagues raised "'str' object has no attribute 'season'" (and then, wrapped in a list, "'dict' object ...")
    integration = YahooIntegration(auth_dir='unused')
    monkeypatch.setattr(integration, '_bare_query', lambda: _make_query_answering_games(seasons))
    monkeypatch.setattr(integration, '_leagues_in', lambda game: [League({'league_id': f'league{game.season}', 'name': 'Ducks+', 'season': game.season})])
    newest_first = sorted(seasons, reverse=True)
    assert integration.list_leagues() == [{'id': f'league{season}', 'name': 'Ducks+', 'season': int(season)} for season in newest_first]


class _RecordingQuery:
    """Stands in for yfpy's query class at construction: records what _make_query gives it, contacts nothing."""
    def __init__(self, auth_dir, league_id, game_code, game_id=None):
        self.league_id, self.game_id, self.league_key = league_id, game_id, None


class _CurrentGameQuery:
    def __init__(self, calls):
        self.calls = calls

    def get_current_game_metadata(self):
        self.calls.append('current game')
        return SimpleNamespace(game_key='478')


def test_yahoo_queries_carry_their_full_league_key(monkeypatch):
    # yfpy builds '<game key>.l.<league id>' with an API call of its own on every request unless league_key is
    # already set, so each draft-state poll cost two Yahoo calls; two tabs left polling one mock room drew Yahoo's
    # rate limit (2026-10-12). The key is now set on every query, and the current season's is fetched once.
    from backend.platform_integration.integrations import yahoo
    monkeypatch.setattr(yahoo, 'YahooFantasySportsQuery', _RecordingQuery)
    monkeypatch.setattr(yahoo, '_GAME_KEY_BY_LEAGUE', {})
    monkeypatch.setattr(yahoo, '_CURRENT_GAME_KEY', None)
    integration = YahooIntegration(auth_dir='unused')
    calls = []
    monkeypatch.setattr(integration, '_bare_query', lambda: _CurrentGameQuery(calls))
    monkeypatch.setattr(integration, '_user_games', lambda: [SimpleNamespace(game_key='466', season='2025')])
    monkeypatch.setattr(integration, '_leagues_in', lambda game: [SimpleNamespace(league_id='56576')])

    assert integration._make_query('56576').league_key == '466.l.56576', "the account's own league: its own season"
    assert calls == [], 'an own league needs no current-season lookup'
    for _ in range(3):
        assert integration._make_query('2719938').league_key == '478.l.2719938', 'a mock room: the current season'
    assert calls == ['current game'], 'the current season is looked up once, not once per query'


def test_yfpy_uses_a_league_key_it_is_given_without_a_lookup():
    # The fix relies on this: with league_key set, yfpy's get_league_key returns it and makes no API call.
    query = object.__new__(YahooFantasySportsQuery)
    query.league_key, query.league_id, query.game_id = '478.l.2719938', '2719938', None
    def refuse():
        raise AssertionError('yfpy looked up the game although the league key was set')
    query.get_current_game_metadata = refuse
    query.get_game_metadata_by_game_id = lambda game_id: refuse()
    assert query.get_league_key() == '478.l.2719938'


def test_yahoo_list_leagues_still_raises_on_other_failures(monkeypatch):
    # an expired token or a network failure must not read as "no leagues": only reconnecting fixes those
    integration = YahooIntegration(auth_dir='unused')
    monkeypatch.setattr(integration, '_bare_query', lambda: _QueryRaising(RuntimeError('token expired')))
    with pytest.raises(RuntimeError):
        integration.list_leagues()


# ── ESPN (composite league id + roster mapping, no espn_api network) ───────────

class _FakeEspnPlayer:
    def __init__(self, name):
        self.name = name
        self.lineupSlot = 'BE'


class _FakeEspnTeam:
    def __init__(self, team_id, team_name, roster_names):
        self.team_id = team_id
        self.team_name = team_name
        self.roster = [_FakeEspnPlayer(n) for n in roster_names]


class _FakeEspnLeague:
    def __init__(self, teams): self.teams = teams


def test_espn_split_league_id_carries_season():
    assert ESPNIntegration._split_league_id('abc:12345::2024') == ('fba', '12345', 2024)
    assert ESPNIntegration._split_league_id('wfba:98765::2024') == ('wfba', '98765', 2024)


def test_yahoo_season_rosters_fall_back_to_the_players_name(monkeypatch):
    """A rostered player whose unified row lacks his Yahoo id (a rookie the table has not caught up with) is found by
    the name Yahoo gives him; one no spelling knows is still the replacement player."""
    integration = YahooIntegration()
    fake = _FakeAllRostersQuery(_fake_yahoo_teams([(1, [(100, 'PG'), (777, 'BN'), (999, 'BN')]), (2, [])]))
    monkeypatch.setattr(integration, '_make_query', lambda league_id: fake)
    lookup = PlatformPlayerLookup({100: _JOKIC_ID}, {'Player 777': _HARDEN_ID}, {})
    state = integration.get_draft_results(_two_team_config(), 'Season Mode', lookup)
    assert state.player_assignments['Team One'] == [_JOKIC_ID, _HARDEN_ID, RP_PLAYER_ID]
    assert state.unmatched_player_names == {'Team One': {2: 'Player 999'}}, 'the unmatched player is named in place'


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
    player_id_lookup = _lookup_by_key({'Nikola Jokic': _JOKIC_ID, 'James Harden': _HARDEN_ID})
    state = integration.get_draft_results(config, 'Season Mode', player_id_lookup)

    assert state.player_assignments == {
        'Team One': [_JOKIC_ID],
        'Team Two': [_HARDEN_ID, RP_PLAYER_ID],
    }


def test_a_short_season_roster_is_filled_with_replacement_players():
    # 13 slots: one team two short, one full. The open slots become the replacement player (-1), and their positions
    # are reported so the roster can show them as empty slots.
    rosters = {'Team A': list(range(11)), 'Team B': list(range(100, 113))}
    filled, empty_slots = fill_empty_slots_with_replacement(rosters, 13)
    assert filled['Team A'] == list(range(11)) + [-1, -1]
    assert filled['Team B'] == rosters['Team B']
    assert empty_slots == {'Team A': [11, 12]}
