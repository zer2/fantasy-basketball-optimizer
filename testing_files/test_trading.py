# testing_files/test_trading.py
# Correctness tests for the trade suggestion engine.
# Uses 2024-25 historical season data, EC scoring, default parameters, and
# the default 12-team rosters from benchmark_trading.py.
#
# Thresholds mirror the frontend defaults:
#   combo threshold  = 2.0  (±2% general-value window: fixed here to pin the snapshot; the frontend now searches
#                            bands of 1 point up to 10 and then beyond, see trade_parameters.ts TRADE_SEARCH_ROUNDS)
#   your_threshold   = 0.0  (0%  → /100 → 0.0,   tp_your_threshold  default)
#   their_threshold  = 0.0  (0%  → /100 → 0.0,   tp_their_threshold default)
#
# Both thresholds require strictly non-negative impact so only mutually beneficial
# trades are shown. Snapshots regenerated on 2026-04-09 after removing
# _identify_trade_candidates (all players evaluated) and tightening defaults.
# These act as a regression guard: if any change shifts the top suggestions,
# this test will catch it.

import os
import time

import pytest

from benchmark_helpers import (
    client, _build_session_request, resolve_player_assignments, resolve_player_ids,
    record_benchmark,
)
from backend.state.session import get_session
from backend.services.trading import run_trade_suggest, run_trade_suggest_across_teams
from backend.models import ComboParam

from test_and_benchmark_trading import _DEFAULT_SEASON_ROSTERS

_YOUR_THRESHOLD  = 0.0
_THEIR_THRESHOLD = 0.0
_COMBO_THRESHOLD = 2.0

_SCORE_TOL = 0.0001   # allowed absolute deviation on H-score diffs (0–1 scale)

# Top expected suggestions per combo size.
# Regenerated 2026-10-13 when complete teams started getting the beth strength correction (it had applied only to
# draft candidates, so a team scored differently on the waiver wire than on the Rosters tab): every score is
# regressed toward average, the 1v1 pairs are unchanged, and the 2v2/3v3 top trades reshuffle.
# Regenerated 2026-08-14 for the team-denominator correction to percentage-stat G-scores
# (the unpublished-paper factor in calculate_scores_from_coefficients) -- the score
# shifts reshuffle the 2v2/3v3 top trades; the 1v1 pairs are unchanged.
# send / receive are sorted lists so order within the group does not matter.
_EXPECTED = {
    '1v1': [
        {
            'send':    ['Daniel Gafford'],
            'receive': ['Ty Jerome'],
            'your_score':  0.001,
            'their_score': 0.0013,
        },
        {
            'send':    ['Rudy Gobert'],
            'receive': ['Onyeka Okongwu'],
            'your_score':  0.001,
            'their_score': 0.0009,
        }
    ],
    '2v2': [
        {
            'send':    ['Daniel Gafford', 'Rudy Gobert'],
            'receive': ['Malik Monk', 'Tyrese Maxey'],
            'your_score':  0.0033,
            'their_score': 0.0008,
        },
        {
            'send':    ['Andrew Nembhard', 'Rudy Gobert'],
            'receive': ['Malik Monk', 'Ty Jerome'],
            'your_score':  0.0028,
            'their_score': 0.0024,
        }
    ],
    '3v3': [
        {
            'send':    ['Andrew Nembhard', 'Isaiah Hartenstein', 'Rudy Gobert'],
            'receive': ['Malik Monk', 'Ty Jerome', 'Tyrese Maxey'],
            'your_score':  0.0058,
            'their_score': 0.0001,
        },
        {
            'send':    ['Andrew Nembhard', 'Daniel Gafford', 'Rudy Gobert'],
            'receive': ['Derrick Jones Jr.', 'Malik Monk', 'Tyrese Maxey'],
            'your_score':  0.0054,
            'their_score': 0.0014,
        }
    ],
}

_COMBO_CONFIGS = [
    pytest.param(
        '1v1', [ComboParam(n_traded=1, n_received=1, threshold=_COMBO_THRESHOLD)],
        id='1v1',
    ),
    pytest.param(
        '2v2', [ComboParam(n_traded=2, n_received=2, threshold=_COMBO_THRESHOLD)],
        id='2v2',
    ),
    pytest.param(
        '3v3', [ComboParam(n_traded=3, n_received=3, threshold=_COMBO_THRESHOLD)],
        id='3v3',
    ),
]


@pytest.fixture(scope='module')
def trading_session():
    """One EC session shared across all trade correctness tests."""
    session_request = _build_session_request(objective='Each Category')
    response = client.post('/sessions', json=session_request)
    assert response.status_code == 201, f'Session creation failed: {response.text}'
    return response.json()['session_id']


@pytest.mark.parametrize('label, combo_params', _COMBO_CONFIGS)
def test_trade_suggest_top4(trading_session, label, combo_params):
    """Top-4 suggestions for Drafter 1 vs Drafter 2 must match the verified snapshot."""
    session = get_session(trading_session)

    start  = time.perf_counter()
    result = run_trade_suggest(
        session              = session
        , player_assignments = resolve_player_assignments(session, _DEFAULT_SEASON_ROSTERS)
        , my_team            = 'Drafter 1'
        , their_team         = 'Drafter 2'
        , combo_params       = combo_params
        , your_threshold     = _YOUR_THRESHOLD
        , their_threshold    = _THEIR_THRESHOLD
        , position_check = True
    )
    elapsed = time.perf_counter() - start

    suggestions = result.suggestions
    record_benchmark(f'Trade suggest snapshot — {label} ({len(suggestions)} suggestions)', elapsed)
    expected    = _EXPECTED[label]

    if os.environ.get('REGEN_GOLDENS'):
        rows = [
            "        {\n"
            f"            'send':    {sorted(session.player_registry[i].name for i in s.send)},\n"
            f"            'receive': {sorted(session.player_registry[i].name for i in s.receive)},\n"
            f"            'your_score':  {round(s.your_score, 4)},\n"
            f"            'their_score': {round(s.their_score, 4)},\n"
            "        }"
            for s in suggestions[:len(expected)]
        ]
        print(f"\n# REGEN trading\n    {label!r}: [\n" + ',\n'.join(rows) + "\n    ],")
        return

    assert len(suggestions) >= len(expected), (
        f'{label}: expected at least {len(expected)} suggestions, got {len(suggestions)}'
    )

    for rank, exp in enumerate(expected):
        actual = suggestions[rank]
        expected_send_ids    = resolve_player_ids(session, exp['send'])
        expected_receive_ids = resolve_player_ids(session, exp['receive'])

        assert sorted(actual.send)    == sorted(expected_send_ids),    f'{label} rank {rank+1}: send mismatch'
        assert sorted(actual.receive) == sorted(expected_receive_ids), f'{label} rank {rank+1}: receive mismatch'

        assert abs(actual.your_score  - exp['your_score'])  <= _SCORE_TOL, (
            f'{label} rank {rank+1}: your_score {actual.your_score} != {exp["your_score"]}'
        )
        assert abs(actual.their_score - exp['their_score']) <= _SCORE_TOL, (
            f'{label} rank {rank+1}: their_score {actual.their_score} != {exp["their_score"]}'
        )


def test_value_gap_bands_together_find_exactly_the_unfiltered_trades(trading_session):
    """The frontend searches in bands of value gap (within 1, 1 to 2, ... 9 to 10, beyond 10). Together the bands must
    find every trade an unfiltered search finds, each exactly once: a band skips the gaps the band before covered."""
    session = get_session(trading_session)
    assignments = resolve_player_assignments(session, _DEFAULT_SEASON_ROSTERS)

    def suggest(threshold, searched_threshold):
        combos = [ComboParam(n_traded=size, n_received=size, threshold=threshold, searched_threshold=searched_threshold)
                  for size in (1, 2)]
        return run_trade_suggest(session, assignments, 'Drafter 1', 'Drafter 2', combos,
                                 _YOUR_THRESHOLD, _THEIR_THRESHOLD, True).suggestions

    bands = [(band + 1, band or None) for band in range(10)] + [(None, 10)]
    found_by_bands = [(tuple(s.send), tuple(s.receive)) for threshold, searched in bands
                      for s in suggest(threshold, searched)]
    unfiltered = [(tuple(s.send), tuple(s.receive)) for s in suggest(None, None)]
    assert len(found_by_bands) == len(set(found_by_bands)), 'no trade is found by two bands'
    assert sorted(found_by_bands) == sorted(unfiltered)


def test_one_search_covers_several_teams_and_names_each_trade_s_team(trading_session):
    session = get_session(trading_session)
    assignments = resolve_player_assignments(session, _DEFAULT_SEASON_ROSTERS)
    combos = [ComboParam(n_traded=1, n_received=1, threshold=None, searched_threshold=None)]
    teams = ['Drafter 2', 'Drafter 3', 'Drafter 4']
    together = run_trade_suggest_across_teams(session, assignments, 'Drafter 1', teams, combos,
                                              _YOUR_THRESHOLD, _THEIR_THRESHOLD, True).suggestions
    separately = [s for team in teams
                  for s in run_trade_suggest(session, assignments, 'Drafter 1', team, combos,
                                             _YOUR_THRESHOLD, _THEIR_THRESHOLD, True).suggestions]
    assert sorted((s.their_team, s.send, s.receive) for s in together)         == sorted((s.their_team, s.send, s.receive) for s in separately)
    assert {s.their_team for s in together} <= set(teams)
    assert [s.your_score for s in together] == sorted((s.your_score for s in together), reverse=True),         'ranked together by your score'


def test_a_trade_search_below_the_threshold_floor_is_refused(trading_session):
    """Every search runs at the floor (-2% for each side) and the frontend filters stricter thresholds itself; below
    it, one search could return hundreds of thousands of trades (350,536 3-for-3 trades with no floor, 2026-10-13)."""
    session = get_session(trading_session)
    body = {
        'player_assignments': {team: [int(p) for p in roster]
                               for team, roster in resolve_player_assignments(session, _DEFAULT_SEASON_ROSTERS).items()},
        'my_team': 'Drafter 1', 'their_teams': ['Drafter 2'],
        'combo_params': [{'n_traded': 1, 'n_received': 1, 'threshold': None, 'searched_threshold': None}],
        'your_differential_threshold': -0.02, 'their_differential_threshold': -0.02,
    }
    assert client.post(f'/sessions/{trading_session}/trade/suggest', json=body).status_code == 200
    for side in ('your_differential_threshold', 'their_differential_threshold'):
        response = client.post(f'/sessions/{trading_session}/trade/suggest', json={**body, side: -0.05})
        assert response.status_code == 422, f'{side} below the floor must be refused'
