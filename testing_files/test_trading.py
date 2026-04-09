# testing_files/test_trading.py
# Correctness tests for the trade suggestion engine.
# Uses 2024-25 historical season data, EC scoring, default parameters, and
# the default 12-team rosters from benchmark_trading.py.
#
# Thresholds mirror the frontend defaults:
#   combo threshold  = 2.0  (±2% general-value filter, from trade_parameters.ts DEFAULT_COMBOS)
#   your_threshold   = 0.0  (0%  → /100 → 0.0,   tp_your_threshold  default)
#   their_threshold  = 0.0  (0%  → /100 → 0.0,   tp_their_threshold default)
#
# Both thresholds require strictly non-negative impact so only mutually beneficial
# trades are shown. Snapshots regenerated on 2026-04-09 after removing
# _identify_trade_candidates (all players evaluated) and tightening defaults.
# These act as a regression guard: if any change shifts the top suggestions,
# this test will catch it.

import time

import pytest

from benchmark_helpers import client, _build_session_request
from backend.session import get_session
from backend.math.trading import run_trade_suggest
from backend.models import ComboParam

from test_and_benchmark_trading import _DEFAULT_SEASON_ROSTERS

_YOUR_THRESHOLD  = 0.0
_THEIR_THRESHOLD = 0.0
_COMBO_THRESHOLD = 2.0

_SCORE_TOL = 0.0001   # allowed absolute deviation on H-score diffs (0–1 scale)

# Top expected suggestions per combo size, verified 2026-04-09.
# Snapshots regenerated after restoring the n_players_selected == n_picks branch
# in perform_iterations (use diff_means directly, no x_scores_available).
# send / receive are sorted lists so order within the group does not matter.
_EXPECTED = {
    '1v1': [
        {
            'send':    ['Isaiah Hartenstein (C)'],
            'receive': ['Malik Monk (PG,SG,SF)'],
            'your_score':  0.0026,
            'their_score': 0.0005,
        },
        {
            'send':    ['Daniel Gafford (C,PF)'],
            'receive': ['Andrew Wiggins (SF,PF)'],
            'your_score':  0.0022,
            'their_score': 0.0009,
        },
    ],
    '2v2': [
        {
            'send':    ['Daniel Gafford (C,PF)', 'Isaiah Hartenstein (C)'],
            'receive': ['Andrew Wiggins (SF,PF)', 'Malik Monk (PG,SG,SF)'],
            'your_score':  0.0034,
            'their_score': 0.0011,
        },
        {
            'send':    ['Andrew Nembhard (PG,SG)', 'Rudy Gobert (C)'],
            'receive': ['Russell Westbrook (PG,SG)', 'Tyrese Maxey (PG,SG)'],
            'your_score':  0.0032,
            'their_score': 0.0008,
        },
    ],
    '3v3': [
        {
            'send':    ['Andrew Nembhard (PG,SG)', 'Isaiah Hartenstein (C)', 'Rudy Gobert (C)'],
            'receive': ['Derrick Jones Jr. (SF,PF)', 'Pascal Siakam (C,SF,PF)', 'Russell Westbrook (PG,SG)'],
            'your_score':  0.0056,
            'their_score': 0.0006,
        },
        {
            'send':    ['Andrew Nembhard (PG,SG)', 'Daniel Gafford (C,PF)', 'Isaiah Hartenstein (C)'],
            'receive': ['Derrick Jones Jr. (SF,PF)', 'Malik Monk (PG,SG,SF)', 'Ty Jerome (PG,SG)'],
            'your_score':  0.0055,
            'their_score': 0.0005,
        },
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
    session_request = _build_session_request(scoring_format='Head to Head: Each Category')
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
        , player_assignments = _DEFAULT_SEASON_ROSTERS
        , my_team            = 'Drafter 1'
        , their_team         = 'Drafter 2'
        , combo_params       = combo_params
        , your_threshold     = _YOUR_THRESHOLD
        , their_threshold    = _THEIR_THRESHOLD
        , ignore_position_check = False
    )
    elapsed = time.perf_counter() - start

    suggestions = result.suggestions
    print(f'\n[benchmark] Trade suggest — {label}: {elapsed:.2f}s  ({len(suggestions)} suggestions)')
    expected    = _EXPECTED[label]

    assert len(suggestions) >= len(expected), (
        f'{label}: expected at least {len(expected)} suggestions, got {len(suggestions)}'
    )

    for rank, exp in enumerate(expected):
        actual = suggestions[rank]

        assert sorted(actual.send)    == sorted(exp['send']),    f'{label} rank {rank+1}: send mismatch'
        assert sorted(actual.receive) == sorted(exp['receive']), f'{label} rank {rank+1}: receive mismatch'

        assert abs(actual.your_score  - exp['your_score'])  <= _SCORE_TOL, (
            f'{label} rank {rank+1}: your_score {actual.your_score} != {exp["your_score"]}'
        )
        assert abs(actual.their_score - exp['their_score']) <= _SCORE_TOL, (
            f'{label} rank {rank+1}: their_score {actual.their_score} != {exp["their_score"]}'
        )
