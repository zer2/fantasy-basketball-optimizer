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

import os
import time

import pytest

from benchmark_helpers import (
    client, _build_session_request, resolve_player_assignments, resolve_player_ids,
)
from backend.state.session import get_session
from backend.services.trading import run_trade_suggest
from backend.models import ComboParam

from test_and_benchmark_trading import _DEFAULT_SEASON_ROSTERS

_YOUR_THRESHOLD  = 0.0
_THEIR_THRESHOLD = 0.0
_COMBO_THRESHOLD = 2.0

_SCORE_TOL = 0.0001   # allowed absolute deviation on H-score diffs (0–1 scale)

# Top expected suggestions per combo size.
# Regenerated 2026-07-19 for the punt-seed init (multi-start / weakest-category, replacing the old
# heuristic) in algorithm_agents -- the H-score shifts reshuffle the 2v2/3v3 top trades (1v1 unchanged).
# send / receive are sorted lists so order within the group does not matter.
_EXPECTED = {
    '1v1': [
        {
            'send':    ['Isaiah Hartenstein (C)'],
            'receive': ['Malik Monk (PG,SG,SF)'],
            'your_score':  0.0019,
            'their_score': 0.0007,
        },
        {
            'send':    ['Daniel Gafford (C,PF)'],
            'receive': ['Andrew Wiggins (SF,PF)'],
            'your_score':  0.0018,
            'their_score': 0.0010,
        },
    ],
    '2v2': [
        {
            'send':    ['Andrew Nembhard (PG,SG)', 'Rudy Gobert (C)'],
            'receive': ['Andrew Wiggins (SF,PF)', 'Tyrese Maxey (PG,SG)'],
            'your_score':  0.0045,
            'their_score': 0.0003,
        },
        {
            'send':    ['Andrew Nembhard (PG,SG)', 'Daniel Gafford (C,PF)'],
            'receive': ['Andrew Wiggins (SF,PF)', 'Derrick Jones Jr. (SF,PF)'],
            'your_score':  0.0042,
            'their_score': 0.0007,
        },
    ],
    '3v3': [
        {
            'send':    ['Andrew Nembhard (PG,SG)', 'Isaiah Hartenstein (C)', 'Rudy Gobert (C)'],
            'receive': ['Kyshawn George (SG,SF)', 'Pascal Siakam (C,SF,PF)', 'Ty Jerome (PG,SG)'],
            'your_score':  0.0059,
            'their_score': 0.0011,
        },
        {
            'send':    ['Daniel Gafford (C,PF)', 'Isaiah Hartenstein (C)', 'Mike Conley (PG)'],
            'receive': ['Andrew Wiggins (SF,PF)', 'Malik Monk (PG,SG,SF)', 'Ty Jerome (PG,SG)'],
            'your_score':  0.0057,
            'their_score': 0.0001,
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
        , player_assignments = resolve_player_assignments(session, _DEFAULT_SEASON_ROSTERS)
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
