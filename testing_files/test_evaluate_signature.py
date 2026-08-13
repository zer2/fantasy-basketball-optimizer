# testing_files/test_evaluate_signature.py
# Characterization ("golden") test: pins a byte-signature — the sha256 of the fully serialized
# EvaluateResponse — for fixed board states. Any unintended drift in the /evaluate payload is caught
# immediately, in particular the expand-view tables (G-score rows, flex allocations, roster
# assignments) that are built by hand-vectorised code in _build_candidates and serialised through
# stdlib dataclasses. A refactor there that changed a value, a field, ordering, or rounding would
# flip the hash.
#
# This is NOT a correctness test — the benchmark suites already assert H-scores with tolerances.
# This guards SERIALIZATION and hot-path STABILITY: the exact bytes the frontend receives.
#
# The agent's neutral baseline (agent.default_h_scores) is computed once at session build, so every
# evaluate is reproducible from the first call. An empty board short-circuits to that baseline — the
# full-exact, un-throttled solve; a non-empty board runs the position-optimiser throttle primed by it.
#
# Values in the payload are already rounded to 2 dp by the backend, which absorbs sub-0.005 numeric
# jitter, so the hash is stable run-to-run. If you intentionally change the algorithm, data, or
# parameters, regenerate the goldens and paste them into _GOLDEN below:
#
#   UPDATE_EVALUATE_SIGNATURE=1 python -m pytest testing_files/test_evaluate_signature.py -s

import hashlib
import json
import os

import pytest

from benchmark_helpers import client, _build_session_request
from backend.state.session import get_session
from backend.services.ranking import rank_candidates

# A fixed mid-draft board using players guaranteed present in the 2024-25 dataset (shared with the
# draft benchmark fixtures). Team 1 is the evaluating team; its own picks are excluded as candidates.
_TEAM_1 = [
    'Nikola Jokic (C)',
    'Giannis Antetokounmpo (C,PF)',
    'Victor Wembanyama (C)',
    'Anthony Edwards (SG,SF)',
]
_TEAM_2 = [
    'Shai Gilgeous-Alexander (PG,SG)',
    'Tyrese Haliburton (PG,SG)',
    'Trae Young (PG)',
    'LeBron James (SF,PF)',
]

# sha256 of json.dumps(EvaluateResponse.model_dump(mode='json'), sort_keys=True), keyed by
# (scoring_format, board). Regenerate with UPDATE_EVALUATE_SIGNATURE=1 (see module docstring).
# Regenerated 2026-08-13 on the player-headshots branch: RosterAssignment now carries
# full_name (display-layer headshot lookups), a payload-shape change with identical scores.
_GOLDEN = {
    ('Head to Head: Each Category',  'empty'): '83edb4d04b9116780e7ac278415649049c50412341bef74756c9a965c0b3d86c',
    ('Head to Head: Each Category',  'mid'):   'caa333334415371eb6bf3f47ee56ebf3711fa0d79cfbf45bca817da174abbb4e',
    ('Head to Head: Most Categories','empty'): '98d02f780477ddbfb222c045f22e569d5dedf81384a8ad78c6bf4c4e830f2377',
    ('Head to Head: Most Categories','mid'):   '7c8d8a7b370463c92b4b2237fad26ef4e45cacbac82dfc6c57e301b00742ec03',
}


def _signature(result) -> str:
    """sha256 of the canonical JSON serialization of the whole EvaluateResponse."""
    payload = json.dumps(result.model_dump(mode='json'), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode()).hexdigest()


@pytest.fixture(
    scope='module',
    params=['Head to Head: Each Category', 'Head to Head: Most Categories'],
)
def warmed_session(request):
    """Create one session per scoring format. The neutral baseline the throttle primes from is built
    at session creation, so no warm-up evaluate is needed for reproducibility."""
    scoring_format = request.param
    response = client.post('/sessions', json=_build_session_request(scoring_format=scoring_format))
    assert response.status_code == 201, f'Session creation failed ({scoring_format}): {response.text}'
    session    = get_session(response.json()['session_id'])
    n_drafters = session.current_params['n_drafters']
    return session, scoring_format, n_drafters


@pytest.mark.parametrize('board', ['empty', 'mid'])
def test_evaluate_signature(warmed_session, board):
    """Pin the serialized /evaluate payload for a fixed board so refactors can't silently change it."""
    session, scoring_format, n_drafters = warmed_session

    assignments = {f'Team {i + 1}': [] for i in range(n_drafters)}
    if board == 'mid':
        assignments['Team 1'] = _TEAM_1
        assignments['Team 2'] = _TEAM_2
        exclusion_list        = _TEAM_1
    else:
        exclusion_list = []

    result = rank_candidates(session, assignments, 'Team 1', exclusion_list, None, 0, None)
    assert len(result.candidates) > 0, 'No candidates returned'

    digest = _signature(result)

    if os.environ.get('UPDATE_EVALUATE_SIGNATURE'):
        print(f"\n[signature] ('{scoring_format}','{board}'): '{digest}',")
        return

    expected = _GOLDEN[(scoring_format, board)]
    assert digest == expected, (
        f'{scoring_format} / {board}: /evaluate signature changed.\n'
        f'  expected {expected}\n'
        f'  actual   {digest}\n'
        'If this change is intentional, regenerate the goldens:\n'
        '  UPDATE_EVALUATE_SIGNATURE=1 python -m pytest testing_files/test_evaluate_signature.py -s'
    )
