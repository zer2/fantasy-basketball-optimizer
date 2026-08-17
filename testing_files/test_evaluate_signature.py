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

from benchmark_helpers import client, _build_session_request, resolve_player_ids
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
# (objective, board). Regenerate with UPDATE_EVALUATE_SIGNATURE=1 (see module docstring).
# Regenerated 2026-08-14 for the expand-view diff reattribution: the displayed Future
# diff nets out the opponents' expected future tilts (res['Opponent-Future-Tilt']), so
# Current diff is the board as it stands. Display-only — H-scores verified identical.
#
# Regenerated 2026-08-16 for the Head-to-Head objective dial. Most Categories is untouched (its
# gradient was already exact); Each Category moved because its gradient is now the gradient of the
# objective it returns rather than n_categories times it, which changes how the kappa penalty
# weighs against it. Measured effect: max 0.21 H-score points across the top 40 on an empty
# 2024-25 board, top five unchanged. Half and Half is new.
_GOLDEN = {
    ('Each Category',  'empty'): '62b727b3ebef0ab93a91dad0a8a572c8ab1193489fe70608f907b4e45126ee70',
    ('Each Category',  'mid'):   '3579d53a98053c9115ed792389367cb7d20b48d29ec1b16b8671f21c8a254919',
    ('Half and Half',  'empty'): '8261c8ace7966996f965adb656ec3d85830c36bfc3fd627905af2fc2477f5261',
    ('Half and Half',  'mid'):   'f5ee76d26604c434650cb416f0cf13c1225b8b97d5907626cfd14e9e1045dd91',
    # Unchanged from before the dial existed — weight 1 runs the same code on the same inputs,
    # which is the evidence that the Most Categories endpoint is untouched.
    ('Most Categories','empty'): 'c31299bf88fc88d67c7b30062a77ec9b27a86acd40e1db4da607e396c1c590eb',
    ('Most Categories','mid'):   '9d4ddcc95b746b852a3924a5ff36a63a5026b02baccf5b3fdb9520c42d5dedb9',
}


def _signature(result) -> str:
    """sha256 of the canonical JSON serialization of the whole EvaluateResponse."""
    payload = json.dumps(result.model_dump(mode='json'), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode()).hexdigest()


@pytest.fixture(
    scope='module',
    params=['Each Category', 'Half and Half', 'Most Categories'],
)
def warmed_session(request):
    """Create one session per objective — both ends of the Head-to-Head dial and a blend of them.
    The blend runs both objectives and combines them, a path neither endpoint exercises, so it is
    pinned like the others. The neutral baseline the throttle primes from is built at session
    creation, so no warm-up evaluate is needed for reproducibility."""
    objective = request.param
    response = client.post('/sessions', json=_build_session_request(objective=objective))
    assert response.status_code == 201, f'Session creation failed ({objective}): {response.text}'
    session    = get_session(response.json()['session_id'])
    n_drafters = session.current_params['n_drafters']
    return session, objective, n_drafters


@pytest.mark.parametrize('board', ['empty', 'mid'])
def test_evaluate_signature(warmed_session, board):
    """Pin the serialized /evaluate payload for a fixed board so refactors can't silently change it."""
    session, objective, n_drafters = warmed_session

    assignments = {f'Team {i + 1}': [] for i in range(n_drafters)}
    if board == 'mid':
        assignments['Team 1'] = resolve_player_ids(session, _TEAM_1)
        assignments['Team 2'] = resolve_player_ids(session, _TEAM_2)
        exclusion_list        = assignments['Team 1']
    else:
        exclusion_list = []

    result = rank_candidates(session, assignments, 'Team 1', exclusion_list, None, 0, None)
    assert len(result.candidates) > 0, 'No candidates returned'

    digest = _signature(result)

    if os.environ.get('UPDATE_EVALUATE_SIGNATURE'):
        print(f"\n[signature] ('{objective}','{board}'): '{digest}',")
        return

    expected = _GOLDEN[(objective, board)]
    assert digest == expected, (
        f'{objective} / {board}: /evaluate signature changed.\n'
        f'  expected {expected}\n'
        f'  actual   {digest}\n'
        'If this change is intentional, regenerate the goldens:\n'
        '  UPDATE_EVALUATE_SIGNATURE=1 python -m pytest testing_files/test_evaluate_signature.py -s'
    )
