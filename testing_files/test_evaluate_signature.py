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
# The signature is captured from a WARMED session (generic_h_scores cached, so the position-optimiser
# throttle is active and the result is reproducible). The very first evaluate on a fresh session runs
# the throttle un-primed and would produce a different, non-reproducible payload.
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
from backend.services.evaluate import run_evaluate

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

# sha256 of json.dumps(EvaluateResponse.model_dump(mode='json'), sort_keys=True) on a warmed session,
# keyed by (scoring_format, board). Regenerate with UPDATE_EVALUATE_SIGNATURE=1 (see module docstring).
# Regenerated 2026-07-17 after the NBA_PLAYER_TABLE dedup: pre-dedup, players with two name spellings
# on one nba id produced duplicate seasonal-average rows, so the historical candidate roster carried
# spurious duplicate-spelling entries; the dedup removed them. Serialization-only change — H-scores are
# unchanged (guarded by the tolerance-based auction/draft/trading correctness goldens).
_GOLDEN = {
    ('Head to Head: Each Category',  'empty'): '3d233126b4e79300fefe946fb86718075f4675ffbd9fae004228ee298efd1787',
    ('Head to Head: Each Category',  'mid'):   'ff2e9a2359d6eeda1eadccd6f0c0bdadce52dba71c41340fa0bbf8e0d5db8d26',
    ('Head to Head: Most Categories','empty'): '6e0d81afc5c4942609d1e55a4a5f116fd9b3ec1b5d9c504a99eebda0daf01427',
    ('Head to Head: Most Categories','mid'):   '10f46b3cc7e9487bda109f052324a6192460e4cadcd48a503669c1a51683ca00',
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
    """Create one session per scoring format and warm it (caches generic_h_scores so the throttle is
    primed and subsequent evaluates are reproducible)."""
    scoring_format = request.param
    response = client.post('/sessions', json=_build_session_request(scoring_format=scoring_format))
    assert response.status_code == 201, f'Session creation failed ({scoring_format}): {response.text}'
    session    = get_session(response.json()['session_id'])
    n_drafters = session.current_params['n_drafters']

    empty_board = {f'Team {i + 1}': [] for i in range(n_drafters)}
    run_evaluate(session, empty_board, 'Team 1', [], None, 0, None)   # warm-up
    return session, scoring_format, n_drafters


@pytest.mark.parametrize('board', ['empty', 'mid'])
def test_evaluate_signature(warmed_session, board):
    """Pin the serialized /evaluate payload for a fixed board so refactors can't silently change it."""
    session, scoring_format, n_drafters = warmed_session

    assignments   = {f'Team {i + 1}': [] for i in range(n_drafters)}
    exclusion_list = []
    if board == 'mid':
        assignments['Team 1'] = _TEAM_1
        assignments['Team 2'] = _TEAM_2
        exclusion_list        = _TEAM_1

    result = run_evaluate(session, assignments, 'Team 1', exclusion_list, None, 0, None)
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
