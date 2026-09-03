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

from benchmark_helpers import (client, _build_session_request, resolve_player_ids,
                               _DEFAULT_CATEGORIES)
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
#
# The 8-category rows were added 2026-08-17 alongside the tiebreaker. Adding them left the nine-
# category hashes below byte-identical, which is the evidence that pricing a tiebreaker into v and
# the G-scores touches nothing in a league that has not named one.
_GOLDEN = {
    ('Each Category',  'empty'): '62b727b3ebef0ab93a91dad0a8a572c8ab1193489fe70608f907b4e45126ee70',
    ('Each Category',  'mid'):   '3579d53a98053c9115ed792389367cb7d20b48d29ec1b16b8671f21c8a254919',
    ('Half and Half',  'empty'): '8261c8ace7966996f965adb656ec3d85830c36bfc3fd627905af2fc2477f5261',
    ('Half and Half',  'mid'):   'f5ee76d26604c434650cb416f0cf13c1225b8b97d5907626cfd14e9e1045dd91',
    # Unchanged from before the dial existed — weight 1 runs the same code on the same inputs,
    # which is the evidence that the Most Categories endpoint is untouched.
    ('Most Categories','empty'): 'c31299bf88fc88d67c7b30062a77ec9b27a86acd40e1db4da607e396c1c590eb',
    ('Most Categories','mid'):   '9d4ddcc95b746b852a3924a5ff36a63a5026b02baccf5b3fdb9520c42d5dedb9',

    # Eight categories (turnovers dropped), which is what a tiebreaker needs: a matchup that can
    # end level. Each objective appears with and without one named, so a change to the weighted
    # win-count DP, to what a category is worth in v, or to the G-score ranking the board is drawn
    # from has to show up here rather than only in a league nobody tested.
    ('8cat Each Category',   'empty'): '99e507ccb473e0f56d4149cf607bed9e2b1e1cea9d976e22f6bda8886a4949b9',
    ('8cat Each Category',   'mid'):   '94d939de939271b85e59a3485f7c0da06df7512bf10924f98946f9d4a552bf13',
    ('8cat Most Categories', 'empty'): '99af582c6badba125f442afa1b368aeac3e7165d9d98c3e02b08d568306384e0',
    ('8cat Most Categories', 'mid'):   '290c2732b011a0518e63e6059f3d0b50c4a0fac39778f2f95a16a269e9621e12',
    ('8cat MC + Points',     'empty'): '9bde70cc3f5c8120af3f382468a5eaf231dfdd97f4c95aed62b8f56d9539d57d',
    ('8cat MC + Points',     'mid'):   '925fb85397bb544238c6b41bc012ba403fd6a3b6f365990a360a79c9e72e2f9e',
    ('8cat Half and Half',   'empty'): '3a3355035a8e6123782682ffca874c47954acffcdb1c55c5cbd853315e04ad38',
    ('8cat Half and Half',   'mid'):   '92f5cb206689ea83e7511ed9b7baf8c801cd3505b47895a8e3c7d9ec3d23a2e6',
    ('8cat Half + Points',   'empty'): '830ac6bc990ecd0c3f31f1248a93ff888e80b3ea0ce4ab4ae42710d33336da3f',
    ('8cat Half + Points',   'mid'):   'ad7589922b8950dc8bfafb305111284cad6608498c311ebfeaae26dfed690ba6',
}


def _signature(result) -> str:
    """sha256 of the canonical JSON serialization of the whole EvaluateResponse."""
    payload = json.dumps(result.model_dump(mode='json'), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode()).hexdigest()


# The configurations pinned below: a label, the objective, the categories, and the tiebreaker.
# Nine categories cover the dial's ends and a blend. Eight (turnovers dropped) cover what a
# tiebreaker needs — a matchup that can end level — with and without one named, since the
# tiebreaker changes the win-count arithmetic, what a category is worth in v and the G-scores, and
# therefore which players the board prefers.
_EIGHT_CATEGORIES = [category for category in _DEFAULT_CATEGORIES if category != 'Turnovers']

_CONFIGURATIONS = [
    ('Each Category',            'Each Category',   None,              None),
    ('Half and Half',            'Half and Half',   None,              None),
    ('Most Categories',          'Most Categories', None,              None),
    ('8cat Each Category',       'Each Category',   _EIGHT_CATEGORIES, None),
    ('8cat Most Categories',     'Most Categories', _EIGHT_CATEGORIES, None),
    ('8cat MC + Points',         'Most Categories', _EIGHT_CATEGORIES, 'Points'),
    ('8cat Half and Half',       'Half and Half',   _EIGHT_CATEGORIES, None),
    ('8cat Half + Points',       'Half and Half',   _EIGHT_CATEGORIES, 'Points'),
]


@pytest.fixture(
    scope='module',
    params=_CONFIGURATIONS,
    ids=[label for label, *_ in _CONFIGURATIONS],
)
def warmed_session(request):
    """Create one session per configuration. The blend runs both objectives and combines them, a
    path neither endpoint exercises, and the tiebreaker rows run the weighted win-count DP against
    a repriced board, so each is pinned separately. The neutral baseline the throttle primes from
    is built at session creation, so no warm-up evaluate is needed for reproducibility."""
    label, objective, categories, tiebreaker = request.param
    response = client.post('/sessions', json=_build_session_request(
        objective=objective, categories=categories, tiebreaker_category=tiebreaker))
    assert response.status_code == 201, f'Session creation failed ({label}): {response.text}'
    session    = get_session(response.json()['session_id'])
    n_drafters = session.current_settings['n_drafters']
    return session, label, n_drafters


@pytest.mark.parametrize('board', ['empty', 'mid'])
def test_evaluate_signature(warmed_session, board):
    """Pin the serialized /evaluate payload for a fixed board so refactors can't silently change it."""
    session, label, n_drafters = warmed_session

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
        print(f"\n[signature] ('{label}','{board}'): '{digest}',")
        return

    expected = _GOLDEN[(label, board)]
    assert digest == expected, (
        f'{label} / {board}: /evaluate signature changed.\n'
        f'  expected {expected}\n'
        f'  actual   {digest}\n'
        'If this change is intentional, regenerate the goldens:\n'
        '  UPDATE_EVALUATE_SIGNATURE=1 python -m pytest testing_files/test_evaluate_signature.py -s'
    )
