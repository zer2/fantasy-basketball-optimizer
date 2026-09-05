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
#
# Regenerated 2026-09-05 for the self-play equilibrium overhaul (solve-half complementary
# groups, read-out serve, seed hysteresis, fixed 32-pass budget — commits 5f2a320..01f77bc)
# plus the mean-field high-confidence redesign committed alongside this regen. All rows run at
# the default confidence 0.5, so the drift here comes from the committed equilibrium mechanics,
# not the mean-field mode (which only engages above 0.5).
_GOLDEN = {
    ('Each Category',  'empty'): 'ee4ce501826e193a80213972e7235df3947902175cc3703af8bc57c7f8fd7114',
    ('Each Category',  'mid'):   'bb4d3664ca945ae6a20aaebc05c4ab4ee491042bd6390dfa45d5c08ee4d56eaf',
    ('Half and Half',  'empty'): 'd9fe9277ee9c67866e616509ab2133539beba8f8c12bd4e0bef4b512a7dca578',
    ('Half and Half',  'mid'):   '507caed5c6f29e2bf55cc9b20a5597e32d11a9e84b2f6bec9b4555c16d687a24',
    ('Most Categories','empty'): '534e05cc1890c29ea53acc463737a17c6ba22892e86a4e5fd84131589eeced0c',
    ('Most Categories','mid'):   '01691a7bf68173b18d81ddd1b4fd30e85c6672d7db4d20577a44b9d788d10e71',

    # Eight categories (turnovers dropped), which is what a tiebreaker needs: a matchup that can
    # end level. Each objective appears with and without one named, so a change to the weighted
    # win-count DP, to what a category is worth in v, or to the G-score ranking the board is drawn
    # from has to show up here rather than only in a league nobody tested.
    ('8cat Each Category',   'empty'): '5ccf089d27384ddc436e2d73be0cca5742c4c9cc90958da72fe8fa28b6907201',
    ('8cat Each Category',   'mid'):   '22d94d22366a60e55596d1a0829bb8ddb8f1341d38ab70c2594c3aabf9056e31',
    ('8cat Most Categories', 'empty'): '3c20b86a5fa8fbfdc5e30e869263bb3f6bee7cae1dedb47a03224358b4d9159f',
    ('8cat Most Categories', 'mid'):   '4b7f5aa2307ee4c1e3852d5a5c21863565abf723ecc864fab52a1e139b41fc80',
    ('8cat MC + Points',     'empty'): '3eff5d8715b9f9f0f0d094944f2fa32001552f878eec9329bf783db5b95708ff',
    ('8cat MC + Points',     'mid'):   'adeba580c1b5b9d1d51b73ffda558bd66adac36907dc5181542579e159ebf872',
    ('8cat Half and Half',   'empty'): 'a78ee820f2abd42a2a52cf97153a04332c7e8cda1fc022732b1324320143426e',
    ('8cat Half and Half',   'mid'):   '15c9e7c3f91eb2f2e8b5f109cf6731c9d9a9db3dac62205624af5660e027a981',
    ('8cat Half + Points',   'empty'): '18dd4889ab7351691cf5aecdaf956b2f5273749079130b8632242ff237e18b48',
    ('8cat Half + Points',   'mid'):   '1942b68615826c903535aecd4d6dc93c0ef34bcf03c140ec19aac4625aea7308',
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
