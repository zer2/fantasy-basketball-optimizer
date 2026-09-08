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
    ('Each Category',  'empty'): 'f96e2826c5648a044850163038e54f719fcaf080330d3ca58fccf00d45c27ce9',
    ('Each Category',  'mid'):   '03d6becd123f8e4c542a6e6f8169d028057871f46799ea10901c108e829be0d9',
    ('Half and Half',  'empty'): '1390cb842e7e0148d5d0bc6add00150222f1f05bcf944f789373744a92c9aa06',
    ('Half and Half',  'mid'):   '16847a24837b071c9b1f8e9655c53707cf9f2112e655051dc6d47390a69c081d',
    ('Most Categories','empty'): 'd03f91b229d087762bae8c9d165ea92dfc97c5f768b72d41bc94b0a3d742799c',
    ('Most Categories','mid'):   'd08c8c368ba0b048482a04252a4c41d0af40dd87604b6123bc2c5d5c38932ae1',

    # Eight categories (turnovers dropped), which is what a tiebreaker needs: a matchup that can
    # end level. Each objective appears with and without one named, so a change to the weighted
    # win-count DP, to what a category is worth in v, or to the G-score ranking the board is drawn
    # from has to show up here rather than only in a league nobody tested.
    ('8cat Each Category',   'empty'): '7a247cad288f6b015e15c362de828d02ced58a49d3fc44297898a9f855cead83',
    ('8cat Each Category',   'mid'):   'e8d2c5b0cb97c26bb36e78de8b066340f3cada0c48d619efaef70332616a6ca0',
    ('8cat Most Categories', 'empty'): 'db23d3cf41c697c1534f2f6cb8b6ec26f9f9684b98e77ded3c0a7f39d2f12ac5',
    ('8cat Most Categories', 'mid'):   '8013471e134feeed4d9ae16e3bb9aeaaeb4d9b67b7ba3c4b01d239fb2aacb40f',
    ('8cat MC + Points',     'empty'): '244c1a0722f6ed8171fc97de5dce44ca4934e07e444e671d417e48d3e35e8919',
    ('8cat MC + Points',     'mid'):   'cb4a525e79613fa36d10dcb0c62743430af6f3c64792f220efba5fb2b7a0d51c',
    ('8cat Half and Half',   'empty'): '73c7353d80e55b201e4da4c1dc7a1f082373e87ac7b902fb0c0d61819d27bdad',
    ('8cat Half and Half',   'mid'):   'e1d3a5bb535d5b85ea9b54ce3afe5ddd16f271e23f795d1e8b6c46cfd66b4168',
    ('8cat Half + Points',   'empty'): '2e2e70fb170035f009b02303984a71ae2e4788d898b0ad0e461ace35fb912be6',
    ('8cat Half + Points',   'mid'):   '0d22c09f46b1ce71646ea55ce0ad6355b2ef29d667f04dc71b38012d10a641fa',
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
