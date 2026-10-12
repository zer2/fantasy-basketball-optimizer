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
#
# Regenerated 2026-10-11 for psi: a stale /100 on Games Played % had left it inert since
# 2025-08-14, and it now credits a replacement's value over an empty game (process_player_data
# games_played_adjustment), so every G- and X-score moves. The seven 'empty' rows moved again the
# same day with get_category_value_direction, which computes the replacement split in a different
# operation order: identical in exact arithmetic, ~1e-16 apart in floating point, and self-play
# amplifies that into a new hash (and 0.3 of an H-score in the 16-category draft golden).
_GOLDEN = {
    ('Each Category',  'empty'): '7fb78f42889af5e987313826de45f9fcb6eb230319c691001f14552a409c2a81',
    ('Each Category',  'mid'):   '8a202c92821ae8f429d7bfbb5265b2ac536fa5ffc9b08cf38f3a5861c335f13c',
    ('Half and Half',  'empty'): '010852f4df59805bcb67df141106c8c05287870e886a70a730c43336085e535d',
    ('Half and Half',  'mid'):   '113a786b893d451a6cff1b23b1e93ece236a93606e6576a7175e49cc30cf7168',
    ('Most Categories','empty'): '05ab6edc2304699e15b8fe4cc73a51e5dcf5ad6a9b5fc8b9e40fff85f4b941e4',
    ('Most Categories','mid'):   '45a8753f5943a2aec06ca9e45bcc79440a0ef658474257d59339707644bfe3f0',

    # Eight categories (turnovers dropped), which is what a tiebreaker needs: a matchup that can
    # end level. Each objective appears with and without one named, so a change to the weighted
    # win-count DP, to what a category is worth in v, or to the G-score ranking the board is drawn
    # from has to show up here rather than only in a league nobody tested.
    ('8cat Each Category',   'empty'): '83f4f54aaf889cece872be684c2e23def5df0f7a576208ce607f6c274eb9362f',
    ('8cat Each Category',   'mid'):   '36a6cfb01c93b06464d843328b8b082a1e375e1e96e7a78fca20c8d1b8690599',
    ('8cat Most Categories', 'empty'): '3f2b343504277bc68fc245cd6d841feb10950f1baf1f3b6026aea667481631ae',
    ('8cat Most Categories', 'mid'):   'e10aa95dd4bc4e60e5c7be35addb84056791065300f20e17e1d98d68ff6f0a2c',
    ('8cat MC + Points',     'empty'): '3b232ffb44f6fd129610b77dba3cd9089b48dee3a658242b163cd71c1e58fb59',
    ('8cat MC + Points',     'mid'):   '31e5bd22806f46f756ed5268fcd7d55273462b3f12308505bbad94259776cdc3',
    ('8cat Half and Half',   'empty'): 'a3b6b722570d812ebd3c27f4c53758057e7bddf308c5c1f8920b7992794e17fd',
    ('8cat Half and Half',   'mid'):   '77eb1f79a7b7b21f023025c079680f49cfaad2f245315f1e105aa4a0937f072a',
    ('8cat Half + Points',   'empty'): 'fdac267162b2a9131213753ea488a7411868aad8d951d11cc5750f0ee5d5534f',
    ('8cat Half + Points',   'mid'):   '2de62d0b0c9c54f460cd37cc57d39027fe791571539b19f130afadffc7c22c9a',
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
