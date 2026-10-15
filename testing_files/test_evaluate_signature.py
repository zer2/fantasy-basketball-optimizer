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
    ('Each Category',  'empty'): 'f4341827e061dd54b23cc3dd341ed6ccbbed6e6c933e629da702cb2d2cdebfdb',
    ('Each Category',  'mid'):   '9c265753f7647dc72a38bd78e51c113f0067ca13730a61218198c979928ad7d8',
    ('Half and Half',  'empty'): '2bea99492204dc73e473b9bc84f2844753df9f0204c84de8a5cce8e80e80ac28',
    ('Half and Half',  'mid'):   'f9f06b2c19bfa0d52a2c43d0f2fa1fb84a541086f9fdbf1192d1d35258d0a70f',
    ('Most Categories','empty'): 'd3148b8d737782e99f9fbd3238a95c758cdda09f5604f504a4550d1785dcc17f',
    ('Most Categories','mid'):   '8335e088c76cab9fee275e624ec7da9ee4568fc9100aed4c49fbbf0a078dc177',

    # Eight categories (turnovers dropped), which is what a tiebreaker needs: a matchup that can
    # end level. Each objective appears with and without one named, so a change to the weighted
    # win-count DP, to what a category is worth in v, or to the G-score ranking the board is drawn
    # from has to show up here rather than only in a league nobody tested.
    ('8cat Each Category',   'empty'): '24928972943ae3dc00e1689e65d5f720c587ce617c28e8a41b868979630c320a',
    ('8cat Each Category',   'mid'):   '787ef40f864fd97fee40bcba6da2d7055af398644917fbb6d7b8dead8ffef018',
    ('8cat Most Categories', 'empty'): '840464f06ee9914c2f8cb3b6626078e6e643a18bf8f5fafbc93a7470bb2972ce',
    ('8cat Most Categories', 'mid'):   '0135622bcf227f561e23eb2ee77ba1a9a51eae505cb25bd2e6122e95ceab4628',
    ('8cat MC + Points',     'empty'): '4a60d96b0bb30b40eca46b4de09b04138810d26a2042b0a94cbf9111dce919cc',
    ('8cat MC + Points',     'mid'):   'e14688bcedcec5cbb8e5e450512ececfeb1a33dacdba1c4b185bc540f6cecde4',
    ('8cat Half and Half',   'empty'): 'eb52265228041cad5ad496ee62edd050520c8d78969a44274e3e1f680e8dcc6f',
    ('8cat Half and Half',   'mid'):   '1259261fec2ea9fd6fae8fe1377653eae9edea6d08ae07ceadb5b98e107ce49b',
    ('8cat Half + Points',   'empty'): 'b21bd12dc282f42c79d8b6f5aa899908bcfb763374c62c82c45d8b286c50b878',
    ('8cat Half + Points',   'mid'):   'c5c9a37791aafeddc2a16fe5f4847eae1984d685eee3b98e44792d5853673f53',
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
