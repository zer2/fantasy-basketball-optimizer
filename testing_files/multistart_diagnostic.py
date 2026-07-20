# testing_files/multistart_diagnostic.py
#
# Diagnostic (NOT a test): does starting the H-score gradient descent from a different punt
# than the heuristic init ever reach a BETTER optimum on an empty (draft-start) board?
#
# It changes nothing in the engine. It uses the existing init-override hook: when
# agent.initial_category_weights is set, get_h_scores broadcasts that vector to every candidate
# as the starting point; when it's None (clear_initial_weights), the per-candidate heuristic is
# used. So for each starting point we just set that attribute and read back the converged score.
#
# For each candidate we compare:
#   - heuristic   : the current behaviour (per-candidate heuristic init)  -> the baseline optimum
#   - neutral     : the plain centre point (agent.v broadcast)
#   - punt <cat>  : the centre with one category down-weighted (one basin per category)
# and report how often, and by how much, the best alternative optimum beats the heuristic's.
#
# Runtime is bounded by scoring only the top-N candidates (candidate_subset) with exact position
# solves. Run from the project root:  python testing_files/multistart_diagnostic.py
# Optional args:  python testing_files/multistart_diagnostic.py <n_candidates> <n_iterations>

import os
import sys
from itertools import combinations

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)                                 # so `import backend` resolves
sys.path.insert(0, os.path.join(_ROOT, 'testing_files'))  # so `import benchmark_helpers` resolves
os.chdir(_ROOT)                                           # so relative data paths (parameters.yaml, backend/data/) resolve

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')   # Windows consoles default to cp1252
except Exception:
    pass

import numpy as np
import pandas as pd

from benchmark_helpers import client, _build_session_request
from backend.state.session import get_session

# ── Config ────────────────────────────────────────────────────────────────────
SCORING_FORMAT = 'Head to Head: Each Category'  # punting bites hardest here
N_CANDIDATES   = int(sys.argv[1]) if len(sys.argv) > 1 else 40    # top-N players to score
N_ITERATIONS   = int(sys.argv[2]) if len(sys.argv) > 2 else 250   # per start; bump to check convergence
MAX_PUNT_COMBO = int(sys.argv[3]) if len(sys.argv) > 3 else 1     # 1 = single-category punts; 2 = also pairs
PUNT_FACTOR    = 0.1     # down-weight the punted category to this fraction of neutral (keep > 0:
                         # an exactly-zero weight divides by zero in the H-score math)
IMPROVEMENT_TOLERANCE_PP = 0.05   # count an alternative as "better" only past this many percentage points


def _build_empty_board_agent():
    """Create a draft session and return (agent, empty_board_assignments, drafter)."""
    response = client.post('/sessions', json=_build_session_request(scoring_format=SCORING_FORMAT))
    assert response.status_code == 201, f'Session creation failed: {response.text}'
    session    = get_session(response.json()['session_id'])
    n_drafters = session.current_params['n_drafters']
    assignments = {f'Team {i + 1}': [] for i in range(n_drafters)}
    return session.agent, assignments, 'Team 1'


def _build_start_points(agent):
    """{label: initial category-weight vector}: the neutral centre plus one punt per category."""
    neutral    = agent.v.reshape(agent.n_categories).astype(float)   # sums to 1
    categories = list(agent.x_scores.columns)

    # Exactly-neutral weights (all == v) are a 0/0 singularity in the H-score term math — one the real
    # init never hits, since the heuristic is always v plus a data-driven offset. Jitter the centre a
    # little so it's representative of "start near the middle" but finite.
    jitter       = 1.0 + 0.05 * np.where(np.arange(agent.n_categories) % 2 == 0, 1.0, -1.0)
    near_neutral = neutral * jitter
    starts = {'near-neutral': near_neutral / near_neutral.sum()}

    short = {c: ''.join(w[0] for w in c.split()) if len(c.split()) > 1 else c[:3] for c in categories}
    for depth in range(1, MAX_PUNT_COMBO + 1):
        for combo in combinations(range(agent.n_categories), depth):
            punted = neutral.copy()
            for category_index in combo:
                punted[category_index] *= PUNT_FACTOR
            label = 'punt ' + '+'.join(short[categories[i]] for i in combo)
            starts[label] = punted / punted.sum()
    return starts


def _uniform_shares(agent):
    """Flex shares held fixed at uniform across all runs, so only the category-weight basin varies."""
    return {
        pos_code: {base: 1.0 / len(pos_info['bases']) for base in pos_info['bases']}
        for pos_code, pos_info in agent.position_structure['flex'].items()
    }


def _run_from_start(agent, assignments, drafter, subset, initial_weights, shares):
    """Full get_h_scores result (Scores + converged Weights) from a fixed weight vector
    (None => heuristic init)."""
    agent._default_result = None            # never short-circuit to the cached empty-board baseline
    if initial_weights is None:
        agent.clear_initial_weights()       # per-candidate heuristic init (current behaviour)
    else:
        agent.initial_category_weights = np.asarray(initial_weights, dtype=float)
        agent.initial_shares           = shares
    return agent.get_h_scores(
        player_assignments = assignments,
        drafter            = drafter,
        n_iterations       = N_ITERATIONS,
        candidate_subset   = subset,
    )


def run_diagnostic():
    agent, assignments, drafter = _build_empty_board_agent()
    agent._position_mode_override = 'exact'   # true optima, not throttle approximations

    subset       = list(agent.x_scores.index[:N_CANDIDATES])
    start_points = _build_start_points(agent)
    shares       = _uniform_shares(agent)

    print(f'Format={SCORING_FORMAT!r}  candidates={len(subset)}  iterations={N_ITERATIONS}  '
          f'punt_factor={PUNT_FACTOR}  max_combo={MAX_PUNT_COMBO}  starts={len(start_points)}')

    heuristic_result = _run_from_start(agent, assignments, drafter, subset, None, shares)
    table           = pd.DataFrame({'heuristic': heuristic_result['Scores']})
    weights_by_seed = {'heuristic': heuristic_result['Weights']}
    for label, vector in start_points.items():
        result               = _run_from_start(agent, assignments, drafter, subset, vector, shares)
        table[label]         = result['Scores']
        weights_by_seed[label] = result['Weights']

    alternative_labels = list(start_points)

    nonfinite = {c: int((~np.isfinite(table[c])).sum()) for c in ['heuristic'] + alternative_labels}
    bad = {c: n for c, n in nonfinite.items() if n}
    if bad:
        print(f'WARNING: non-finite scores from some starts (count per start): {bad}')

    table['best_alt']       = table[alternative_labels].max(axis=1)
    table['best_alt_start'] = table[alternative_labels].idxmax(axis=1)
    table['improvement_pp'] = (table['best_alt'] - table['heuristic']) * 100

    improved = table[table['improvement_pp'] > IMPROVEMENT_TOLERANCE_PP].sort_values(
        'improvement_pp', ascending=False
    )

    print('\n-- Result ' + '-' * 60)
    print(f'candidates where an alternative start beats the heuristic (> {IMPROVEMENT_TOLERANCE_PP} pp): '
          f'{len(improved)} / {len(table)}')
    if len(improved) > 0:
        print(f'improvement (pp)  max={improved.improvement_pp.max():.3f}  '
              f'mean={improved.improvement_pp.mean():.3f}  median={improved.improvement_pp.median():.3f}')
        print('\nwinning start among improved candidates:')
        print(improved['best_alt_start'].value_counts().to_string())
        print('\ntop improvements:')
        top = improved.head(15)[['heuristic', 'best_alt', 'best_alt_start', 'improvement_pp']].copy()
        top['heuristic'] = (top['heuristic'] * 100).round(3)
        top['best_alt']  = (top['best_alt'] * 100).round(3)
        top['improvement_pp'] = top['improvement_pp'].round(3)
        print(top.to_string())
    else:
        print('No alternative start reached a better optimum than the heuristic — '
              'the surface looks effectively unimodal here (given these iterations/candidates).')

    neutral_beats = int(((table['near-neutral'] - table['heuristic']) * 100 > IMPROVEMENT_TOLERANCE_PP).sum())
    print(f'\n(aside) near-neutral centre beats the heuristic on {neutral_beats} / {len(table)} candidates')

    # Which SINGLE init is best if you had to pick one for everyone? Rank by mean converged score.
    all_methods = ['heuristic'] + alternative_labels
    argmax_all  = table[all_methods].idxmax(axis=1)
    per_start = pd.DataFrame({
        'mean_score_pp':   (table[all_methods].mean() * 100).round(3),
        'median_score_pp': (table[all_methods].median() * 100).round(3),
        'beats_heuristic': [int((table[m] > table['heuristic'] + IMPROVEMENT_TOLERANCE_PP / 100).sum())
                            for m in all_methods],
        'outright_best':   [int((argmax_all == m).sum()) for m in all_methods],
    }).sort_values('mean_score_pp', ascending=False)
    heuristic_rank = list(per_start.index).index('heuristic') + 1
    print('\n-- Best single initialization (one init used for all candidates) ' + '-' * 6)
    print(per_start.head(15).to_string())
    print(f'... heuristic (current) ranks {heuristic_rank} / {len(per_start)} by mean score')

    # Are the pair seeds necessary, or do they just reach the same punt a single seed already finds?
    single_cols = ['heuristic'] + [c for c in alternative_labels if '+' not in c]
    pair_cols   = [c for c in alternative_labels if '+' in c]
    if pair_cols:
        best_single_score = table[single_cols].max(axis=1)
        best_single_col   = table[single_cols].idxmax(axis=1)
        best_all_score    = table[all_methods].max(axis=1)
        best_all_col      = table[all_methods].idxmax(axis=1)
        pair_gain_pp      = (best_all_score - best_single_score) * 100

        pair_wins = [c for c in table.index
                     if best_all_col[c] in pair_cols and pair_gain_pp[c] > IMPROVEMENT_TOLERANCE_PP]

        print('\n-- Are pair seeds necessary? ' + '-' * 41)
        print(f'a pair beats the best single seed (> {IMPROVEMENT_TOLERANCE_PP} pp) on '
              f'{len(pair_wins)} / {len(table)} candidates')
        if pair_wins and all(weights_by_seed[c] is not None for c in [best_all_col[x] for x in pair_wins]):
            gains  = pair_gain_pp[pair_wins]
            shifts = pd.Series({                          # L1 distance in category-weight space
                c: float(np.abs(weights_by_seed[best_all_col[c]].loc[c].values
                                - weights_by_seed[best_single_col[c]].loc[c].values).sum())
                for c in pair_wins
            })
            print(f'  extra score from pairs (pp):    max={gains.max():.3f}  mean={gains.mean():.3f}  median={gains.median():.3f}')
            print(f'  weight L1 shift vs best single:  max={shifts.max():.3f}  mean={shifts.mean():.3f}  median={shifts.median():.3f}'
                  '   (0 = identical punt, 2 = disjoint)')
            print('  -> small gain AND small shift = the pair just re-finds the single-seed punt (redundant).')

    # THE decision question: does multi-start (best punt per candidate) change the DRAFT ORDER, or
    # just nudge scores while leaving the ranking — and therefore every pick — unchanged?
    current_score    = table['heuristic']                 # current behaviour: heuristic init only
    multistart_score = table[all_methods].max(axis=1)     # best over heuristic + every punt seed
    rank_current = current_score.rank(ascending=False, method='min')
    rank_multi   = multistart_score.rank(ascending=False, method='min')
    shift        = (rank_current - rank_multi)            # positive = moves up under multi-start
    moved        = int((shift != 0).sum())

    print('\n-- Does multi-start change the draft order? ' + '-' * 27)
    print(f'candidates whose rank changed: {moved} / {len(table)}   '
          f'Spearman(scores) = {current_score.corr(multistart_score, method="spearman"):.4f}   '
          f'max rank shift = {int(shift.abs().max())}')
    comparison = pd.DataFrame({
        'multistart_rank': rank_multi.astype(int),
        'heuristic_rank':  rank_current.astype(int),
        'shift':           shift.astype(int),
        'best_seed':       table[all_methods].idxmax(axis=1),
    }).sort_values('multistart_rank').head(15)
    print(comparison.to_string())


if __name__ == '__main__':
    run_diagnostic()
