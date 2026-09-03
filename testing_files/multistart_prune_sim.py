# testing_files/multistart_prune_sim.py
#
# Simulate pruned multi-start on an empty (draft-start) board and compare to the current single
# (heuristic) start. Scheme: seed the 9 single-category punts, run 2 iterations and keep the best 3,
# run to 10 iterations and keep the best 1, then run that one to convergence.
#
# The descent from a fixed seed is deterministic, so a seed's score "at iteration k" is just that seed
# run for k iterations -- no checkpoint/continue needed. We therefore run every seed at 2 / 10 / N
# iterations and apply the pruning rule per candidate. Changes nothing in the engine.
#
# Run from the project root:  python testing_files/multistart_prune_sim.py [n_candidates] [n_iterations]

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'testing_files'))
os.chdir(_ROOT)

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import pandas as pd

from benchmark_helpers import client, _build_session_request
from backend.state.session import get_session

OBJECTIVE = sys.argv[5] if len(sys.argv) > 5 else 'Each Category'
N_CANDIDATES   = int(sys.argv[1]) if len(sys.argv) > 1 else 40
N_ITERATIONS   = int(sys.argv[2]) if len(sys.argv) > 2 else 250   # convergence budget for the survivor
PUNT_FACTOR    = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1  # punted category -> this fraction of neutral
# (iterations so far, seeds kept). Pass e.g. "2:1" or "2:3,10:1" as argv[4]; default 9 -> 3 at 2, 3 -> 1 at 10.
PRUNE_STAGES   = ([tuple(int(x) for x in stage.split(':')) for stage in sys.argv[4].split(',')]
                  if len(sys.argv) > 4 else [(2, 3), (10, 1)])


def _build_empty_board_agent():
    response = client.post('/sessions', json=_build_session_request(objective=OBJECTIVE))
    assert response.status_code == 201, f'Session creation failed: {response.text}'
    session    = get_session(response.json()['session_id'])
    n_drafters = session.current_settings['n_drafters']
    return session.agent, {f'Team {i + 1}': [] for i in range(n_drafters)}, 'Team 1'


def _uniform_shares(agent):
    return {
        pos_code: {base: 1.0 / len(info['bases']) for base in info['bases']}
        for pos_code, info in agent.position_structure['flex'].items()
    }


def _punt_vectors(agent):
    neutral = agent.v.reshape(agent.n_categories).astype(float)
    vectors = {}
    for category_index, category in enumerate(agent.x_scores.columns):
        punted = neutral.copy()
        punted[category_index] *= PUNT_FACTOR
        vectors[f'punt {category}'] = punted / punted.sum()
    return vectors


def _score(agent, assignments, drafter, subset, iters, initial_weights, shares):
    agent._default_result = None
    if initial_weights is None:
        agent.clear_initial_weights()
    else:
        agent.initial_category_weights = np.asarray(initial_weights, dtype=float)
        agent.initial_shares           = shares
    return agent.get_h_scores(
        player_assignments = assignments,
        drafter            = drafter,
        n_iterations       = iters,
        candidate_subset   = subset,
    )['Scores']


def run_simulation():
    agent, assignments, drafter = _build_empty_board_agent()
    agent._position_mode_override = 'exact'
    subset  = list(agent.x_scores.index[:N_CANDIDATES])
    shares  = _uniform_shares(agent)
    punts   = _punt_vectors(agent)
    checkpoints = sorted({it for it, _ in PRUNE_STAGES} | {N_ITERATIONS})

    print(f'Format={OBJECTIVE!r}  candidates={len(subset)}  convergence={N_ITERATIONS}  '
          f'seeds={len(punts)}  prune={PRUNE_STAGES}')

    # scores[iters] = DataFrame(candidate x seed) of the H-score after `iters` iterations from each seed.
    scores = {}
    for iters in checkpoints:
        scores[iters] = pd.DataFrame({label: _score(agent, assignments, drafter, subset, iters, vec, shares)
                                      for label, vec in punts.items()})
    baseline = _score(agent, assignments, drafter, subset, N_ITERATIONS, None, shares)   # current heuristic

    # Apply the pruning rule per candidate: keep top-k by score at each checkpoint, carry survivors forward.
    survivors = {c: list(punts) for c in subset}
    for iters, keep in PRUNE_STAGES:
        ranked = scores[iters]
        for c in subset:
            survivors[c] = list(ranked.loc[c, survivors[c]].sort_values(ascending=False).index[:keep])
    winner       = {c: survivors[c][0] for c in subset}
    pruned_score = pd.Series({c: scores[N_ITERATIONS].loc[c, winner[c]] for c in subset})

    full_multistart = scores[N_ITERATIONS].max(axis=1)                       # best of all 9 at convergence
    true_best_seed  = scores[N_ITERATIONS].idxmax(axis=1)
    kept_true_best  = int(pd.Series(winner).eq(true_best_seed).sum())

    imp_vs_heuristic = (pruned_score - baseline) * 100
    gap_vs_full      = (full_multistart - pruned_score) * 100

    print('\n-- Pruned multi-start vs current heuristic ' + '-' * 27)
    print(f'improvement (pp):  mean={imp_vs_heuristic.mean():.3f}  median={imp_vs_heuristic.median():.3f}  '
          f'max={imp_vs_heuristic.max():.3f}  helped={int((imp_vs_heuristic > 0.05).sum())}/{len(subset)}')
    print('\n-- Pruning quality (vs running all 9 to convergence) ' + '-' * 17)
    print(f'pruning kept the true-best seed on {kept_true_best}/{len(subset)} candidates')
    print(f'score given up by pruning (pp):  mean={gap_vs_full.mean():.3f}  median={gap_vs_full.median():.3f}  '
          f'max={gap_vs_full.max():.3f}')

    # Draft-order change vs the current heuristic ranking.
    rank_heuristic = baseline.rank(ascending=False, method='min')
    rank_pruned    = pruned_score.rank(ascending=False, method='min')
    shift          = (rank_heuristic - rank_pruned)
    print('\n-- Draft-order change vs heuristic ' + '-' * 35)
    print(f'candidates whose rank changed: {int((shift != 0).sum())}/{len(subset)}   '
          f'max rank shift = {int(shift.abs().max())}')

    # iterations x (seeds running in that interval): all 9 up to the first prune, then the kept counts.
    # A prune checkpoint costs at least 1 iteration to score (the engine floors the loop at 1), so a
    # "0-iteration" prune still runs every seed once -- floor the checkpoints at 1.
    counts = [len(punts)] + [k for _, k in PRUNE_STAGES]
    bounds = [0] + [max(1, it) for it, _ in PRUNE_STAGES] + [N_ITERATIONS]
    ideal  = sum(counts[i] * (bounds[i + 1] - bounds[i]) for i in range(len(counts)))
    print(f'\napprox compute vs single start (ideal per-candidate): {ideal / N_ITERATIONS:.2f}x')


if __name__ == '__main__':
    run_simulation()
