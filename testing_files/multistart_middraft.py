# testing_files/multistart_middraft.py
#
# Diagnostic (NOT a test): mid-draft, seed a single x0.9 punt of the category the drafting team is
# CURRENTLY WEAKEST in, and ask whether that roster-aware single seed (a) beats the heuristic init and
# (b) captures what blind 9-punt multi-start would get (i.e. is the weakest category usually the best
# seed anyway). Runs at the app's real 30-iteration budget. Changes nothing in the engine.
#
# Run from the project root:  python testing_files/multistart_middraft.py [n_candidates] [n_iterations]

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

OBJECTIVE = 'Each Category'
N_CANDIDATES   = int(sys.argv[1]) if len(sys.argv) > 1 else 40
N_ITERATIONS   = int(sys.argv[2]) if len(sys.argv) > 2 else 30
PUNT_FACTOR    = 0.9
ROSTER_SIZE    = 5     # how many players the drafting team already holds


def _build_agent():
    response = client.post('/sessions', json=_build_session_request(objective=OBJECTIVE))
    assert response.status_code == 201, f'Session creation failed: {response.text}'
    session    = get_session(response.json()['session_id'])
    n_drafters = session.current_settings['n_drafters']
    return session.agent, n_drafters


def _uniform_shares(agent):
    return {pos_code: {base: 1.0 / len(info['bases']) for base in info['bases']}
            for pos_code, info in agent.position_structure['flex'].items()}


def _punt(neutral, category_index):
    w = neutral.copy()
    w[category_index] *= PUNT_FACTOR
    return w / w.sum()


def _run(agent, assignments, drafter, subset, initial_weights, shares):
    agent._default_result = None
    if initial_weights is None:
        agent.clear_initial_weights()
    else:
        agent.initial_category_weights = np.asarray(initial_weights, dtype=float)
        agent.initial_shares           = shares
    return agent.get_h_scores(player_assignments=assignments, drafter=drafter,
                              n_iterations=N_ITERATIONS, candidate_subset=subset)['Scores']


def run_diagnostic():
    agent, n_drafters = _build_agent()
    agent._position_mode_override = 'exact'
    neutral    = agent.v.reshape(agent.n_categories).astype(float)
    categories = list(agent.x_scores.columns)
    shares     = _uniform_shares(agent)

    # Bigs-heavy roster: the top players whose position includes centre -> should be weak in assists.
    centers = [p for p in agent.x_scores.index if 'C' in p.split('(')[-1]][:ROSTER_SIZE]
    assignments = {f'Team {i + 1}': [] for i in range(n_drafters)}
    assignments['Team 1'] = centers

    # Weakest category = smallest summed team strength (x-scores) across the roster.
    team_totals   = agent.x_scores.loc[centers].sum(axis=0)
    weakest_index = int(np.argmin(team_totals.values))
    weakest_cat   = categories[weakest_index]

    print(f'Format={OBJECTIVE!r}  drafter roster ({len(centers)}): {centers}')
    print(f'team strength by category (x-score sum):')
    print(team_totals.round(2).to_string())
    print(f'--> weakest category = {weakest_cat!r}  (seed = x0.9 punt of it)\n')

    subset = [p for p in agent.x_scores.index if p not in centers][:N_CANDIDATES]

    heuristic     = _run(agent, assignments, 'Team 1', subset, None, shares)
    weakest_seed  = _run(agent, assignments, 'Team 1', subset, _punt(neutral, weakest_index), shares)
    all_punts     = pd.DataFrame({cat: _run(agent, assignments, 'Team 1', subset, _punt(neutral, i), shares)
                                  for i, cat in enumerate(categories)})

    full_best     = all_punts.max(axis=1)
    best_seed     = all_punts.idxmax(axis=1)

    imp_weakest = (weakest_seed - heuristic) * 100
    imp_full    = (full_best - heuristic) * 100
    gap         = (full_best - weakest_seed) * 100          # what the single seed gives up vs all 9

    print('-- Weakest-category single seed vs heuristic ' + '-' * 24)
    print(f'improvement (pp):  mean={imp_weakest.mean():.3f}  median={imp_weakest.median():.3f}  '
          f'helped={int((imp_weakest > 0.05).sum())}/{len(subset)}')
    print('\n-- Full 9-punt multi-start vs heuristic (upper bound) ' + '-' * 15)
    print(f'improvement (pp):  mean={imp_full.mean():.3f}  median={imp_full.median():.3f}  '
          f'helped={int((imp_full > 0.05).sum())}/{len(subset)}')
    print('\n-- Does the single weakest-category seed capture the multi-start gain? ' + '-' * 2)
    print(f'gap vs full 9-punt (pp):  mean={gap.mean():.3f}  median={gap.median():.3f}  max={gap.max():.3f}')
    print(f'weakest category IS the best-of-9 seed on {int((best_seed == weakest_cat).sum())}/{len(subset)} candidates')
    print(f'best-of-9 seed distribution:')
    print(best_seed.value_counts().to_string())


if __name__ == '__main__':
    run_diagnostic()
