# testing_files/multistart_flex_diagnostic.py
#
# Diagnostic (NOT a test): can you reach the same ultimate PUNT strategies by seeding the flex
# POSITION allocation instead of the category weights?
#
# Two seed groups, run on an empty (draft-start) board, both descending category weights AND shares:
#   category seeds : punt one category (fixed weight vector), shares start uniform
#   position seeds : neutral category weights, shares start loaded toward one base position
# For each seed we read back the converged category weights = the punt it lands on. Then we ask:
#   - do the position seeds reach *distinct* punts (does flex allocation couple to punting)?
#   - do those punts *match* the ones the category seeds reach (nearest-neighbour in weight space)?
#
# Changes nothing in the engine. Run from the project root:
#   python testing_files/multistart_flex_diagnostic.py [n_candidates] [n_iterations]

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
N_CANDIDATES   = int(sys.argv[1]) if len(sys.argv) > 1 else 24
N_ITERATIONS   = int(sys.argv[2]) if len(sys.argv) > 2 else 250
PUNT_FACTOR    = 0.1     # down-weight a punted category to this fraction of neutral
LOAD_FACTOR    = 1.0     # share on the seeded base in each slot that can hold it (1.0 = fully, "2/0")


def _build_empty_board_agent():
    response = client.post('/sessions', json=_build_session_request(objective=OBJECTIVE))
    assert response.status_code == 201, f'Session creation failed: {response.text}'
    session    = get_session(response.json()['session_id'])
    n_drafters = session.current_params['n_drafters']
    return session.agent, {f'Team {i + 1}': [] for i in range(n_drafters)}, 'Team 1'


def _uniform_shares(agent):
    return {
        pos_code: {base: 1.0 / len(info['bases']) for base in info['bases']}
        for pos_code, info in agent.position_structure['flex'].items()
    }


def _shares_loaded_toward(agent, target_base):
    """Every flex slot that can hold target_base is committed to it (a PG seed sends both the G slot
    and the Util slot to PG -> guards 2/0); slots that can't hold it stay balanced (F 1/1)."""
    shares = {}
    for pos_code, info in agent.position_structure['flex'].items():
        bases = info['bases']
        if target_base in bases and len(bases) > 1:
            spill = (1.0 - LOAD_FACTOR) / (len(bases) - 1)
            shares[pos_code] = {b: (LOAD_FACTOR if b == target_base else spill) for b in bases}
        else:
            shares[pos_code] = {b: 1.0 / len(bases) for b in bases}
    return shares


def _near_neutral(agent):
    neutral = agent.v.reshape(agent.n_categories).astype(float)
    jitter  = 1.0 + 0.05 * np.where(np.arange(agent.n_categories) % 2 == 0, 1.0, -1.0)
    return (neutral * jitter) / (neutral * jitter).sum()


def _build_seeds(agent):
    """{label: (category_weight_vector, shares, group)}."""
    neutral      = agent.v.reshape(agent.n_categories).astype(float)
    near_neutral = _near_neutral(agent)
    categories   = list(agent.x_scores.columns)
    uniform      = _uniform_shares(agent)

    seeds = {'neutral/uniform-shares': (near_neutral, uniform, 'position')}
    for category_index, category in enumerate(categories):          # category seeds: punt one category
        punted = neutral.copy()
        punted[category_index] *= PUNT_FACTOR
        seeds[f'punt {category}'] = (punted / punted.sum(), uniform, 'category')
    for base in agent.position_structure['base_list']:              # position seeds: load one base
        seeds[f'load {base}'] = (near_neutral.copy(), _shares_loaded_toward(agent, base), 'position')
    return seeds


def _run(agent, assignments, drafter, subset, category_vector, shares):
    agent._default_result = None
    agent.initial_category_weights = np.asarray(category_vector, dtype=float)
    agent.initial_shares           = shares
    return agent.get_h_scores(
        player_assignments = assignments,
        drafter            = drafter,
        n_iterations       = N_ITERATIONS,
        candidate_subset   = subset,
    )


def _punt_signature(weight_row, neutral, categories, k=2):
    """The k most down-weighted categories relative to neutral = the punt this weight vector encodes."""
    ratio = np.asarray(weight_row) / neutral
    return [categories[i] for i in np.argsort(ratio)[:k]]


def run_diagnostic():
    agent, assignments, drafter = _build_empty_board_agent()
    agent._position_mode_override = 'exact'
    neutral    = agent.v.reshape(agent.n_categories).astype(float)
    categories = list(agent.x_scores.columns)
    subset     = list(agent.x_scores.index[:N_CANDIDATES])
    seeds      = _build_seeds(agent)

    groups = {label: group for label, (_, _, group) in seeds.items()}
    print(f'Format={OBJECTIVE!r}  candidates={len(subset)}  iterations={N_ITERATIONS}  '
          f'category-seeds={sum(g == "category" for g in groups.values())}  '
          f'position-seeds={sum(g == "position" for g in groups.values())}')

    scores  = {}
    weights = {}
    for label, (vector, shares, _group) in seeds.items():
        result         = _run(agent, assignments, drafter, subset, vector, shares)
        scores[label]  = result['Scores']
        weights[label] = result['Weights']

    cat_labels = [l for l, g in groups.items() if g == 'category']
    pos_labels = [l for l, g in groups.items() if g == 'position']

    # ── Illustrative: for the top candidate, what punt each seed lands on ──────────────────
    example = subset[0]
    print(f'\n-- What punt each seed reaches for {example} ' + '-' * 20)
    rows = []
    for label in seeds:
        rows.append({
            'seed':   label,
            'group':  groups[label],
            'punt (converged)': ' / '.join(_punt_signature(weights[label].loc[example].values, neutral, categories)),
            'score_pp': round(float(scores[label][example]) * 100, 3),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # Raw converged weights (as % of neutral) for the example, so the "punt" is visible, not a noisy label.
    print(f'\n-- Converged weights as % of neutral for {example} (100 = neutral; low = punted) ' + '-' * 4)
    pct = pd.DataFrame(
        {label: np.round(weights[label].loc[example].values / neutral * 100, 0).astype(int)
         for label in seeds},
        index=[c[:8] for c in categories],
    ).T
    pct['group'] = [groups[l] for l in seeds]
    print(pct.to_string())

    # ── Aggregate across candidates ───────────────────────────────────────────────────────
    # Compare the spread of converged punts from category seeds vs position seeds, and whether a
    # position seed ever lands on the same punt a category seed reaches (all L1 in weight space).
    pos_diversity, cat_diversity, coverage = [], [], []
    for candidate in subset:
        pos_vectors = np.vstack([weights[l].loc[candidate].values for l in pos_labels])
        cat_vectors = np.vstack([weights[l].loc[candidate].values for l in cat_labels])
        pos_diversity.append(np.abs(pos_vectors[:, None, :] - pos_vectors[None, :, :]).sum(axis=2).max())
        cat_diversity.append(np.abs(cat_vectors[:, None, :] - cat_vectors[None, :, :]).sum(axis=2).max())
        for cat_vector in cat_vectors:
            coverage.append(np.abs(pos_vectors - cat_vector).sum(axis=1).min())

    pos_diversity, cat_diversity, coverage = map(np.array, (pos_diversity, cat_diversity, coverage))

    print('\n-- Aggregate ' + '-' * 57)
    print(f'category-seed punt diversity (max pairwise L1 per candidate): '
          f'mean={cat_diversity.mean():.3f}  median={np.median(cat_diversity):.3f}')
    print(f'position-seed punt diversity (max pairwise L1 per candidate): '
          f'mean={pos_diversity.mean():.3f}  median={np.median(pos_diversity):.3f}')
    print(f'  (0 = seeds collapse to one punt; compare position vs category to see if flex seeding spans as much)')
    print(f'coverage of category punts by nearest position punt (L1):     '
          f'mean={coverage.mean():.3f}  median={np.median(coverage):.3f}  '
          f'(0 = a position seed reaches that punt, large = never)')

    cat_best = pd.DataFrame({l: scores[l] for l in cat_labels}).max(axis=1)
    pos_best = pd.DataFrame({l: scores[l] for l in pos_labels}).max(axis=1)
    print(f'best score per candidate (pp):  category seeds mean={cat_best.mean() * 100:.3f}   '
          f'position seeds mean={pos_best.mean() * 100:.3f}')


if __name__ == '__main__':
    run_diagnostic()
