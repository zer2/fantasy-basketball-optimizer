"""Run a twelve-seat field under three update rules, and record what each one does.

`algorithm_agents.py` recomputes the base H-scores over repeated passes, and each pass best-
responds to the RUNNING AVERAGE of every prior pass rather than to the latest one. The comments
there say why: pure best-response oscillates, with every seat flipping onto the same punt at
once. Averaging damps it into a mixed equilibrium where the punts spread across archetypes.

This is the smallest model that exhibits that. Nine identical categories, twelve seats, nine
units of effort each. A seat's chance in a category is Phi(its own effort minus the FIELD's
average effort there), so a category everyone contests is expensive and a category everyone has
abandoned is cheap. Each seat best-responds to what it is shown, and the runs differ only in
what it is shown and in how far it is allowed to jump.

    global / latest     a fresh search over every punt structure, against the previous pass
    global / averaged   the same fresh search, against the mean of every pass so far
    local  / averaged   a search started from where the seat already is, against that mean

What came out was NOT what the note predicts, and the runs are kept as measured:

    global / latest     herds completely -- all twelve seats punt the SAME three categories --
                        and locks into a perfect limit cycle, drift pinned at 1.33 forever
    global / averaged   still oscillates, drift around 1.0. Averaging alone does not damp this
    local  / averaged   settles in a few passes, drift under 0.005, and the punts SPREAD across
                        categories rather than herding

So in this model the thing that produces a spread equilibrium is responding locally -- starting
each pass from where the seat already was, which is what a warm start does -- and not the
averaging on its own. That may be a property of the toy rather than of the shipped algorithm,
whose categories are not identical and whose seats are not interchangeable. It is recorded here
as what the model does, not as a claim about the real one.

    python visualizations/prepare_self_play_data.py
"""

from __future__ import annotations

import json
from math import erf, sqrt
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

_VISUALIZATIONS_DIR = Path(__file__).resolve().parent

SEAT_COUNT = 12
CATEGORY_COUNT = 9
EFFORT_BUDGET = 9.0
PASS_COUNT = 24
STARTING_JITTER = 0.05          # what breaks the tie in an otherwise symmetric field

CATEGORY_NAMES = [
    'Field Goal %', 'Free Throw %', 'Threes',
    'Points', 'Rebounds', 'Assists',
    'Steals', 'Blocks', 'Turnovers',
]


def categories_won_against(weights: np.ndarray, field: np.ndarray) -> float:
    """Expected categories taken, against a field contesting each one at `field`."""
    margins = (weights - field) / sqrt(2.0)
    return float(np.sum([0.5 * (1.0 + erf(margin)) for margin in margins]))


def best_response(field: np.ndarray, start: np.ndarray) -> np.ndarray:
    """The spread of one seat's budget that does best against a given field."""
    result = minimize(
        lambda weights: -categories_won_against(weights, field), start,
        method      = 'SLSQP',
        bounds      = [(0.0, EFFORT_BUDGET)] * CATEGORY_COUNT,
        constraints = {'type': 'eq',
                       'fun': lambda weights: float(np.sum(weights) - EFFORT_BUDGET)},
        options     = {'maxiter': 300, 'ftol': 1e-10},
    )
    if not result.success:
        raise RuntimeError(f'a seat failed to best-respond: {result.message}')
    return np.clip(np.array(result.x), 0.0, None)


def global_best_response(field: np.ndarray, own: np.ndarray) -> np.ndarray:
    """The best response found from every punt structure, not just from where the seat is.

    The distinction turns out to be the whole story. A local search warm-started at a seat's
    own current weights cannot cross the ridge between one punt structure and another, so a
    seat stays in the basin it is in; a global search jumps to whichever structure is cheapest
    right now, and since every seat is shown the same field, every seat jumps to the same one.
    """
    order = np.argsort(field)                    # cheapest categories first
    starts = [own, np.full(CATEGORY_COUNT, EFFORT_BUDGET / CATEGORY_COUNT)]
    for punts in range(1, 5):
        contest_cheapest = np.zeros(CATEGORY_COUNT)
        contest_cheapest[order[:CATEGORY_COUNT - punts]] = EFFORT_BUDGET / (CATEGORY_COUNT - punts)
        contest_dearest = np.zeros(CATEGORY_COUNT)
        contest_dearest[order[punts:]] = EFFORT_BUDGET / (CATEGORY_COUNT - punts)
        starts.extend([contest_cheapest, contest_dearest])

    best, best_score = None, -np.inf
    for start in starts:
        candidate = best_response(field, start)
        score = categories_won_against(candidate, field)
        if score > best_score:
            best, best_score = candidate, score
    return best


def run_field(respond_to_running_average: bool, search_globally: bool) -> dict:
    """Every pass's weights for every seat, plus how far the field moved each pass."""
    generator = np.random.default_rng(5150)
    weights = (np.full((SEAT_COUNT, CATEGORY_COUNT), EFFORT_BUDGET / CATEGORY_COUNT)
               + generator.normal(scale=STARTING_JITTER, size=(SEAT_COUNT, CATEGORY_COUNT)))
    weights = EFFORT_BUDGET * weights / weights.sum(axis=1, keepdims=True)

    history, drift = [weights.copy()], []
    running_total = weights.copy()
    for pass_index in range(PASS_COUNT):
        # What each seat is shown: the latest field, or the average of every field so far.
        shown = (running_total / (pass_index + 1)) if respond_to_running_average else weights
        field = shown.mean(axis=0)
        respond = global_best_response if search_globally else best_response
        updated = np.array([respond(field, weights[seat]) for seat in range(SEAT_COUNT)])
        drift.append(float(np.abs(updated - weights).mean()))
        weights = updated
        running_total += weights
        history.append(weights.copy())

    return {
        'weights': [pass_weights.round(4).tolist() for pass_weights in history],
        'drift':   [round(value, 5) for value in drift],
    }


def count_punts(weights: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """Which categories the field has abandoned, per category, at the end of a run."""
    return (weights < threshold).sum(axis=0)


def main() -> None:
    runs = {}
    for key, averaged, globally in (
        ('global_latest',   False, True),
        ('global_averaged', True,  True),
        ('local_averaged',  True,  False),
    ):
        runs[key] = run_field(averaged, globally)
        final = np.array(runs[key]['weights'][-1])
        drift = runs[key]['drift']
        print(f'{key:16} drift  first {drift[0]:.4f}  last {drift[-1]:.4f}  '
              f'mean of last five {np.mean(drift[-5:]):.4f}')
        print(f'{"":16} punts per category at the end: {count_punts(final).tolist()}')

    data_path = _VISUALIZATIONS_DIR / 'data' / 'self_play.json'
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps({
        'seat_count':     SEAT_COUNT,
        'category_count': CATEGORY_COUNT,
        'category_names': CATEGORY_NAMES,
        'effort_budget':  EFFORT_BUDGET,
        'runs':           runs,
    }), encoding='utf-8')
    print(f'\nWrote {data_path.relative_to(_VISUALIZATIONS_DIR.parent)}')


if __name__ == '__main__':
    main()
