"""Find how hard each scoring format wants to punt, by running the same search under each.

The claim the scene is built on -- Most Categories punts harder than Each Category, Rotisserie
punts less -- comes from the docs, and the scene is only worth making if it is true. So it is
measured here rather than assumed, and the scene plots whatever comes back.

The apparatus is the punting scene's: nine identical categories, nine units of effort to spread
across them, and an opponent who spreads theirs evenly so parity is a coin flip. A unit of effort
in a category moves your margin there by one standard deviation, making the chance of taking it
Phi(w - 1). Only the PAYOFF changes between runs:

    Each Category    one point per category won            sum of Phi
    Most Categories  all or nothing on winning 5 of 9      P(wins >= 5), a Poisson binomial
    Rotisserie       rank among twelve, per category       sum of Phi(margin / sqrt 2)

Rotisserie's square root of two is not a fudge. In the other two formats you are measured against
ONE opponent's total, so the margin's spread is the pairwise one. In Rotisserie you are placed
among eleven independent opponents, and what decides your rank is your own draw against each of
theirs -- two independent draws, so the spread widens by root two and every category's payoff
flattens. A flatter payoff is the mechanism behind punting less, if that is what comes out.

The script also writes, for each format, what one category is worth as a function of your margin
in it while the other eight stay at that format's own optimum. That curve is the scene's
explanation: the optima differ because these curves differ in shape.

    python visualizations/prepare_payoff_formats_data.py
"""

from __future__ import annotations

import json
from math import erf, sqrt
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

_VISUALIZATIONS_DIR = Path(__file__).resolve().parent.parent

CATEGORY_COUNT = 9
EFFORT_BUDGET = 9.0
OPPONENT_EFFORT = 1.0
CATEGORIES_NEEDED = 5          # 5 of 9 takes the week in Most Categories
ROTISSERIE_RIVALS = 11         # the other eleven teams in a twelve-team league

_PUNTED_BELOW = 0.05           # effort under this is an abandoned category, not a thin one
_MARGIN_SAMPLES = 161


def win_probability(weight: float | np.ndarray) -> np.ndarray:
    """The chance of taking a category against an opponent at parity."""
    margin = np.asarray(weight, dtype=float) - OPPONENT_EFFORT
    return 0.5 * (1.0 + np.vectorize(erf)(margin / sqrt(2.0)))


def rank_share(weight: float | np.ndarray) -> np.ndarray:
    """The share of eleven rivals beaten in a category -- Rotisserie's payoff, normalised.

    Widened by root two against `win_probability` because a rank is settled by your own draw
    against each rival's, which is two independent draws rather than one margin.
    """
    margin = np.asarray(weight, dtype=float) - OPPONENT_EFFORT
    return 0.5 * (1.0 + np.vectorize(erf)(margin / (sqrt(2.0) * sqrt(2.0))))


def distribution_of_wins(probabilities: np.ndarray) -> np.ndarray:
    """The Poisson binomial: P(exactly k categories won), k = 0..9, by convolution."""
    distribution = np.array([1.0])
    for probability in probabilities:
        shifted = np.zeros(len(distribution) + 1)
        shifted[:-1] += distribution * (1.0 - probability)
        shifted[1:] += distribution * probability
        distribution = shifted
    return distribution


def score_each_category(weights: np.ndarray) -> float:
    return float(np.sum(win_probability(weights)))


def score_most_categories(weights: np.ndarray) -> float:
    distribution = distribution_of_wins(win_probability(weights))
    return float(np.sum(distribution[CATEGORIES_NEEDED:]))


def score_rotisserie(weights: np.ndarray) -> float:
    return float(np.sum(rank_share(weights)) * ROTISSERIE_RIVALS)


PAYOFFS = {
    'each_category':   ('Each Category',   score_each_category),
    'most_categories': ('Most Categories', score_most_categories),
    'rotisserie':      ('Rotisserie',      score_rotisserie),
}


def _starting_points() -> list[np.ndarray]:
    """Balanced, plus every number of punts from one to four, plus a few random spreads.

    Multi-start rather than one descent because the punt structure makes this landscape
    multi-peaked -- each number of abandoned categories is its own local optimum, which is the
    subject of another scene entirely.
    """
    starts = []
    for punts in range(0, 5):
        weights = np.zeros(CATEGORY_COUNT)
        weights[punts:] = EFFORT_BUDGET / (CATEGORY_COUNT - punts)
        starts.append(weights)
    generator = np.random.default_rng(20260913)
    for _ in range(12):
        draw = generator.random(CATEGORY_COUNT)
        starts.append(EFFORT_BUDGET * draw / draw.sum())
    return starts


def find_optimum(score) -> tuple[np.ndarray, float]:
    """The best spread of the budget under one payoff, over every start."""
    constraints = {'type': 'eq', 'fun': lambda weights: float(np.sum(weights) - EFFORT_BUDGET)}
    bounds = [(0.0, EFFORT_BUDGET)] * CATEGORY_COUNT

    best_weights, best_score = None, -np.inf
    for start in _starting_points():
        result = minimize(
            lambda weights: -score(weights), start,
            method      = 'SLSQP',
            bounds      = bounds,
            constraints = constraints,
            options     = {'maxiter': 400, 'ftol': 1e-10},
        )
        if result.success and -result.fun > best_score:
            best_weights, best_score = np.array(result.x), float(-result.fun)
    if best_weights is None:
        raise RuntimeError('no start converged; the payoff or the constraints are wrong')
    return best_weights, best_score


def measure_single_category_payoff(score, optimum: np.ndarray) -> dict:
    """What one category is worth, as your margin in it moves and the other eight hold still.

    This is the shape that decides everything: a payoff that saturates early makes a near-won
    category cheap to abandon, and one that stays steep makes abandoning anything expensive.
    """
    margins = np.linspace(-3.0, 3.0, _MARGIN_SAMPLES)
    values = []
    for margin in margins:
        weights = np.array(optimum, dtype=float)
        weights[0] = OPPONENT_EFFORT + margin
        values.append(score(weights))
    values = np.array(values)
    return {
        'margins': [round(float(margin), 3) for margin in margins],
        # Normalised to its own range: the three payoffs are in different units (categories,
        # a probability, rank points) and what the scene compares is their SHAPE.
        'payoff':  [round(float(value), 6)
                    for value in (values - values.min()) / (values.max() - values.min())],
    }


def weights_after_draining(punted: float) -> np.ndarray:
    """The budget once `punted` categories have been given up, sharing the rest equally.

    The same continuous drain the punting scene walks, so the three formats are compared along
    one path rather than at three unrelated points.
    """
    whole = int(punted)
    weights = np.zeros(CATEGORY_COUNT)
    contested = CATEGORY_COUNT - whole
    weights[whole:] = EFFORT_BUDGET / contested
    if whole < CATEGORY_COUNT:
        draining = (EFFORT_BUDGET / contested) * (1.0 - (punted - whole))
        weights[whole] = draining
        remaining = [index for index in range(whole + 1, CATEGORY_COUNT)]
        if remaining:
            weights[remaining] = (EFFORT_BUDGET - draining) / len(remaining)
    return weights


def measure_drain_path(score) -> dict:
    """Each format's score along the same walk from perfect balance to four punts.

    This is what actually separates the formats. Where the optimum SITS turned out to be the
    same for all three, so a scene claiming otherwise would be wrong; what differs is how much
    the detour is worth and how sharply the payoff falls away past the answer.
    """
    punted = np.linspace(0.0, 4.0, 161)
    values = np.array([score(weights_after_draining(point)) for point in punted])
    return {
        'punted': [round(float(point), 3) for point in punted],
        'score':  [round(float(value), 6) for value in values],
        'best_at': round(float(punted[int(np.argmax(values))]), 3),
    }


def main() -> None:
    measured = []
    for key, (label, score) in PAYOFFS.items():
        optimum, best = find_optimum(score)
        sorted_weights = np.sort(optimum)[::-1]
        punts = int(np.sum(sorted_weights < _PUNTED_BELOW))
        measured.append({
            'key':       key,
            'label':     label,
            'weights':   [round(float(weight), 4) for weight in sorted_weights],
            'punts':     punts,
            'score':     round(best, 4),
            'balanced':  round(score(np.full(CATEGORY_COUNT, EFFORT_BUDGET / CATEGORY_COUNT)), 4),
            'single_category_payoff': measure_single_category_payoff(score, optimum),
            'drain_path': measure_drain_path(score),
        })
        balanced = measured[-1]['balanced']
        print(f'{label:17} punts {punts}   optimum {best:.4f}   balanced {balanced:.4f}   '
              f'gain {100 * (best - balanced) / abs(balanced):.2f}%   '
              f'peak of drain path at {measured[-1]["drain_path"]["best_at"]}')
        print(f'{"":17} weights {np.round(sorted_weights, 3)}')

    data_path = _VISUALIZATIONS_DIR / 'data' / 'payoff_formats.json'
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps({
        'category_count':    CATEGORY_COUNT,
        'effort_budget':     EFFORT_BUDGET,
        'categories_needed': CATEGORIES_NEEDED,
        'formats':           measured,
    }), encoding='utf-8')
    print(f'\nWrote {data_path.relative_to(_VISUALIZATIONS_DIR.parent)}')


if __name__ == '__main__':
    main()
