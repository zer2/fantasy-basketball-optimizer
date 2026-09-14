"""Draw the objective over a slice of weight space, and climb it from several starting points.

Gradient descent finds local optima, and in this objective every punt structure is one -- so
where you start decides which one you reach. `algorithm_agents.py` answers that with a seed
menu: one gentle punt per category, plus the balanced build, plus each candidate's own converged
build from earlier passes, all climbed, best one kept. A comment there records the payoff:
challenger seeds win 24% of bootstrap solves.

The slice. Nine identical categories and nine units of effort, scored as the sum of Phi(effort
minus parity) -- the punting scene's model. Two of its optima are "abandon the first three
categories" and "abandon the last three"; both score 4.6247 where perfect balance scores 4.5000.
The plane through those two points and the balanced build is the smallest picture containing
more than one peak, and it is the honest one: the multiple peaks are not a feature of a toy, they
are what having to CHOOSE which categories to abandon looks like.

Coordinates are how far along you are toward each of the two optima, so (0, 0) is perfect
balance, (1, 0) is the first optimum and (0, 1) is the second. Outside the region where every
weight is still non-negative there is nothing to draw, and the image leaves it black.

Writes a baked PNG of the surface -- never a dimmed one. Manim's `ImageMobject.set_opacity()`
replaces per-pixel alpha wholesale, so any dimming has to be in the file.

    python visualizations/prepare_seed_landscape_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import norm

_VISUALIZATIONS_DIR = Path(__file__).resolve().parent

CATEGORY_COUNT = 9
EFFORT_BUDGET = 9.0
PUNTS_PER_OPTIMUM = 3
# Wide enough to hold all THREE optima this plane passes through: abandoning the first
# three categories, abandoning the last three, and -- at (-1, -1) -- abandoning the middle
# three, which is where a seed placed between the other two actually ends up.
SLICE_LOW, SLICE_HIGH = -1.45, 1.45
IMAGE_RESOLUTION = 420
CONTRAST_WINDOW = 0.22        # how far below the best score the colour ramp reaches

ASCENT_STEPS = 90
ASCENT_STEP_SIZE = 0.035
ASCENT_STOP_BELOW = 1e-8


def score(weights: np.ndarray) -> float:
    """Expected categories won against an opponent contesting every category at parity."""
    return float(np.sum(norm.cdf(np.asarray(weights) - 1.0)))


def punt_optimum(punted: range) -> np.ndarray:
    weights = np.full(CATEGORY_COUNT, EFFORT_BUDGET / (CATEGORY_COUNT - PUNTS_PER_OPTIMUM))
    weights[list(punted)] = 0.0
    return weights


BALANCED = np.full(CATEGORY_COUNT, EFFORT_BUDGET / CATEGORY_COUNT)
FIRST_OPTIMUM = punt_optimum(range(0, PUNTS_PER_OPTIMUM))
SECOND_OPTIMUM = punt_optimum(range(CATEGORY_COUNT - PUNTS_PER_OPTIMUM, CATEGORY_COUNT))


def weights_at(along_first: float, along_second: float) -> np.ndarray:
    return (BALANCED
            + along_first * (FIRST_OPTIMUM - BALANCED)
            + along_second * (SECOND_OPTIMUM - BALANCED))


def is_feasible(weights: np.ndarray) -> bool:
    return bool((weights >= -1e-9).all())


def score_at(along_first: float, along_second: float) -> float:
    weights = weights_at(along_first, along_second)
    return score(weights) if is_feasible(weights) else float('nan')


def build_surface() -> np.ndarray:
    axis = np.linspace(SLICE_LOW, SLICE_HIGH, IMAGE_RESOLUTION)
    return np.array([[score_at(first, second) for first in axis] for second in axis])


def paint_surface(surface: np.ndarray, path: Path) -> None:
    """Dark where the objective is poor, bright where it is good, black where there is nothing.

    A two-stop ramp rather than a rainbow: the scene is about WHERE the high ground is, and a
    ramp that only ever gets brighter keeps "higher" and "brighter" the same statement.
    """
    finite = np.isfinite(surface)
    # Contrast is spent on the top of the range only. The feasible region is a triangle whose
    # three CORNERS are the three optima, and over the whole region the objective varies by
    # about a third of a category -- so a ramp stretched across everything renders the interior
    # as one bright wash and hides the peaks it exists to show.
    high = np.nanmax(surface)
    low = high - CONTRAST_WINDOW
    height = np.zeros_like(surface)
    height[finite] = np.clip((surface[finite] - low) / (high - low), 0.0, 1.0) ** 2

    dark = np.array([10, 18, 38])
    bright = np.array([248, 232, 92])
    pixels = (dark + height[..., None] * (bright - dark)).astype(np.uint8)
    pixels[~finite] = 0
    # Row zero of an image is the TOP, and row zero of the surface is the LOW end of the axis.
    Image.fromarray(pixels[::-1], mode='RGB').save(path)


def climb(start: tuple[float, float]) -> dict:
    """Gradient ascent from one seed, staying inside the feasible region.

    Plain steepest ascent with a fixed step, which is what makes the point: an optimiser that
    only ever goes uphill from where it was put cannot cross the valley between two peaks, so
    the seed decides the answer.
    """
    position = np.array(start, dtype=float)
    path = [position.copy()]
    for _ in range(ASCENT_STEPS):
        here = score_at(*position)
        gradient = np.zeros(2)
        for axis in range(2):
            step = np.zeros(2)
            step[axis] = 1e-4
            ahead, behind = score_at(*(position + step)), score_at(*(position - step))
            if not (np.isfinite(ahead) and np.isfinite(behind)):
                return _finish(path, position)
            gradient[axis] = (ahead - behind) / 2e-4

        magnitude = float(np.linalg.norm(gradient))
        if magnitude < 1e-9:
            break
        proposal = position + ASCENT_STEP_SIZE * gradient / magnitude
        # Backtrack rather than step outside the region or downhill: the picture is of an
        # optimiser that stops when it can no longer improve, not one that wanders.
        for _ in range(6):
            value = score_at(*proposal)
            if np.isfinite(value) and value > here + ASCENT_STOP_BELOW:
                break
            proposal = position + (proposal - position) / 2.0
        else:
            break
        position = proposal
        path.append(position.copy())
    return _finish(path, position)


def _finish(path: list[np.ndarray], position: np.ndarray) -> dict:
    return {
        'path':  [[round(float(point[0]), 4), round(float(point[1]), 4)] for point in path],
        'final': [round(float(position[0]), 4), round(float(position[1]), 4)],
        'score': round(score_at(*position), 5),
    }


# The seed menu, in this slice. "Gentle punt" is a short step toward one optimum, which is what
# the shipped menu's one-gentle-punt-per-category seeds are; the warm start is a build already
# most of the way up a peak, which is what a previous pass hands over.
SEEDS = [
    ('cold',   'balanced',        (0.0,  0.0)),
    ('menu',   'gentle punt A',   (0.35, 0.0)),
    ('menu',   'gentle punt B',   (0.0,  0.35)),
    ('menu',   'gentle both',     (0.3,  0.3)),
    ('warm',   'last pass',       (0.85, 0.08)),
]


def main() -> None:
    surface = build_surface()
    image_path = _VISUALIZATIONS_DIR / 'assets' / 'seed_landscape' / 'objective_slice.png'
    image_path.parent.mkdir(parents=True, exist_ok=True)
    paint_surface(surface, image_path)
    print(f'Wrote {image_path.relative_to(_VISUALIZATIONS_DIR.parent)}  '
          f'({IMAGE_RESOLUTION} x {IMAGE_RESOLUTION})')

    climbs = []
    for kind, label, start in SEEDS:
        result = climb(start)
        result.update({'kind': kind, 'label': label, 'start': list(start)})
        climbs.append(result)
        print(f'{label:16} {kind:5} start {start} -> {result["final"]}  '
              f'score {result["score"]:.5f}  ({len(result["path"])} steps)')

    best = max(climb_result['score'] for climb_result in climbs)
    cold = next(c for c in climbs if c['kind'] == 'cold')
    print(f'\nbest seed reaches {best:.5f}; the cold start reaches {cold["score"]:.5f}')

    data_path = _VISUALIZATIONS_DIR / 'data' / 'seed_landscape.json'
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps({
        'slice_low':        SLICE_LOW,
        'slice_high':       SLICE_HIGH,
        'balanced_score':   round(score(BALANCED), 5),
        'optimum_score':    round(score(FIRST_OPTIMUM), 5),
        'image':            f'assets/seed_landscape/{image_path.name}',
        'climbs':           climbs,
    }), encoding='utf-8')
    print(f'Wrote {data_path.relative_to(_VISUALIZATIONS_DIR.parent)}')


if __name__ == '__main__':
    main()
