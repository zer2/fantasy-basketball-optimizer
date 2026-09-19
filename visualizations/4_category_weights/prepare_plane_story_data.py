"""Build the one-plane telling of the truncated-max model.

Everything the scene shows lives in two real categories, so the model can be watched rather
than derived: a pool of players scattered in the plane, a value bar that has already taken
everyone above it, and a score line that slides across to find the best survivor left.

Writes `data/plane_story.json` plus two image layers into `assets/plane_story/`:

    pick_density_neutral.png      where the pick lands under equal weights
    pick_density_alternative.png  and where it lands under the tilted ones

plus a dimmed copy of each. Only where the picks END UP is drawn; the pool's own density is the
setup rather than the answer, and the frame is worth more to the answer.

The images are written here rather than drawn in the scene because a smooth field is an image,
not ten thousand mobjects: Manim would spend the whole render on it and still band.

    python visualizations/prepare_plane_story_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CATEGORIES = ['Free Throw %', 'Blocks']
POOL_SIZE = 25                  # M, the app's default: how many survivors a pick chooses among
EXPERIMENT_POOLS = 6            # independent drafts shown one at a time in the scene
TAKEN_PLAYERS_SHOWN = 14        # players above the bar, drawn only so the bar means something
DENSITY_SIMULATIONS = 60000     # pools simulated to map where the pick lands
GRID_RESOLUTION = 320
PLANE_HALF_EXTENT = 3.0         # standard deviations drawn either side of the origin
RANDOM_SEED = 5150

# The two weightings the scene compares, written unnormalised as a reader would think of them:
# equal care for both categories, against three times the care on one. Both run side by side from
# the experiments onward, which is why each gets a colour and keeps it -- yellow is the neutral
# one everywhere in the scene, blue is the alternative everywhere.
NEUTRAL_WEIGHTS = np.array([1.0, 1.0])
ALTERNATIVE_WEIGHTS = np.array([1.5, 0.5])
NEUTRAL_COLOUR = (255, 214, 80)
ALTERNATIVE_COLOUR = (110, 170, 255)

_VISUALIZATIONS_DIR = Path(__file__).resolve().parent.parent
_IMAGE_DIR = _VISUALIZATIONS_DIR / 'assets' / 'plane_story'
_CORRELATION_PATH = (_VISUALIZATIONS_DIR.parent / 'coefficient_exploration_output'
                     / 'correlations_2024-25.csv')


def load_category_covariance() -> np.ndarray:
    """The real 2x2 correlation between the two categories, from the shipped exploration output."""
    correlations = pd.read_csv(_CORRELATION_PATH, index_col=0)
    missing = [category for category in CATEGORIES if category not in correlations.index]
    if missing:
        raise SystemExit(f'{missing} absent from {_CORRELATION_PATH.name}; the plane needs two '
                         f'categories that season actually measured.')
    return correlations.loc[CATEGORIES, CATEGORIES].to_numpy()


def draw_survivors(
    generator: np.random.Generator
    , covariance: np.ndarray
    , value_direction: np.ndarray
    , count: int
) -> np.ndarray:
    """`count` players from the pool a pick actually chooses among.

    The pool is conditioned on value <= 0: everyone better than the bar has already been drafted,
    which is the truncation the whole model is built around. Rejection sampling is the honest way
    to draw it -- about half of any draw survives, so it is also the cheap way.
    """
    survivors = []
    while len(survivors) < count:
        batch = generator.multivariate_normal([0.0, 0.0], covariance, size=4 * count)
        survivors.extend(batch[batch @ value_direction <= 0.0])
    return np.array(survivors[:count])


def simulate_pick_positions(
    generator: np.random.Generator
    , covariance: np.ndarray
    , value_direction: np.ndarray
    , weights: np.ndarray
    , pool_count: int
) -> np.ndarray:
    """Where the pick lands, over many independent drafts, for one set of weights.

    This is x(w)'s whole meaning made concrete: run the draft situation again and again, keep
    the player chosen each time, and the cloud of those winners IS the distribution the model
    reports the mean of.
    """
    picks = np.empty((pool_count, 2))
    for index in range(pool_count):
        pool = draw_survivors(generator, covariance, value_direction, POOL_SIZE)
        picks[index] = pool[np.argmax(pool @ weights)]
    return picks


def pick_density_grid(picks: np.ndarray) -> np.ndarray:
    """Where the picks landed, as a smoothed field on the same grid as the pool's density."""
    edges = np.linspace(-PLANE_HALF_EXTENT, PLANE_HALF_EXTENT, GRID_RESOLUTION + 1)
    counts, _, _ = np.histogram2d(picks[:, 0], picks[:, 1], bins=[edges, edges])

    # A little smoothing, because the scene shows a field and a raw histogram of sixty thousand
    # points over a hundred thousand cells is speckle. Separable box blurs, repeated, approach a
    # Gaussian closely enough and need no extra dependency.
    field = counts.T
    for _ in range(3):
        field = _blur_once(field)
    return field


def _blur_once(field: np.ndarray, radius: int = 4) -> np.ndarray:
    """One separable box blur pass over a 2D field."""
    window = np.ones(2 * radius + 1) / (2 * radius + 1)
    blurred = np.apply_along_axis(lambda row: np.convolve(row, window, mode='same'), 1, field)
    return np.apply_along_axis(lambda col: np.convolve(col, window, mode='same'), 0, blurred)


def write_density_image(
    field: np.ndarray
    , colour: tuple[int, int, int]
    , path: Path
    , alpha_scale: float = 1.0
) -> None:
    """A density field as a single-hue RGBA image, transparent where there is nothing.

    Alpha carries the density and the hue stays flat, so two of these can be laid over one
    another and still be told apart -- which is the whole point of the last act.

    `alpha_scale` writes a dimmed copy. It exists because Manim's ImageMobject.set_opacity does
    not dim an image, it REPLACES every pixel's alpha with one value -- a density map put through
    it comes out a solid rectangle. Dimming has to be baked in here, where the per-pixel alpha
    still means something.
    """
    peak = field.max()
    if peak <= 0:
        raise SystemExit(f'{path.name} would be blank; the field has no mass in it.')

    # Square-rooted so the thin outskirts of the distribution stay visible: on a linear ramp
    # everything but the core reads as black, and the shape of the tail is what is being compared.
    normalised = np.sqrt(field / peak)
    image = np.zeros((*field.shape, 4), dtype=np.uint8)
    image[..., 0], image[..., 1], image[..., 2] = colour
    image[..., 3] = (normalised * alpha_scale * 255).astype(np.uint8)

    # Image rows run top-down; the grid's first row is the BOTTOM of the plane.
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image[::-1], mode='RGBA').save(path)


def main() -> None:
    covariance = load_category_covariance()
    value_direction = np.array([1.0, 1.0]) / np.sqrt(2.0)
    generator = np.random.default_rng(RANDOM_SEED)
    print(f'Categories {CATEGORIES[0]} / {CATEGORIES[1]}, '
          f'real correlation {covariance[0, 1]:+.4f}')

    neutral = NEUTRAL_WEIGHTS / np.linalg.norm(NEUTRAL_WEIGHTS)
    alternative = ALTERNATIVE_WEIGHTS / np.linalg.norm(ALTERNATIVE_WEIGHTS)

    # The drafts the scene walks through one at a time. Same weights every time -- what changes
    # between experiments is only which players turned up, which is the point.
    experiments = []
    for _ in range(EXPERIMENT_POOLS):
        pool = draw_survivors(generator, covariance, value_direction, POOL_SIZE)
        # A handful of players from above the bar go with each pool. They are not part of the
        # problem -- they were drafted before this pick came round -- but without them on screen
        # the bar is a line through nothing, and the truncation the model is built on is the one
        # thing a viewer has to see.
        taken = []
        while len(taken) < TAKEN_PLAYERS_SHOWN:
            batch = generator.multivariate_normal([0.0, 0.0], covariance, size=32)
            taken.extend(batch[batch @ value_direction > 0.0])
        # Both weightings pick from the SAME pool, which is the comparison the scene is for:
        # where the two disagree, the strategies genuinely want different players.
        experiments.append({
            'players': pool.round(4).tolist(),
            'taken': np.array(taken[:TAKEN_PLAYERS_SHOWN]).round(4).tolist(),
            'neutral_scores': (pool @ neutral).round(4).tolist(),
            'alternative_scores': (pool @ alternative).round(4).tolist(),
            'neutral_pick': int(np.argmax(pool @ neutral)),
            'alternative_pick': int(np.argmax(pool @ alternative)),
        })
    disagreements = sum(experiment['neutral_pick'] != experiment['alternative_pick']
                        for experiment in experiments)
    print(f'{EXPERIMENT_POOLS} experiment pools; the two weightings choose differently in '
          f'{disagreements} of them')

    _IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for label, weights, colour in (('neutral', neutral, NEUTRAL_COLOUR),
                                   ('alternative', alternative, ALTERNATIVE_COLOUR)):
        picks = simulate_pick_positions(
            generator, covariance, value_direction, weights, DENSITY_SIMULATIONS)
        field = pick_density_grid(picks)
        write_density_image(field, colour, _IMAGE_DIR / f'pick_density_{label}.png')
        # A dimmed copy, for the moment the second distribution arrives and the first has to
        # stay visible underneath it without competing.
        write_density_image(field, colour, _IMAGE_DIR / f'pick_density_{label}_faded.png',
                            alpha_scale=0.30)
        summary[label] = {
            'weights': weights.round(4).tolist(),
            'pick_mean': picks.mean(axis=0).round(4).tolist(),
        }
        print(f'{label:8} weights {weights.round(3).tolist()} -> '
              f'expected pick {picks.mean(axis=0).round(3).tolist()}')

    tilt = (np.array(summary['alternative']['pick_mean'])
            - np.array(summary['neutral']['pick_mean']))
    print(f'tilt (alternative minus neutral) {tilt.round(3).tolist()}')

    data_path = _VISUALIZATIONS_DIR / 'data' / 'plane_story.json'
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps({
        'categories':        CATEGORIES,
        'covariance':        covariance.round(6).tolist(),
        'value_direction':   value_direction.round(6).tolist(),
        'pool_size':         POOL_SIZE,
        'plane_half_extent': PLANE_HALF_EXTENT,
        'random_seed':       RANDOM_SEED,
        'density_simulations': DENSITY_SIMULATIONS,
        'experiments':       experiments,
        'weight_settings':   summary,
        'tilt':              tilt.round(4).tolist(),
    }), encoding='utf-8')
    print(f'Wrote {data_path.relative_to(_VISUALIZATIONS_DIR.parent)}')


if __name__ == '__main__':
    main()
