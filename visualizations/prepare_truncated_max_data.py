"""Precompute everything the truncated-max scenes draw, so a render samples nothing.

Run once (or whenever the covariance file or the pool size changes):

    python visualizations/prepare_truncated_max_data.py

Writes `data/truncated_max.json`. The three scenes read only that file: the pools, the
selection statistics, the density curves and the scalar core all come out of here with a
fixed seed, so re-rendering a scene gives back the same video instead of a new sample.

Everything that the shipped model computes is computed BY the shipped model
(`backend.math.truncated_max_pick_model`) rather than reimplemented here -- the point of
the animation is to explain the code that runs, so any drift between the two would be a
lie told in slow motion. The one thing this file adds is the ABSOLUTE expected pick
E[z | selected]; the shipped entry point returns the difference between that and the
generic team's pick, because only the difference is a tilt. The difference of two absolute
picks computed here is checked against the shipped function below.

Two real categories carry the whole story, because a player has to be a POINT on screen for
scene A to work at all. Free Throw % and Blocks are the pair chosen: their real correlation
is the closest to zero in the matrix, which is what makes the score/value correlation rho
sweep a wide range (0.70 to 1.00) as the weights tilt across the quadrant. A strongly
correlated pair such as Rebounds/Assists pins rho above 0.90 everywhere and there is
nothing left to watch. They are also the fantasy tension everyone already knows: the
centre who swats everything and cannot shoot a free throw.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Import the app itself rather than reimplementing its numbers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.math.truncated_max_pick_model import (                       # noqa: E402
    _evaluate_expected_max_score,
    _score_value_geometry,
    _skew_normal_survival,
    compute_expected_pick_tilts,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VISUALIZATIONS_DIR = Path(__file__).resolve().parent
_CORRELATION_PATH = _PROJECT_ROOT / 'coefficient_exploration_output' / 'correlations_2024-25.csv'

CATEGORY_PAIR = ('Free Throw %', 'Blocks')
PICK_POOL_SIZE = 25           # the app default (backend/api/schemas.py)
RANDOM_SEED = 90125           # fixed so every render of every scene shows the same draws

# The weight sweep. Zero degrees is all weight on the first category, ninety on the second,
# forty-five is the generic team (w parallel to v). The sweep stays inside the quadrant:
# negative weights would push rho lower still, but "I want FEWER blocks" is not a thing a
# fantasy team wants, and the story should stay inside the space the app searches.
ANGLE_GRID_DEGREES = np.linspace(0.0, 90.0, 31)
# Angles the scenes stop on. The generic team is deliberately NOT one of them: at forty-five
# degrees rho is clamped at one and the picture is degenerate, which is exactly the kink the
# shipped model smooths over.
SHOWCASE_ANGLES_DEGREES = (8.0, 45.0, 82.0)

SELECTION_POOL_COUNT = 4000   # pools averaged over to locate the selection cloud's centre
SELECTION_CLOUD_SIZE = 260    # of those, how many selections are drawn as dots

SCORE_GRID = np.linspace(-3.6, 3.6, 181)          # the s axis, in units of sigma_s
SCALAR_GRID = np.linspace(-3.6, 5.2, 221)         # the standardised t axis for scene C
# Pool sizes scene C animates through. Three to a hundred and twenty brackets the app's
# twenty-five on both sides, and is logarithmic because what the eye reads is the ORDER of
# magnitude of M -- the 1/M tail moves like log M, not like M.
SCALAR_POOL_SIZES = np.round(np.geomspace(3.0, 120.0, 45), 4)

_SQRT_TWO_PI = np.sqrt(2.0 * np.pi)


def load_category_covariance(
    category_pair
):
    """The 2x2 slice of the real category correlation matrix, in standardised units.

    Category values reach the model as z-scores, so the correlation matrix IS the covariance
    -- no rescaling, and every category has unit variance by construction.
    """
    correlations = pd.read_csv(_CORRELATION_PATH, index_col=0)
    missing = [name for name in category_pair if name not in correlations.index]
    if missing:
        raise KeyError(f'{_CORRELATION_PATH.name} has no row for: {", ".join(missing)}')
    covariance = correlations.loc[list(category_pair), list(category_pair)].to_numpy(dtype=float)
    return covariance


def weights_at_angle(
    angle_degrees
):
    """A unit weight vector at `angle_degrees` off the first category's axis."""
    angle_radians = np.radians(np.asarray(angle_degrees, dtype=float))
    return np.stack([np.cos(angle_radians), np.sin(angle_radians)], axis=-1)


def draw_pool_until_survivors(
    covariance
    , value_direction
    , survivor_target
    , random_generator
):
    """One pick's worth of draws: keep drawing until `survivor_target` clear the value bar.

    Returns every draw in the order it was made together with a survivor flag, because the
    scene wants to SHOW the ones that are already gone rather than start from a pool that
    has mysteriously been pre-filtered.
    """
    cholesky_factor = np.linalg.cholesky(covariance)
    drawn_points = []
    survivors_found = 0
    while survivors_found < survivor_target:
        point = random_generator.standard_normal(covariance.shape[0]) @ cholesky_factor.T
        drawn_points.append(point)
        if point @ value_direction <= 0.0:
            survivors_found += 1
    drawn_points = np.array(drawn_points)
    return drawn_points, drawn_points @ value_direction <= 0.0


def draw_truncated_pool_bank(
    covariance
    , value_direction
    , pool_count
    , pick_pool_size
    , random_generator
):
    """(pool_count, pick_pool_size, categories) draws, every one of them below the value bar.

    ONE bank is shared by every weight angle on purpose. The pools do not depend on the
    weights -- who is available is decided before I choose what I want -- so re-drawing them
    per angle would let sampling noise masquerade as the effect the scene is about.
    """
    cholesky_factor = np.linalg.cholesky(covariance)
    needed = pool_count * pick_pool_size
    kept = []
    kept_count = 0
    while kept_count < needed:
        # Draw generously and keep the survivors: roughly half of any normal sample sits
        # below a bar through the origin, so twice what is left over is one pass in practice.
        batch = random_generator.standard_normal((2 * (needed - kept_count) + 64,
                                                  covariance.shape[0])) @ cholesky_factor.T
        survivors = batch[batch @ value_direction <= 0.0]
        kept.append(survivors)
        kept_count += len(survivors)
    return np.concatenate(kept)[:needed].reshape(pool_count, pick_pool_size, -1)


def compute_absolute_expected_pick(
    category_weights
    , covariance
    , value_direction
    , pick_pool_size
):
    """E[z | value bar, score is the max of M] in absolute terms, (P, C).

    This is grad_w [sigma_s * e(rho)], the quantity the shipped module differentiates and
    then differences against the generic team's pick. Assembled here out of the shipped
    pieces so the scene's arrows and the app's arithmetic cannot drift apart.
    """
    weights = np.asarray(category_weights, dtype=float).reshape(-1, covariance.shape[0])
    sigma_w, sigma_v, _, sigma_s, sigma_u, rho, _ = _score_value_geometry(
        weights, covariance, value_direction)
    expected_max, expected_max_slope, _ = _evaluate_expected_max_score(rho, pick_pool_size)
    score_coefficient = (expected_max - rho * expected_max_slope) / sigma_s
    value_coefficient = expected_max_slope / sigma_u
    return (sigma_w * score_coefficient[:, None]
            + sigma_v[None, :] * value_coefficient[:, None])


def compute_score_value_geometry_by_angle(
    covariance
    , value_direction
    , angle_grid_degrees
):
    """sigma_s and rho at every angle on the sweep -- the two numbers scene B is about."""
    weights = weights_at_angle(angle_grid_degrees)
    _, _, _, sigma_s, sigma_u, rho, _ = _score_value_geometry(
        weights, covariance, value_direction)
    return sigma_s, float(sigma_u), rho


def standard_normal_density(
    standardised_value
):
    return np.exp(-0.5 * standardised_value ** 2) / _SQRT_TWO_PI


def skew_normal_density(
    standardised_value
    , shape
):
    """The survivor's standardised score density: 2 phi(t) Phi(shape t).

    This is P(u <= 0 | s) / P(u <= 0) times the marginal: the conditional probability that a
    player with score t was still available, which falls away as t rises because score and
    value are correlated. Phi comes out of the shipped survival function at shape zero (where
    the skew-normal is just the normal), so both curves in scene B come from one piece of code.
    """
    normal_cdf = 1.0 - _skew_normal_survival(shape * standardised_value, 0.0)
    return 2.0 * standard_normal_density(standardised_value) * normal_cdf


def skew_normal_cumulative(
    standardised_value
    , shape
):
    return 1.0 - _skew_normal_survival(standardised_value, shape)


def solve_tail_quantile(
    shape
    , pick_pool_size
):
    """The t where the survival function is 1/M -- the ledge the Gumbel gap is measured from.

    The same bisection the shipped model runs, on the same monotone survival function; it is
    private in there because nothing outside needed it until this animation did.
    """
    lower, upper = -4.0, 12.0
    for _ in range(60):
        middle = 0.5 * (lower + upper)
        if _skew_normal_survival(middle, shape) * pick_pool_size > 1.0:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


def build_plane_act(
    covariance
    , value_direction
    , random_generator
):
    """Scene A: one visible pool, and the selection statistics behind it."""
    drawn_points, is_survivor = draw_pool_until_survivors(
        covariance, value_direction, PICK_POOL_SIZE, random_generator)

    pool_bank = draw_truncated_pool_bank(
        covariance, value_direction, SELECTION_POOL_COUNT, PICK_POOL_SIZE, random_generator)

    selection_means = []
    selection_clouds = []
    for angle_degrees in ANGLE_GRID_DEGREES:
        scores = pool_bank @ weights_at_angle(angle_degrees)
        selected = pool_bank[np.arange(SELECTION_POOL_COUNT), scores.argmax(axis=1)]
        selection_means.append(selected.mean(axis=0))
        selection_clouds.append(selected[:SELECTION_CLOUD_SIZE])

    model_picks = compute_absolute_expected_pick(
        weights_at_angle(ANGLE_GRID_DEGREES), covariance, value_direction, PICK_POOL_SIZE)

    # The claim the scene rests on, checked rather than asserted: differencing two absolute
    # picks reproduces the shipped tilt exactly, so the arrows drawn here are the app's.
    shipped_tilts = compute_expected_pick_tilts(
        weights_at_angle(ANGLE_GRID_DEGREES), covariance,
        np.asarray(value_direction).reshape(-1, 1), PICK_POOL_SIZE).reshape(-1, 2)
    generic_pick = compute_absolute_expected_pick(
        np.asarray(value_direction).reshape(1, -1), covariance, value_direction, PICK_POOL_SIZE)
    tilt_discrepancy = np.abs((model_picks - generic_pick) - shipped_tilts).max()
    if tilt_discrepancy > 1e-12:
        raise AssertionError(
            f'absolute picks differ from the shipped tilt by {tilt_discrepancy:.3e}; the '
            f'animation would be describing something the app does not do')

    monte_carlo_discrepancy = np.abs(np.array(selection_means) - model_picks).max()
    print(f'Scene A: {len(drawn_points)} draws to find {PICK_POOL_SIZE} survivors; '
          f'model against {SELECTION_POOL_COUNT} simulated pools, '
          f'worst category gap {monte_carlo_discrepancy:.4f} standard deviations')

    return {
        'drawn_points':           drawn_points.round(5).tolist(),
        'is_survivor':            is_survivor.tolist(),
        'selection_mean_by_angle': np.array(selection_means).round(5).tolist(),
        'model_pick_by_angle':    model_picks.round(5).tolist(),
        'selection_cloud_by_angle': np.array(selection_clouds).round(4).tolist(),
        'selection_pool_count':   SELECTION_POOL_COUNT,
    }


def build_reduction_act(
    covariance
    , value_direction
):
    """Scene B: the (s, u) geometry and the two marginals, at every angle on the sweep."""
    sigma_s, sigma_u, correlation = compute_score_value_geometry_by_angle(
        covariance, value_direction, ANGLE_GRID_DEGREES)
    shape = -correlation / np.sqrt(1.0 - correlation ** 2)

    # Both marginals live on the same axis in units of sigma_s, so the truncation reads as a
    # change of SHAPE rather than a change of scale: before conditioning the score is normal,
    # after it the low-value half is all that is left and the high scores are thinned.
    before_conditioning = standard_normal_density(SCORE_GRID)
    after_conditioning = np.array([
        skew_normal_density(SCORE_GRID, one_shape) for one_shape in shape
    ])

    # Every conditioned curve has to still be a probability density on the plotted window.
    total_mass = np.trapz(after_conditioning, SCORE_GRID, axis=1)
    if np.abs(total_mass - 1.0).max() > 5e-3:
        raise AssertionError(
            f'a conditioned score density integrates to {total_mass.min():.4f}-'
            f'{total_mass.max():.4f} over the plotted window, not to one')

    print(f'Scene B: conditioned score mean runs '
          f'{np.trapz(after_conditioning * SCORE_GRID, SCORE_GRID, axis=1).min():+.3f} to '
          f'{np.trapz(after_conditioning * SCORE_GRID, SCORE_GRID, axis=1).max():+.3f}')
    print(f'Scene B: rho runs {correlation.min():.3f} to {correlation.max():.3f}, '
          f'sigma_s runs {sigma_s.min():.3f} to {sigma_s.max():.3f}, sigma_u {sigma_u:.3f}')

    return {
        'score_grid':                     SCORE_GRID.round(4).tolist(),
        'score_density_before_conditioning': before_conditioning.round(6).tolist(),
        'score_density_after_conditioning':  after_conditioning.round(6).tolist(),
        'value_density':                  standard_normal_density(SCORE_GRID).round(6).tolist(),
    }


def build_scalar_core_act(
    correlation_by_angle
):
    """Scene C: the skew-normal, the max-of-M density, and where the Gumbel gap reaches.

    The reference correlation is the sweep's lowest -- the most tilted team the scene shows.
    At rho near one the skew-normal is a half-normal and the picture has no shape to it; the
    interesting regime, and the one a punting team actually sits in, is rho well below one.
    """
    reference_correlation = float(correlation_by_angle.min())
    shape = -reference_correlation / np.sqrt(1.0 - reference_correlation ** 2)

    density = skew_normal_density(SCALAR_GRID, shape)
    cumulative = skew_normal_cumulative(SCALAR_GRID, shape)

    # The two are assembled from different pieces of the shipped module -- one from its
    # survival function at shape zero, the other from the same function at the real shape --
    # and scene C multiplies them together as M g G^(M-1). If they ever disagree the scene
    # draws a curve that is not a density at all, and a mirrored skew is not obvious by eye,
    # so the derivative is checked rather than trusted.
    derivative_discrepancy = np.abs(np.gradient(cumulative, SCALAR_GRID) - density).max()
    if derivative_discrepancy > 1e-3:
        raise AssertionError(
            f'the skew-normal density is not the derivative of its own cumulative '
            f'(worst gap {derivative_discrepancy:.3e})')

    tail_quantiles = []
    expected_maxima = []
    exact_means = []
    for pool_size in SCALAR_POOL_SIZES:
        tail_quantile = solve_tail_quantile(shape, pool_size)
        expected_max, _, _ = _evaluate_expected_max_score(
            np.array([reference_correlation]), pool_size)
        # What the model is approximating, integrated directly off the max-of-M density the
        # scene draws. Both go on screen: the quantile-plus-Gumbel construction is an
        # approximation, and an animation that hid the gap would be selling it.
        max_density = pool_size * density * cumulative ** (pool_size - 1.0)
        total_mass = np.trapz(max_density, SCALAR_GRID)
        if abs(total_mass - 1.0) > 5e-3:
            raise AssertionError(
                f'the max-of-{pool_size:.0f} density integrates to {total_mass:.4f} over the '
                f'plotted window, not to one')
        tail_quantiles.append(tail_quantile)
        expected_maxima.append(float(expected_max[0]))
        exact_means.append(float(np.trapz(max_density * SCALAR_GRID, SCALAR_GRID)))

    at_app_default = int(np.argmin(np.abs(SCALAR_POOL_SIZES - PICK_POOL_SIZE)))
    print(f'Scene C: rho {reference_correlation:.3f}, shape {shape:.3f}; '
          f'at M={SCALAR_POOL_SIZES[at_app_default]:.0f} the ledge is '
          f't_q={tail_quantiles[at_app_default]:.3f} and e={expected_maxima[at_app_default]:.3f} '
          f'(Gumbel gap {expected_maxima[at_app_default] - tail_quantiles[at_app_default]:.3f}, '
          f'exact mean {exact_means[at_app_default]:.3f})')

    return {
        'reference_correlation': reference_correlation,
        'shape':                 float(shape),
        'scalar_grid':           SCALAR_GRID.round(4).tolist(),
        'skew_normal_density':   density.round(6).tolist(),
        'skew_normal_cumulative': cumulative.round(8).tolist(),
        'pool_sizes':            SCALAR_POOL_SIZES.tolist(),
        'tail_quantile_by_pool_size': np.round(tail_quantiles, 5).tolist(),
        'expected_max_by_pool_size': np.round(expected_maxima, 5).tolist(),
        'exact_mean_by_pool_size': np.round(exact_means, 5).tolist(),
    }


def main() -> None:
    covariance = load_category_covariance(CATEGORY_PAIR)
    # The generic team values both categories equally, normalised so that a weight vector
    # anywhere on the sweep has the same length as the value direction and only its DIRECTION
    # distinguishes it -- which is the whole content of "w enters only through selection".
    value_direction = np.array([1.0, 1.0]) / np.sqrt(2.0)
    random_generator = np.random.default_rng(RANDOM_SEED)

    print(f'Categories {CATEGORY_PAIR[0]} / {CATEGORY_PAIR[1]}, '
          f'real correlation {covariance[0, 1]:+.4f}')

    sigma_s, sigma_u, correlation = compute_score_value_geometry_by_angle(
        covariance, value_direction, ANGLE_GRID_DEGREES)

    prepared = {
        'categories':            list(CATEGORY_PAIR),
        'category_correlation':  float(covariance[0, 1]),
        'covariance':            covariance.round(8).tolist(),
        'value_direction':       value_direction.round(8).tolist(),
        'pick_pool_size':        PICK_POOL_SIZE,
        'random_seed':           RANDOM_SEED,
        'angle_grid_degrees':    ANGLE_GRID_DEGREES.round(4).tolist(),
        'showcase_angles_degrees': list(SHOWCASE_ANGLES_DEGREES),
        'score_standard_deviation_by_angle': sigma_s.round(6).tolist(),
        'correlation_by_angle':  correlation.round(6).tolist(),
        'value_standard_deviation': sigma_u,
        'plane':                 build_plane_act(covariance, value_direction, random_generator),
        'reduction':             build_reduction_act(covariance, value_direction),
        'scalar_core':           build_scalar_core_act(correlation),
    }

    output_path = _VISUALIZATIONS_DIR / 'data' / 'truncated_max.json'
    output_path.write_text(json.dumps(prepared), encoding='utf-8')
    print(f'Wrote {output_path.relative_to(_PROJECT_ROOT)} '
          f'({output_path.stat().st_size / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
