"""Verification of the truncated-max pick model against its own ground truths.

Three layers, each checking the one above it: a seeded Monte Carlo simulation of the
actual selection process checks the exact order-statistic integral; the exact integral
checks the closed form (quantile + Gumbel correction + Mills ratio); and w-space finite
differences of the closed form check the assembled Jacobian. Plus the structural
properties the surrounding algorithm relies on: exact zero at neutral weights, and the
emergent supply curve (tilting away from a category monotonically buys tilt at a
monotonically increasing price in value).
"""

import numpy as np
import pytest
from scipy.stats import norm

from backend.math.truncated_max_pick_model import (
    compute_expected_pick_tilt_jacobian,
    compute_expected_pick_tilts,
)

COVARIANCE = np.array([
    [1.0, 0.3, 0.2, -0.2],
    [0.3, 1.0, 0.4, 0.1],
    [0.2, 0.4, 1.0, 0.0],
    [-0.2, 0.1, 0.0, 1.0],
])
VALUE_DIRECTION = (np.ones(4) / 4).reshape(-1, 1)


def build_tilted_weights(tilted_category_weight: float) -> np.ndarray:
    weights = np.array([1.0, 1.0, 1.0, tilted_category_weight])
    return (weights / weights.sum()).reshape(1, -1)


def compute_exact_pick(weights_row, covariance, value_direction, pick_pool_size):
    """The exact order-statistic integral this module's closed form approximates,
    on a dense grid — the reference the closed form is held to."""
    w = weights_row.reshape(-1)
    v = value_direction.reshape(-1)
    ss = w @ covariance @ w
    uu = v @ covariance @ v
    c = w @ covariance @ v
    sigma_s = np.sqrt(ss)
    sigma_value_given_score = np.sqrt(uu - c * c / ss)

    grid = np.linspace(-8 * sigma_s, 8 * sigma_s, 4001)
    conditional_mean = (c / ss) * grid
    survivor_density = (
        norm.pdf(grid, 0, sigma_s)
        * np.clip(norm.cdf(-conditional_mean / sigma_value_given_score), 1e-300, None)
        / 0.5
    )
    cumulative = np.cumsum(survivor_density)
    cumulative /= cumulative[-1]
    max_density = pick_pool_size * survivor_density * cumulative ** (pick_pool_size - 1)
    max_density /= np.trapz(max_density, grid)

    expected_score = np.trapz(grid * max_density, grid)
    standardized_bar = -conditional_mean / sigma_value_given_score
    conditional_value = conditional_mean - sigma_value_given_score * (
        norm.pdf(standardized_bar) / np.clip(norm.cdf(standardized_bar), 1e-300, None)
    )
    expected_value = np.trapz(conditional_value * max_density, grid)

    lift = np.column_stack([covariance @ w, covariance @ v]) @ np.linalg.inv(
        np.array([[ss, c], [c, uu]])
    )
    return lift @ np.array([expected_score, expected_value])


def compute_exact_tilt(weights_row, covariance, value_direction, pick_pool_size):
    """Exact pick minus the exact generic pick: the quantity the module returns."""
    my_pick = compute_exact_pick(weights_row, covariance, value_direction, pick_pool_size)
    # The generic pick degenerates to a one-dimensional truncated-normal max; a
    # near-parallel weight vector stands in for it (the exact integral cannot take
    # w exactly parallel to v — the 2x2 Gram matrix is singular there).
    near_generic = value_direction.reshape(-1) + 1e-6 * np.array([1.0, -1.0, 1.0, -1.0])
    generic_pick = compute_exact_pick(
        near_generic, covariance, value_direction, pick_pool_size)
    return my_pick - generic_pick


@pytest.mark.parametrize('pick_pool_size', [10, 25, 100])
@pytest.mark.parametrize('tilted_category_weight', [0.7, 0.4, 0.1])
def test_closed_form_tracks_exact_integral(pick_pool_size, tilted_category_weight):
    weights = build_tilted_weights(tilted_category_weight)
    closed = compute_expected_pick_tilts(
        weights, COVARIANCE, VALUE_DIRECTION, pick_pool_size).reshape(-1)
    exact = compute_exact_tilt(weights, COVARIANCE, VALUE_DIRECTION, pick_pool_size)
    # The Gumbel-corrected closed form runs 2-6% relative error against the exact
    # integral across this grid (see the derivation note); 10% is the regression alarm.
    assert np.max(np.abs(closed - exact)) < 0.10 * np.max(np.abs(exact))


def test_exact_integral_matches_monte_carlo():
    pick_pool_size = 25
    weights = build_tilted_weights(0.4)
    rng = np.random.default_rng(20260903)
    draws = rng.multivariate_normal(np.zeros(4), COVARIANCE, size=4_000_000)
    survivors = draws[draws @ VALUE_DIRECTION.reshape(-1) <= 0.0]
    n_pools = survivors.shape[0] // pick_pool_size
    pools = survivors[: n_pools * pick_pool_size].reshape(n_pools, pick_pool_size, 4)
    scores = pools @ weights.reshape(-1)
    selected = pools[np.arange(n_pools), scores.argmax(axis=1)]

    simulated = selected.mean(axis=0)
    exact = compute_exact_pick(weights, COVARIANCE, VALUE_DIRECTION, pick_pool_size)
    assert np.max(np.abs(simulated - exact)) < 0.02


def test_tilt_is_exactly_zero_at_neutral_weights():
    tilts = compute_expected_pick_tilts(
        VALUE_DIRECTION.reshape(1, -1), COVARIANCE, VALUE_DIRECTION, 25)
    assert np.all(tilts == 0.0)


def test_jacobian_matches_finite_differences_away_from_neutral():
    pick_pool_size = 25
    weights = np.vstack([
        build_tilted_weights(0.7),
        build_tilted_weights(0.4),
        np.array([[0.4, 0.15, 0.25, 0.2]]),
    ])
    jacobian = compute_expected_pick_tilt_jacobian(
        weights, COVARIANCE, VALUE_DIRECTION, pick_pool_size)

    step = 1e-6
    for row in range(weights.shape[0]):
        for k in range(4):
            bumped_up = weights.copy()
            bumped_down = weights.copy()
            bumped_up[row, k] += step
            bumped_down[row, k] -= step
            difference = (
                compute_expected_pick_tilts(
                    bumped_up, COVARIANCE, VALUE_DIRECTION, pick_pool_size)
                - compute_expected_pick_tilts(
                    bumped_down, COVARIANCE, VALUE_DIRECTION, pick_pool_size)
            ).reshape(weights.shape[0], 4)[row] / (2 * step)
            # The Jacobian is fully closed-form; the tolerance here is set by this test's
            # own finite differences (~1e-7), not by the Jacobian.
            assert np.max(np.abs(jacobian[row, :, k] - difference)) < 1e-5, (
                f'row {row}, weight {k}')


def test_jacobian_is_symmetric_and_annihilates_weights():
    """The pick is the exact gradient of the scalar sigma_s * e(rho), so its Jacobian is
    a Hessian: symmetric, and (by degree-1 homogeneity of the value function) J w = 0.
    This is the scoring-rule form — the algebra behind weights matching gradients at
    optima — preserved exactly through the approximation."""
    weights = np.vstack([
        build_tilted_weights(0.7),
        build_tilted_weights(0.1),
        np.array([[0.4, 0.15, 0.25, 0.2]]),
        VALUE_DIRECTION.reshape(1, -1),          # the clamped neutral point too
    ])
    jacobian = compute_expected_pick_tilt_jacobian(
        weights, COVARIANCE, VALUE_DIRECTION, 25)
    for row in range(weights.shape[0]):
        asymmetry = np.max(np.abs(jacobian[row] - jacobian[row].T))
        radial = np.max(np.abs(jacobian[row] @ weights[row]))
        scale = np.max(np.abs(jacobian[row]))
        assert asymmetry < 1e-12 * scale, f'row {row}: asymmetry {asymmetry}'
        assert radial < 1e-12 * scale, f'row {row}: J w = {radial}'


def test_supply_curve_is_monotone():
    """Less weight on a category means monotonically more tilt away from it, and a
    monotonically lower expected pick value — the emergent cost with no gamma."""
    pick_pool_size = 25
    value_vector = VALUE_DIRECTION.reshape(-1)
    tilted_weights = [1.0, 0.7, 0.4, 0.1, 0.0]
    category_tilts = []
    pick_values = []
    for tilted_category_weight in tilted_weights:
        tilt = compute_expected_pick_tilts(
            build_tilted_weights(tilted_category_weight),
            COVARIANCE, VALUE_DIRECTION, pick_pool_size).reshape(-1)
        category_tilts.append(tilt[3])
        pick_values.append(float(value_vector @ tilt))
    assert all(np.diff(category_tilts) < 0.0)
    assert all(np.diff(pick_values) < 0.0)


def test_field_truncation_creates_the_anti_crowding_supply_effect():
    """When the field's drafting direction abandons a category, players rich in it
    survive into the window: leaning into the abandoned category must get cheaper
    (higher tilt harvest at lower value cost) than under a neutral field, and the
    default (no field direction) must be bit-identical to passing v explicitly."""
    pick_pool_size = 25
    value_vector = VALUE_DIRECTION.reshape(-1)
    field = np.array([1.0, 1.0, 1.0, 0.3])          # the crowd punts category 4
    field = (field / field.sum()).reshape(-1, 1)
    lean_in = np.array([[1.0, 1.0, 1.0, 1.5]]) / 4.5

    # Tilts are measured against the generic reference picking from the SAME pool, so the
    # shared surplus of the abandoned category cancels; the crowding shows up in the VALUE
    # ECONOMICS of each direction instead.
    join_punt = np.array([[1.0, 1.0, 1.0, 0.4]]) / 3.4

    def value_cost(weights, field_direction):
        tilt = compute_expected_pick_tilts(
            weights, COVARIANCE, VALUE_DIRECTION, pick_pool_size,
            field_direction=field_direction).reshape(-1)
        return float(value_vector @ tilt)

    # Leaning into the abandoned category gets cheaper (the crowd left its players
    # on the board), and joining the crowded punt gets more expensive (you forfeit
    # surplus the generic drafter collects).
    assert value_cost(lean_in, field) > value_cost(lean_in, None)
    assert value_cost(join_punt, field) < value_cost(join_punt, None)

    explicit_v = compute_expected_pick_tilts(
        lean_in, COVARIANCE, VALUE_DIRECTION, pick_pool_size,
        field_direction=VALUE_DIRECTION)
    default = compute_expected_pick_tilts(
        lean_in, COVARIANCE, VALUE_DIRECTION, pick_pool_size)
    assert np.array_equal(explicit_v, default)

    jacobian = compute_expected_pick_tilt_jacobian(
        lean_in, COVARIANCE, VALUE_DIRECTION, pick_pool_size, field_direction=field)[0]
    assert np.max(np.abs(jacobian - jacobian.T)) < 1e-12 * np.max(np.abs(jacobian))
    assert np.max(np.abs(jacobian @ lean_in.reshape(-1))) < 1e-12 * np.max(np.abs(jacobian))


def test_neutral_neighbourhood_is_continuous():
    """Crossing the clamp seam must not jump: tiny tilts produce tiny outputs."""
    deviation = np.array([1.0, -1.0, 1.0, -1.0])
    for scale in (1e-8, 1e-6, 1e-4, 1e-3):
        weights = (VALUE_DIRECTION.reshape(-1) + scale * deviation).reshape(1, -1)
        tilt = compute_expected_pick_tilts(weights, COVARIANCE, VALUE_DIRECTION, 25)
        assert np.max(np.abs(tilt)) < 0.5, f'scale {scale}'
