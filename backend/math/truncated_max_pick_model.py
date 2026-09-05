"""The truncated-max model of a future pick, and its Jacobian with respect to the weights.

A future pick is modeled as the best of a window of surviving players: a player is a
category vector z ~ N(0, Sigma); the pool at a pick is `pick_pool_size` independent draws
conditioned on v'z <= 0 (everyone above the value bar is already taken); the team selects
the survivor with the largest score w'z. The model's output is the expected category
profile of that selection RELATIVE to the pick a generic team (w = v) would make from the
same pool — the same "pure tilt" contract as the simplified-form x_mu it replaces, which
also vanishes at neutral weights. Tilting the score direction away from value therefore
buys category tilt at an emergent, accelerating price in value; there is no priced
weight-aberration term (no gamma) and no generic-level term (no omega): the level cancels
in the diff, and the cost comes out of the selection geometry itself.

The construction is built to preserve the scoring-rule form EXACTLY, not just
approximately. The exact model's expected pick is the gradient of its value function
W(w) = E[max score] (the envelope theorem), which makes the Jacobian a symmetric PSD
Hessian annihilating w — the algebra behind "optimal weights are proportional to the
objective's category gradients". Scale-invariance collapses that value function to

    W(w) = sigma_s * e(rho),      rho = w'Sigma*v / (sigma_s * sigma_u),

with e a UNIVARIATE function (the skew-normal shape is alpha = -rho / sqrt(1 - rho^2),
so the standardized max location depends on rho alone). We approximate e by the
1 - 1/M quantile plus the Gumbel mean correction — and then take the pick to be the
EXACT gradient of that approximate scalar:

    x = grad_w [sigma_s * e(rho)]
      = ((e - rho * e') / sigma_s) * Sigma w  +  (e' / sigma_u) * Sigma v.

Approximating the scalar and differentiating exactly keeps the Hessian symmetric with
J w = 0 by construction, so weights match gradients at optima of the approximate model
too (verified to solver precision); it is also measurably MORE accurate per category
than lifting E[score] and E[value] separately, because the envelope identity
E[u | selected] = sigma_u * (rho e + (1 - rho^2) e') is a better estimate of the value
cost than a second independent approximation. Accuracy against the exact
order-statistic integral and against Monte Carlo is pinned in
testing_files/test_truncated_max_pick_model.py.

The Jacobian is the exact Hessian of sigma_s * e(rho): all it needs beyond the forward
pass is e''(rho), and e, e', e'' are all closed-form — the quantile comes from one
bisection of the monotone survival function, its derivatives from the implicit function
theorem, where Owen's T supplies the elementary shape-partial
dT/dalpha = exp(-t^2 (1 + alpha^2) / 2) / (2 pi (1 + alpha^2)). No numerical
differentiation anywhere.

At w exactly parallel to v the true model has a kink (score and value become the same
variable, and the tilt response grows like the norm of the deviation), so rho is clamped
just inside +/-1, with derivative zero through the clamp. That smooths the kink in a tiny
neighbourhood of neutral, keeps every formula finite, and — because the generic reference
is evaluated through the same clamped machinery — preserves x(v) = 0 exactly.
"""

import numpy as np
from scipy.special import ndtr, owens_t

EULER_GAMMA = 0.5772156649015329
_SQRT_TWO_PI = np.sqrt(2.0 * np.pi)

# rho^2 is kept this far inside 1: the smoothing radius of the neutral-point kink. Within
# it the tilt response is linearized; outside it the model is untouched.
_MAX_SQUARED_CORRELATION = 1.0 - 1e-4

_BISECTION_ITERATIONS = 50

_DENSITY_FLOOR = 1e-300


def _standard_normal_pdf(t):
    return np.exp(-0.5 * t * t) / _SQRT_TWO_PI


def _skew_normal_survival(t, shape):
    """Survival function of the standardized skew-normal: 1 - Phi(t) + 2*OwensT(t, shape)."""
    return (1.0 - ndtr(t)) + 2.0 * owens_t(t, shape)


def _evaluate_expected_max_score(
    rho
    , pick_pool_size
):
    """The univariate core: e(rho) and its first two derivatives, all closed-form.

    For a standardized pool (unit score and value variances, correlation rho), the
    selected survivor's expected score is e(rho); real units are sigma_s * e. e is the
    1 - 1/M quantile of the skew-normal survivor score plus the Gumbel mean correction.
    Derivatives come from the implicit function theorem on M * SF(t; alpha) = 1: the
    quantile's alpha-sensitivity is Owen's T's elementary shape-partial, and everything
    else is phi/Phi algebra. Returns (e, de_drho, dde_drho2).
    """
    s2 = 1.0 - rho * rho                              # >= 1 - _MAX_SQUARED_CORRELATION
    d_s2 = -2.0 * rho
    shape = -rho / np.sqrt(s2)
    d_shape = -s2 ** -1.5
    dd_shape = -3.0 * rho * s2 ** -2.5

    # The 1 - 1/M quantile, by bisection on the monotone survival function.
    lower = np.full_like(rho, -4.0)
    upper = np.full_like(rho, 12.0)
    for _ in range(_BISECTION_ITERATIONS):
        middle = 0.5 * (lower + upper)
        above = _skew_normal_survival(middle, shape) * pick_pool_size > 1.0
        lower = np.where(above, middle, lower)
        upper = np.where(above, upper, middle)
    t = 0.5 * (lower + upper)

    pdf_t = _standard_normal_pdf(t)
    pdf_shaped = _standard_normal_pdf(shape * t)
    cdf_shaped = ndtr(shape * t)
    density = np.maximum(2.0 * pdf_t * cdf_shaped, _DENSITY_FLOOR)
    cross = 2.0 * pdf_t * pdf_shaped                  # the recurring phi(t)*phi(alpha*t) pair

    # First derivatives. dSF/dt = -density; dSF/dshape is Owen's T's elementary partial
    # (1 + shape^2 = 1 / s2), so the quantile moves as d_t = (dSF/dshape / density) * d_shape.
    survival_shape_partial = (s2 / np.pi) * np.exp(-0.5 * t * t / s2)
    d_t = (survival_shape_partial / density) * d_shape

    density_t = -t * density + shape * cross          # d(density)/dt
    density_shape = t * cross                         # d(density)/dshape
    d_density = density_t * d_t + density_shape * d_shape

    gumbel = EULER_GAMMA / (pick_pool_size * density)
    e = t + gumbel
    d_e = d_t - (gumbel / density) * d_density

    # Second derivatives, same chain one layer down.
    one_plus_shape2 = 1.0 / s2
    d_survival_shape_partial = survival_shape_partial * (
        d_s2 / s2 - t * d_t / s2 + t * t * d_s2 / (2.0 * s2 * s2))
    dd_t = (
        (d_survival_shape_partial * d_shape + survival_shape_partial * dd_shape) / density
        - (survival_shape_partial * d_shape) * d_density / (density * density)
    )
    # Partials of density_t and density_shape (mixed partials agree, a useful identity:
    # d(density_t)/dshape == d(density_shape)/dt == cross * (1 - t^2 (1 + shape^2))).
    mixed = cross * (1.0 - t * t * one_plus_shape2)
    density_tt = -density - t * density_t - shape * t * one_plus_shape2 * cross
    density_shape_shape = -shape * t ** 3 * cross
    d_density_t = density_tt * d_t + mixed * d_shape
    d_density_shape = mixed * d_t + density_shape_shape * d_shape
    dd_density = (
        d_density_t * d_t + density_t * dd_t
        + d_density_shape * d_shape + density_shape * dd_shape
    )
    dd_e = dd_t - (EULER_GAMMA / pick_pool_size) * (
        dd_density / (density * density)
        - 2.0 * d_density * d_density / (density ** 3)
    )
    return e, d_e, dd_e


def _score_value_geometry(
    weights
    , covariance
    , value_direction
):
    """The quadratic forms and the clamped correlation shared by the pick and its Jacobian."""
    v = np.asarray(value_direction, dtype=float).reshape(-1)
    sigma_w = weights @ covariance                    # rows are Sigma * w
    sigma_v = covariance @ v
    ss = np.einsum('pc,pc -> p', sigma_w, weights)
    uu = float(v @ sigma_v)
    if np.any(ss <= 0.0) or uu <= 0.0:
        raise ValueError('score and value variances must be positive')
    c = sigma_w @ v
    sigma_s = np.sqrt(ss)
    sigma_u = np.sqrt(uu)
    rho_limit = np.sqrt(_MAX_SQUARED_CORRELATION)
    rho_raw = c / (sigma_s * sigma_u)
    rho = np.clip(rho_raw, -rho_limit, rho_limit)
    clamp_active = np.abs(rho_raw) > rho_limit
    return sigma_w, sigma_v, ss, sigma_s, sigma_u, rho, clamp_active


def compute_expected_pick_tilts(
    category_weights
    , covariance
    , value_direction
    , pick_pool_size
    , field_direction=None
):
    """Expected category profile of one future pick, relative to a generic team's pick.

    category_weights: (P, C) batch of weight vectors; covariance: (C, C); value_direction:
    (C, 1) the neutral weights v; pick_pool_size: the window M. Returns (P, C, 1), zero
    exactly at category_weights == v (the generic reference runs through the same clamped
    machinery, so the diff cancels identically there).

    field_direction (C, 1), default value_direction: the direction the FIELD drafts by,
    which is what actually determines who is gone from the pool. Truncating on it instead
    of generic value is the anti-crowding supply effect: categories the field tilts away
    from survive past the bar into the window, so tilting toward an abandoned category
    gets cheap and joining a crowded punt forfeits surplus — emergent, priced in effect
    space, moderated upstream by how confidently the field's behavior is modelled. The
    generic reference picks from the SAME contorted pool, so x(v) stays exactly zero.
    """
    weights = np.asarray(category_weights, dtype=float)
    v = np.asarray(value_direction, dtype=float).reshape(-1)
    # The generic team rides along as one extra batch row, so it goes through
    # bit-for-bit the same arithmetic as every candidate row — which is what makes the
    # tilt EXACTLY zero (not merely tiny) for a weight row equal to v.
    weights_with_reference = np.vstack([weights, v])
    sigma_w, sigma_v, ss, sigma_s, sigma_u, rho, _ = _score_value_geometry(
        weights_with_reference, covariance,
        value_direction if field_direction is None else field_direction)

    e, d_e, _ = _evaluate_expected_max_score(rho, pick_pool_size)
    score_coefficient = (e - rho * d_e) / sigma_s     # p~: the Sigma*w loading of grad W
    value_coefficient = d_e / sigma_u                 # q~: the Sigma*v loading of grad W
    picks = (
        sigma_w * score_coefficient[:, None]
        + sigma_v[None, :] * value_coefficient[:, None]
    )
    return (picks[:-1] - picks[-1][None, :]).reshape(-1, v.shape[0], 1)


def compute_expected_pick_tilt_jacobian(
    category_weights
    , covariance
    , value_direction
    , pick_pool_size
    , field_direction=None
):
    """d(expected pick tilt)_i / d(weight)_k, shape (P, C, C) — the exact Hessian of the
    approximate value function sigma_s * e(rho), so it is symmetric and annihilates w by
    construction (the scoring-rule form, preserved exactly through the approximation).

    Chain rule: the coefficients p~ = (e - rho e')/sigma_s and q~ = e'/sigma_u depend on w
    only through ss = w'Sigma*w and c = w'Sigma*t (t the truncation direction: the field's
    drafting direction, defaulting to generic value), and every scalar y obeys
    dy/dw = (dy/dss) * 2*Sigma*w + (dy/dc) * Sigma*t. The field direction is constant with
    respect to my weights, so the Hessian structure is untouched by the anti-crowding pool.
    """
    weights = np.asarray(category_weights, dtype=float)
    v = np.asarray(value_direction, dtype=float).reshape(-1)
    n_categories = v.shape[0]
    sigma_w, sigma_v, ss, sigma_s, sigma_u, rho, clamp_active = _score_value_geometry(
        weights, covariance,
        value_direction if field_direction is None else field_direction)

    e, d_e, dd_e = _evaluate_expected_max_score(rho, pick_pool_size)
    score_coefficient = (e - rho * d_e) / sigma_s
    value_coefficient = d_e / sigma_u

    # rho = c / (sigma_s * sigma_u): the two scalar routes into the coefficients — both
    # zero through an active clamp (the smoothing is flat there).
    d_rho_d_ss = np.where(clamp_active, 0.0, -rho / (2.0 * ss))
    d_rho_d_c = np.where(clamp_active, 0.0, 1.0 / (sigma_s * sigma_u))

    d_score_coefficient_d_rho = -rho * dd_e / sigma_s
    d_p_d_ss = (
        -score_coefficient / (2.0 * ss)
        + d_score_coefficient_d_rho * d_rho_d_ss
    )
    d_p_d_c = d_score_coefficient_d_rho * d_rho_d_c
    d_q_d_ss = (dd_e / sigma_u) * d_rho_d_ss
    d_q_d_c = (dd_e / sigma_u) * d_rho_d_c

    grad_p = 2.0 * sigma_w * d_p_d_ss[:, None] + sigma_v[None, :] * d_p_d_c[:, None]
    grad_q = 2.0 * sigma_w * d_q_d_ss[:, None] + sigma_v[None, :] * d_q_d_c[:, None]

    jacobian = (
        score_coefficient[:, None, None] * covariance[None, :, :]
        + sigma_w[:, :, None] * grad_p[:, None, :]
        + sigma_v[None, :, None] * grad_q[:, None, :]
    )
    return jacobian.reshape(-1, n_categories, n_categories)
