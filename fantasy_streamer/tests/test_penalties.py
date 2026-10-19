"""Unit tests for probabilistic drop penalties, VORP, snipe probability, and urgency."""

import math
import pytest

from fantasy_streamer.models.penalties import (
    DropPenaltyCalculator,
    calculate_drop_penalty,
    calculate_effective_survival_horizon,
    calculate_matchup_urgency,
    calculate_waiver_snipe_probability,
    calculate_weekly_vorp,
)


def test_weekly_vorp_zero_when_below_or_equal_replacement():
    """Verify that players at or below replacement baseline have exactly 0.0 VORP."""
    mu_rep = 22.0
    assert calculate_weekly_vorp(mu_p=20.0, mu_rep=mu_rep) == 0.0
    assert calculate_weekly_vorp(mu_p=22.0, mu_rep=mu_rep) == 0.0


def test_weekly_vorp_positive_scaling():
    """Verify linear VORP scaling with effective games = 3.0."""
    mu_rep = 20.0
    mu_p = 25.0
    expected_vorp = 3.0 * (25.0 - 20.0)  # 15.0
    assert pytest.approx(calculate_weekly_vorp(mu_p, mu_rep), rel=1e-5) == expected_vorp


def test_waiver_snipe_probability_inflection():
    """Verify logistic snipe probability equals 0.5 at the offset threshold (mu_rep + 2.5)."""
    mu_rep = 20.0
    # At mu_p = 22.5, exponent is -0.9 * (22.5 - 20.0 - 2.5) = 0 -> 1 / (1 + 1) = 0.5
    p_mid = calculate_waiver_snipe_probability(mu_p=22.5, mu_rep=mu_rep)
    assert pytest.approx(p_mid, abs=1e-5) == 0.50


def test_waiver_snipe_probability_asymptotes():
    """Verify sigmoid bounds between 0 and 1 with realistic asymptotic behavior."""
    mu_rep = 20.0
    p_low = calculate_waiver_snipe_probability(mu_p=10.0, mu_rep=mu_rep)
    p_high = calculate_waiver_snipe_probability(mu_p=40.0, mu_rep=mu_rep)

    assert 0.0 <= p_low < 0.01
    assert 0.99 < p_high <= 1.0


def test_effective_survival_horizon_geometric_series():
    """Verify geometric decay horizon for various remaining week counts."""
    gamma = 0.75
    # W_rem = 1 -> H_eff = 0.75
    h_1 = calculate_effective_survival_horizon(weeks_remaining=1, gamma=gamma)
    assert pytest.approx(h_1, rel=1e-5) == 0.75

    # W_rem = 2 -> 0.75 + 0.75^2 = 0.75 + 0.5625 = 1.3125
    h_2 = calculate_effective_survival_horizon(weeks_remaining=2, gamma=gamma)
    assert pytest.approx(h_2, rel=1e-5) == 1.3125

    # As W_rem -> large, sum converges to gamma / (1 - gamma) = 0.75 / 0.25 = 3.0
    h_large = calculate_effective_survival_horizon(weeks_remaining=50, gamma=gamma)
    assert pytest.approx(h_large, abs=1e-4) == 3.0

    # Non-positive weeks remaining
    assert calculate_effective_survival_horizon(weeks_remaining=0, gamma=gamma) == 0.0


def test_matchup_urgency_scalar_clamping_and_neutral():
    """Verify exponential scaling with delta_S and clamping at 3.0."""
    sigma_s = 65.0
    # At delta_S = 0, exp(0) = 1.0
    u_0 = calculate_matchup_urgency(delta_s=0.0, sigma_s=sigma_s)
    assert pytest.approx(u_0, rel=1e-5) == 1.0

    # At delta_S = 65.0, exp(65^2 / (2 * 65^2)) = exp(0.5) ~= 1.6487
    u_65 = calculate_matchup_urgency(delta_s=65.0, sigma_s=sigma_s)
    assert pytest.approx(u_65, rel=1e-4) == math.exp(0.5)

    # At extreme delta_S (e.g. 300), clamped at max 3.0
    u_extreme = calculate_matchup_urgency(delta_s=300.0, sigma_s=sigma_s, max_clamp=3.0)
    assert u_extreme == 3.0


def test_drop_penalty_calculator_integration():
    """Verify full c_p product calculation and calculator class interface."""
    calc = DropPenaltyCalculator(
        mu_rep=20.0,
        weeks_remaining=8,
        delta_s=40.0,
        effective_games=3.0,
        gamma=0.75,
        sigma_s=65.0,
        max_urgency_clamp=3.0,
    )

    # Sub-replacement player should have zero penalty
    res_scrub = calc.evaluate_player(1, "Deep Reserve", 18.0)
    assert res_scrub.weekly_vorp == 0.0
    assert res_scrub.c_p == 0.0

    # Quality hold player
    res_star = calc.evaluate_player(2, "Core Starter", 32.0)
    assert res_star.weekly_vorp == 3.0 * (32.0 - 20.0)  # 36.0
    assert res_star.p_snipe > 0.95
    assert res_star.h_eff > 2.5
    assert res_star.c_p > 50.0  # High opportunity cost prevents frivolous drop
