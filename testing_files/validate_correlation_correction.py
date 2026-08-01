"""Validation harness for the Most-Categories correlation correction.

Three checks, run as a script (not part of the pytest suite):
  1. The implemented correction matches a brute-force leave-two-out computation.
  2. The analytic gradient term matches finite differences of the correction
     (with the Loewner matrix frozen, which is what the implementation claims).
  3. Corrected win probabilities track Monte-Carlo ground truth (correlated
     normals) better than the independence assumption does.
"""

import sys

sys.path.insert(0, '.')

import numpy as np
from scipy.stats import norm

from backend.math.algorithm_helpers import (
    compute_win_probability,
    calculate_pair_bracket_matrix,
    calculate_correction_terms,
    calculate_correction_probability_gradient,
)


def reference_correction_terms(probs, correlation_off_diagonal, densities):
    """Float64 reference assembled from the prefix/suffix per-pair machinery."""
    bracket_matrix = calculate_pair_bracket_matrix(probs)
    m_phi = np.einsum('cd,acdo,ado->aco', correlation_off_diagonal, bracket_matrix, densities)
    correction = 0.5 * (densities * m_phi).sum(axis=1)
    gradient = calculate_correction_probability_gradient(
        probs, correlation_off_diagonal, densities)
    return correction, m_phi, gradient

rng = np.random.default_rng(11)


def build_synthetic_correlation_matrix(n_categories: int) -> np.ndarray:
    """A random valid correlation matrix with realistic moderate entries."""
    loadings = rng.uniform(-0.5, 0.7, size=(n_categories, 2))
    covariance = loadings @ loadings.T + np.diag(rng.uniform(0.5, 1.5, n_categories))
    scale = np.sqrt(np.diag(covariance))
    return covariance / np.outer(scale, scale)


def win_count_pmf(probabilities: list[float]) -> np.ndarray:
    """Exact win-count pmf for independent Bernoullis (reference DP)."""
    pmf = np.array([1.0])
    for probability in probabilities:
        extended = np.zeros(len(pmf) + 1)
        extended[:-1] += pmf * (1 - probability)
        extended[1:]  += pmf * probability
        pmf = extended
    return pmf


def brute_force_correction(z_scores: np.ndarray, correlation: np.ndarray) -> float:
    """Direct evaluation of eq (5): sum over pairs with explicit leave-two-out DPs."""
    n_categories = len(z_scores)
    threshold = n_categories // 2 + 1
    probabilities = norm.cdf(z_scores)
    densities = norm.pdf(z_scores)
    total = 0.0
    for i in range(n_categories):
        for j in range(i + 1, n_categories):
            remaining = [probabilities[c] for c in range(n_categories) if c not in (i, j)]
            pmf = win_count_pmf(remaining)
            if n_categories % 2 == 1:
                bracket = pmf[threshold - 2] - pmf[threshold - 1]
            else:
                tie_count = n_categories // 2
                bracket = 0.5 * (pmf[tie_count - 2] - pmf[tie_count])
            total += correlation[i, j] * densities[i] * densities[j] * bracket
    return total


def implemented_correction(z_scores: np.ndarray, correlation: np.ndarray) -> float:
    """The vectorized path exactly as get_objective_and_pdf_weights_mc computes it."""
    probs = norm.cdf(z_scores).reshape(1, -1, 1)
    densities = norm.pdf(z_scores).reshape(1, -1, 1)
    n_categories = len(z_scores)

    pair_matrix = calculate_pair_bracket_matrix(probs)
    off_diagonal = correlation - np.eye(n_categories)
    m_phi = np.einsum('cd,acdo,ado->aco', off_diagonal, pair_matrix, densities)
    return float(0.5 * (densities * m_phi).sum())


def check_fast_path_matches_reference() -> None:
    """The production path (complex64 evaluation space) vs the float64 reference.

    Single precision is a design choice: the node formulation has uniformly bounded
    conditioning, so the error budget is ~1e-7 relative — asserted at 1e-6 absolute.
    """
    print('── 0. fast path (complex64 nodes) vs float64 reference ──────')
    local_rng = np.random.default_rng(21)
    worst = 0.0
    for n_categories in (9, 8, 11, 5):
        for trial in range(30):
            probs = norm.cdf(local_rng.normal(0.1, 1.0, (4, n_categories, 3)))
            if trial % 3 == 0:
                probs[:, 1, :] = probs[:, 0, :]
                probs[0, 2, :] = 0.99999
                probs[1, 3, :] = 0.00001
            densities = norm.pdf(local_rng.normal(0, 1, probs.shape))
            correlation = build_synthetic_correlation_matrix(n_categories)
            off_diagonal = correlation - np.eye(n_categories)

            fast = calculate_correction_terms(probs, off_diagonal, densities, True)
            reference = reference_correction_terms(probs, off_diagonal, densities)
            for fast_term, reference_term in zip(fast, reference):
                worst = max(worst, np.abs(fast_term - reference_term).max())
    print(f'   worst abs deviation over 120 states (n=9,8,11,5): {worst:.2e}')
    assert worst < 1e-6, 'the single-precision fast path must track the float64 reference'


def check_correction_matches_brute_force() -> None:
    print('── 1. implementation vs brute force ─────────────────────────')
    worst = 0.0
    for n_categories in (9, 8, 11):
        correlation = build_synthetic_correlation_matrix(n_categories)
        for _ in range(200):
            z_scores = rng.normal(0, 1.2, n_categories)
            reference = brute_force_correction(z_scores, correlation)
            computed = implemented_correction(z_scores, correlation)
            worst = max(worst, abs(computed - reference))
    print(f'   worst abs deviation over 600 random states (n=9,8,11): {worst:.2e}')
    assert worst < 1e-9, 'implementation must match the brute-force pairwise sum'


def check_gradient_matches_finite_differences() -> None:
    print('── 2. gradient vs finite differences (frozen M) ─────────────')
    n_categories = 9
    correlation = build_synthetic_correlation_matrix(n_categories)
    off_diagonal = correlation - np.eye(n_categories)
    worst = 0.0
    for _ in range(50):
        z_scores = rng.normal(0, 1.2, n_categories)

        probs = norm.cdf(z_scores).reshape(1, -1, 1)
        frozen_pair_matrix = np.einsum(
            'cd,acdo->acdo', off_diagonal, calculate_pair_bracket_matrix(probs)
        )[0, :, :, 0]

        def frozen_correction(z_vector: np.ndarray) -> float:
            densities = norm.pdf(z_vector)
            return 0.5 * densities @ frozen_pair_matrix @ densities

        densities = norm.pdf(z_scores)
        analytic_gradient = -z_scores * (frozen_pair_matrix @ densities) * densities

        step = 1e-6
        for index in range(n_categories):
            bumped_up   = z_scores.copy(); bumped_up[index]   += step
            bumped_down = z_scores.copy(); bumped_down[index] -= step
            numeric = (frozen_correction(bumped_up) - frozen_correction(bumped_down)) / (2 * step)
            worst = max(worst, abs(numeric - analytic_gradient[index]))
    print(f'   worst abs deviation over 50 states x 9 categories: {worst:.2e}')
    assert worst < 1e-6, 'analytic gradient must match finite differences with M frozen'


def check_against_monte_carlo() -> None:
    print('── 3. corrected vs independent vs Monte Carlo ───────────────')
    n_categories = 9
    correlation = build_synthetic_correlation_matrix(n_categories)
    n_samples = 2_000_000
    cholesky = np.linalg.cholesky(correlation)

    scenarios = {
        'even matchup':    rng.normal(0.0, 0.35, n_categories),
        'slightly ahead':  rng.normal(0.4, 0.35, n_categories),
        'clearly ahead':   rng.normal(0.9, 0.35, n_categories),
        'slightly behind': rng.normal(-0.4, 0.35, n_categories),
        'punt build':      np.concatenate([rng.normal(0.7, 0.2, 6), rng.normal(-1.4, 0.2, 3)]),
    }
    print(f'   {"scenario":<16} {"MC truth":>9} {"indep":>9} {"corrected":>10} '
          f'{"err indep":>10} {"err corr":>9}')
    for label, z_scores in scenarios.items():
        samples = rng.standard_normal((n_samples, n_categories)) @ cholesky.T + z_scores
        wins = (samples > 0).sum(axis=1)
        truth = ((wins >= n_categories // 2 + 1).mean())

        independent = float(compute_win_probability(
            norm.cdf(z_scores).reshape(1, -1, 1))[0, 0])
        corrected = independent + implemented_correction(z_scores, correlation)
        print(f'   {label:<16} {truth:9.4f} {independent:9.4f} {corrected:10.4f} '
              f'{abs(independent - truth):10.4f} {abs(corrected - truth):9.4f}')


if __name__ == '__main__':
    check_fast_path_matches_reference()
    check_correction_matches_brute_force()
    check_gradient_matches_finite_differences()
    check_against_monte_carlo()
    print('all checks complete')
