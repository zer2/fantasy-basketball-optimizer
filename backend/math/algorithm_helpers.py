"""
Backend-only copy of src/math/algorithm_helpers.py.

Changes vs original:
- `import streamlit as st` removed.
- @st.cache_data() replaced with @functools.lru_cache on get_win_grid / get_tie_grid.
The original src/ file is untouched.
"""

import functools
import pandas as pd
from scipy.stats import norm
import numpy as np
from itertools import combinations

def auction_value_adjuster(raw_values_unselected: pd.Series
                            , n_remaining_players: int
                            , remaining_cash: int
                            , noise: float) -> pd.Series:
    dollar_values = dollar_scale_adjustment(raw_values_unselected,
                                            remaining_cash,
                                            n_remaining_players)
    return savor_calculation(dollar_values, noise)


def dollar_scale_adjustment(raw_values_unselected: pd.Series
                             , remaining_cash: int
                             , n_remaining_players: int) -> pd.Series:
    raw_values_unselected = raw_values_unselected.sort_values(ascending=False)
    replacement_value = raw_values_unselected.iloc[n_remaining_players]
    value_above_replacement = np.clip(raw_values_unselected - replacement_value, 0, None)
    remaining_value = value_above_replacement.iloc[0:n_remaining_players].sum()
    dollar_per_value = remaining_cash / remaining_value
    return value_above_replacement * dollar_per_value


def savor_calculation(dollar_value: pd.Series, noise: float) -> pd.Series:
    if noise == 0:
        return dollar_value
    probability_of_non_streaming = norm.cdf(dollar_value / noise)
    adjustment_factor = (noise) / (2 * np.pi) ** 0.5 * (1 - np.exp((-dollar_value ** 2) / (2 * noise ** 2)))
    adjusted_value = dollar_value * probability_of_non_streaming - adjustment_factor
    return adjusted_value * dollar_value.sum() / adjusted_value.sum()


def combinatorial_calculation(c: np.ndarray
                               , c_comp: np.ndarray
                               , data=1
                               , level: int = 0
                               , n_false: int = 0):
    if n_false > c.shape[1] / 2:
        return 0
    elif level < c.shape[1]:
        return (combinatorial_calculation(c, c_comp, data * c[:, level, :], level + 1, n_false) +
                combinatorial_calculation(c, c_comp, data * c_comp[:, level, :], level + 1, n_false + 1))
    elif n_false == c.shape[1] / 2:
        return data / 2
    else:
        return data


@functools.lru_cache(maxsize=None)
def get_win_grid(n_categories: int) -> np.ndarray:
    which = np.array([list(combinations(range(n_categories), int(n_categories / 2) + 1))])
    grid = np.zeros((which.shape[1], n_categories), dtype='bool')
    grid[np.arange(which.shape[1])[None].T, which] = True
    return np.expand_dims(grid, axis=2)


@functools.lru_cache(maxsize=None)
def get_tie_grid(n_categories: int) -> np.ndarray:
    which = np.array([list(combinations(range(n_categories), int(n_categories / 2)))])
    grid = np.zeros((which.shape[1], n_categories), dtype='bool')
    grid[np.arange(which.shape[1])[None].T, which] = True
    return np.expand_dims(grid, axis=2)


def compute_win_probability(probs: np.ndarray) -> np.ndarray:
    """Compute P(winning the matchup) for each player vs each opponent via DP.

    Uses the same win-count polynomial DP as combinatorial_calculation, but
    expressed as a single forward pass rather than recursive enumeration.

    Args:
        probs: Win probability per category, shape (n_players, n_categories, n_opponents).

    Returns:
        shape (n_players, n_opponents): probability of winning the matchup.
    """
    n_players, n_categories, n_opponents = probs.shape
    k_to_win = n_categories // 2 + 1

    # dp[:, k, :] = P(win exactly k of the categories processed so far)
    dp = np.ones((n_players, 1, n_opponents))
    for i in range(n_categories):
        p_i     = probs[:, i, :][:, np.newaxis, :]
        updated = np.zeros((n_players, i + 2, n_opponents))
        updated[:, :i + 1, :] += dp * (1 - p_i)
        updated[:, 1:, :]     += dp * p_i
        dp = updated

    result = dp[:, k_to_win:, :].sum(axis=1)
    if n_categories % 2 == 0:
        result += dp[:, n_categories // 2, :] / 2
    return result


def _build_win_count_prefix_suffix(probs: np.ndarray):
    """Build prefix and suffix DP tables for win-count distributions.

    Each prefix[i] has shape (n_players, i+1, n_opponents) where entry [p, k, o]
    is the probability of winning exactly k of the first i categories.
    Suffix tables mirror this from the right.
    This is the same polynomial-multiplication DP that combinatorial_calculation
    uses recursively, expressed here as explicit tables so we can do leave-one-out
    convolutions for every category in one pass.
    """
    n_players, n_categories, n_opponents = probs.shape

    prefix = [None] * (n_categories + 1)
    prefix[0] = np.ones((n_players, 1, n_opponents))
    for i in range(n_categories):
        p_i = probs[:, i, :][:, np.newaxis, :]   # (n_players, 1, n_opponents)
        previous = prefix[i]                       # (n_players, i+1, n_opponents)
        updated = np.zeros((n_players, i + 2, n_opponents))
        updated[:, :i + 1, :] += previous * (1 - p_i)
        updated[:, 1:, :]     += previous * p_i
        prefix[i + 1] = updated

    suffix = [None] * (n_categories + 1)
    suffix[n_categories] = np.ones((n_players, 1, n_opponents))
    for i in range(n_categories - 1, -1, -1):
        p_i = probs[:, i, :][:, np.newaxis, :]
        following = suffix[i + 1]                  # (n_players, n_categories-i, n_opponents)
        n_following = n_categories - i
        updated = np.zeros((n_players, n_following + 1, n_opponents))
        updated[:, :n_following, :] += following * (1 - p_i)
        updated[:, 1:, :]           += following * p_i
        suffix[i] = updated

    return prefix, suffix


def _bracket_targets(n_categories_total: int) -> tuple[int, int, float]:
    """Win-count targets and scale for the correlation-correction bracket.

    The bracket is the mixed second difference of the matchup objective with respect
    to a pair of category win probabilities. With the majority threshold
    v = n//2 + 1 it reduces to a difference of two point masses:

        odd n:   P(= v-2) - P(= v-1),           scale 1
        even n:  (P(= t-2) - P(= t)) / 2,       t = n//2 (ties count half)

    Targets are indexed on the win count of the REMAINING categories, so they depend
    only on the original total n, never on how many categories were left out.
    """
    if n_categories_total % 2 == 1:
        return n_categories_total // 2 - 1, n_categories_total // 2, 1.0
    return n_categories_total // 2 - 2, n_categories_total // 2, 0.5


def _bracket_target_weights(n_categories_total: int) -> tuple[tuple[int, float], ...]:
    """The bracket as (target, weight) pairs on the remaining categories' win count."""
    lower_target, upper_target, scale = _bracket_targets(n_categories_total)
    return ((lower_target, scale), (upper_target, -scale))


def _bracket_derivative_target_weights(n_categories_total: int) -> tuple[tuple[int, float], ...]:
    """(target, weight) pairs for ∂(bracket)/∂p of one more removed category.

    Differentiating each mass with respect to a remaining category's win
    probability shifts it into a first difference: ∂P(W = t)/∂p = P(W' = t-1) - P(W' = t)
    with W' excluding that category too. Applied to the bracket's (target, weight)
    pairs this yields a second-difference stencil.
    """
    derivative: dict[int, float] = {}
    for target, weight in _bracket_target_weights(n_categories_total):
        derivative[target - 1] = derivative.get(target - 1, 0.0) + weight
        derivative[target]     = derivative.get(target, 0.0) - weight
    return tuple(sorted(derivative.items()))


def _leave_one_out_mass_combination(probs: np.ndarray
                                     , target_weights: tuple[tuple[int, float], ...]) -> np.ndarray:
    """sum_t weight_t * P(W_-c = target_t) for every left-out category c.

    Built from the same prefix/suffix DP tables as calculate_tipping_points.
    Shapes: probs (n_players, n_columns, n_opponents) -> same shape out.
    """
    n_players, n_columns, n_opponents = probs.shape
    prefix, suffix = _build_win_count_prefix_suffix(probs)

    def leave_one_out_mass(column_index: int, target: int) -> np.ndarray:
        mass = np.zeros((n_players, n_opponents))
        if target < 0:
            return mass
        pre = prefix[column_index]
        suf = suffix[column_index + 1]
        for wins_before in range(min(target + 1, pre.shape[1])):
            wins_after = target - wins_before
            if 0 <= wins_after < suf.shape[1]:
                mass += pre[:, wins_before, :] * suf[:, wins_after, :]
        return mass

    combination = np.zeros_like(probs)
    for column_index in range(n_columns):
        for target, weight in target_weights:
            combination[:, column_index, :] += weight * leave_one_out_mass(column_index, target)
    return combination


def _stack_single_exclusions(probs: np.ndarray) -> np.ndarray:
    """All n single-category exclusions of probs, stacked into the players axis.

    (n_players, n_columns, n_opponents) -> (n_columns * n_players, n_columns - 1,
    n_opponents), ordered so row block e holds probs with column e removed. Lets
    every exclusion share one DP pass instead of one pass per exclusion — the DP
    cost is dominated by python-level loop steps, not arithmetic.
    """
    n_players, n_columns, n_opponents = probs.shape
    remaining_indices = np.array(
        [[c for c in range(n_columns) if c != excluded] for excluded in range(n_columns)]
    )   # (n_columns, n_columns - 1)
    stacked = probs[:, remaining_indices, :]               # (n_players, n_columns, n_columns-1, n_opponents)
    return stacked.transpose(1, 0, 2, 3).reshape(
        n_columns * n_players, n_columns - 1, n_opponents
    )


def _pair_mass_combination(probs: np.ndarray
                            , target_weights: tuple[tuple[int, float], ...]) -> np.ndarray:
    """sum_t weight_t * P(W_-ij = target_t) for every left-out pair (i, j).

    All single exclusions are stacked into the players axis so the leave-one-out
    machinery runs over every remaining column in one prefix/suffix pass. Exact —
    no divided differences, so pairs with equal win probabilities (common
    mid-draft) need no regularization. Symmetric in (i, j) with a zero, unused
    diagonal.

    Shapes: probs (n_players, n_columns, n_opponents)
            -> (n_players, n_columns, n_columns, n_opponents).
    """
    n_players, n_columns, n_opponents = probs.shape
    combined = _leave_one_out_mass_combination(
        _stack_single_exclusions(probs), target_weights
    ).reshape(n_columns, n_players, n_columns - 1, n_opponents)

    pair_matrix = np.zeros((n_players, n_columns, n_columns, n_opponents))
    for excluded in range(n_columns):
        remaining = [c for c in range(n_columns) if c != excluded]
        pair_matrix[:, remaining, excluded, :] = combined[excluded]
    return pair_matrix


def calculate_pair_bracket_matrix(probs: np.ndarray) -> np.ndarray:
    """Exact leave-two-out bracket matrix B for the correlation correction.

    B[p, i, j, o] = P(W_-ij = k-2) - P(W_-ij = k-1) (odd n; even-n analogue with
    ties counting half), where W_-ij is the win count over all categories except
    i and j.

    Args:
        probs: Win probability per category, shape (n_players, n_categories, n_opponents).

    Returns:
        shape (n_players, n_categories, n_categories, n_opponents).
    """
    return _pair_mass_combination(probs, _bracket_target_weights(probs.shape[1]))


# Node machinery for the evaluation-space correction, cached per category count.
_CORRECTION_NODE_CACHE: dict[int, tuple] = {}


def _correction_node_machinery(n_categories: int) -> tuple:
    """Complex nodes and stencil functionals for the evaluation-space correction.

    Nodes are the upper-half roots of x^m = −1 (m even, m > n): rotated off the real
    axis so every factor 1 − p + p·x stays uniformly bounded away from zero for every
    p in [0, 1] (the factor's zero lies on the negative real axis; the nearest node is
    π/m away). Real-coefficient polynomials take conjugate values at conjugate nodes,
    so the upper half carries everything and each stencil functional folds the
    conjugate pair into 2·Re(...) — halving all downstream array sizes.
    """
    machinery = _CORRECTION_NODE_CACHE.get(n_categories)
    if machinery is not None:
        return machinery

    # Fewer nodes than degree+1 alias coefficient u with u+m, u+2m, ... — but every
    # coefficient we ever read back sits near the majority threshold, and its aliasing
    # partners all lie ABOVE the relevant polynomial's degree, where coefficients are
    # zero. So the minimal node count keeping every stencil read exact is
    #     m >= degree - lowest_read_target + 1
    # over both reads: the bracket stencil against T (degree n-2) and the derivative
    # stencil against the triple level (degree n-3). Rounded up to even so conjugate
    # halving applies. For n = 9 this gives 6 nodes instead of 10 — exact, not approximate.
    bracket_floor = min(target for target, _ in _bracket_target_weights(n_categories))
    derivative_floor = min(target for target, _ in _bracket_derivative_target_weights(n_categories))
    node_count = max(
        (n_categories - 2) + 1 - max(bracket_floor, 0),
        (n_categories - 3) + 1 - max(derivative_floor, 0),
        2,
    )
    node_count += node_count % 2
    half_count = node_count // 2
    odd_multiples = 2 * np.arange(half_count) + 1
    nodes = np.exp(1j * np.pi * odd_multiples / node_count).astype(np.complex64)

    coefficient_index = np.arange(node_count).reshape(-1, 1)
    inverse_half = np.exp(-1j * np.pi * odd_multiples * coefficient_index / node_count) / node_count

    def stencil_functional(target_weights: tuple[tuple[int, float], ...]) -> np.ndarray:
        weights = np.zeros(half_count, dtype=complex)
        for target, weight in target_weights:
            if 0 <= target < node_count:
                weights += weight * inverse_half[target]
        return (2 * weights).astype(np.complex64)

    machinery = (
        nodes,
        stencil_functional(_bracket_target_weights(n_categories)),
        stencil_functional(_bracket_derivative_target_weights(n_categories)),
    )
    _CORRECTION_NODE_CACHE[n_categories] = machinery
    return machinery


def calculate_correction_terms(probs: np.ndarray
                                , correlation_off_diagonal: np.ndarray
                                , standard_pdf: np.ndarray
                                , calculate_gradient: bool = False):
    """Correlation correction and (optionally) its probability-gradient — fast path.

    Works in polynomial-evaluation space: every win-count polynomial is represented
    by its values at fixed complex nodes, where building the full product is a prod,
    removing a category is POINTWISE division (no long-division carry, hence no
    sequential scans), and coefficient-space stencil reads are fixed weighted sums
    over the node values. Runs in single precision: unlike the coefficient-space
    recursions (which amplify rounding step by step and need float64), the node
    formulation has uniformly bounded conditioning — divisors stay ≥ ~0.16 and the
    transform is DFT-like — so complex64 costs ~1e-7 relative error while halving
    the memory traffic that dominates the runtime.

    Validated against the prefix/suffix per-pair reference (calculate_pair_bracket_matrix
    and calculate_correction_probability_gradient): worst deviation ~1e-8 absolute on
    corrections of magnitude ~1e-1.
    """
    n_players, n_categories, n_opponents = probs.shape
    nodes, bracket_weights, derivative_weights = _correction_node_machinery(n_categories)

    probs_single = probs.astype(np.float32)
    pdf_single = standard_pdf.astype(np.float32)
    correlation_single = correlation_off_diagonal.astype(np.float32)

    # factor values f_c(x_k) = 1 − p_c + p_c x_k, shape (a, n, k, o)
    p = probs_single[:, :, np.newaxis, :]
    factor_values = (1 - p + p * nodes.reshape(1, 1, -1, 1)).astype(np.complex64)

    full_values = factor_values.prod(axis=1)                                  # F at nodes
    leave_one_out_values = full_values[:, np.newaxis, :, :] / factor_values   # Q rows at nodes
    mixture_values = np.einsum('ij,aio,aiko->ajko',
                               correlation_single, pdf_single, leave_one_out_values)
    pair_aggregate_values = mixture_values / factor_values                    # T at nodes

    m_phi = np.einsum('acko,k->aco', pair_aggregate_values, bracket_weights).real
    correction = 0.5 * (pdf_single * m_phi).sum(axis=1)

    if not calculate_gradient:
        return correction, m_phi, None

    all_pairs_values = np.einsum('ajo,ajko->ako', pdf_single, pair_aggregate_values)
    own_pair_values = pdf_single[:, :, np.newaxis, :] * pair_aggregate_values
    triple_values = (all_pairs_values[:, np.newaxis, :, :] - 2 * own_pair_values) / factor_values
    probability_gradient = 0.5 * np.einsum('amko,k->amo', triple_values, derivative_weights).real
    return correction, m_phi, probability_gradient


def calculate_correction_probability_gradient(probs: np.ndarray
                                               , correlation_off_diagonal: np.ndarray
                                               , standard_pdf: np.ndarray) -> np.ndarray:
    """∂(correlation correction)/∂p_m — the part the frozen-matrix gradient misses.

    The correction is ½ sum over pairs of (R−I)_ij φ_i φ_j B_ij, and B_ij does not
    depend on p_i or p_j (both are excluded from W_-ij) — so its p_m-derivative
    only involves pairs avoiding m, each contributing the fully symmetric third
    mixed partial: a second-difference of leave-three-out masses.

    Args:
        probs:                    (n_players, n_categories, n_opponents) win probabilities.
        correlation_off_diagonal: (n_categories, n_categories), R − I.
        standard_pdf:             φ(z), same shape as probs.

    Returns:
        (n_players, n_categories, n_opponents): ∂(correction)/∂p_m per category m.
    """
    n_players, n_categories, n_opponents = probs.shape
    derivative_targets = _bracket_derivative_target_weights(n_categories)

    # One batched call computes the second differences for every excluded m at once:
    # the stacked (m-excluded) sets each get their full pair matrix in shared DP passes.
    second_differences = _pair_mass_combination(
        _stack_single_exclusions(probs), derivative_targets
    ).reshape(n_categories, n_players, n_categories - 1, n_categories - 1, n_opponents)

    gradient = np.zeros_like(probs)
    for excluded in range(n_categories):
        remaining = [c for c in range(n_categories) if c != excluded]
        reduced_correlation = correlation_off_diagonal[np.ix_(remaining, remaining)]
        reduced_pdf = standard_pdf[:, remaining, :]
        gradient[:, excluded, :] = 0.5 * np.einsum(
            'cd,acdo,aco,ado->ao',
            reduced_correlation, second_differences[excluded], reduced_pdf, reduced_pdf,
        )
    return gradient


def calculate_tipping_points(x: np.ndarray) -> np.ndarray:
    """Compute per-category tipping-point probabilities using prefix-suffix DP.

    tipping_point[p, c, o] = P(category c is decisive in player p's matchup vs opponent o)
                            = x[p,c,o] * P(win exactly k of other n-1 cats | probs=x)
                            + (1-x[p,c,o]) * P(win exactly k of other n-1 cats | probs=1-x)
    where k = n_categories // 2.

    Uses the same win-count polynomial DP as combinatorial_calculation, but organised as
    prefix/suffix tables so every leave-one-out result is computed in a single forward and
    backward pass instead of enumerating C(n, k) winning combinations.
    """
    n_players, n_categories, n_opponents = x.shape
    k = n_categories // 2

    prefix,      suffix      = _build_win_count_prefix_suffix(x)
    prefix_comp, suffix_comp = _build_win_count_prefix_suffix(1 - x)

    result = np.zeros((n_players, n_categories, n_opponents))

    for c in range(n_categories):
        pre      = prefix[c]          # (n_players, c+1,           n_opponents)
        suf      = suffix[c + 1]      # (n_players, n_categories-c, n_opponents)
        pre_comp = prefix_comp[c]
        suf_comp = suffix_comp[c + 1]

        # P(win exactly k of other n-1 categories) via convolution at sum = k
        win_exactly_k      = np.zeros((n_players, n_opponents))
        win_exactly_k_comp = np.zeros((n_players, n_opponents))
        for a in range(min(k + 1, pre.shape[1])):
            b = k - a
            if 0 <= b < suf.shape[1]:
                win_exactly_k      += pre[:, a, :]      * suf[:, b, :]
            if 0 <= b < suf_comp.shape[1]:
                win_exactly_k_comp += pre_comp[:, a, :] * suf_comp[:, b, :]

        x_c = x[:, c, :]
        result[:, c, :] = x_c * win_exactly_k + (1 - x_c) * win_exactly_k_comp

    if n_categories % 2 == 0:
        # For even n, scale win contribution by 1/2 and add tie probability / 2.
        # Tie probability (same for all categories) = P(win exactly k of all n).
        tie_prob = prefix[n_categories][:, k, :]   # (n_players, n_opponents)
        result = result / 2 + tie_prob[:, np.newaxis, :] / 2

    return result
