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
