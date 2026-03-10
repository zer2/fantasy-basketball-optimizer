"""
Backend-only copy of src/math/algorithm_helpers.py.

Changes vs original:
- `import streamlit as st` removed.
- @st.cache_data() decorators removed from get_win_grid / get_tie_grid.
The original src/ file is untouched.
"""

import pandas as pd
from scipy.stats import norm
import numpy as np
from itertools import combinations
import numexpr as ne


def auction_value_adjuster(raw_values_unselected: pd.Series,
                            n_remaining_players: int,
                            remaining_cash: int,
                            noise: float) -> pd.Series:
    dollar_values = dollar_scale_adjustment(raw_values_unselected,
                                            remaining_cash,
                                            n_remaining_players)
    return savor_calculation(dollar_values, noise)


def dollar_scale_adjustment(raw_values_unselected: pd.Series,
                             remaining_cash: int,
                             n_remaining_players: int) -> pd.Series:
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


def combinatorial_calculation(c: np.ndarray,
                               c_comp: np.ndarray,
                               data=1,
                               level: int = 0,
                               n_false: int = 0):
    if n_false > c.shape[1] / 2:
        return 0
    elif level < c.shape[1]:
        return (combinatorial_calculation(c, c_comp, data * c[:, level, :], level + 1, n_false) +
                combinatorial_calculation(c, c_comp, data * c_comp[:, level, :], level + 1, n_false + 1))
    elif n_false == c.shape[1] / 2:
        return data / 2
    else:
        return data


def get_win_grid(n_categories: int) -> np.ndarray:
    which = np.array([list(combinations(range(n_categories), int(n_categories / 2) + 1))])
    grid = np.zeros((which.shape[1], n_categories), dtype='bool')
    grid[np.arange(which.shape[1])[None].T, which] = True
    return np.expand_dims(grid, axis=2)


def get_tie_grid(n_categories: int) -> np.ndarray:
    which = np.array([list(combinations(range(n_categories), int(n_categories / 2)))])
    grid = np.zeros((which.shape[1], n_categories), dtype='bool')
    grid[np.arange(which.shape[1])[None].T, which] = True
    return np.expand_dims(grid, axis=2)


def calculate_tipping_points(x: np.ndarray) -> np.ndarray:
    n_categories = x.shape[1]
    grid = get_win_grid(n_categories)
    grid = np.array([grid] * x.shape[0])
    x = x.reshape(x.shape[0], 1, n_categories, x.shape[2])

    positive_first_part = np.prod(ne.evaluate('grid * x + (1-grid) * (1-x)'), axis=2).reshape(
        x.shape[0], grid.shape[1], 1, x.shape[3])
    positive_case_probabilities = np.sum(ne.evaluate('positive_first_part * grid'), axis=1)

    negative_first_part = np.prod(ne.evaluate('(1 - grid) * x + grid * (1-x)'), axis=2).reshape(
        x.shape[0], grid.shape[1], 1, x.shape[3])
    negative_case_probabilities = np.sum(ne.evaluate('negative_first_part * grid'), axis=1)

    final_probabilities = ne.evaluate('positive_case_probabilities + negative_case_probabilities')

    if n_categories % 2 == 0:
        tie_grid = get_tie_grid(n_categories)
        tie_grid = np.array([tie_grid] * x.shape[0])
        tie_probabilities = np.prod(ne.evaluate('tie_grid * x + (1-tie_grid) * (1-x)'), axis=2)
        tie_probabilities = tie_probabilities.reshape(
            x.shape[0], tie_grid.shape[1], 1, x.shape[3]).sum(axis=1)
        final_probabilities = final_probabilities / 2 + tie_probabilities / 2

    return final_probabilities
