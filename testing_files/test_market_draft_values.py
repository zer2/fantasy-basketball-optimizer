# testing_files/test_market_draft_values.py
# The market blend behind the example team's predicted draft order and auction prices
# (backend/services/market_draft_values): H-scoring's base order and generic dollars, averaged with Yahoo's average
# draft position and average auction cost.

import pandas as pd
import pytest

from backend.services.market_draft_values import (
    MARKET_WEIGHT, compute_availability_order, compute_blended_prices, uses_market_draft_values,
)


def test_without_market_values_the_order_is_h_scorings():
    assert compute_availability_order([7, 3, 9], None, n_total_picks=20) == [7, 3, 9]


def test_the_order_averages_h_rank_and_adp_with_no_adp_counting_as_the_last_pick():
    assert MARKET_WEIGHT == 0.5
    market = pd.DataFrame({'average_pick': [4.0, 1.0], 'average_cost': [10.0, 50.0]}, index=[101, 103])
    # predicted picks: 101 -> (1 + 4) / 2 = 2.5; 102 -> (2 + 10) / 2 = 6; 103 -> (3 + 1) / 2 = 2; 104 -> (4 + 10) / 2 = 7
    assert compute_availability_order([101, 102, 103, 104], market, n_total_picks=10) == [103, 101, 102, 104]


def test_equal_predictions_keep_h_scorings_order():
    market = pd.DataFrame({'average_pick': [3.0, 1.0], 'average_cost': [1.0, 1.0]}, index=[201, 202])
    # 201 -> (1 + 3) / 2 = 2; 202 -> (2 + 1) / 2 = 1.5; 203 and 204 -> no ADP: (3 + 5) / 2 = 4, (4 + 5) / 2 = 4.5
    assert compute_availability_order([201, 202, 203, 204], market, n_total_picks=5) == [202, 201, 203, 204]


def test_prices_blend_generic_dollars_with_market_costs_rescaled_to_the_same_total():
    generic = pd.Series([40.0, 30.0, 30.0], index=[1, 2, 3])
    market = pd.DataFrame({'average_pick': [1.0, 2.0], 'average_cost': [60.0, 20.0]}, index=[1, 2])
    blended = compute_blended_prices(generic, market)
    # market side: 1 -> 60, 2 -> 20, 3 -> 0 (nobody bought him), rescaled from 80 to 100: 75, 25, 0
    assert blended.to_dict() == pytest.approx({1: 57.5, 2: 27.5, 3: 15.0})
    assert blended.sum() == pytest.approx(generic.sum())


def test_without_market_values_the_prices_are_the_generic_dollars():
    generic = pd.Series([40.0, 30.0], index=[1, 2])
    assert compute_blended_prices(generic, None) is generic


def test_only_projection_sessions_use_market_values():
    assert uses_market_draft_values({'data_source_type': 'projections'})
    assert not uses_market_draft_values({'data_source_type': 'historical'})
