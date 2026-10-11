# testing_files/test_auction_pricing.py
# Auction pricing (backend/services/auction_pricing.py): a team keeps $1 for every other empty slot, and the
# cash a team's best options leave unspent is split evenly among them in "Your $". Pure functions, no session
# needed.

import pandas as pd
import pytest

from backend.services.auction_pricing import add_unusable_cash_shares, compute_max_allowed_bid, compute_max_allowed_bids


def _price(
    values: dict[str, float]
    , team_cash: float
    , open_active_slots: int
    , empty_bench_slots: int = 0
) -> dict[str, float]:
    your_dollar_series = pd.Series(values, dtype=float)
    adjusted = add_unusable_cash_shares(
        your_dollar_series = your_dollar_series
        , rosterable_players = your_dollar_series.index
        , team_cash          = team_cash
        , open_active_slots  = open_active_slots
        , empty_roster_slots = open_active_slots + empty_bench_slots
    )
    return {player: round(value, 6) for player, value in adjusted.items()}


def test_surplus_is_split_evenly_among_best_options():
    # Best options cost $30 of $60, so each of the two gets $15 more; the third is not one of them.
    assert _price({'A': 20, 'B': 10, 'C': 5}, team_cash=60, open_active_slots=2) == {'A': 35, 'B': 25, 'C': 5}


def test_no_surplus_changes_nothing():
    # The best options already cost more than the team's cash.
    assert _price({'A': 30, 'B': 20, 'C': 10}, team_cash=40, open_active_slots=2) == {'A': 30, 'B': 20, 'C': 10}


def test_last_slot_puts_the_surplus_on_the_best_option():
    # One slot, $100: the best option absorbs the whole $90 surplus.
    assert _price({'A': 10, 'B': 9}, team_cash=100, open_active_slots=1) == {'A': 100, 'B': 9}


def test_bench_slots_hold_back_a_dollar_each():
    # $60 - $30 for the best options - $2 for two bench slots leaves $28, $14 each.
    assert _price({'A': 20, 'B': 10, 'C': 5}, team_cash=60, open_active_slots=2, empty_bench_slots=2) == {
        'A': 34, 'B': 24, 'C': 5}


def test_best_options_cost_at_least_the_minimum_bid():
    # Players worth under $1 still cost $1 to buy: $10 - 2 x $1 leaves $8, $4 each.
    assert _price({'A': 0.5, 'B': 0.2, 'C': 0.1}, team_cash=10, open_active_slots=2) == {
        'A': 4.5, 'B': 4.2, 'C': 0.1}


def test_values_may_exceed_what_the_team_can_bid():
    # A cash-poor team: $20 with three open slots can bid at most $18, but "Your $" is what the players are
    # worth to it, so they keep their values.
    assert _price({'A': 45, 'B': 30, 'C': 10}, team_cash=20, open_active_slots=3) == {'A': 45, 'B': 30, 'C': 10}


def test_maximum_allowable_bid_keeps_a_dollar_per_other_empty_slot():
    assert compute_max_allowed_bid(remaining_cash=50, empty_slots=4) == 47
    assert compute_max_allowed_bid(remaining_cash=50, empty_slots=1) == 50
    assert compute_max_allowed_bid(remaining_cash=50, empty_slots=0) == 0


def test_maximum_allowable_bid_is_never_negative():
    # $2 with five empty slots cannot even cover the $1 minimums.
    assert compute_max_allowed_bid(remaining_cash=2, empty_slots=5) == 0


def test_maximum_allowable_bids_count_each_team_s_empty_slots():
    # Five picks per team: Team 1 has two empty slots, Team 2 has four, Team 3 is full.
    player_assignments = {'Team 1': [1, 2, 3], 'Team 2': [4], 'Team 3': [5, 6, 7, 8, 9]}
    remaining_cash     = {'Team 1': 30, 'Team 2': 80, 'Team 3': 12}
    assert compute_max_allowed_bids(player_assignments, remaining_cash, total_roster_picks=5) == {
        'Team 1': 29, 'Team 2': 77, 'Team 3': 0}


def test_a_team_without_an_open_active_slot_is_refused():
    with pytest.raises(ValueError):
        _price({'A': 20}, team_cash=60, open_active_slots=0)
