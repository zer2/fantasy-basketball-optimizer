# testing_files/test_auction_pricing.py
# Auction pricing (backend/services/auction_pricing.py): a team keeps $1 for every other empty slot, and the
# cash a team's best options leave unspent is added to "Your $" (simple scaling, or the smooth rule). Pure
# functions, no session needed.

import numpy as np
import pandas as pd
import pytest

from backend.services.auction_pricing import (
    add_unusable_cash_shares, compute_cash_to_cost_ratio, compute_max_allowed_bid, compute_max_allowed_bids,
    smooth_spending_scale, sum_spendable_league_cash,
)


def _price(
    values: dict[str, float]
    , team_cash: float
    , open_active_slots: int
    , empty_bench_slots: int = 0
    , rule: str = 'simple_scaling'
    , smooth_rule_softness: float = 0.05
) -> dict[str, float]:
    your_dollar_series = pd.Series(values, dtype=float)
    adjusted = add_unusable_cash_shares(
        your_dollar_series = your_dollar_series
        , rosterable_players = your_dollar_series.index
        , team_cash          = team_cash
        , open_active_slots  = open_active_slots
        , empty_roster_slots = open_active_slots + empty_bench_slots
        , rule               = rule
        , smooth_rule_softness = smooth_rule_softness
    )
    return {player: round(value, 6) for player, value in adjusted.items()}


def test_surplus_is_split_among_best_options_in_proportion():
    # Best options cost $30 of $60: the $30 surplus is split 2:1 like their values, so they keep their ratio and
    # together spend the cash; the third is not one of them.
    assert _price({'A': 20, 'B': 10, 'C': 5}, team_cash=60, open_active_slots=2) == {'A': 40, 'B': 20, 'C': 5}


def test_no_surplus_changes_nothing():
    # The best options already cost more than the team's cash.
    assert _price({'A': 30, 'B': 20, 'C': 10}, team_cash=40, open_active_slots=2) == {'A': 30, 'B': 20, 'C': 10}


def test_last_slot_puts_the_surplus_on_the_best_option():
    # One slot, $100: the best option absorbs the whole $90 surplus.
    assert _price({'A': 10, 'B': 9}, team_cash=100, open_active_slots=1) == {'A': 100, 'B': 9}


def test_bench_slots_hold_back_a_dollar_each():
    # $60 - $30 for the best options - $2 for two bench slots leaves $28, split 2:1.
    assert _price({'A': 20, 'B': 10, 'C': 5}, team_cash=60, open_active_slots=2, empty_bench_slots=2) == {
        'A': round(20 + 28 * 2 / 3, 6), 'B': round(10 + 28 / 3, 6), 'C': 5}


def test_best_options_cost_at_least_the_minimum_bid():
    # Players worth under $1 still cost $1 to buy, and are weighted by that cost: $10 - 2 x $1 leaves $8, $4 each.
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


# ── The smooth rule ───────────────────────────────────────────────────────────

def test_the_smooth_scale_follows_the_ratio_above_one_and_fades_below():
    assert smooth_spending_scale(2.0, 0.05) == pytest.approx(2.0, abs=1e-6)
    assert smooth_spending_scale(1.2, 0.05) == pytest.approx(1.2, abs=1e-3)
    assert smooth_spending_scale(1.0, 0.05) == pytest.approx(1.0 + 0.05 * np.log(2))
    assert smooth_spending_scale(0.7, 0.05) == pytest.approx(1.0, abs=1e-3)


def test_a_softer_scale_starts_earlier_and_bends_wider():
    assert smooth_spending_scale(1.0, 0.2) == pytest.approx(1.0 + 0.2 * np.log(2))
    assert smooth_spending_scale(0.8, 0.2) > smooth_spending_scale(0.8, 0.1) > 1.0
    with pytest.raises(ValueError):
        smooth_spending_scale(1.0, -0.1)


def test_a_softness_of_zero_waits_for_the_surplus():
    assert smooth_spending_scale(0.9, 0.0) == 1.0
    assert smooth_spending_scale(1.3, 0.0) == 1.3


def test_the_softness_sets_how_much_the_smooth_rule_adds_near_the_bend():
    # $30 of best options and $30 of cash (a ratio of 1): the pair takes 1 + s ln 2 of its cost.
    for softness in (0.1, 0.2):
        priced = _price({'A': 20, 'B': 10}, team_cash=30, open_active_slots=2, rule='smooth',
                        smooth_rule_softness=softness)
        assert priced['A'] + priced['B'] == pytest.approx(30 * (1 + softness * np.log(2)), abs=1e-6)


def test_the_smooth_rule_boosts_bigger_players_more_and_has_no_cliff():
    # $40, two slots, $20 / $10.10 / $9.90: the best options spend the $40, and the third, just below them, is
    # boosted almost as much as the second rather than not at all.
    priced = _price({'A': 20, 'B': 10.1, 'C': 9.9}, team_cash=40, open_active_slots=2, rule='smooth')
    assert priced['A'] + priced['B'] == pytest.approx(40, abs=0.01)
    assert priced['A'] == pytest.approx(27.89, abs=0.01)
    assert priced['B'] == pytest.approx(12.11, abs=0.01)
    assert priced['C'] == pytest.approx(11.83, abs=0.01)
    assert priced['A'] / 20 > priced['B'] / 10.1 > 1, 'a bigger player gets a bigger multiplier'


def test_the_smooth_rule_starts_a_little_before_the_cash_covers_the_best_options():
    just_short = _price({'A': 20, 'B': 10}, team_cash=29, open_active_slots=2, rule='smooth')
    assert just_short['A'] > 20 and just_short['B'] > 10, 'a ratio of 0.97 already boosts'
    far_short = _price({'A': 20, 'B': 10}, team_cash=15, open_active_slots=2, rule='smooth')
    assert far_short == pytest.approx({'A': 20, 'B': 10}, abs=1e-3), 'a ratio of 0.5 does nothing'


def test_an_unknown_rule_is_refused():
    with pytest.raises(ValueError):
        _price({'A': 20}, team_cash=60, open_active_slots=1, rule='generous')


def test_the_smooth_rule_shifts_a_teams_valuation_toward_its_top_players():
    # A draft-like curve over 40 rosterable players, a team with $150 for five open slots: every player gains, but the
    # multiplier rises with value all the way down, so the top players' share of the team's valuation grows and the
    # bottom players' shrinks.
    values = {f'P{rank}': 40 * 0.92 ** rank for rank in range(40)}
    priced = _price(values, team_cash=150, open_active_slots=5, rule='smooth', smooth_rule_softness=0.2)
    multipliers = np.array([priced[player] / value for player, value in values.items()])
    assert (multipliers >= 1).all(), 'the rule only adds'
    assert (np.diff(multipliers) <= 1e-9).all(), 'the multiplier falls with value'
    before = np.array(list(values.values())) / sum(values.values())
    after = np.array(list(priced.values())) / sum(priced.values())
    assert after[0] > before[0] and after[-1] < before[-1], 'the top gains share and the bottom loses it'


def test_the_cash_to_cost_ratio_is_spendable_cash_over_the_best_options():
    # $45 with two open active slots and one empty bench slot: $44 spendable against $20 + $10 for the best two.
    values = pd.Series({'A': 20.0, 'B': 10.0, 'C': 5.0})
    assert compute_cash_to_cost_ratio(values, values.index, 45, 2, 3) == pytest.approx(44 / 30)
    # A best option worth under $1 still costs the $1 minimum bid.
    cheap = pd.Series({'A': 0.4, 'B': 0.2})
    assert compute_cash_to_cost_ratio(cheap, cheap.index, 3, 2, 2) == pytest.approx(3 / 2)
    assert compute_cash_to_cost_ratio(values, pd.Index([]), 45, 2, 3) is None, 'no player fits an open slot'


def test_a_team_with_no_open_slot_adds_nothing_to_the_league_cash():
    # Three active slots each: B has filled them and kept $24, which it can no longer spend on anyone the dollars price.
    rosters = {'A': [1], 'B': [2, 3, 4], 'C': []}
    cash = {'A': 120.0, 'B': 24.0, 'C': 200.0}
    assert sum_spendable_league_cash(rosters, cash, 3) == 320.0
