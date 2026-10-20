# testing_files/test_auction_pricing.py
# The auction autodraft's maximum allowable bid (backend/services/auction_pricing.py): a team keeps $1 for every
# other empty slot. Pure functions, no session needed.

from backend.services.auction_pricing import compute_max_allowed_bid, compute_max_allowed_bids


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
