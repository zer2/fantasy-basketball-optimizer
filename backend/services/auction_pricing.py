"""
Auction pricing rules for the auction autodraft.

compute_max_allowed_bid(s): the most a team (each team) may bid while keeping $1 for every other empty slot.
"""

from __future__ import annotations

import math


def compute_max_allowed_bid(
    remaining_cash: float
    , empty_slots: int
) -> int:
    """The most a team may bid while keeping $1 for every other empty roster slot (bench included).
    A team with no empty slot cannot bid at all."""
    if empty_slots <= 0:
        return 0
    return max(0, math.floor(remaining_cash - (empty_slots - 1)))


def compute_max_allowed_bids(
    player_assignments: dict[str, list[int]]
    , remaining_cash: dict[str, float]
    , total_roster_picks: int
) -> dict[str, int]:
    """compute_max_allowed_bid for every team, from its roster size and the league's picks per team."""
    return {team: compute_max_allowed_bid(remaining_cash[team], total_roster_picks - len(roster))
            for team, roster in player_assignments.items()}
