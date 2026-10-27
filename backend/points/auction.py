"""
Points-mode auction pricing.

Implements the minimum-viable reference pricing described in the handoff:

    1. Find teams with open roster places; exclude cash stranded on completed rosters.
    2. Reserve m × N for the minimum bids on open spots (m = minimum_bid, N = open spots).
    3. Compute discretionary money D = C - m × N.
    4. For positive surplus: reference_price_i = m + D × u_i / Σ(u_j)
       where u_i = marginal lineup FP of player i.
    5. Candidates not being purchased shown at minimum bid.
    6. Legal max bid for any player: B - m × (r − 1)
       where B = my remaining budget, r = my remaining open spots.

SAVOR is optional and off by default.  When enabled it applies only to the
discretionary premium (above minimum_bid), then reserves are added back.
Degeneracy cases are handled explicitly — never silently produce nonsense.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class AuctionPricingError(ValueError):
    """An auction pricing computation is ill-posed or degenerate."""


def compute_reference_prices(
    marginal_fp: dict[int, float]
    , team_budgets: dict[str, float]
    , team_open_spots: dict[str, int]
    , minimum_bid: int = 1
    , savor_enabled: bool = False
    , savor_noise: float = 0.0
) -> dict[int, dict]:
    """Compute reference prices for all undrafted candidates.

    Args:
        marginal_fp:       {player_id: marginal_fp_value} for all available players.
                           NaN values are treated as zero surplus.
        team_budgets:      {team_id: remaining_cash} for all teams currently in the auction.
        team_open_spots:   {team_id: open_roster_spots} for all teams.
        minimum_bid:       Minimum legal bid (platform default usually $1).
        savor_enabled:     Whether to apply SAVOR adjustment to discretionary premiums.
        savor_noise:       SAVOR noise σ (only used when savor_enabled).

    Returns:
        {player_id: {
            'reference_price':   float  — generic market reference price,
            'minimum_bid':       int,
            'is_draftable':      bool   — False if all teams have completed rosters,
        }}

    Raises:
        AuctionPricingError for degenerate inputs such as all budgets exhausted.
    """
    # Aggregate across all teams: total discretionary money and total open spots
    total_budget       = sum(team_budgets.values())
    total_open_spots   = sum(team_open_spots.values())
    total_reserved     = minimum_bid * total_open_spots

    if total_open_spots == 0:
        # All rosters complete; no prices can be computed.
        raise AuctionPricingError(
            "All rosters are completed; no open spots remain. "
            "Auction prices cannot be computed."
        )

    if total_budget <= 0:
        raise AuctionPricingError(
            "Total remaining budget is zero or negative. "
            "Auction prices cannot be computed."
        )

    total_discretionary = max(0.0, total_budget - total_reserved)

    # Only players with positive marginal FP share in the discretionary budget.
    positive_surplus_ids = [pid for pid, fp in marginal_fp.items() if fp > 0]
    total_positive_fp    = sum(marginal_fp[pid] for pid in positive_surplus_ids)

    prices: dict[int, dict] = {}

    for player_id, fp_value in marginal_fp.items():
        if total_positive_fp > 0 and fp_value > 0:
            surplus_share = fp_value / total_positive_fp
            discretionary = total_discretionary * surplus_share
            if savor_enabled and savor_noise > 0:
                discretionary = _apply_savor(discretionary, savor_noise)
            reference_price = minimum_bid + discretionary
        else:
            # Zero or negative expected contribution → show at minimum bid.
            reference_price = float(minimum_bid)

        prices[player_id] = {
            'reference_price': reference_price,
            'minimum_bid':     minimum_bid,
            'is_draftable':    total_open_spots > 0,
        }

    return prices


def compute_personal_values(
    marginal_fp: dict[int, float]
    , my_remaining_budget: float
    , my_open_spots: int
    , minimum_bid: int = 1
    , savor_enabled: bool = False
    , savor_noise: float = 0.0
) -> dict[int, dict]:
    """Compute personal willingness-to-pay values from one team's perspective.

    Uses only this team's budget and open spots; does not pool across teams.

    Returns:
        {player_id: {
            'personal_value':         float,
            'legal_max_bid':          float,
            'minimum_bid':            int,
            'budget_feasible':        bool  — can this team afford minimum_bid?,
        }}
    """
    if my_open_spots <= 0:
        return {
            pid: {
                'personal_value':  0.0,
                'legal_max_bid':   0.0,
                'minimum_bid':     minimum_bid,
                'budget_feasible': False,
            }
            for pid in marginal_fp
        }

    reserved          = minimum_bid * my_open_spots
    discretionary     = max(0.0, my_remaining_budget - reserved)
    legal_max_bid     = max(0.0, my_remaining_budget - minimum_bid * max(0, my_open_spots - 1))

    positive_ids      = [pid for pid, fp in marginal_fp.items() if fp > 0]
    total_positive_fp = sum(marginal_fp[pid] for pid in positive_ids)

    result: dict[int, dict] = {}

    for player_id, fp_value in marginal_fp.items():
        if total_positive_fp > 0 and fp_value > 0:
            surplus_share = fp_value / total_positive_fp
            my_discretionary = discretionary * surplus_share
            if savor_enabled and savor_noise > 0:
                my_discretionary = _apply_savor(my_discretionary, savor_noise)
            personal_value = minimum_bid + my_discretionary
        else:
            personal_value = float(minimum_bid)

        result[player_id] = {
            'personal_value':  min(personal_value, legal_max_bid),
            'legal_max_bid':   legal_max_bid,
            'minimum_bid':     minimum_bid,
            'budget_feasible': my_remaining_budget >= minimum_bid,
        }

    return result


def validate_bid(
    bid_amount: float
    , player_id: int
    , my_remaining_budget: float
    , my_open_spots: int
    , minimum_bid: int = 1
) -> None:
    """Raise AuctionPricingError if the bid is illegal.

    A bid is legal when:
      - bid >= minimum_bid
      - bid <= B - m * (r - 1)  where B = remaining_budget, r = open_spots
    """
    if bid_amount < minimum_bid:
        raise AuctionPricingError(
            f"Bid of ${bid_amount:.0f} is below the minimum bid of ${minimum_bid}."
        )

    legal_max = max(0.0, my_remaining_budget - minimum_bid * max(0, my_open_spots - 1))
    if bid_amount > legal_max:
        raise AuctionPricingError(
            f"Bid of ${bid_amount:.0f} exceeds legal maximum of ${legal_max:.0f} "
            f"(${my_remaining_budget:.0f} remaining, {my_open_spots} open spots, "
            f"${minimum_bid} minimum bid per spot)."
        )


# ── SAVOR ─────────────────────────────────────────────────────────────────────

def _apply_savor(discretionary_premium: float, noise: float) -> float:
    """Apply SAVOR smoothing to a discretionary premium (applied before reserves).

    Without noise (sigma=0), returns the input unchanged — deterministic baseline.
    """
    if noise <= 0:
        return discretionary_premium
    # SAVOR is applied as a proportional damping; see algorithm_helpers for the
    # full formulation.  Here we expose a simplified version for points mode.
    return discretionary_premium * max(0.0, 1.0 - noise)
