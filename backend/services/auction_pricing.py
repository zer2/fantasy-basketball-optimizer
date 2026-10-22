"""
Auction pricing rules shared by the H-score table's "Your $" and the auction autodraft.

compute_max_allowed_bid(s): the most a team (each team) may bid while keeping $1 for every other empty slot.

add_unusable_cash_shares: "Your $" alone is priced on the league's money, so a team holding more cash than
its open slots can use never bids it and ends the auction with money left over. Each candidate's "Your $"
gets a share of the cash the team could not otherwise use if it bought him.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd


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


# How a team's otherwise unusable cash is added to "Your $" (compute_unusable_cash_shares):
#   'simple_scaling' (PR #464): once the team's cash exceeds what its best options cost, the surplus is added to those
#       best options in proportion to their cost; nothing below that point, and no one else;
#   'smooth': the best options are scaled by smooth_spending_scale(r) in all, which follows the surplus smoothly and
#       starts a little before the cash covers them; every player the team can roster is multiplied by
#       1 + (scale - 1) x his value / the best options' value-weighted average value, so a player worth that average
#       gets the scale itself, bigger players more, and there is no cliff below the best options (the user,
#       2026-10-21).
CASH_SHARE_RULES = ('simple_scaling', 'smooth')
# The rule the H-score table, and every autodrafter not given one, applies: the smooth rule at softness 0.2 beat
# simple scaling head to head by +0.33 own-view H-score points (+-0.04 over 240 pairs of auctions, ahead in all six
# seasons), where 0.1 gained +0.11 and 0.3 lost (it overpaid early); experiment report, Smooth vs simple scaling,
# 2026-10-21.
APP_CASH_SHARE_RULE = 'smooth'


def add_unusable_cash_shares(
    your_dollar_series: pd.Series
    , rosterable_players: pd.Index
    , team_cash: float
    , open_active_slots: int
    , empty_roster_slots: int
    , rule: str
    , smooth_rule_softness: float
) -> pd.Series:
    """Each player's "Your $" plus his share of the cash the team could not otherwise use
    (compute_unusable_cash_shares)."""
    return your_dollar_series + compute_unusable_cash_shares(
        your_dollar_series, rosterable_players, team_cash, open_active_slots, empty_roster_slots, rule,
        smooth_rule_softness)


def smooth_spending_scale(
    ratio: float
    , softness: float
) -> float:
    """1 + s ln(1 + e^((ratio - 1) / s)), s the softness: a smoothed max(1, ratio). About 1 well below a ratio of 1
    and the ratio itself well above it; 1 + s ln 2 at 1, where it departs from max(1, ratio) the most. At a softness
    of 0 it is max(1, ratio) exactly: no spending before the surplus."""
    if softness < 0:
        raise ValueError(f'The smooth rule needs a softness of at least 0; got {softness}.')
    if softness == 0:
        return max(1.0, ratio)
    return 1.0 + softness * float(np.logaddexp(0.0, (ratio - 1.0) / softness))


def sum_spendable_league_cash(
    player_assignments: dict[str, list[int]]
    , remaining_cash: dict[str, float]
    , active_slots_per_team: int
) -> float:
    """The league's cash the dollar values are scaled to: each team's cash while it still has an open active slot. A
    team that has filled its active slots cannot bid on any player the dollars price (perma-bench picks are not
    scored), so counting its leftover money would only inflate everyone else's values (the user, 2026-10-21)."""
    return sum(cash for team, cash in remaining_cash.items() if len(player_assignments[team]) < active_slots_per_team)


def select_best_option_costs(
    your_dollar_series: pd.Series
    , rosterable_players: pd.Index
    , team_cash: float
    , open_active_slots: int
    , empty_roster_slots: int
) -> tuple[float, pd.Series, pd.Series]:
    """(spendable, the rosterable players' "Your $", the best options' costs) for compute_unusable_cash_shares and
    compute_cash_to_cost_ratio: spendable is the team's cash minus $1 for each empty bench slot; the best options are
    its top rosterable players, one per open active slot, each costing his "Your $" but no less than the $1 minimum
    bid."""
    if open_active_slots < 1:
        raise ValueError(f'Pricing needs an open active slot; the team has {open_active_slots}.')
    if empty_roster_slots < open_active_slots:
        raise ValueError(f'The team has {empty_roster_slots} empty roster slots but {open_active_slots} open '
                         f'active slots; active slots are part of the roster.')
    spendable  = team_cash - (empty_roster_slots - open_active_slots)
    rosterable = your_dollar_series.loc[rosterable_players]
    best_options = rosterable.sort_values(ascending=False).iloc[:open_active_slots]
    return spendable, rosterable, best_options.clip(lower=1.0)


def compute_cash_to_cost_ratio(
    your_dollar_series: pd.Series
    , rosterable_players: pd.Index
    , team_cash: float
    , open_active_slots: int
    , empty_roster_slots: int
) -> Optional[float]:
    """r, the ratio the smooth rule acts on: the team's spendable cash over what its best options cost. Below 1 the
    team cannot afford all of them; above 1 it holds cash they will not use. None when no available player fits an
    open slot."""
    spendable, _, best_option_costs = select_best_option_costs(
        your_dollar_series, rosterable_players, team_cash, open_active_slots, empty_roster_slots)
    if len(best_option_costs) == 0:
        return None
    return spendable / float(best_option_costs.sum())


def compute_unusable_cash_shares(
    your_dollar_series: pd.Series
    , rosterable_players: pd.Index
    , team_cash: float
    , open_active_slots: int
    , empty_roster_slots: int
    , rule: str
    , smooth_rule_softness: float
) -> pd.Series:
    """Each player's share of the cash the team could not otherwise use, $0 for most: the amount
    add_unusable_cash_shares adds to his "Your $", kept apart so the table can show which values carry one. "Your $"
    is what a player is worth to the team, so it is not capped at what the team can bid.

      - the team's best options: its top rosterable players by "Your $", one per open active slot. They are
        the players it would fill its slots with; their cost is their "Your $" but no less than the $1 minimum bid;
      - spendable: the team's cash minus $1 for each empty bench slot.

    'simple_scaling': the surplus (spendable minus the best options' cost) is added to the best options in proportion to
    their cost, so they keep their ratios and together spend the cash; everyone else is unchanged. Example: $60, two
    open slots, players worth $20, $10 and $5: the surplus is $30, and the first two become $40 and $20.

    'smooth': with ratio = spendable / the best options' cost, scale = smooth_spending_scale(ratio,
    smooth_rule_softness): about 1 well below a ratio of 1, about the ratio well above it, starting a little below 1.
    Every rosterable player's multiplier is 1 + (scale - 1) x cost / the best options' value-weighted average cost
    (sum of cost^2 over sum of cost). A player worth that average gets the scale itself and bigger players more, and
    the best options together come to exactly scale x their cost, with no cliff at the last of them. Example
    (softness 0.05): $40, two open slots, players worth $20, $10.10 and $9.90: they become $27.89, $12.11 and $11.83.
    smooth_rule_softness is ignored by 'simple_scaling'.

    your_dollar_series covers the available pool; rosterable_players are those whose positions fit one of
    the team's open active slots (the candidates the table shows)."""
    if rule not in CASH_SHARE_RULES:
        raise ValueError(f'Unknown cash-share rule {rule!r}; expected one of {CASH_SHARE_RULES}.')
    spendable, rosterable, best_option_costs = select_best_option_costs(
        your_dollar_series, rosterable_players, team_cash, open_active_slots, empty_roster_slots)
    shares = pd.Series(0.0, index=your_dollar_series.index)
    if len(best_option_costs) == 0:
        return shares

    if rule == 'simple_scaling':
        surplus = max(0.0, spendable - best_option_costs.sum())
        shares.loc[best_option_costs.index] = surplus * best_option_costs / best_option_costs.sum()
        return shares

    scale = smooth_spending_scale(spendable / best_option_costs.sum(), smooth_rule_softness)
    # The best options' value-weighted average cost: a player worth exactly this gets the multiplier scale.
    weighted_average_cost = float((best_option_costs ** 2).sum() / best_option_costs.sum())
    costs = rosterable.clip(lower=1.0)
    shares.loc[rosterable.index] = (scale - 1.0) * costs / weighted_average_cost * costs
    return shares
