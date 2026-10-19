"""
Roster-level fantasy points valuation.

Computes marginal lineup value: how much does adding a candidate to the current roster
improve the best achievable active-lineup FP over the declared evaluation window?

Keeps the following concepts strictly separate:
  - Draft roster legality: 14 total players (incl. rotating bench).
  - Active lineup value: only eligible active slots score each day.
  - IR: does not create extra active slots.

Marginal value is defined as:
    marginal_fp(candidate) = F(roster + candidate) - F(roster)

where F(R) = sum over days in window of optimal_lineup_fp(R, day).

For replacement evaluation:
    replacement_delta(candidate, dropped) = F(roster - dropped + candidate) - F(roster)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from backend.points.lineups import (
    assign_optimal_daily_lineup
    , compute_lineup_fp
    , build_slot_configs_from_position_config
)


# ── Public margin functions ───────────────────────────────────────────────────

def compute_marginal_fp(
    roster_player_ids: list[int]
    , candidate_id: int
    , per_game_fp: dict[int, float]
    , slot_configs: list[dict]
    , position_eligibility: dict[int, list[str]]
    , n_days: int = 1
) -> float:
    """Expected FP added to the active lineup by drafting candidate_id.

    For a single day (n_days=1) this is the pure marginal contribution.
    For a multi-day window it is the sum of daily marginal gains (no schedule
    model — each day is treated as if all players are available).

    Args:
        roster_player_ids: Current draft roster (without the candidate).
        candidate_id:      The player being evaluated.
        per_game_fp:       {player_id: fp_per_game}; NaN = not playing.
        slot_configs:      Active-slot definitions.
        position_eligibility: {player_id: [positions]}.
        n_days:            Number of days to project over (simple multiplier here;
                           schedule-aware version lives in roster_schedule.py).

    Returns:
        Marginal expected FP (always ≥ 0 when adding a non-negative player).
    """
    base_fp = _evaluate_roster(
        roster_player_ids  , per_game_fp, slot_configs, position_eligibility
    )
    augmented_fp = _evaluate_roster(
        roster_player_ids + [candidate_id], per_game_fp, slot_configs, position_eligibility
    )
    return (augmented_fp - base_fp) * n_days


def compute_replacement_delta(
    roster_player_ids: list[int]
    , candidate_id: int
    , dropped_id: int
    , per_game_fp: dict[int, float]
    , slot_configs: list[dict]
    , position_eligibility: dict[int, list[str]]
    , n_days: int = 1
) -> dict:
    """Evaluate the gain from trading dropped_id for candidate_id.

    Returns:
        {
          'delta_fp': float          — net FP change (positive = improvement),
          'before_fp': float         — baseline active-lineup FP,
          'after_fp': float          — post-swap FP,
          'displaced_player_id': int — the player whose starts are displaced,
        }
    """
    before_fp = _evaluate_roster(
        roster_player_ids, per_game_fp, slot_configs, position_eligibility
    )
    new_roster = [pid for pid in roster_player_ids if pid != dropped_id] + [candidate_id]
    after_fp = _evaluate_roster(
        new_roster, per_game_fp, slot_configs, position_eligibility
    )
    return {
        'delta_fp':           (after_fp - before_fp) * n_days,
        'before_fp':          before_fp * n_days,
        'after_fp':           after_fp * n_days,
        'displaced_player_id': dropped_id,
    }


def compute_all_marginals(
    roster_player_ids: list[int]
    , candidate_ids: list[int]
    , per_game_fp: dict[int, float]
    , slot_configs: list[dict]
    , position_eligibility: dict[int, list[str]]
    , n_days: int = 1
) -> dict[int, float]:
    """Compute marginal FP for each candidate in one batch.

    Computes the base roster FP once, then evaluates each augmented roster.

    Returns:
        {candidate_id: marginal_fp}
    """
    base_fp = _evaluate_roster(
        roster_player_ids, per_game_fp, slot_configs, position_eligibility
    )
    result: dict[int, float] = {}
    for cid in candidate_ids:
        augmented_fp = _evaluate_roster(
            roster_player_ids + [cid], per_game_fp, slot_configs, position_eligibility
        )
        result[cid] = (augmented_fp - base_fp) * n_days
    return result


# ── Internal ──────────────────────────────────────────────────────────────────

def _evaluate_roster(
    roster_player_ids: list[int]
    , per_game_fp: dict[int, float]
    , slot_configs: list[dict]
    , position_eligibility: dict[int, list[str]]
) -> float:
    """Optimal active-lineup FP for this roster on one generic day."""
    assignment = assign_optimal_daily_lineup(
        roster_player_ids  = roster_player_ids
        , per_game_fp      = per_game_fp
        , slot_configs     = slot_configs
        , position_eligibility = position_eligibility
        , locked_assignments   = None
    )
    return compute_lineup_fp(assignment, per_game_fp)
