"""
Backend-only copy of src/math/position_optimization.py.

The only change vs the original: every function that read position structure
from st.session_state now takes an explicit `pos_cfg: PositionConfig` parameter.
The original src/ file is untouched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from backend.session import PositionConfig


# ── internal helpers ──────────────────────────────────────────────────────────

def get_future_player_rows(position_rewards: np.ndarray
                           , pos_cfg: PositionConfig) -> np.ndarray:
    """Takes an array of rewards by simplified position (5 cols) and
    translates them to rewards per slot (13) by player."""

    position_numbers   = pos_cfg.position_numbers
    position_structure = pos_cfg.position_structure
    position_indices   = pos_cfg.position_indices
    base_list          = position_structure['base_list']

    base_rewards = {
        position_code: np.array([position_rewards[:, i]] * position_numbers[position_code])
                        .reshape(position_numbers[position_code], len(position_rewards))
        for i, position_code in enumerate(base_list)
    }

    flex_rewards = {
        position_code:
            np.array([
                np.max(position_rewards[:, position_indices[position_code]], axis=1)
                + 0.0001 * len(position_indices[position_code])
            ] * position_numbers[position_code])
            .reshape(position_numbers[position_code], len(position_rewards))
        for position_code in position_structure['flex_list']
    }

    row = np.concatenate(
        [base_rewards[p] for p in position_structure['base_list']]
        + [flex_rewards[p] for p in position_structure['flex_list']],
        axis=0,
    ).T

    return row


def get_player_rows(players: list, pos_cfg: PositionConfig) -> np.ndarray:
    """Turns a list of player eligibilities into a row array for the assignment
    problem: one row per player, one column per roster slot, 0 where the player is
    eligible for that slot and -inf where not.

    Slots are ordered base positions first (each repeated position_numbers[code]
    times) then flex positions, matching get_future_player_rows / the optimiser.
    """
    n_players          = len(players)
    position_structure = pos_cfg.position_structure
    position_numbers   = pos_cfg.position_numbers
    base_list          = position_structure['base_list']
    flex_list          = position_structure['flex_list']
    base_index         = {position_code: i for i, position_code in enumerate(base_list)}

    # (n_players, n_base) eligibility for each base position. (reshape covers n_players == 0,
    # where the list comprehension would otherwise collapse to shape (0,).)
    is_base = np.array(
        [[position_code in player for position_code in base_list] for player in players],
        dtype=bool,
    ).reshape(n_players, len(base_list))

    # Per-slot-group eligibility in final slot order: a base group is eligible iff the player
    # has that base position; a flex group iff the player has ANY of its eligible bases.
    group_eligible = [is_base[:, base_index[position_code]] for position_code in base_list]
    group_eligible += [
        is_base[:, [base_index[b] for b in position_structure['flex'][position_code]['bases']]].any(axis=1)
        for position_code in flex_list
    ]
    group_slot_counts = [position_numbers[position_code] for position_code in base_list + flex_list]

    # Map eligibility to 0 / -inf on the compact (n_players, n_groups) matrix, then expand each
    # group to its slot count with one vectorised np.repeat — replacing the per-group Python
    # list-of-lists builds + dict + concatenate of the original.
    group_values = np.where(np.stack(group_eligible, axis=1), 0.0, -np.inf)
    return np.repeat(group_values, group_slot_counts, axis=1)


def _optimize_positions_for_prospective_player(
    candidate_player_row: np.ndarray
    , reward_vector: np.ndarray
    , team_so_far_array: np.ndarray
    , n_remaining_players: int
) -> np.ndarray:
    """Pure function — no session state needed (unchanged from original)."""
    future_player_rows = np.array([reward_vector] * n_remaining_players).reshape(
        n_remaining_players, reward_vector.shape[0]
    )
    full_array = np.concatenate(
        [team_so_far_array, [candidate_player_row], future_player_rows], axis=0
    )
    # scipy raises ValueError both for infeasible assignments (legitimate — the
    # team can't fit this player anywhere) and for NaN/Inf in the cost matrix
    # (a bug we want surfaced). The -1 sentinel signals infeasibility downstream
    # in get_position_array_from_res. Narrowed to ValueError so unrelated bugs
    # (TypeError, KeyError, etc.) propagate instead of being swallowed.
    try:
        res = linear_sum_assignment(full_array, maximize=True)
        return res[1]
    except ValueError:
        return np.array([-1] * len(reward_vector))


def get_position_array_from_res(res: np.ndarray
                                 , position_shares: dict
                                 , n_remaining_players: int
                                 , pos_cfg: PositionConfig):
    position_ranges    = pos_cfg.position_ranges
    position_structure = pos_cfg.position_structure

    future_positions = res[:, -n_remaining_players:]
    position_sums: dict = {}

    for position_code, position_range in position_ranges.items():
        position_sums[position_code] = (
            (future_positions >= position_range['start'])
            & (future_positions < position_range['end'])
        ).sum(axis=1).astype(float)

    for position_code in position_structure['flex_list']:
        share_values  = position_shares[position_code].values
        share_columns = list(position_shares[position_code].columns)
        flex_split    = share_values * position_sums[position_code].reshape(-1, 1)
        for base_position_code in position_structure['flex'][position_code]['bases']:
            position_sums[base_position_code] += flex_split[:, share_columns.index(base_position_code)]

    res_main = np.concatenate(
        [[position_sums[p]] for p in position_structure['base_list']], axis=0
    ).T

    flex_shares_out = {
        p: position_sums[p] for p in position_structure['flex_list']
    }
    return res_main, flex_shares_out


# ── public API ────────────────────────────────────────────────────────────────

def optimize_positions_all_players(
    candidate_player_array: np.ndarray
    , position_rewards: np.ndarray
    , team_so_far_array: np.ndarray
    , position_shares: dict
    , pos_cfg: PositionConfig
    , scale_down: bool = True
    , active_count: int | None = None
    , cached_rosters: np.ndarray | None = None
):
    # The eligibility rows (candidate_player_array, team_so_far_array) are weight-independent, so the
    # caller builds them once per evaluate via get_player_rows rather than once per gradient iteration.
    n_total_picks       = sum(pos_cfg.position_numbers.values())
    n_remaining_players = n_total_picks - 1 - team_so_far_array.shape[0]
    n_candidates        = candidate_player_array.shape[0]

    reward_array = get_future_player_rows(position_rewards, pos_cfg)

    # Throttle: solve the roster assignment only for the top `active_count` candidates and reuse the
    # previous iteration's assignment for the rest. The per-candidate Hungarian solve dominates the
    # cost, and a lower-ranked candidate's optimal slots barely move between iterations. A full solve
    # runs when there is no cache yet or active_count already covers everyone.
    if cached_rosters is None or active_count is None or active_count >= n_candidates:
        active_count = n_candidates

    active_rosters = np.array([
        _optimize_positions_for_prospective_player(
            player, reward_vector, team_so_far_array, n_remaining_players
        )
        for player, reward_vector in zip(candidate_player_array[:active_count], reward_array[:active_count])
    ])
    rosters = (active_rosters if active_count >= n_candidates
               else np.concatenate([active_rosters, cached_rosters[active_count:]], axis=0))

    final_positions, flex_shares = get_position_array_from_res(
        rosters, position_shares, n_remaining_players, pos_cfg
    )

    if scale_down:
        final_positions = final_positions / n_remaining_players

    return rosters, final_positions, flex_shares


def check_single_player_eligibility(
    player: list
    , team_so_far: list
    , pos_cfg: PositionConfig
) -> bool:
    position_numbers = pos_cfg.position_numbers
    n_total_picks    = sum(position_numbers.values())
    n_base_positions = len(pos_cfg.position_structure['base_list'])

    position_rewards    = np.array([[0] * n_base_positions])
    n_remaining_players = n_total_picks - 1 - len(team_so_far)
    reward_vector       = get_future_player_rows(position_rewards, pos_cfg)[0]
    team_so_far_array   = (get_player_rows(team_so_far, pos_cfg)
                           if len(team_so_far) > 0
                           else np.empty((0, n_total_picks)))
    candidate_vector    = get_player_rows([player], pos_cfg)[0]

    all_res = _optimize_positions_for_prospective_player(
        candidate_vector, reward_vector, team_so_far_array, n_remaining_players
    )
    return bool(all(all_res >= 0))


def check_team_eligibility(team: list, pos_cfg: PositionConfig) -> bool:
    """Checks if a full team satisfies position constraints.

    scipy's linear_sum_assignment raises ValueError when no feasible assignment
    exists — that is the signal for "team is ineligible." Narrowed to ValueError
    so genuine bugs (NaN/Inf in the cost matrix, TypeError, etc.) propagate.
    """
    if len(team) == 0:
        return True
    team_array = get_player_rows(team, pos_cfg)
    try:
        linear_sum_assignment(team_array, maximize=True)
        return True
    except ValueError:
        return False


def check_all_player_eligibility(
    players: list
    , team_so_far: list
    , pos_cfg: PositionConfig
) -> list[bool]:
    position_numbers = pos_cfg.position_numbers
    n_total_picks    = sum(position_numbers.values())
    n_base_positions = len(pos_cfg.position_structure['base_list'])

    position_rewards    = np.array([[0] * n_base_positions])
    n_remaining_players = n_total_picks - 1 - len(team_so_far)
    reward_vector       = get_future_player_rows(position_rewards, pos_cfg)[0]
    team_so_far_array   = (get_player_rows(team_so_far, pos_cfg)
                           if len(team_so_far) > 0
                           else np.empty((0, n_total_picks)))
    player_rows = get_player_rows(players, pos_cfg)

    return [
        all(
            _optimize_positions_for_prospective_player(
                row, reward_vector, team_so_far_array, n_remaining_players
            ) >= 0
        )
        for row in player_rows
    ]
