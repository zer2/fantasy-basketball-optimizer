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

def get_future_player_rows(position_rewards: np.ndarray,
                           pos_cfg: PositionConfig) -> np.ndarray:
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
    """Turns a list of player eligibilities into a row array for the
    assignment problem."""

    n_players          = len(players)
    position_structure = pos_cfg.position_structure
    position_numbers   = pos_cfg.position_numbers
    base_list          = position_structure['base_list']
    flex_list          = position_structure['flex_list']

    base_index     = {position_code: i for i, position_code in enumerate(base_list)}
    is_base        = np.array([[position_code in player for position_code in base_list] for player in players])

    base_slots = {
        position_code: np.where(
            is_base[:, base_index[position_code]].reshape(-1, 1),
            [[0]       * position_numbers[position_code]] * n_players,
            [[-np.inf] * position_numbers[position_code]] * n_players,
        )
        for position_code in base_list
    }

    flex_slots = {
        position_code: np.where(
            is_base[:, [base_index[b] for b in position_structure['flex'][position_code]['bases']]]
            .any(axis=1).reshape(-1, 1),
            [[0]       * position_numbers[position_code]] * n_players,
            [[-np.inf] * position_numbers[position_code]] * n_players,
        )
        for position_code in flex_list
    }

    res = np.concatenate(
        [base_slots[p] for p in base_list]
        + [flex_slots[p] for p in flex_list],
        axis=1,
    )
    return res


def _optimize_positions_for_prospective_player(
    candidate_player_row: np.ndarray,
    reward_vector: np.ndarray,
    team_so_far_array: np.ndarray,
    n_remaining_players: int,
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


def get_position_array_from_res(res: np.ndarray,
                                 position_shares: dict,
                                 n_remaining_players: int,
                                 pos_cfg: PositionConfig):
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
    candidate_players: list,
    position_rewards: np.ndarray,
    team_so_far: list,
    position_shares: dict,
    pos_cfg: PositionConfig,
    scale_down: bool = True,
):
    position_numbers   = pos_cfg.position_numbers
    n_total_picks      = sum(position_numbers.values())
    n_remaining_players = n_total_picks - 1 - len(team_so_far)

    reward_array          = get_future_player_rows(position_rewards, pos_cfg)
    team_so_far_array     = (get_player_rows(team_so_far, pos_cfg)
                             if len(team_so_far) > 0
                             else np.empty((0, n_total_picks)))
    candidate_player_array = get_player_rows(candidate_players, pos_cfg)

    rosters = np.concatenate(
        [[
            _optimize_positions_for_prospective_player(
                player, reward_vector, team_so_far_array, n_remaining_players
            )
            for player, reward_vector in zip(candidate_player_array, reward_array)
        ]],
        axis=0,
    )

    final_positions, flex_shares = get_position_array_from_res(
        rosters, position_shares, n_remaining_players, pos_cfg
    )

    if scale_down:
        final_positions = final_positions / n_remaining_players

    return rosters, final_positions, flex_shares


def check_single_player_eligibility(
    player: list,
    team_so_far: list,
    pos_cfg: PositionConfig,
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
    players: list,
    team_so_far: list,
    pos_cfg: PositionConfig,
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
