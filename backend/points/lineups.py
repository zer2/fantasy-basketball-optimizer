"""
Daily and weekly active-lineup optimisation for points mode.

Maximises the sum of projected fantasy points over eligible player-slot assignments
using scipy's linear_sum_assignment (exact optimal for this bipartite problem).

Handles:
  - Multi-position player eligibility and flex slot routing.
  - Daily pregame lock and weekly matchup lock modes.
  - Frozen pre-locked slot assignments with input validation.
  - Rational benching of zero-game or negative-projected players.
  - Clean partitioning into active starters, bench, and IR slots.
  - Complete backward compatibility with legacy dict callers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
from scipy.optimize import linear_sum_assignment


# ── Canonical Position Structure Default ─────────────────────────────────────

DEFAULT_POSITION_STRUCTURE: dict = {
    'base_list': ['PG', 'SG', 'SF', 'PF', 'C'],
    'flex_list': ['G', 'F', 'Util'],
    'flex': {
        'G': {'bases': ['PG', 'SG']},
        'F': {'bases': ['SF', 'PF']},
        'Util': {'bases': ['PG', 'SG', 'SF', 'PF', 'C']},
    },
}

INELIGIBLE_COST: float = 1e6


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DailyLineupAssignment:
    """Optimal assignment of players to active slots on a single day."""
    active_assignments: dict[str, Optional[int]]
    bench_players: list[int]
    ir_players: list[int]
    total_expected_fp: float

    # Dict-like mapping interface for backward compatibility
    def __getitem__(self, slot_id: str) -> Optional[int]:
        return self.active_assignments[slot_id]

    def get(self, slot_id: str, default: Optional[int] = None) -> Optional[int]:
        return self.active_assignments.get(slot_id, default)

    def items(self):
        return self.active_assignments.items()

    def values(self):
        return self.active_assignments.values()

    def keys(self):
        return self.active_assignments.keys()

    def __iter__(self):
        return iter(self.active_assignments)

    def __contains__(self, slot_id: str) -> bool:
        return slot_id in self.active_assignments

    def __len__(self) -> int:
        return len(self.active_assignments)


@dataclass(frozen=True)
class WeeklyLineupAssignment:
    """Optimal assignment of players to active slots for a matchup week."""
    active_assignments: dict[str, Optional[int]]
    bench_players: list[int]
    ir_players: list[int]
    total_expected_fp: float
    player_weekly_fp: dict[int, float] = field(default_factory=dict)

    # Dict-like mapping interface for backward compatibility
    def __getitem__(self, slot_id: str) -> Optional[int]:
        return self.active_assignments[slot_id]

    def get(self, slot_id: str, default: Optional[int] = None) -> Optional[int]:
        return self.active_assignments.get(slot_id, default)

    def items(self):
        return self.active_assignments.items()

    def values(self):
        return self.active_assignments.values()

    def keys(self):
        return self.active_assignments.keys()

    def __iter__(self):
        return iter(self.active_assignments)

    def __contains__(self, slot_id: str) -> bool:
        return slot_id in self.active_assignments

    def __len__(self) -> int:
        return len(self.active_assignments)


# ── Slot Config Helper ───────────────────────────────────────────────────────

def _is_eligible(player_positions: list[str], slot_eligible: list[str]) -> bool:
    """True if the player holds at least one position the slot accepts."""
    p_set = {str(p).upper() for p in player_positions}
    s_set = {str(s).upper() for s in slot_eligible}
    if 'UTIL' in s_set or 'UTIL' in p_set:
        return True
    return bool(p_set & s_set)


def build_slot_configs_from_position_config(
    active_slots: dict[str, int],
    position_structure: Optional[dict] = None,
) -> list[dict]:
    """Convert an active_slots dict into the canonical slot_configs format."""
    ps = position_structure or DEFAULT_POSITION_STRUCTURE
    base_list = list(ps.get('base_list', ['PG', 'SG', 'SF', 'PF', 'C']))
    flex_list = list(ps.get('flex_list', ['G', 'F', 'Util']))
    flex_info = ps.get('flex', {})

    all_positions = base_list + [p for p in flex_list if p not in base_list]

    for k in active_slots:
        if k not in all_positions:
            all_positions.append(k)

    slots = []
    for pos in all_positions:
        count = active_slots.get(pos, 0)
        if count <= 0:
            continue
        if pos in base_list:
            eligible = [pos]
        elif pos in flex_info:
            eligible = list(flex_info[pos].get('bases', [pos])) + [pos]
        elif pos.upper() in ('UTIL', 'BENCH', 'IR'):
            eligible = list(base_list) + [pos]
        else:
            eligible = [pos]

        for i in range(1, count + 1):
            slots.append({
                'slot_id': f'{pos}{i}',
                'eligible_positions': eligible,
            })
    return slots


# ── Public Entry Points ──────────────────────────────────────────────────────

def assign_optimal_daily_lineup(
    roster_player_ids: list[int],
    player_positions: Optional[dict[int, list[str]]] = None,
    player_fp: Optional[dict[int, float]] = None,
    active_slots: Optional[Union[dict[str, int], list[dict]]] = None,
    playing_today: Optional[set[int]] = None,
    locked_assignments: Optional[dict[str, int]] = None,
    bench_slots: int = 4,
    ir_slots: int = 3,
    *,
    per_game_fp: Optional[dict[int, float]] = None,
    slot_configs: Optional[list[dict]] = None,
    position_eligibility: Optional[dict[int, list[str]]] = None,
    injured_players: Optional[set[int]] = None,
    position_structure: Optional[dict] = None,
) -> DailyLineupAssignment:
    """Assign players to active scoring slots to maximise total daily expected FP.

    Supports both current PROJECT.md interface contracts and legacy caller patterns.
    """
    # Detect legacy positional call:
    # assign_optimal_daily_lineup(roster, per_game_fp, slot_configs, position_eligibility, locked_assignments)
    if player_positions is not None and (
        isinstance(player_fp, list)
        or (isinstance(player_positions, dict) and player_positions and isinstance(next(iter(player_positions.values())), (int, float)))
    ):
        actual_fp = player_positions
        actual_slots = player_fp
        actual_pos = active_slots if isinstance(active_slots, dict) else position_eligibility
        actual_locked = playing_today if isinstance(playing_today, dict) else locked_assignments
        actual_playing = None
    else:
        actual_pos = player_positions if player_positions is not None else position_eligibility
        actual_fp = player_fp if player_fp is not None else per_game_fp
        actual_slots = active_slots if active_slots is not None else slot_configs
        actual_locked = locked_assignments
        actual_playing = playing_today

    fp_map = actual_fp or {}
    pos_map = actual_pos or {}

    effective_fp: dict[int, float] = {}
    non_playing: set[int] = set()
    for pid in roster_player_ids:
        if actual_playing is not None and pid not in actual_playing:
            effective_fp[pid] = 0.0
            non_playing.add(pid)
        else:
            val = fp_map.get(pid, np.nan)
            effective_fp[pid] = float(val) if (val is not None and np.isfinite(val)) else np.nan

    resolved_slots = _resolve_slot_configs(actual_slots, slot_configs, position_structure)

    return _solve_bipartite_lineup(
        roster_player_ids=roster_player_ids,
        player_positions=pos_map,
        effective_fp=effective_fp,
        slot_configs=resolved_slots,
        locked_assignments=actual_locked,
        bench_slots=bench_slots,
        ir_slots=ir_slots,
        injured_players=injured_players,
        raw_fp_map=fp_map,
        assignment_factory=lambda a, b, ir, tot: DailyLineupAssignment(a, b, ir, tot),
        non_playing=non_playing,
    )


def assign_optimal_weekly_lineup(
    roster_player_ids: list[int],
    player_positions: dict[int, list[str]],
    player_fp_per_game: dict[int, float],
    player_games_in_week: dict[int, int],
    active_slots: Union[dict[str, int], list[dict]],
    locked_assignments: Optional[dict[str, int]] = None,
    bench_slots: int = 4,
    ir_slots: int = 3,
    *,
    injured_players: Optional[set[int]] = None,
    position_structure: Optional[dict] = None,
) -> WeeklyLineupAssignment:
    """Assign players to active scoring slots to maximise total weekly expected FP (W_i = FP/G_i * Games_i)."""
    weekly_fp: dict[int, float] = {}
    non_playing: set[int] = set()
    for pid in roster_player_ids:
        fpg = player_fp_per_game.get(pid, np.nan)
        games = player_games_in_week.get(pid, 0)
        if games is None:
            games = 0
        if games <= 0:
            non_playing.add(pid)
        if fpg is not None and np.isfinite(fpg) and games > 0:
            weekly_fp[pid] = float(fpg * games)
        elif fpg is not None and np.isfinite(fpg) and games == 0:
            weekly_fp[pid] = 0.0
        else:
            weekly_fp[pid] = np.nan

    resolved_slots = _resolve_slot_configs(active_slots, None, position_structure)

    return _solve_bipartite_lineup(
        roster_player_ids=roster_player_ids,
        player_positions=player_positions,
        effective_fp=weekly_fp,
        slot_configs=resolved_slots,
        locked_assignments=locked_assignments,
        bench_slots=bench_slots,
        ir_slots=ir_slots,
        injured_players=injured_players,
        raw_fp_map=player_fp_per_game,
        assignment_factory=lambda a, b, ir, tot: WeeklyLineupAssignment(a, b, ir, tot, weekly_fp),
        non_playing=non_playing,
    )


def compute_lineup_fp(
    assignment: Union[dict[str, Optional[int]], DailyLineupAssignment, WeeklyLineupAssignment],
    per_game_fp: Optional[dict[int, float]] = None,
) -> float:
    """Sum projected FP over non-empty slot assignments."""
    if hasattr(assignment, 'total_expected_fp') and per_game_fp is None:
        return float(assignment.total_expected_fp)

    slot_dict = assignment.active_assignments if hasattr(assignment, 'active_assignments') else assignment
    fp_map = per_game_fp or {}
    total = 0.0
    for pid in slot_dict.values():
        if pid is not None:
            fp = fp_map.get(pid, np.nan)
            if fp is not None and np.isfinite(fp) and fp > 0:
                total += fp
    return total


# ── Internal Core Bipartite Solver ───────────────────────────────────────────

def _resolve_slot_configs(
    active_slots: Union[dict[str, int], list[dict], None],
    slot_configs: Optional[list[dict]],
    position_structure: Optional[dict],
) -> list[dict]:
    if slot_configs is not None:
        return slot_configs
    if isinstance(active_slots, list):
        return active_slots
    if isinstance(active_slots, dict):
        return build_slot_configs_from_position_config(active_slots, position_structure)
    return []


def _solve_bipartite_lineup(
    roster_player_ids: list[int],
    player_positions: dict[int, list[str]],
    effective_fp: dict[int, float],
    slot_configs: list[dict],
    locked_assignments: Optional[dict[str, int]],
    bench_slots: int,
    ir_slots: int,
    injured_players: Optional[set[int]],
    raw_fp_map: dict[int, float],
    assignment_factory,
    non_playing: Optional[set[int]] = None,
):
    # Validation: duplicate player IDs in roster
    if len(roster_player_ids) != len(set(roster_player_ids)):
        raise ValueError("Duplicate player IDs in roster")

    locked = dict(locked_assignments or {})

    # Validation: locked_assignments checks
    if locked:
        locked_pids = list(locked.values())
        if len(locked_pids) != len(set(locked_pids)):
            raise ValueError(f"Duplicate player IDs found in locked_assignments: {locked_pids}")
        roster_set = set(roster_player_ids)
        valid_slot_ids = {s['slot_id'] for s in slot_configs}
        slot_map = {s['slot_id']: s for s in slot_configs}
        for slot_id, pid in locked.items():
            if slot_id not in valid_slot_ids:
                raise ValueError(
                    f"Locked slot '{slot_id}' does not exist in active slots {sorted(valid_slot_ids)}"
                )
            if pid not in roster_set:
                raise ValueError(
                    f"Locked player {pid} in slot '{slot_id}' is not on roster {roster_player_ids}"
                )
            slot = slot_map[slot_id]
            p_pos = player_positions.get(pid, [])
            if not _is_eligible(p_pos, slot['eligible_positions']):
                raise ValueError(
                    f"Player {pid} with positions {p_pos} is not eligible "
                    f"for locked slot '{slot_id}' ({slot['eligible_positions']})"
                )

    locked_player_ids = set(locked.values())

    free_slots = [s for s in slot_configs if s['slot_id'] not in locked]
    free_players = [
        pid for pid in roster_player_ids
        if pid not in locked_player_ids
        and effective_fp.get(pid) is not None
        and np.isfinite(effective_fp.get(pid, np.nan))
    ]

    assignment: dict[str, Optional[int]] = dict(locked)
    np_set = non_playing or set()

    if free_slots and free_players:
        slot_ids = [s['slot_id'] for s in free_slots]
        slot_eligib = [s['eligible_positions'] for s in free_slots]
        n_slots = len(free_slots)
        n_players = len(free_players)
        size = max(n_slots, n_players)

        # Pad with 0.0 for dummy slots and dummy players
        cost = np.full((size, size), 0.0)

        for s_idx, eligible_positions in enumerate(slot_eligib):
            for p_idx, pid in enumerate(free_players):
                p_pos = player_positions.get(pid, [])
                if _is_eligible(p_pos, eligible_positions):
                    fp = effective_fp.get(pid, 0.0)
                    if fp is None or not np.isfinite(fp):
                        fp = 0.0
                    if pid in np_set:
                        # Non-playing player (off-day or 0 games): positive cost so dummy slots (0.0) are preferred
                        cost[s_idx, p_idx] = 1.0
                    elif fp > 0.0:
                        cost[s_idx, p_idx] = -fp - 1e-9
                    elif fp == 0.0:
                        # Healthy active player playing today with 0.0 projected FP
                        cost[s_idx, p_idx] = -1e-9
                    else:
                        # Negative FP player has positive cost penalty: solver prefers dummy (cost 0.0)
                        cost[s_idx, p_idx] = -fp
                else:
                    # Ineligible pairing has massive penalty
                    cost[s_idx, p_idx] = INELIGIBLE_COST

        row_ind, col_ind = linear_sum_assignment(cost)

        for s_idx, p_idx in zip(row_ind, col_ind):
            if s_idx < n_slots:
                slot_id = slot_ids[s_idx]
                if p_idx < n_players and cost[s_idx, p_idx] < INELIGIBLE_COST and cost[s_idx, p_idx] <= 0.0:
                    assignment[slot_id] = free_players[p_idx]
                else:
                    assignment[slot_id] = None

    # Ensure all configured slots are present in assignment
    for slot in slot_configs:
        assignment.setdefault(slot['slot_id'], None)

    # Compute total expected FP
    total_fp = 0.0
    for slot_id, pid in assignment.items():
        if pid is not None:
            fp = effective_fp.get(pid, 0.0)
            if fp is not None and np.isfinite(fp) and fp > 0:
                total_fp += fp

    # Partition unassigned roster players into IR and Bench
    active_starters = {pid for pid in assignment.values() if pid is not None}
    unassigned = [pid for pid in roster_player_ids if pid not in active_starters]

    if injured_players is not None:
        inj_set = set(injured_players)
    else:
        inj_set = {
            pid for pid in unassigned
            if raw_fp_map.get(pid) is None or not np.isfinite(raw_fp_map.get(pid, np.nan))
        }

    ir_candidates = [pid for pid in unassigned if pid in inj_set]
    ir_assigned = ir_candidates[:ir_slots]
    ir_set = set(ir_assigned)

    bench_assigned = [pid for pid in unassigned if pid not in ir_set]

    return assignment_factory(
        assignment,
        bench_assigned,
        ir_assigned,
        round(total_fp, 4),
    )
