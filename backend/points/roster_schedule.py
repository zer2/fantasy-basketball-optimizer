"""
Roster Schedule & Multi-Day Valuation Engine for Points Mode.

Simulates daily active starting lineups across multi-day evaluation windows using
ScheduleCalendar and assign_optimal_daily_lineup. Evaluates marginal starting lineup
fantasy points (Delta FP) gained by waiver wire acquisitions and trades, accurately
accounting for positional eligibility, daily active slot limits, schedule density,
and displacement of lower-scoring starters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Iterable, Optional, Union

import numpy as np

from backend.points.schedule import ScheduleCalendar, normalize_date
from backend.points.lineups import (
    DailyLineupAssignment,
    assign_optimal_daily_lineup,
)

logger = logging.getLogger('fbbo.points.roster_schedule')


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WaiverValuationResult:
    """Valuation result for a candidate add/drop transaction."""
    candidate_id: int
    drop_player_id: int
    delta_fp: float
    baseline_fp: float
    modified_fp: float
    candidate_games: int = 0
    dropped_games: int = 0
    daily_deltas: dict[str, float] = field(default_factory=dict)
    drop_penalty: float = 0.0
    net_delta_fp: float = 0.0

    # Dict-like mapping interface for backward compatibility
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict[str, Any]:
        return {
            'candidate_id': self.candidate_id,
            'drop_player_id': self.drop_player_id,
            'delta_fp': self.delta_fp,
            'baseline_fp': self.baseline_fp,
            'modified_fp': self.modified_fp,
            'candidate_games': self.candidate_games,
            'dropped_games': self.dropped_games,
            'daily_deltas': self.daily_deltas,
            'drop_penalty': self.drop_penalty,
            'net_delta_fp': self.net_delta_fp,
        }


# ── Multi-Day Lineup Simulation ──────────────────────────────────────────────

def compute_multi_day_lineup_fp(
    roster_player_ids: list[int],
    schedule: ScheduleCalendar,
    evaluation_window: list[str],
    player_positions: dict[int, list[str]],
    player_fp_per_game: dict[int, float],
    active_slots: dict[str, int],
    bench_slots: int = 4,
    ir_slots: int = 3,
    injured_players: Optional[set[int]] = None,
) -> tuple[float, dict[str, DailyLineupAssignment]]:
    """Simulate daily optimal lineups across evaluation_window dates.

    Args:
        roster_player_ids: List of player IDs on the roster.
        schedule: ScheduleCalendar providing team and player schedules.
        evaluation_window: List of ISO-8601 date strings.
        player_positions: Mapping of player_id to list of eligible positions.
        player_fp_per_game: Mapping of player_id to projected FP per game.
        active_slots: Mapping of slot names to capacity (e.g. {'PG': 1, 'SG': 1, 'Util': 2}).
        bench_slots: Number of bench slots.
        ir_slots: Number of IR slots.
        injured_players: Optional set of injured player IDs.

    Returns:
        tuple of (total_starting_fp, daily_assignments_dict)
    """
    if not evaluation_window:
        return 0.0, {}

    normalized_dates = schedule.normalize_window(evaluation_window)
    daily_assignments: dict[str, DailyLineupAssignment] = {}
    total_starting_fp = 0.0

    for date_str in normalized_dates:
        playing_today = schedule.get_players_playing_on_date(roster_player_ids, date_str)
        assignment = assign_optimal_daily_lineup(
            roster_player_ids=roster_player_ids,
            player_positions=player_positions,
            player_fp=player_fp_per_game,
            active_slots=active_slots,
            playing_today=playing_today,
            bench_slots=bench_slots,
            ir_slots=ir_slots,
            injured_players=injured_players,
        )
        daily_assignments[date_str] = assignment
        total_starting_fp += assignment.total_expected_fp

    return round(total_starting_fp, 4), daily_assignments


# ── Marginal Lineup Delta FP (F3 / R1 Contract) ──────────────────────────────

def compute_marginal_lineup_delta_fp(
    current_roster_ids: list[int],
    drop_player_id: int,
    add_player_id: int,
    schedule: ScheduleCalendar,
    evaluation_window: list[str],
    player_positions: dict[int, list[str]],
    player_fp_per_game: dict[int, float],
    active_slots: dict[str, int],
    bench_slots: int = 4,
    ir_slots: int = 3,
    injured_players: Optional[set[int]] = None,
) -> float:
    """Calculate the marginal starting lineup fantasy points gained over evaluation_window.

    Simulates daily active starting lineups across evaluation_window dates using ScheduleCalendar
    and assign_optimal_daily_lineup.

    Baseline starting FP = sum of daily optimal starting lineup FP with current_roster_ids.
    Modified starting FP = sum of daily optimal starting lineup FP with (current_roster_ids - {drop} + {add}).
    Delta FP = Modified starting FP - Baseline starting FP.

    Displaces lower-scoring starters rather than assuming raw projected points.
    Respects positional eligibility and daily active slot limits.

    Returns:
        Delta FP (positive = net gain, negative = net loss).
    """
    # Validation checks
    roster_set = set(current_roster_ids)
    if drop_player_id not in roster_set:
        raise ValueError(
            f"Drop player {drop_player_id} is not in current roster {current_roster_ids}"
        )
    if add_player_id in roster_set and add_player_id != drop_player_id:
        raise ValueError(
            f"Add player {add_player_id} is already in current roster {current_roster_ids}"
        )

    # Empty window or identity swap edge cases
    dates = schedule.normalize_window(evaluation_window)
    if not dates or drop_player_id == add_player_id:
        return 0.0

    # 1. Baseline starting FP
    baseline_fp, _ = compute_multi_day_lineup_fp(
        roster_player_ids=current_roster_ids,
        schedule=schedule,
        evaluation_window=dates,
        player_positions=player_positions,
        player_fp_per_game=player_fp_per_game,
        active_slots=active_slots,
        bench_slots=bench_slots,
        ir_slots=ir_slots,
        injured_players=injured_players,
    )

    # 2. Modified roster: replace drop_player_id with add_player_id
    modified_roster = [pid for pid in current_roster_ids if pid != drop_player_id]
    if add_player_id not in modified_roster:
        modified_roster.append(add_player_id)

    # 3. Modified starting FP
    modified_fp, _ = compute_multi_day_lineup_fp(
        roster_player_ids=modified_roster,
        schedule=schedule,
        evaluation_window=dates,
        player_positions=player_positions,
        player_fp_per_game=player_fp_per_game,
        active_slots=active_slots,
        bench_slots=bench_slots,
        ir_slots=ir_slots,
        injured_players=injured_players,
    )

    return round(modified_fp - baseline_fp, 4)


def compute_marginal_lineup_detailed(
    current_roster_ids: list[int],
    drop_player_id: int,
    add_player_id: int,
    schedule: ScheduleCalendar,
    evaluation_window: list[str],
    player_positions: dict[int, list[str]],
    player_fp_per_game: dict[int, float],
    active_slots: dict[str, int],
    bench_slots: int = 4,
    ir_slots: int = 3,
    injured_players: Optional[set[int]] = None,
) -> dict[str, Any]:
    """Detailed version of compute_marginal_lineup_delta_fp with per-day breakdowns."""
    roster_set = set(current_roster_ids)
    if drop_player_id not in roster_set:
        raise ValueError(
            f"Drop player {drop_player_id} is not in current roster {current_roster_ids}"
        )
    if add_player_id in roster_set and add_player_id != drop_player_id:
        raise ValueError(
            f"Add player {add_player_id} is already in current roster {current_roster_ids}"
        )

    dates = schedule.normalize_window(evaluation_window)
    if not dates:
        return {
            'delta_fp': 0.0,
            'baseline_fp': 0.0,
            'modified_fp': 0.0,
            'daily_deltas': {},
            'daily_baseline': {},
            'daily_modified': {},
            'candidate_games': 0,
            'dropped_games': 0,
        }

    baseline_fp, base_daily = compute_multi_day_lineup_fp(
        roster_player_ids=current_roster_ids,
        schedule=schedule,
        evaluation_window=dates,
        player_positions=player_positions,
        player_fp_per_game=player_fp_per_game,
        active_slots=active_slots,
        bench_slots=bench_slots,
        ir_slots=ir_slots,
        injured_players=injured_players,
    )

    modified_roster = [pid for pid in current_roster_ids if pid != drop_player_id]
    if add_player_id not in modified_roster:
        modified_roster.append(add_player_id)

    modified_fp, mod_daily = compute_multi_day_lineup_fp(
        roster_player_ids=modified_roster,
        schedule=schedule,
        evaluation_window=dates,
        player_positions=player_positions,
        player_fp_per_game=player_fp_per_game,
        active_slots=active_slots,
        bench_slots=bench_slots,
        ir_slots=ir_slots,
        injured_players=injured_players,
    )

    daily_deltas: dict[str, float] = {}
    daily_baseline: dict[str, float] = {}
    daily_modified: dict[str, float] = {}

    for d in dates:
        b_fp = base_daily[d].total_expected_fp
        m_fp = mod_daily[d].total_expected_fp
        daily_baseline[d] = b_fp
        daily_modified[d] = m_fp
        daily_deltas[d] = round(m_fp - b_fp, 4)

    start_date, end_date = dates[0], dates[-1]
    candidate_games = schedule.get_player_game_count(add_player_id, start_date, end_date)
    dropped_games = schedule.get_player_game_count(drop_player_id, start_date, end_date)

    return {
        'delta_fp': round(modified_fp - baseline_fp, 4),
        'baseline_fp': baseline_fp,
        'modified_fp': modified_fp,
        'daily_deltas': daily_deltas,
        'daily_baseline': daily_baseline,
        'daily_modified': daily_modified,
        'candidate_games': candidate_games,
        'dropped_games': dropped_games,
    }


# ── Waiver Wire Candidates Evaluator ────────────────────────────────────────

def evaluate_waiver_candidates(
    current_roster_ids: list[int],
    *args,
    **kwargs,
) -> list[WaiverValuationResult]:
    """Evaluate multiple waiver candidates and compute Delta FP for each.

    Flexible interface supporting both positional and keyword conventions:
      evaluate_waiver_candidates(roster, candidate_ids, drop_player_id, schedule, window, ...)
      evaluate_waiver_candidates(roster, drop_candidates, candidate_ids, schedule, window, ...)
      evaluate_waiver_candidates(roster, candidate_ids, schedule, window, ..., drop_player_id=...)

    Returns:
        List of WaiverValuationResult objects sorted by delta_fp descending.
    """
    # Parse flexible arguments
    candidate_ids: list[int] = []
    drop_candidates: list[int] = []
    schedule: Optional[ScheduleCalendar] = kwargs.get('schedule')
    evaluation_window: Optional[list[str]] = kwargs.get('evaluation_window')
    player_positions: dict[int, list[str]] = kwargs.get('player_positions', {})
    player_fp_per_game: dict[int, float] = kwargs.get('player_fp_per_game', {})
    active_slots: dict[str, int] = kwargs.get('active_slots', {})
    bench_slots: int = kwargs.get('bench_slots', 4)
    ir_slots: int = kwargs.get('ir_slots', 3)
    injured_players: Optional[set[int]] = kwargs.get('injured_players')
    top_n: Optional[int] = kwargs.get('top_n')

    # Resolve drop player(s) from kwargs if present
    kw_drop = (
        kwargs.get('drop_player_id')
        or kwargs.get('drop_player_ids')
        or kwargs.get('drop_candidate_ids')
        or kwargs.get('drop_candidates')
    )
    if kw_drop is not None:
        if isinstance(kw_drop, (list, tuple, set)):
            drop_candidates = [int(p) for p in kw_drop]
        else:
            drop_candidates = [int(kw_drop)]

    # Disambiguate positional arguments
    # Pattern A: (roster, candidate_ids, drop_candidates, schedule, evaluation_window, ...)
    # Pattern B: (roster, drop_candidates, candidate_ids, schedule, evaluation_window, ...)
    # Pattern C: (roster, candidate_ids, schedule, evaluation_window, ...)
    remaining_args = list(args)

    if remaining_args:
        first = remaining_args.pop(0)
        if isinstance(first, ScheduleCalendar):
            schedule = first
        elif isinstance(first, (int, np.integer)):
            drop_candidates = [int(first)]
        elif isinstance(first, (list, tuple, set)):
            # Could be candidate_ids or drop_candidates
            if remaining_args and isinstance(remaining_args[0], ScheduleCalendar):
                candidate_ids = [int(p) for p in first]
                schedule = remaining_args.pop(0)
            elif remaining_args and isinstance(remaining_args[0], (list, tuple, set, int, np.integer)):
                second = remaining_args.pop(0)
                # Check which one contains roster players
                first_list = [int(p) for p in first]
                second_list = [int(second)] if isinstance(second, (int, np.integer)) else [int(p) for p in second]
                # If first is subset of roster, it is drop_candidates
                if set(first_list).issubset(set(current_roster_ids)):
                    drop_candidates = first_list
                    candidate_ids = second_list
                else:
                    candidate_ids = first_list
                    drop_candidates = second_list
            else:
                candidate_ids = [int(p) for p in first]

    if remaining_args and schedule is None:
        if isinstance(remaining_args[0], ScheduleCalendar):
            schedule = remaining_args.pop(0)

    if remaining_args and evaluation_window is None:
        evaluation_window = remaining_args.pop(0)

    if remaining_args and not player_positions:
        player_positions = remaining_args.pop(0)

    if remaining_args and not player_fp_per_game:
        player_fp_per_game = remaining_args.pop(0)

    if remaining_args and not active_slots:
        active_slots = remaining_args.pop(0)

    # Fallbacks and validation
    if schedule is None:
        schedule = ScheduleCalendar()

    dates = schedule.normalize_window(evaluation_window)

    if not candidate_ids and 'candidate_ids' in kwargs:
        candidate_ids = [int(p) for p in kwargs['candidate_ids']]
    if not candidate_ids and 'available_waiver_ids' in kwargs:
        candidate_ids = [int(p) for p in kwargs['available_waiver_ids']]

    # If no drop candidates provided, consider all current roster players
    if not drop_candidates:
        drop_candidates = list(current_roster_ids)

    # Baseline calculation across the window (calculated once for efficiency)
    baseline_fp, base_daily = compute_multi_day_lineup_fp(
        roster_player_ids=current_roster_ids,
        schedule=schedule,
        evaluation_window=dates,
        player_positions=player_positions,
        player_fp_per_game=player_fp_per_game,
        active_slots=active_slots,
        bench_slots=bench_slots,
        ir_slots=ir_slots,
        injured_players=injured_players,
    )

    # Drop penalty calculator integration from fantasy_streamer
    drop_penalty_calculator = kwargs.get('drop_penalty_calculator')
    consider_drop_penalties = kwargs.get('consider_drop_penalties', False)
    if consider_drop_penalties and drop_penalty_calculator is None:
        try:
            from fantasy_streamer.models.penalties import DropPenaltyCalculator
            from fantasy_streamer.models.projections import compute_replacement_baseline, Player
            baseline = 22.5
            if candidate_ids and player_fp_per_game:
                fa_players = [
                    Player(
                        id=c,
                        name=f"Player_{c}",
                        team="",
                        positions=player_positions.get(c, ["UTIL"]),
                        fppg=float(player_fp_per_game.get(c, 0.0)),
                    )
                    for c in candidate_ids
                ]
                baseline = compute_replacement_baseline(fa_players, top_k=min(10, len(fa_players)))
            drop_penalty_calculator = DropPenaltyCalculator(replacement_baseline=baseline)
        except Exception as exc:
            logger.debug("Failed to initialize DropPenaltyCalculator from fantasy_streamer: %s", exc)
            drop_penalty_calculator = None

    results: list[WaiverValuationResult] = []
    roster_set = set(current_roster_ids)
    start_date = dates[0] if dates else ""
    end_date = dates[-1] if dates else ""

    # Evaluate each candidate
    for cand_id in candidate_ids:
        if cand_id in roster_set:
            continue  # Already rostered

        best_result: Optional[WaiverValuationResult] = None

        # Compare against each drop candidate and keep the optimal drop pair
        for drop_id in drop_candidates:
            if drop_id not in roster_set:
                continue

            mod_roster = [p for p in current_roster_ids if p != drop_id] + [cand_id]

            mod_fp, mod_daily = compute_multi_day_lineup_fp(
                roster_player_ids=mod_roster,
                schedule=schedule,
                evaluation_window=dates,
                player_positions=player_positions,
                player_fp_per_game=player_fp_per_game,
                active_slots=active_slots,
                bench_slots=bench_slots,
                ir_slots=ir_slots,
                injured_players=injured_players,
            )

            delta_fp = round(mod_fp - baseline_fp, 4)

            daily_deltas = {}
            for d in dates:
                daily_deltas[d] = round(
                    mod_daily[d].total_expected_fp - base_daily[d].total_expected_fp, 4
                )

            cand_games = schedule.get_player_game_count(cand_id, start_date, end_date) if dates else 0
            drop_games = schedule.get_player_game_count(drop_id, start_date, end_date) if dates else 0

            # Calculate drop penalty if enabled
            drop_penalty = 0.0
            if drop_penalty_calculator is not None:
                drop_fppg = float(player_fp_per_game.get(drop_id, 0.0))
                weeks_in_window = max(1.0, len(dates) / 7.0)
                drop_penalty = round(
                    float(drop_penalty_calculator.calculate_penalty(
                        fppg=drop_fppg,
                        games_per_week=drop_games / weeks_in_window,
                    )),
                    4,
                )

            net_delta_fp = round(delta_fp - drop_penalty, 4)

            cand_res = WaiverValuationResult(
                candidate_id=cand_id,
                drop_player_id=drop_id,
                delta_fp=delta_fp,
                baseline_fp=baseline_fp,
                modified_fp=mod_fp,
                candidate_games=cand_games,
                dropped_games=drop_games,
                daily_deltas=daily_deltas,
                drop_penalty=drop_penalty,
                net_delta_fp=net_delta_fp,
            )

            score = cand_res.net_delta_fp if consider_drop_penalties else cand_res.delta_fp
            best_score = (best_result.net_delta_fp if consider_drop_penalties else best_result.delta_fp) if best_result else -float('inf')

            if best_result is None or score > best_score:
                best_result = cand_res

        if best_result is not None:
            results.append(best_result)

    # Sort results by net_delta_fp or delta_fp descending
    if consider_drop_penalties:
        results.sort(key=lambda r: (r.net_delta_fp, r.delta_fp), reverse=True)
    else:
        results.sort(key=lambda r: r.delta_fp, reverse=True)

    if top_n is not None and top_n > 0:
        return results[:top_n]

    return results
