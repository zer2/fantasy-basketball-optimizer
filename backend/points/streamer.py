"""
Streamer Optimization Engine for Points Leagues.

Determines the optimal streaming acquisition plan to maximize total starting fantasy
points and points per add under weekly acquisition limits (e.g. ESPN 7 adds).

Provides:
  - Schedule density analysis: prioritizes off-nights / light slates where active
    starter slots are not crowded, ensuring streamed games actually start and displace nobody.
  - Daily single-slot churn: rotating a dedicated streaming slot every day (e.g. 7 adds for 7 games).
  - Multi-slot tactical adds: targeting back-to-backs (2 games in 2 days) or 3 games in 4 nights
    across 1 or 2+ streaming slots to maximize games played while conserving acquisition budget.
  - Mathematical optimization between churn and tactical multi-slot adds via dynamic programming.
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

logger = logging.getLogger('fbbo.points.streamer')


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class StreamingMove:
    """A planned streamer acquisition and start."""
    date: str
    add_player_id: int
    drop_player_id: Optional[int]
    games_gained: int
    projected_fp: float
    slot_index: int = 1
    dates_active: list[str] = field(default_factory=list)
    dates_played: list[str] = field(default_factory=list)

    # Property aliases for flexibility
    @property
    def day(self) -> str:
        return self.date

    @property
    def player_id(self) -> int:
        return self.add_player_id

    @property
    def fp(self) -> float:
        return self.projected_fp

    @property
    def games(self) -> int:
        return self.games_gained

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict[str, Any]:
        return {
            'date': self.date,
            'day': self.date,
            'add_player_id': self.add_player_id,
            'player_id': self.add_player_id,
            'drop_player_id': self.drop_player_id,
            'games_gained': self.games_gained,
            'games': self.games_gained,
            'projected_fp': self.projected_fp,
            'fp': self.projected_fp,
            'slot_index': self.slot_index,
            'dates_active': list(self.dates_active),
            'dates_played': list(self.dates_played),
        }


@dataclass
class StreamerSchedulePlan:
    """Schedule plan representing a sequence of streaming acquisitions."""
    strategy: str
    moves: list[StreamingMove]
    total_acquisitions: int
    total_games_streamed: int
    total_projected_fp: float
    points_per_add: float
    net_delta_fp: float

    @property
    def acquisitions(self) -> int:
        return self.total_acquisitions

    @property
    def games(self) -> int:
        return self.total_games_streamed

    @property
    def points(self) -> float:
        return self.total_projected_fp

    @property
    def delta_fp(self) -> float:
        return self.net_delta_fp

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict[str, Any]:
        return {
            'strategy': self.strategy,
            'moves': [m.to_dict() for m in self.moves],
            'total_acquisitions': self.total_acquisitions,
            'total_games_streamed': self.total_games_streamed,
            'total_projected_fp': self.total_projected_fp,
            'points_per_add': self.points_per_add,
            'net_delta_fp': self.net_delta_fp,
        }


@dataclass
class StreamerOptimizationResult:
    """Overall result of the streamer optimization engine."""
    optimal_plan: StreamerSchedulePlan
    mode: str
    total_acquisitions_used: int
    total_streamed_games: int
    total_net_delta_fp: float
    points_per_add: float
    alternative_plans: dict[str, StreamerSchedulePlan] = field(default_factory=dict)
    daily_density_analysis: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def acquisitions_used(self) -> int:
        return self.total_acquisitions_used

    @property
    def games_streamed(self) -> int:
        return self.total_streamed_games

    @property
    def net_delta_fp(self) -> float:
        return self.total_net_delta_fp

    @property
    def plan(self) -> StreamerSchedulePlan:
        return self.optimal_plan

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def items(self):
        return self.to_dict().items()

    def keys(self):
        return self.to_dict().keys()

    def values(self):
        return self.to_dict().values()

    def __iter__(self):
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            'optimal_plan': self.optimal_plan.to_dict(),
            'mode': self.mode,
            'total_acquisitions_used': self.total_acquisitions_used,
            'total_streamed_games': self.total_streamed_games,
            'total_net_delta_fp': self.total_net_delta_fp,
            'points_per_add': self.points_per_add,
            'alternative_plans': {k: p.to_dict() for k, p in self.alternative_plans.items()},
            'daily_density_analysis': self.daily_density_analysis,
        }


# ── Schedule Density Analyzer ────────────────────────────────────────────────

def analyze_schedule_density(
    current_roster_ids: list[int],
    schedule: ScheduleCalendar,
    evaluation_window: list[str],
    player_positions: dict[int, list[str]],
    player_fp_per_game: dict[int, float],
    active_slots: dict[str, int],
    bench_slots: int = 4,
    ir_slots: int = 3,
) -> dict[str, dict[str, Any]]:
    """Analyze schedule slate density and active slot capacity per day in window."""
    dates = schedule.normalize_window(evaluation_window)
    density_map: dict[str, dict[str, Any]] = {}
    total_active_slots_count = sum(active_slots.values())

    for d in dates:
        nba_teams_today = schedule.get_teams_playing_on_date(d)
        nba_games_count = len(nba_teams_today) // 2

        roster_playing = schedule.get_players_playing_on_date(current_roster_ids, d)
        daily_assign = assign_optimal_daily_lineup(
            roster_player_ids=current_roster_ids,
            player_positions=player_positions,
            player_fp=player_fp_per_game,
            active_slots=active_slots,
            playing_today=roster_playing,
            bench_slots=bench_slots,
            ir_slots=ir_slots,
        )

        starters = [pid for pid in daily_assign.active_assignments.values() if pid is not None]
        active_count = len(starters)
        open_slots_count = max(0, total_active_slots_count - active_count)

        # Off-night / light slate if <= 5 games or open active slots
        is_light_slate = bool(nba_games_count <= 5 or open_slots_count > 0)

        density_map[d] = {
            'date': d,
            'nba_games_count': nba_games_count,
            'nba_teams_count': len(nba_teams_today),
            'is_light_slate': is_light_slate,
            'roster_playing_count': len(roster_playing),
            'active_starters_count': active_count,
            'open_slots_count': open_slots_count,
            'baseline_starting_fp': daily_assign.total_expected_fp,
        }

    return density_map


# ── Core Streaming Optimization Solver ───────────────────────────────────────

def optimize_streaming_acquisitions(
    current_roster_ids: list[int],
    available_waiver_ids: list[int],
    schedule: ScheduleCalendar,
    evaluation_window: list[str],
    player_positions: dict[int, list[str]],
    player_fp_per_game: dict[int, float],
    active_slots: dict[str, int],
    max_acquisitions: int = 7,
    mode: str = "auto",  # "daily_churn", "tactical_multi", "auto"
    streaming_slots: int = 1,
    drop_candidate_ids: Optional[list[int]] = None,
    bench_slots: int = 4,
    ir_slots: int = 3,
    solver_backend: str = "ortools",
    **kwargs: Any,
) -> StreamerOptimizationResult:
    """Optimize streaming acquisitions over an evaluation window.

    Determines maximum points per add and optimal schedule plan under weekly acquisition caps.

    Optimizes between:
      a) Daily single-slot churn: rotating a dedicated streaming slot every day (e.g. 7 adds for 7 games).
      b) Multi-slot tactical adds: targeting back-to-backs (2 games in 2 days) or 3 games in 4 nights
         across 1 or 2+ streaming slots to maximize games played and points while conserving acquisitions.
      c) Schedule density analysis: prioritizes off-nights / light slates where active starter slots
         are not crowded, ensuring streamed games actually start and displace nobody.

    Args:
        current_roster_ids: List of roster player IDs.
        available_waiver_ids: Free agent / streamer candidates available.
        schedule: ScheduleCalendar instance.
        evaluation_window: List of dates or tuple range.
        player_positions: Dict mapping player_id to list of positions.
        player_fp_per_game: Dict mapping player_id to FP/G.
        active_slots: Dict defining slot capacities.
        max_acquisitions: Weekly transaction limit (e.g. ESPN 7).
        mode: "daily_churn", "tactical_multi", or "auto".
        streaming_slots: Number of roster slots dedicated to streaming (1 or 2).
        drop_candidate_ids: Optional list of player IDs eligible to be dropped.
        bench_slots: Number of bench slots.
        ir_slots: Number of IR slots.

    Returns:
        StreamerOptimizationResult with optimal plan, alternative plans, and density metrics.
    """
    dates = schedule.normalize_window(evaluation_window)

    # Empty window or 0 acquisitions boundary case
    if not dates or max_acquisitions <= 0 or not available_waiver_ids:
        empty_plan = StreamerSchedulePlan(
            strategy=mode,
            moves=[],
            total_acquisitions=0,
            total_games_streamed=0,
            total_projected_fp=0.0,
            points_per_add=0.0,
            net_delta_fp=0.0,
        )
        density = analyze_schedule_density(
            current_roster_ids=current_roster_ids,
            schedule=schedule,
            evaluation_window=dates,
            player_positions=player_positions,
            player_fp_per_game=player_fp_per_game,
            active_slots=active_slots,
            bench_slots=bench_slots,
            ir_slots=ir_slots,
        )
        return StreamerOptimizationResult(
            optimal_plan=empty_plan,
            mode=mode,
            total_acquisitions_used=0,
            total_streamed_games=0,
            total_net_delta_fp=0.0,
            points_per_add=0.0,
            alternative_plans={mode: empty_plan},
            daily_density_analysis=density,
        )

    # 1. Schedule Density Analysis & Baseline Lineup Performance
    daily_density = analyze_schedule_density(
        current_roster_ids=current_roster_ids,
        schedule=schedule,
        evaluation_window=dates,
        player_positions=player_positions,
        player_fp_per_game=player_fp_per_game,
        active_slots=active_slots,
        bench_slots=bench_slots,
        ir_slots=ir_slots,
    )

    total_baseline_starting_fp = sum(d['baseline_starting_fp'] for d in daily_density.values())

    # 2. Determine Designated Drop Player(s) for the Streaming Slot(s)
    designated_drops: list[int] = []
    if drop_candidate_ids:
        valid_drops = [pid for pid in drop_candidate_ids if pid in current_roster_ids]
        if valid_drops:
            contributions = []
            for d_pid in valid_drops:
                mod_roster = [p for p in current_roster_ids if p != d_pid]
                tot = 0.0
                for d in dates:
                    pl = schedule.get_players_playing_on_date(mod_roster, d)
                    asgn = assign_optimal_daily_lineup(
                        roster_player_ids=mod_roster,
                        player_positions=player_positions,
                        player_fp=player_fp_per_game,
                        active_slots=active_slots,
                        playing_today=pl,
                        bench_slots=bench_slots,
                        ir_slots=ir_slots,
                    )
                    tot += asgn.total_expected_fp
                contributions.append((tot, d_pid))
            # Sort so that highest total when dropped (meaning lowest loss) comes first
            contributions.sort(key=lambda x: x[0], reverse=True)
            designated_drops = [x[1] for x in contributions[:streaming_slots]]

    if not designated_drops and current_roster_ids:
        player_losses = []
        for d_pid in current_roster_ids:
            mod_roster = [p for p in current_roster_ids if p != d_pid]
            tot = 0.0
            for d in dates:
                pl = schedule.get_players_playing_on_date(mod_roster, d)
                asgn = assign_optimal_daily_lineup(
                    roster_player_ids=mod_roster,
                    player_positions=player_positions,
                    player_fp=player_fp_per_game,
                    active_slots=active_slots,
                    playing_today=pl,
                    bench_slots=bench_slots,
                    ir_slots=ir_slots,
                )
                tot += asgn.total_expected_fp
            loss = total_baseline_starting_fp - tot
            player_losses.append((loss, d_pid))
        player_losses.sort(key=lambda x: x[0])
        designated_drops = [x[1] for x in player_losses[:streaming_slots]]

    primary_drop_id = designated_drops[0] if designated_drops else None
    core_roster = [p for p in current_roster_ids if p not in designated_drops]

    # Pre-calculate baseline starting FP of core roster on each day
    core_daily_fp: dict[str, float] = {}
    for d in dates:
        pl = schedule.get_players_playing_on_date(core_roster, d)
        asgn = assign_optimal_daily_lineup(
            roster_player_ids=core_roster,
            player_positions=player_positions,
            player_fp=player_fp_per_game,
            active_slots=active_slots,
            playing_today=pl,
            bench_slots=bench_slots,
            ir_slots=ir_slots,
        )
        core_daily_fp[d] = asgn.total_expected_fp

    # 3. Filter and Index Waiver Candidates
    roster_set = set(current_roster_ids)
    viable_candidates = [
        int(pid) for pid in available_waiver_ids
        if pid not in roster_set
        and player_fp_per_game.get(pid, 0.0) is not None
        and np.isfinite(player_fp_per_game.get(pid, 0.0))
        and player_fp_per_game.get(pid, 0.0) > 0.0
    ]

    candidate_game_dates: dict[int, set[str]] = {}
    date_set = set(dates)
    start_date, end_date = dates[0], dates[-1]
    for cid in viable_candidates:
        g_dates = set(schedule.get_player_game_dates(cid, start_date, end_date)) & date_set
        if g_dates:
            candidate_game_dates[cid] = g_dates

    # 4. Daily Marginal Gain Helper
    def evaluate_candidate_daily_gain(cid: int, d: str, base_roster: list[int]) -> float:
        if d not in candidate_game_dates.get(cid, set()):
            return 0.0
        test_roster = base_roster + [cid]
        pl = schedule.get_players_playing_on_date(test_roster, d)
        asgn = assign_optimal_daily_lineup(
            roster_player_ids=test_roster,
            player_positions=player_positions,
            player_fp=player_fp_per_game,
            active_slots=active_slots,
            playing_today=pl,
            bench_slots=bench_slots,
            ir_slots=ir_slots,
        )
        if cid not in asgn.active_assignments.values():
            return 0.0
        base_fp = core_daily_fp.get(d, 0.0)
        return max(0.0, round(asgn.total_expected_fp - base_fp, 4))

    # ── Strategy A: Daily Single-Slot Churn Optimizer ────────────────────────
    daily_churn_moves: list[StreamingMove] = []
    daily_best_options: list[tuple[str, int, float]] = []

    for d in dates:
        best_cid: Optional[int] = None
        best_gain: float = 0.0
        for cid in candidate_game_dates:
            gain = evaluate_candidate_daily_gain(cid, d, core_roster)
            if gain > best_gain:
                best_gain = gain
                best_cid = cid

        if best_cid is not None and best_gain > 0.0:
            daily_best_options.append((d, best_cid, best_gain))

    # Prioritize highest gain days up to max_acquisitions
    daily_best_options.sort(key=lambda x: x[2], reverse=True)
    selected_churn_days = sorted(daily_best_options[:max_acquisitions], key=lambda x: x[0])

    churn_total_fp = 0.0
    for d, cid, gain in selected_churn_days:
        daily_churn_moves.append(StreamingMove(
            date=d,
            add_player_id=cid,
            drop_player_id=primary_drop_id,
            games_gained=1,
            projected_fp=gain,
            slot_index=1,
            dates_active=[d],
            dates_played=[d],
        ))
        churn_total_fp += gain

    churn_adds = len(daily_churn_moves)
    churn_games = churn_adds
    churn_ppa = round(churn_total_fp / churn_adds, 4) if churn_adds > 0 else 0.0

    streamed_days_map = {m.date: m.add_player_id for m in daily_churn_moves}
    churn_window_starting_fp = 0.0
    for d in dates:
        if d in streamed_days_map:
            cid = streamed_days_map[d]
            pl = schedule.get_players_playing_on_date(core_roster + [cid], d)
            asgn = assign_optimal_daily_lineup(
                roster_player_ids=core_roster + [cid],
                player_positions=player_positions,
                player_fp=player_fp_per_game,
                active_slots=active_slots,
                playing_today=pl,
                bench_slots=bench_slots,
                ir_slots=ir_slots,
            )
            churn_window_starting_fp += asgn.total_expected_fp
        else:
            churn_window_starting_fp += core_daily_fp[d]

    churn_net_delta_fp = round(churn_window_starting_fp - total_baseline_starting_fp, 4)

    daily_churn_plan = StreamerSchedulePlan(
        strategy="daily_churn",
        moves=daily_churn_moves,
        total_acquisitions=churn_adds,
        total_games_streamed=churn_games,
        total_projected_fp=round(churn_total_fp, 4),
        points_per_add=churn_ppa,
        net_delta_fp=churn_net_delta_fp,
    )

    # ── Strategy B: Multi-Slot Tactical Adds Optimizer ───────────────────────

    def build_tactical_stints_for_roster(
        base_roster: list[int],
        allowed_candidates: list[int],
        slot_idx: int,
        drop_id: Optional[int],
    ) -> list[dict[str, Any]]:
        stints: list[dict[str, Any]] = []

        for cid in allowed_candidates:
            c_dates = candidate_game_dates.get(cid, set())
            if not c_dates:
                continue

            c_date_indices = sorted([i for i, d in enumerate(dates) if d in c_dates])
            if not c_date_indices:
                continue

            m = len(c_date_indices)
            for i_idx in range(m):
                start_i = c_date_indices[i_idx]
                start_d = dates[start_i]

                for j_idx in range(i_idx, m):
                    end_i = c_date_indices[j_idx]
                    end_d = dates[end_i]

                    stint_dates = dates[start_i: end_i + 1]
                    games_started = 0
                    stint_fp = 0.0
                    played_dates = []

                    for d in stint_dates:
                        gain = evaluate_candidate_daily_gain(cid, d, base_roster)
                        if gain > 0.0 and d in c_dates:
                            games_started += 1
                            stint_fp += gain
                            played_dates.append(d)

                    if games_started > 0 and stint_fp > 0.0:
                        stints.append({
                            'cid': cid,
                            'start_idx': start_i,
                            'end_idx': end_i,
                            'start_date': start_d,
                            'end_date': end_d,
                            'games_gained': games_started,
                            'projected_fp': round(stint_fp, 4),
                            'dates_active': stint_dates,
                            'dates_played': played_dates,
                            'slot_index': slot_idx,
                            'drop_player_id': drop_id,
                        })
        return stints

    def solve_interval_dp(
        stints: list[dict[str, Any]],
        max_adds: int,
    ) -> list[StreamingMove]:
        if not stints or max_adds <= 0:
            return []

        stints_by_start: dict[int, list[dict[str, Any]]] = {}
        for st in stints:
            stints_by_start.setdefault(st['start_idx'], []).append(st)

        n_days = len(dates)
        # memo: (day_idx, adds_left) -> (max_fp, adds_used, games_count, choice)
        memo: dict[tuple[int, int], tuple[float, int, int, Any]] = {}

        def dp(day_idx: int, adds_left: int) -> tuple[float, int, int, Any]:
            if day_idx >= n_days or adds_left <= 0:
                return (0.0, 0, 0, None)

            state = (day_idx, adds_left)
            if state in memo:
                return memo[state]

            # Option 1: Skip day_idx
            best_val, best_adds, best_games, best_choice = dp(day_idx + 1, adds_left)

            # Option 2: Take a stint starting at day_idx
            for st in stints_by_start.get(day_idx, []):
                next_day = st['end_idx'] + 1
                future_val, future_adds, future_games, _ = dp(next_day, adds_left - 1)
                total_val = st['projected_fp'] + future_val
                total_adds = 1 + future_adds
                total_games = st['games_gained'] + future_games

                is_better = False
                if total_val > best_val + 1e-6:
                    is_better = True
                elif abs(total_val - best_val) <= 1e-6:
                    # Tiebreak 1: Fewer acquisitions used (higher points per add)
                    if total_adds < best_adds:
                        is_better = True
                    elif total_adds == best_adds and total_games > best_games:
                        # Tiebreak 2: More games played
                        is_better = True

                if is_better:
                    best_val = total_val
                    best_adds = total_adds
                    best_games = total_games
                    best_choice = st

            memo[state] = (round(best_val, 4), best_adds, best_games, best_choice)
            return memo[state]

        dp(0, max_adds)

        curr_day = 0
        curr_adds = max_adds
        chosen_moves: list[StreamingMove] = []

        while curr_day < n_days and curr_adds > 0:
            val, adds_used, games_count, choice = memo.get((curr_day, curr_adds), (0.0, 0, 0, None))
            if choice is None:
                curr_day += 1
            else:
                st = choice
                chosen_moves.append(StreamingMove(
                    date=st['start_date'],
                    add_player_id=st['cid'],
                    drop_player_id=st['drop_player_id'],
                    games_gained=st['games_gained'],
                    projected_fp=st['projected_fp'],
                    slot_index=st['slot_index'],
                    dates_active=st['dates_active'],
                    dates_played=st['dates_played'],
                ))
                curr_day = st['end_idx'] + 1
                curr_adds -= 1

        return chosen_moves

    # Solve tactical moves across 1 or 2 streaming slots
    best_tactical_moves: list[StreamingMove] = []
    best_tactical_fp = -1.0
    best_tactical_games = -1

    all_cids = list(candidate_game_dates.keys())

    if streaming_slots >= 2 and len(designated_drops) >= 2 and max_acquisitions >= 2:
        # Search over acquisition budget splits between Slot 1 and Slot 2
        drop1, drop2 = designated_drops[0], designated_drops[1]
        stints_slot1 = build_tactical_stints_for_roster(core_roster, all_cids, slot_idx=1, drop_id=drop1)

        splits = [(a1, max_acquisitions - a1) for a1 in range(1, max_acquisitions)]
        # Also consider single slot with all adds
        splits.append((max_acquisitions, 0))

        for a1, a2 in splits:
            m1 = solve_interval_dp(stints_slot1, a1)
            if a2 > 0:
                used_cids = {m.add_player_id for m in m1}
                rem_cids = [c for c in all_cids if c not in used_cids]
                stints_slot2 = build_tactical_stints_for_roster(core_roster, rem_cids, slot_idx=2, drop_id=drop2)
                m2 = solve_interval_dp(stints_slot2, a2)
            else:
                m2 = []

            comb_moves = m1 + m2
            comb_fp = sum(m.projected_fp for m in comb_moves)
            comb_games = sum(m.games_gained for m in comb_moves)

            is_better = False
            if comb_fp > best_tactical_fp + 1e-6:
                is_better = True
            elif abs(comb_fp - best_tactical_fp) <= 1e-6 and comb_games > best_tactical_games:
                is_better = True
            elif abs(comb_fp - best_tactical_fp) <= 1e-6 and comb_games == best_tactical_games:
                if len(comb_moves) < len(best_tactical_moves):
                    is_better = True

            if is_better:
                best_tactical_fp = comb_fp
                best_tactical_games = comb_games
                best_tactical_moves = sorted(comb_moves, key=lambda m: m.date)
    else:
        stints_slot1 = build_tactical_stints_for_roster(core_roster, all_cids, slot_idx=1, drop_id=primary_drop_id)
        best_tactical_moves = solve_interval_dp(stints_slot1, max_acquisitions)

    tactical_moves = best_tactical_moves
    tactical_total_fp = sum(m.projected_fp for m in tactical_moves)
    tactical_adds = len(tactical_moves)
    tactical_games = sum(m.games_gained for m in tactical_moves)
    tactical_ppa = round(tactical_total_fp / tactical_adds, 4) if tactical_adds > 0 else 0.0

    # Calculate actual net delta FP over baseline for tactical plan
    tactical_active_by_day: dict[str, list[int]] = {d: [] for d in dates}
    for m in tactical_moves:
        for d in m.dates_active:
            if d in tactical_active_by_day:
                tactical_active_by_day[d].append(m.add_player_id)

    tactical_window_starting_fp = 0.0
    for d in dates:
        active_streamers = tactical_active_by_day[d]
        if active_streamers:
            test_roster = core_roster + active_streamers
            pl = schedule.get_players_playing_on_date(test_roster, d)
            asgn = assign_optimal_daily_lineup(
                roster_player_ids=test_roster,
                player_positions=player_positions,
                player_fp=player_fp_per_game,
                active_slots=active_slots,
                playing_today=pl,
                bench_slots=bench_slots,
                ir_slots=ir_slots,
            )
            tactical_window_starting_fp += asgn.total_expected_fp
        else:
            tactical_window_starting_fp += core_daily_fp[d]

    tactical_net_delta_fp = round(tactical_window_starting_fp - total_baseline_starting_fp, 4)

    tactical_multi_plan = StreamerSchedulePlan(
        strategy="tactical_multi",
        moves=tactical_moves,
        total_acquisitions=tactical_adds,
        total_games_streamed=tactical_games,
        total_projected_fp=round(tactical_total_fp, 4),
        points_per_add=tactical_ppa,
        net_delta_fp=tactical_net_delta_fp,
    )

    alternative_plans: dict[str, StreamerSchedulePlan] = {
        'daily_churn': daily_churn_plan,
        'tactical_multi': tactical_multi_plan,
    }

    # ── 5. Attempt MILP Solve via fantasy_streamer Integration ──────────────
    milp_plan: Optional[StreamerSchedulePlan] = None
    if mode in ("milp", "auto"):
        try:
            milp_res = optimize_streaming_milp(
                current_roster_ids=current_roster_ids,
                available_waiver_ids=available_waiver_ids,
                schedule=schedule,
                evaluation_window=dates,
                player_positions=player_positions,
                player_fp_per_game=player_fp_per_game,
                active_slots=active_slots,
                max_acquisitions=max_acquisitions,
                designated_drops=designated_drops,
                bench_slots=bench_slots,
                ir_slots=ir_slots,
                candidate_game_dates=candidate_game_dates,
                solver_backend=solver_backend,
            )
            milp_plan = milp_res.optimal_plan
            alternative_plans['milp'] = milp_plan
        except Exception as exc:
            logger.debug("fantasy_streamer MILP solve failed or unavailable: %s", exc)

    # ── 6. Mode Resolution & Plan Selection ──────────────────────────────────
    if mode == "milp":
        optimal_plan = milp_plan if milp_plan is not None else tactical_multi_plan
    elif mode == "daily_churn":
        optimal_plan = daily_churn_plan
    elif mode == "tactical_multi":
        optimal_plan = tactical_multi_plan
    else:  # mode == "auto"
        # Compare available plans (milp, tactical_multi, daily_churn)
        candidate_plans = [p for p in [milp_plan, tactical_multi_plan, daily_churn_plan] if p is not None]
        positive_plans = [p for p in candidate_plans if p.total_acquisitions > 0 and p.net_delta_fp > 0.0]
        if positive_plans:
            positive_plans.sort(key=lambda p: (p.net_delta_fp, p.points_per_add), reverse=True)
            optimal_plan = positive_plans[0]
        elif drop_candidate_ids:
            plans_with_moves = [p for p in candidate_plans if p.total_acquisitions > 0]
            if plans_with_moves:
                plans_with_moves.sort(key=lambda p: (p.net_delta_fp, p.total_projected_fp), reverse=True)
                optimal_plan = plans_with_moves[0]
            else:
                optimal_plan = daily_churn_plan
        else:
            optimal_plan = StreamerSchedulePlan(
                strategy=mode,
                moves=[],
                total_acquisitions=0,
                total_games_streamed=0,
                total_projected_fp=0.0,
                points_per_add=0.0,
                net_delta_fp=0.0,
            )

    return StreamerOptimizationResult(
        optimal_plan=optimal_plan,
        mode=mode,
        total_acquisitions_used=optimal_plan.total_acquisitions,
        total_streamed_games=optimal_plan.total_games_streamed,
        total_net_delta_fp=optimal_plan.net_delta_fp,
        points_per_add=optimal_plan.points_per_add,
        alternative_plans=alternative_plans,
        daily_density_analysis=daily_density,
    )


# ── MILP Streamer Engine Integration ─────────────────────────────────────────

def optimize_streaming_milp(
    current_roster_ids: list[int],
    available_waiver_ids: list[int],
    schedule: ScheduleCalendar,
    evaluation_window: list[str],
    player_positions: dict[int, list[str]],
    player_fp_per_game: dict[int, float],
    active_slots: dict[str, int],
    max_acquisitions: int = 7,
    designated_drops: Optional[list[int]] = None,
    bench_slots: int = 4,
    ir_slots: int = 3,
    candidate_game_dates: Optional[dict[int, set[str]]] = None,
    solver_backend: str = "ortools",
    **kwargs: Any,
) -> StreamerOptimizationResult:
    """Optimize streaming schedule using the rolling-horizon MILP solver from fantasy_streamer."""
    from datetime import date
    from fantasy_streamer.config import LeagueConfig
    from fantasy_streamer.models.projections import Player, compute_replacement_baseline
    from fantasy_streamer.models.penalties import DropPenaltyCalculator
    from fantasy_streamer.models.schedule import MatchupHorizon, ScheduleMatrix
    from fantasy_streamer.optimization.constraints import OptimizationProblemData
    from fantasy_streamer.optimization.milp_engine import MILPStreamerEngine

    dates = schedule.normalize_window(evaluation_window)
    if not dates:
        raise ValueError("evaluation_window must contain at least one valid ISO date string.")

    # 1. Parse start date & horizon
    y, m, d_num = [int(x) for x in dates[0].split("-")]
    start_dt = date(y, m, d_num)
    horizon = MatchupHorizon(start_date=start_dt, horizon_days=len(dates))

    # 2. Build ScheduleMatrix
    sched_matrix = ScheduleMatrix(horizon=horizon)
    for i, d_str in enumerate(dates):
        playing_teams = schedule.get_teams_playing_on_date(d_str)
        for t in playing_teams:
            sched_matrix.team_games.setdefault(t.upper(), set()).add(i)

        if candidate_game_dates:
            for cid, c_dates in candidate_game_dates.items():
                if d_str in c_dates:
                    sched_matrix.player_games.setdefault(cid, set()).add(i)
        else:
            cand_playing = schedule.get_players_playing_on_date(available_waiver_ids, d_str)
            for cid in cand_playing:
                sched_matrix.player_games.setdefault(cid, set()).add(i)

        roster_playing = schedule.get_players_playing_on_date(current_roster_ids, d_str)
        for pid in roster_playing:
            sched_matrix.player_games.setdefault(pid, set()).add(i)

    # 3. Setup candidate pool
    cands_fppg = [float(player_fp_per_game.get(c, 20.0)) for c in available_waiver_ids]
    fa_objs = [
        Player(
            player_id=c,
            name=f"FreeAgent_{c}",
            team=schedule.get_player_team(c) or "BOS",
            positions=player_positions.get(c, ["UTIL"]),
            fppg=float(player_fp_per_game.get(c, 20.0)),
        )
        for c in available_waiver_ids
    ]
    mu_rep = compute_replacement_baseline(fa_objs, k=min(10, len(fa_objs))) if fa_objs else 22.5
    drop_calc = DropPenaltyCalculator(replacement_baseline=mu_rep)

    # 4. Build roster Player objects
    droppable_set = set(designated_drops) if designated_drops is not None else set()
    if not droppable_set and len(current_roster_ids) > sum(active_slots.values()):
        sorted_roster = sorted(current_roster_ids, key=lambda pid: player_fp_per_game.get(pid, 0.0))
        droppable_set = set(sorted_roster[:max(1, len(current_roster_ids) - sum(active_slots.values()))])
    elif not droppable_set:
        droppable_set = set(current_roster_ids)

    roster_players: list[Player] = []
    for pid in current_roster_ids:
        fppg = float(player_fp_per_game.get(pid, 20.0))
        is_drop = pid in droppable_set
        cp = float(drop_calc.calculate_penalty(fppg=fppg, games_per_week=3.5)) if is_drop else 0.0
        roster_players.append(Player(
            player_id=pid,
            name=f"Player_{pid}",
            team=schedule.get_player_team(pid) or "LAL",
            positions=player_positions.get(pid, ["UTIL"]),
            fppg=fppg,
            is_rostered=True,
            is_droppable=is_drop,
            drop_penalty_c_p=cp,
        ))

    # 5. League configuration
    custom_active_slots: list[str] = []
    for s_name, s_count in active_slots.items():
        if s_name == "UTIL" and s_count > 1:
            for s_idx in range(1, s_count + 1):
                custom_active_slots.append(f"UTIL{s_idx}")
        elif s_count == 1:
            custom_active_slots.append(s_name)
        else:
            for _ in range(s_count):
                custom_active_slots.append(s_name)

    from fantasy_streamer.config import ACTIVE_SLOTS
    config = LeagueConfig(
        roster_size=len(current_roster_ids),
        active_starters=sum(active_slots.values()),
        bench_spots=bench_slots,
        active_slots=custom_active_slots if custom_active_slots else list(ACTIVE_SLOTS),
    )

    # 6. Optimization Problem Data
    prob_data = OptimizationProblemData(
        config=config,
        rostered_players=roster_players,
        candidate_free_agents=fa_objs,
        schedule=sched_matrix,
        horizon=horizon,
        rem_acquisitions_w1=max_acquisitions,
        cap_acquisitions_w2=max_acquisitions,
    )

    # 7. Solve via MILP engine
    milp_engine = MILPStreamerEngine(backend=solver_backend)
    res = milp_engine.solve(prob_data)

    # 8. Convert transactions to StreamingMove list
    moves: list[StreamingMove] = []
    drop_candidates_queue = list(droppable_set)
    for act in res.actions:
        if act.action_type == "ADD":
            drop_id = drop_candidates_queue.pop(0) if drop_candidates_queue else None
            cid = act.player_id
            act_team = getattr(act, "player_team", "") or schedule.get_player_team(cid) or ""
            played_indices = [
                i for i in range(act.day_idx, len(dates))
                if sched_matrix.has_game(act_team, i, cid)
            ]
            played_dates = [dates[i] for i in played_indices]
            moves.append(StreamingMove(
                date=act.date_str,
                add_player_id=cid,
                drop_player_id=drop_id,
                games_gained=len(played_dates),
                projected_fp=round(act.fppg * len(played_dates), 4),
                dates_active=dates[act.day_idx:],
                dates_played=played_dates,
            ))
            if len(moves) >= max_acquisitions:
                break

    total_streamed_fp = sum(m.projected_fp for m in moves)
    total_games_streamed = sum(m.games_gained for m in moves)
    ppa = round(total_streamed_fp / len(moves), 4) if moves else 0.0

    density = analyze_schedule_density(
        current_roster_ids=current_roster_ids,
        schedule=schedule,
        evaluation_window=dates,
        player_positions=player_positions,
        player_fp_per_game=player_fp_per_game,
        active_slots=active_slots,
        bench_slots=bench_slots,
        ir_slots=ir_slots,
    )
    total_baseline_starting_fp = sum(d["baseline_starting_fp"] for d in density.values())

    # Calculate actual net delta FP over baseline for the MILP plan
    milp_active_by_day: dict[str, list[int]] = {d: [] for d in dates}
    for m in moves:
        for d in m.dates_active:
            if d in milp_active_by_day:
                milp_active_by_day[d].append(m.add_player_id)

    milp_window_starting_fp = 0.0
    for d in dates:
        active_streamers = milp_active_by_day[d]
        if active_streamers:
            core_roster = [p for p in current_roster_ids if p not in droppable_set]
            test_roster = core_roster + active_streamers
            pl = schedule.get_players_playing_on_date(test_roster, d)
            asgn = assign_optimal_daily_lineup(
                roster_player_ids=test_roster,
                player_positions=player_positions,
                player_fp=player_fp_per_game,
                active_slots=active_slots,
                playing_today=pl,
                bench_slots=bench_slots,
                ir_slots=ir_slots,
            )
            milp_window_starting_fp += asgn.total_expected_fp
        else:
            milp_window_starting_fp += density[d]["baseline_starting_fp"]

    milp_net_delta_fp = round(milp_window_starting_fp - total_baseline_starting_fp, 4) if moves else 0.0

    plan = StreamerSchedulePlan(
        strategy="milp",
        moves=moves,
        total_acquisitions=len(moves),
        total_games_streamed=total_games_streamed,
        total_projected_fp=round(total_streamed_fp, 4),
        points_per_add=ppa,
        net_delta_fp=milp_net_delta_fp,
    )

    return StreamerOptimizationResult(
        optimal_plan=plan,
        mode="milp",
        total_acquisitions_used=plan.total_acquisitions,
        total_streamed_games=plan.total_games_streamed,
        total_net_delta_fp=plan.net_delta_fp,
        points_per_add=plan.points_per_add,
        alternative_plans={"milp": plan},
        daily_density_analysis=density,
    )


# Alias for compatibility with external callers and router
optimize_streaming_schedule = optimize_streaming_acquisitions
