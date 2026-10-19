"""
Trade Analyzer, Win-Win Suggestions & League Impact Engine for Points Mode.

Provides:
- PointsTradeAnalysisResult: Pre/post starting lineup FP, Net Delta FP, streamer spot boost.
- LeagueTradeImpact: Standings shifts, projected total points, competitive balance delta.
- PointsTradeSuggestion: Win-win trade proposals matching format and gain criteria.
- analyze_points_trade: Multi-day schedule-aware trade simulation and legality validation.
- search_win_win_points_trades: Fast combinatorial search for mutually beneficial trades.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from backend.points.schedule import ScheduleCalendar
from backend.points.lineups import assign_optimal_daily_lineup

logger = logging.getLogger('fbbo.points.trade')


# ── Exceptions ───────────────────────────────────────────────────────────────

class IllegalTradeError(ValueError):
    """Raised when a proposed trade violates roster legality or positional caps."""
    pass


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LeagueTradeImpact:
    """Projected league-wide impact of a trade on standings and competitive balance."""
    pre_trade_points: dict[str, float]
    post_trade_points: dict[str, float]
    pre_trade_ranks: dict[str, int]
    post_trade_ranks: dict[str, int]
    rank_shifts: dict[str, int]
    pre_trade_std: float
    post_trade_std: float
    pre_trade_spread: float
    post_trade_spread: float
    competitive_balance_delta: float

    def to_dict(self) -> dict:
        return {
            'pre_trade_points': self.pre_trade_points,
            'post_trade_points': self.post_trade_points,
            'pre_trade_ranks': self.pre_trade_ranks,
            'post_trade_ranks': self.post_trade_ranks,
            'rank_shifts': self.rank_shifts,
            'pre_trade_std': self.pre_trade_std,
            'post_trade_std': self.post_trade_std,
            'pre_trade_spread': self.pre_trade_spread,
            'post_trade_spread': self.post_trade_spread,
            'competitive_balance_delta': self.competitive_balance_delta,
        }


@dataclass(frozen=True)
class PointsTradeAnalysisResult:
    """Complete evaluation of a fantasy points trade between two teams."""
    team_a_pre_fp: float
    team_a_post_fp: float
    team_a_delta_fp: float
    team_a_net_delta_fp: float
    team_a_streamer_boost: float
    team_b_pre_fp: float
    team_b_post_fp: float
    team_b_delta_fp: float
    team_b_net_delta_fp: float
    team_b_streamer_boost: float
    is_legal: bool = True
    rejection_reason: Optional[str] = None
    league_impact: Optional[LeagueTradeImpact] = None
    team_a_opened_spots: int = 0
    team_b_opened_spots: int = 0

    def to_dict(self) -> dict:
        return {
            'team_a_pre_fp': self.team_a_pre_fp,
            'team_a_post_fp': self.team_a_post_fp,
            'team_a_delta_fp': self.team_a_delta_fp,
            'team_a_net_delta_fp': self.team_a_net_delta_fp,
            'team_a_streamer_boost': self.team_a_streamer_boost,
            'team_b_pre_fp': self.team_b_pre_fp,
            'team_b_post_fp': self.team_b_post_fp,
            'team_b_delta_fp': self.team_b_delta_fp,
            'team_b_net_delta_fp': self.team_b_net_delta_fp,
            'team_b_streamer_boost': self.team_b_streamer_boost,
            'is_legal': self.is_legal,
            'rejection_reason': self.rejection_reason,
            'team_a_opened_spots': self.team_a_opened_spots,
            'team_b_opened_spots': self.team_b_opened_spots,
            'league_impact': self.league_impact.to_dict() if self.league_impact else None,
        }


@dataclass(frozen=True)
class PointsTradeSuggestion:
    """A mutually beneficial win-win trade proposal."""
    partner_team_id: str
    user_sends: list[int]
    partner_sends: list[int]
    user_gain: float
    partner_gain: float
    trade_format: str
    analysis: Optional[PointsTradeAnalysisResult] = None

    @property
    def send(self) -> list[int]:
        return self.user_sends

    @property
    def receive(self) -> list[int]:
        return self.partner_sends

    @property
    def user_delta_fp(self) -> float:
        return self.user_gain

    @property
    def partner_delta_fp(self) -> float:
        return self.partner_gain

    @property
    def total_gain(self) -> float:
        return round(self.user_gain + self.partner_gain, 4)

    def to_dict(self) -> dict:
        return {
            'partner_team_id': self.partner_team_id,
            'user_sends': self.user_sends,
            'partner_sends': self.partner_sends,
            'user_gain': self.user_gain,
            'partner_gain': self.partner_gain,
            'total_gain': self.total_gain,
            'trade_format': self.trade_format,
            'analysis': self.analysis.to_dict() if self.analysis else None,
        }


# ── Helper Calculations ──────────────────────────────────────────────────────

def calculate_streamer_spot_boost(
    opened_spots: int,
    evaluation_window: list[str],
    waiver_baseline_fp_per_game: float = 20.0,
    streamer_multiplier: Optional[float] = None,
    streamer_games_per_week: float = 6.5,
    baseline_games_per_week: float = 3.4,
) -> float:
    """Opportunity value of opened streaming spot(s).

    A normal player plays ~3.3-3.5 games/week, but a streaming slot utilized
    with daily adds yields 6-7 games/week (~1.5x-2.0x volume multiplier).
    """
    if opened_spots <= 0 or not evaluation_window:
        return 0.0

    weeks = len(evaluation_window) / 7.0
    if streamer_multiplier is not None:
        effective_games = baseline_games_per_week * streamer_multiplier
    else:
        effective_games = streamer_games_per_week

    boost = opened_spots * weeks * effective_games * waiver_baseline_fp_per_game
    return round(float(boost), 4)


def simulate_roster_lineup_fp(
    roster: list[int],
    schedule: ScheduleCalendar,
    evaluation_window: list[str],
    player_positions: dict[int, list[str]],
    player_fp_per_game: dict[int, float],
    active_slots: dict[str, int],
    bench_slots: int = 4,
    ir_slots: int = 3,
) -> float:
    """Compute total starting lineup FP for a roster across an evaluation window."""
    if not roster:
        return 0.0

    norm_window = schedule.normalize_window(evaluation_window)
    total_fp = 0.0
    for date_str in norm_window:
        playing_today = schedule.get_players_playing_on_date(roster, date_str)
        assignment = assign_optimal_daily_lineup(
            roster_player_ids=roster,
            player_positions=player_positions,
            player_fp=player_fp_per_game,
            active_slots=active_slots,
            playing_today=playing_today,
            bench_slots=bench_slots,
            ir_slots=ir_slots,
        )
        total_fp += assignment.total_expected_fp

    return round(float(total_fp), 4)


def validate_trade_legality(
    team_a_roster: list[int],
    team_b_roster: list[int],
    team_a_sends: list[int],
    team_b_sends: list[int],
    roster_size_limit: int,
    player_positions: dict[int, list[str]],
    active_slots: dict[str, int],
    position_caps: Optional[dict[str, int]] = None,
    ignore_position_check: bool = False,
) -> tuple[bool, Optional[str]]:
    """Verify roster ownership, size limits, positional caps, and active slot validity."""
    set_a = set(team_a_roster)
    set_b = set(team_b_roster)

    # 1. Non-empty trade
    if not team_a_sends and not team_b_sends:
        return False, "Trade cannot be empty"

    # 2. Player ownership
    for pid in team_a_sends:
        if pid not in set_a:
            return False, f"Player {pid} is not on Team A roster"

    for pid in team_b_sends:
        if pid not in set_b:
            return False, f"Player {pid} is not on Team B roster"

    # 3. Duplicate checks
    if len(team_a_sends) != len(set(team_a_sends)):
        return False, "Duplicate player IDs found in Team A sends"
    if len(team_b_sends) != len(set(team_b_sends)):
        return False, "Duplicate player IDs found in Team B sends"

    # 4. Post-trade roster construction
    post_a = [p for p in team_a_roster if p not in team_a_sends] + list(team_b_sends)
    post_b = [p for p in team_b_roster if p not in team_b_sends] + list(team_a_sends)

    if len(post_a) != len(set(post_a)):
        return False, "Duplicate player IDs produced in Team A post-trade roster"
    if len(post_b) != len(set(post_b)):
        return False, "Duplicate player IDs produced in Team B post-trade roster"

    # 5. Roster size limit
    if len(post_a) > roster_size_limit:
        return False, f"Team A exceeds roster size limit ({len(post_a)} > {roster_size_limit})"
    if len(post_b) > roster_size_limit:
        return False, f"Team B exceeds roster size limit ({len(post_b)} > {roster_size_limit})"

    # 6. Positional caps & slot validity (skipped if ignore_position_check is True)
    if not ignore_position_check:
        if position_caps:
            for pos, max_count in position_caps.items():
                norm_pos = pos.upper()
                count_a = sum(1 for pid in post_a if norm_pos in [p.upper() for p in player_positions.get(pid, [])])
                if count_a > max_count:
                    return False, f"Team A exceeds positional cap for {norm_pos} ({count_a} > {max_count})"

                count_b = sum(1 for pid in post_b if norm_pos in [p.upper() for p in player_positions.get(pid, [])])
                if count_b > max_count:
                    return False, f"Team B exceeds positional cap for {norm_pos} ({count_b} > {max_count})"

        # 7. Positional validity (ensuring required base slots can be filled if roster has enough players)
        total_active = sum(active_slots.values())
        for label, roster in (("Team A", post_a), ("Team B", post_b)):
            if len(roster) >= total_active:
                for slot_pos, count in active_slots.items():
                    if count > 0 and slot_pos.upper() not in ('UTIL', 'BENCH', 'IR', 'G', 'F'):
                        norm_pos = slot_pos.upper()
                        eligible_players = [
                            pid for pid in roster
                            if norm_pos in [p.upper() for p in player_positions.get(pid, [])]
                        ]
                        if len(eligible_players) < count:
                            msg = (
                                f"{label} cannot satisfy required active slot {norm_pos} "
                                f"({len(eligible_players)} < {count})"
                            )
                            return False, msg

    return True, None


def compute_league_trade_impact(
    all_team_rosters: dict[str, list[int]],
    team_a_id: str,
    team_b_id: str,
    team_a_pre_fp: float,
    team_a_post_fp: float,
    team_b_pre_fp: float,
    team_b_post_fp: float,
    schedule: ScheduleCalendar,
    evaluation_window: list[str],
    player_positions: dict[int, list[str]],
    player_fp_per_game: dict[int, float],
    active_slots: dict[str, int],
    bench_slots: int = 4,
    ir_slots: int = 3,
) -> LeagueTradeImpact:
    """Project trade impact on league standings, points, and competitive balance."""
    norm_window = schedule.normalize_window(evaluation_window)

    pre_trade_points: dict[str, float] = {}
    post_trade_points: dict[str, float] = {}

    for tid, rost in all_team_rosters.items():
        if tid == team_a_id:
            pre_trade_points[tid] = round(float(team_a_pre_fp), 4)
            post_trade_points[tid] = round(float(team_a_post_fp), 4)
        elif tid == team_b_id:
            pre_trade_points[tid] = round(float(team_b_pre_fp), 4)
            post_trade_points[tid] = round(float(team_b_post_fp), 4)
        else:
            pts = simulate_roster_lineup_fp(
                roster=rost,
                schedule=schedule,
                evaluation_window=norm_window,
                player_positions=player_positions,
                player_fp_per_game=player_fp_per_game,
                active_slots=active_slots,
                bench_slots=bench_slots,
                ir_slots=ir_slots,
            )
            pre_trade_points[tid] = pts
            post_trade_points[tid] = pts

    def _rank_teams(pts_map: dict[str, float]) -> dict[str, int]:
        sorted_teams = sorted(pts_map.keys(), key=lambda t: (-pts_map[t], str(t)))
        return {t: rank + 1 for rank, t in enumerate(sorted_teams)}

    pre_ranks = _rank_teams(pre_trade_points)
    post_ranks = _rank_teams(post_trade_points)

    # rank_shifts: positive shift means improved rank (e.g. 4 -> 2 is +2)
    rank_shifts = {t: pre_ranks[t] - post_ranks[t] for t in all_team_rosters}

    pre_pts_vals = list(pre_trade_points.values())
    post_pts_vals = list(post_trade_points.values())

    pre_std = float(np.std(pre_pts_vals)) if pre_pts_vals else 0.0
    post_std = float(np.std(post_pts_vals)) if post_pts_vals else 0.0
    pre_spread = float(max(pre_pts_vals) - min(pre_pts_vals)) if pre_pts_vals else 0.0
    post_spread = float(max(post_pts_vals) - min(post_pts_vals)) if post_pts_vals else 0.0
    balance_delta = post_std - pre_std

    return LeagueTradeImpact(
        pre_trade_points=pre_trade_points,
        post_trade_points=post_trade_points,
        pre_trade_ranks=pre_ranks,
        post_trade_ranks=post_ranks,
        rank_shifts=rank_shifts,
        pre_trade_std=round(pre_std, 4),
        post_trade_std=round(post_std, 4),
        pre_trade_spread=round(pre_spread, 4),
        post_trade_spread=round(post_spread, 4),
        competitive_balance_delta=round(balance_delta, 4),
    )


# ── Public Engine Functions ──────────────────────────────────────────────────

def analyze_points_trade(
    team_a_roster: list[int],
    team_b_roster: list[int],
    team_a_sends: list[int],
    team_b_sends: list[int],
    schedule: ScheduleCalendar,
    evaluation_window: list[str],
    player_positions: dict[int, list[str]],
    player_fp_per_game: dict[int, float],
    active_slots: dict[str, int],
    roster_size_limit: int,
    waiver_baseline_fp_per_game: float = 20.0,
    all_team_rosters: Optional[dict[str, list[int]]] = None,
    bench_slots: int = 4,
    ir_slots: int = 3,
    position_caps: Optional[dict[str, int]] = None,
    streamer_multiplier: Optional[float] = None,
    streamer_games_per_week: float = 6.5,
    raise_on_illegal: bool = False,
    team_a_id: str = "team_a",
    team_b_id: str = "team_b",
    ignore_position_check: bool = False,
) -> PointsTradeAnalysisResult:
    """Evaluate a proposed trade between two fantasy points teams.

    Computes:
    - Pre- and post-trade starting lineup fantasy points.
    - Streamer spot valuation adjustment for opened roster spots.
    - Net Delta FP for both sides.
    - Full legality validation (roster caps, non-rostered players, positional caps).
    - League-wide standings and competitive balance impact when all_team_rosters is provided.
    """
    norm_window = schedule.normalize_window(evaluation_window)

    # 1. Legality check
    is_legal, rejection_reason = validate_trade_legality(
        team_a_roster=team_a_roster,
        team_b_roster=team_b_roster,
        team_a_sends=team_a_sends,
        team_b_sends=team_b_sends,
        roster_size_limit=roster_size_limit,
        player_positions=player_positions,
        active_slots=active_slots,
        position_caps=position_caps,
        ignore_position_check=ignore_position_check,
    )

    if not is_legal:
        if raise_on_illegal:
            raise IllegalTradeError(rejection_reason)
        return PointsTradeAnalysisResult(
            team_a_pre_fp=0.0,
            team_a_post_fp=0.0,
            team_a_delta_fp=0.0,
            team_a_net_delta_fp=0.0,
            team_a_streamer_boost=0.0,
            team_b_pre_fp=0.0,
            team_b_post_fp=0.0,
            team_b_delta_fp=0.0,
            team_b_net_delta_fp=0.0,
            team_b_streamer_boost=0.0,
            is_legal=False,
            rejection_reason=rejection_reason,
            league_impact=None,
            team_a_opened_spots=0,
            team_b_opened_spots=0,
        )

    # 2. Pre-trade starting lineup FP simulation
    team_a_pre = simulate_roster_lineup_fp(
        roster=team_a_roster,
        schedule=schedule,
        evaluation_window=norm_window,
        player_positions=player_positions,
        player_fp_per_game=player_fp_per_game,
        active_slots=active_slots,
        bench_slots=bench_slots,
        ir_slots=ir_slots,
    )

    team_b_pre = simulate_roster_lineup_fp(
        roster=team_b_roster,
        schedule=schedule,
        evaluation_window=norm_window,
        player_positions=player_positions,
        player_fp_per_game=player_fp_per_game,
        active_slots=active_slots,
        bench_slots=bench_slots,
        ir_slots=ir_slots,
    )

    # 3. Post-trade rosters
    post_a = [p for p in team_a_roster if p not in team_a_sends] + list(team_b_sends)
    post_b = [p for p in team_b_roster if p not in team_b_sends] + list(team_a_sends)

    team_a_post_lineup = simulate_roster_lineup_fp(
        roster=post_a,
        schedule=schedule,
        evaluation_window=norm_window,
        player_positions=player_positions,
        player_fp_per_game=player_fp_per_game,
        active_slots=active_slots,
        bench_slots=bench_slots,
        ir_slots=ir_slots,
    )

    team_b_post_lineup = simulate_roster_lineup_fp(
        roster=post_b,
        schedule=schedule,
        evaluation_window=norm_window,
        player_positions=player_positions,
        player_fp_per_game=player_fp_per_game,
        active_slots=active_slots,
        bench_slots=bench_slots,
        ir_slots=ir_slots,
    )

    # 4. Streamer Spot Valuation Adjustment
    opened_a = max(0, len(team_a_sends) - len(team_b_sends))
    opened_b = max(0, len(team_b_sends) - len(team_a_sends))

    boost_a = calculate_streamer_spot_boost(
        opened_spots=opened_a,
        evaluation_window=norm_window,
        waiver_baseline_fp_per_game=waiver_baseline_fp_per_game,
        streamer_multiplier=streamer_multiplier,
        streamer_games_per_week=streamer_games_per_week,
    )

    boost_b = calculate_streamer_spot_boost(
        opened_spots=opened_b,
        evaluation_window=norm_window,
        waiver_baseline_fp_per_game=waiver_baseline_fp_per_game,
        streamer_multiplier=streamer_multiplier,
        streamer_games_per_week=streamer_games_per_week,
    )

    team_a_post_total = round(team_a_post_lineup + boost_a, 4)
    team_b_post_total = round(team_b_post_lineup + boost_b, 4)

    delta_a = round(team_a_post_total - team_a_pre, 4)
    delta_b = round(team_b_post_total - team_b_pre, 4)

    # 5. League-wide impact projection
    league_impact = None
    if all_team_rosters:
        # Resolve team IDs if not explicitly matching keys
        resolved_all = dict(all_team_rosters)
        resolved_a_id = team_a_id
        resolved_b_id = team_b_id

        if resolved_a_id not in resolved_all:
            for k, r in resolved_all.items():
                if set(r) == set(team_a_roster):
                    resolved_a_id = k
                    break
            else:
                resolved_all[resolved_a_id] = team_a_roster

        if resolved_b_id not in resolved_all:
            for k, r in resolved_all.items():
                if set(r) == set(team_b_roster):
                    resolved_b_id = k
                    break
            else:
                resolved_all[resolved_b_id] = team_b_roster

        league_impact = compute_league_trade_impact(
            all_team_rosters=resolved_all,
            team_a_id=resolved_a_id,
            team_b_id=resolved_b_id,
            team_a_pre_fp=team_a_pre,
            team_a_post_fp=team_a_post_total,
            team_b_pre_fp=team_b_pre,
            team_b_post_fp=team_b_post_total,
            schedule=schedule,
            evaluation_window=norm_window,
            player_positions=player_positions,
            player_fp_per_game=player_fp_per_game,
            active_slots=active_slots,
            bench_slots=bench_slots,
            ir_slots=ir_slots,
        )

    return PointsTradeAnalysisResult(
        team_a_pre_fp=round(team_a_pre, 4),
        team_a_post_fp=team_a_post_total,
        team_a_delta_fp=delta_a,
        team_a_net_delta_fp=delta_a,
        team_a_streamer_boost=boost_a,
        team_b_pre_fp=round(team_b_pre, 4),
        team_b_post_fp=team_b_post_total,
        team_b_delta_fp=delta_b,
        team_b_net_delta_fp=delta_b,
        team_b_streamer_boost=boost_b,
        is_legal=True,
        rejection_reason=None,
        league_impact=league_impact,
        team_a_opened_spots=opened_a,
        team_b_opened_spots=opened_b,
    )


def search_win_win_points_trades(
    user_team_id: str,
    all_team_rosters: dict[str, list[int]],
    schedule: ScheduleCalendar,
    evaluation_window: list[str],
    player_positions: dict[int, list[str]],
    player_fp_per_game: dict[int, float],
    active_slots: dict[str, int],
    roster_size_limit: int,
    min_user_gain: float = 1.0,
    min_partner_gain: float = 1.0,
    trade_formats: list[str] = ["1v1", "2v1", "2v2"],
    waiver_baseline_fp_per_game: float = 20.0,
    bench_slots: int = 4,
    ir_slots: int = 3,
    position_caps: Optional[dict[str, int]] = None,
    streamer_multiplier: Optional[float] = None,
    streamer_games_per_week: float = 6.5,
    max_suggestions: int = 50,
    ignore_position_check: bool = False,
) -> list[PointsTradeSuggestion]:
    """Search league rosters for mutually beneficial win-win trade proposals."""
    if user_team_id not in all_team_rosters:
        raise ValueError(f"user_team_id '{user_team_id}' not found in all_team_rosters")

    user_roster = all_team_rosters[user_team_id]
    norm_window = schedule.normalize_window(evaluation_window)

    # Pre-compute and cache pre-trade FP for all teams
    pre_fp_cache: dict[str, float] = {}
    for tid, rost in all_team_rosters.items():
        pre_fp_cache[tid] = simulate_roster_lineup_fp(
            roster=rost,
            schedule=schedule,
            evaluation_window=norm_window,
            player_positions=player_positions,
            player_fp_per_game=player_fp_per_game,
            active_slots=active_slots,
            bench_slots=bench_slots,
            ir_slots=ir_slots,
        )

    user_pre = pre_fp_cache[user_team_id]
    suggestions: list[PointsTradeSuggestion] = []

    # Normalize formats
    active_formats = set(f.lower().strip() for f in trade_formats)

    for partner_id, partner_roster in all_team_rosters.items():
        if partner_id == user_team_id:
            continue

        partner_pre = pre_fp_cache[partner_id]

        # Generator of candidate packages: (u_sends, p_sends, format_str)
        packages: list[tuple[list[int], list[int], str]] = []

        if "1v1" in active_formats:
            for u in user_roster:
                for p in partner_roster:
                    packages.append(([u], [p], "1v1"))

        if "2v1" in active_formats:
            # User sends 2, partner sends 1 (user opens streamer spot)
            if len(partner_roster) + 1 <= roster_size_limit and len(user_roster) >= 2:
                for u1, u2 in itertools.combinations(user_roster, 2):
                    for p in partner_roster:
                        packages.append(([u1, u2], [p], "2v1"))

            # User sends 1, partner sends 2 (partner opens streamer spot)
            if len(user_roster) + 1 <= roster_size_limit and len(partner_roster) >= 2:
                for u in user_roster:
                    for p1, p2 in itertools.combinations(partner_roster, 2):
                        packages.append(([u], [p1, p2], "1v2"))

        if "1v2" in active_formats and "2v1" not in active_formats:
            if len(user_roster) + 1 <= roster_size_limit and len(partner_roster) >= 2:
                for u in user_roster:
                    for p1, p2 in itertools.combinations(partner_roster, 2):
                        packages.append(([u], [p1, p2], "1v2"))

        if "2v2" in active_formats:
            if len(user_roster) >= 2 and len(partner_roster) >= 2:
                for u_pair in itertools.combinations(user_roster, 2):
                    for p_pair in itertools.combinations(partner_roster, 2):
                        packages.append((list(u_pair), list(p_pair), "2v2"))

        for u_sends, p_sends, fmt in packages:
            # 1. Quick legality prune
            is_legal, _ = validate_trade_legality(
                team_a_roster=user_roster,
                team_b_roster=partner_roster,
                team_a_sends=u_sends,
                team_b_sends=p_sends,
                roster_size_limit=roster_size_limit,
                player_positions=player_positions,
                active_slots=active_slots,
                position_caps=position_caps,
                ignore_position_check=ignore_position_check,
            )
            if not is_legal:
                continue

            # 2. User side simulation & threshold pruning
            post_user = [p for p in user_roster if p not in u_sends] + p_sends
            opened_u = max(0, len(u_sends) - len(p_sends))
            boost_u = calculate_streamer_spot_boost(
                opened_spots=opened_u,
                evaluation_window=norm_window,
                waiver_baseline_fp_per_game=waiver_baseline_fp_per_game,
                streamer_multiplier=streamer_multiplier,
                streamer_games_per_week=streamer_games_per_week,
            )
            user_post_lineup = simulate_roster_lineup_fp(
                roster=post_user,
                schedule=schedule,
                evaluation_window=norm_window,
                player_positions=player_positions,
                player_fp_per_game=player_fp_per_game,
                active_slots=active_slots,
                bench_slots=bench_slots,
                ir_slots=ir_slots,
            )
            user_post = round(user_post_lineup + boost_u, 4)
            user_gain = round(user_post - user_pre, 4)

            if user_gain < min_user_gain:
                continue  # User doesn't benefit enough -> skip partner simulation

            # 3. Partner side simulation & threshold check
            post_partner = [p for p in partner_roster if p not in p_sends] + u_sends
            opened_p = max(0, len(p_sends) - len(u_sends))
            boost_p = calculate_streamer_spot_boost(
                opened_spots=opened_p,
                evaluation_window=norm_window,
                waiver_baseline_fp_per_game=waiver_baseline_fp_per_game,
                streamer_multiplier=streamer_multiplier,
                streamer_games_per_week=streamer_games_per_week,
            )
            partner_post_lineup = simulate_roster_lineup_fp(
                roster=post_partner,
                schedule=schedule,
                evaluation_window=norm_window,
                player_positions=player_positions,
                player_fp_per_game=player_fp_per_game,
                active_slots=active_slots,
                bench_slots=bench_slots,
                ir_slots=ir_slots,
            )
            partner_post = round(partner_post_lineup + boost_p, 4)
            partner_gain = round(partner_post - partner_pre, 4)

            if partner_gain < min_partner_gain:
                continue  # Partner doesn't benefit enough

            # Found mutually beneficial trade!
            analysis = PointsTradeAnalysisResult(
                team_a_pre_fp=round(user_pre, 4),
                team_a_post_fp=user_post,
                team_a_delta_fp=user_gain,
                team_a_net_delta_fp=user_gain,
                team_a_streamer_boost=boost_u,
                team_b_pre_fp=round(partner_pre, 4),
                team_b_post_fp=partner_post,
                team_b_delta_fp=partner_gain,
                team_b_net_delta_fp=partner_gain,
                team_b_streamer_boost=boost_p,
                is_legal=True,
                rejection_reason=None,
                team_a_opened_spots=opened_u,
                team_b_opened_spots=opened_p,
            )

            suggestions.append(
                PointsTradeSuggestion(
                    partner_team_id=partner_id,
                    user_sends=u_sends,
                    partner_sends=p_sends,
                    user_gain=user_gain,
                    partner_gain=partner_gain,
                    trade_format=fmt,
                    analysis=analysis,
                )
            )

    # Sort suggestions by total gain descending, then by user gain descending
    suggestions.sort(key=lambda s: (s.total_gain, s.user_gain), reverse=True)
    return suggestions[:max_suggestions]
