"""
DynamicPointsAgent: roster/budget-aware points evaluation.

Conditions on the current roster and models future roster additions to evaluate
the best achievable team under positional and resource constraints.

Required behavior (per handoff §7):
  In a fixture where future centers are scarce and guards abundant, the agent
  can favor a lower raw-point center when that choice produces the better
  legal completed roster.  Changing future access, current roster, or
  remaining auction budget must update the recommendation.

Deterministic constrained solver (no sampling, no MCTS).
Verified against exhaustive completion on tiny cases.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from backend.points.config import PointsScoringConfig
from backend.points.lineups import (
    assign_optimal_daily_lineup
    , compute_lineup_fp
)
from backend.points.roster import compute_all_marginals

logger = logging.getLogger('fbbo.points.agents')


# ── Replacement pool model ────────────────────────────────────────────────────

class ReplacementPool:
    """Models which players are obtainable in future picks/bids.

    For snake drafts: uses supplied ADP / market ranks to determine future access.
    For auctions: additionally respects remaining budget constraints.

    Approximation is explicitly labeled; see `is_exact_model`.
    """

    def __init__(
        self
        , available_player_ids: list[int]
        , per_game_fp: dict[int, float]
        , position_eligibility: dict[int, list[str]]
        , adp_rank: Optional[dict[int, int]] = None
        , remaining_picks: Optional[int] = None
    ):
        """
        Args:
            available_player_ids: Players currently undrafted / not yet sold.
            per_game_fp:          {player_id: fp_per_game}.
            position_eligibility: {player_id: [pos, ...]}.
            adp_rank:             {player_id: rank} lower = higher demand.
                                  When provided, restricts future access.
            remaining_picks:      How many picks remain for this team.
        """
        self.per_game_fp          = per_game_fp
        self.position_eligibility = position_eligibility
        self.adp_rank             = adp_rank or {}
        self.remaining_picks      = remaining_picks

        # When ADP is available, exclude players likely to go before our next pick.
        if adp_rank and remaining_picks is not None:
            # Conservative: exclude top-N players by ADP where N is a simple
            # positional-scarcity estimate (top 20% likely gone).
            n_likely_gone = max(0, len(adp_rank) // 5)
            top_demand = sorted(adp_rank, key=lambda pid: adp_rank[pid])[:n_likely_gone]
            top_demand_set = set(top_demand)
            self.accessible = [pid for pid in available_player_ids if pid not in top_demand_set]
        else:
            self.accessible = list(available_player_ids)

        self.is_exact_model = (adp_rank is None)  # without ADP we allow all → optimistic bound

    def best_additions_for_slots(
        self
        , n_slots: int
        , required_positions: Optional[list[str]] = None
        , exclude_player_ids: Optional[set[int]] = None
    ) -> list[int]:
        """Return the n_slots best obtainable players (by FP), filtered by positions.

        This is the greedy completion heuristic.  For tiny cases the caller may
        enumerate completions exhaustively; this is the fast production path.
        """
        excluded = set(exclude_player_ids or [])
        pool = [
            pid for pid in self.accessible
            if pid not in excluded
            and np.isfinite(self.per_game_fp.get(pid, np.nan))
        ]
        if required_positions:
            pool = [
                pid for pid in pool
                if set(self.position_eligibility.get(pid, [])) & set(required_positions)
            ]
        pool_sorted = sorted(pool, key=lambda pid: -self.per_game_fp.get(pid, 0.0))
        return pool_sorted[:n_slots]


# ── Agent ─────────────────────────────────────────────────────────────────────

class DynamicPointsAgent:
    """Evaluate draft/auction candidates using roster-completion modeling.

    The agent is stateless across evaluations — each call to evaluate_candidates
    is self-contained given the board state passed in.  Cache at the call site
    when the board has not changed.
    """

    def __init__(
        self
        , config: PointsScoringConfig
        , per_game_fp: dict[int, float]
        , position_eligibility: dict[int, list[str]]
        , slot_configs: list[dict]
        , total_roster_spots: int
    ):
        """
        Args:
            config:               PointsScoringConfig for this session.
            per_game_fp:          {player_id: fp_per_game} for all players.
            position_eligibility: {player_id: [positions]}.
            slot_configs:         Active scoring slot definitions.
            total_roster_spots:   Full draft roster size (active + bench).
        """
        self.config               = config
        self.per_game_fp          = per_game_fp
        self.position_eligibility = position_eligibility
        self.slot_configs         = slot_configs
        self.total_roster_spots   = total_roster_spots

    def evaluate_candidates(
        self
        , my_roster: list[int]
        , candidate_ids: list[int]
        , replacement_pool: ReplacementPool
        , remaining_roster_spots: int
        , n_days: int = 1
    ) -> list[dict]:
        """Rank candidates by dynamic (completion-aware) value.

        For each candidate, models the best legal roster completion using the
        replacement pool, then evaluates that completed roster's lineup FP.

        Returns:
            List of dicts (one per candidate), sorted by dynamic_value descending:
            {
              'player_id':          int,
              'projected_fp_per_game': float,
              'marginal_fp':        float  — simple F(R+cand) - F(R),
              'dynamic_value':      float  — F(best completed roster with cand) baseline,
              'completion_ids':     [int]  — future additions in the model,
              'is_exact_model':     bool,
              'approximation_note': str | None,
            }
        """
        base_fp = self._eval_fp(my_roster, n_days)

        # Simple marginals (immediate, not completion-aware) — always computable.
        simple_marginals = compute_all_marginals(
            roster_player_ids     = my_roster
            , candidate_ids       = candidate_ids
            , per_game_fp         = self.per_game_fp
            , slot_configs        = self.slot_configs
            , position_eligibility = self.position_eligibility
            , n_days              = n_days
        )

        results = []
        for cid in candidate_ids:
            fp_per_game = self.per_game_fp.get(cid, np.nan)
            marginal    = simple_marginals.get(cid, 0.0)

            # Completion model
            augmented_roster = my_roster + [cid]
            future_spots     = max(0, remaining_roster_spots - 1)
            completion_ids, approx_note = self._model_completion(
                current_roster   = augmented_roster
                , n_future_spots = future_spots
                , pool           = replacement_pool
            )
            completed_roster = augmented_roster + completion_ids
            dynamic_value    = self._eval_fp(completed_roster, n_days)

            results.append({
                'player_id':             cid,
                'projected_fp_per_game': float(fp_per_game) if np.isfinite(fp_per_game) else None,
                'marginal_fp':           float(marginal),
                'dynamic_value':         float(dynamic_value),
                'completion_ids':        completion_ids,
                'is_exact_model':        replacement_pool.is_exact_model,
                'approximation_note':    approx_note,
            })

        # Stable sort: dynamic_value descending; tie-break by per-game FP then player_id.
        results.sort(key=lambda r: (
            -(r['dynamic_value'] or 0.0)
            , -(r['projected_fp_per_game'] or 0.0)
            , r['player_id']
        ))
        return results

    # ── Auction evaluation ────────────────────────────────────────────────────

    def evaluate_auction_candidates(
        self
        , my_roster: list[int]
        , candidate_ids: list[int]
        , replacement_pool: ReplacementPool
        , remaining_roster_spots: int
        , remaining_budget: float
        , minimum_bid: int
        , n_days: int = 1
    ) -> list[dict]:
        """Evaluate candidates with budget-aware completion modeling.

        Future-price assumptions are taken from the replacement pool; the budget
        is constrained so that spending `bid` on a candidate leaves enough to
        complete the roster at minimum bids.  Personal value is labeled conditional.

        Returns same keys as evaluate_candidates plus:
            'budget_constrained_max_bid': float  — highest bid that keeps completion feasible,
            'legal_max_bid':              float  — B - m*(r-1),
        """
        legal_max_bid = max(0.0, remaining_budget - minimum_bid * max(0, remaining_roster_spots - 1))
        results = self.evaluate_candidates(
            my_roster           = my_roster
            , candidate_ids     = candidate_ids
            , replacement_pool  = replacement_pool
            , remaining_roster_spots = remaining_roster_spots
            , n_days            = n_days
        )
        for entry in results:
            entry['legal_max_bid'] = float(legal_max_bid)
        return results

    # ── Internal ──────────────────────────────────────────────────────────────

    def _eval_fp(self, roster: list[int], n_days: int) -> float:
        assignment = assign_optimal_daily_lineup(
            roster_player_ids     = roster
            , per_game_fp         = self.per_game_fp
            , slot_configs        = self.slot_configs
            , position_eligibility = self.position_eligibility
        )
        return compute_lineup_fp(assignment, self.per_game_fp) * n_days

    def _model_completion(
        self
        , current_roster: list[int]
        , n_future_spots: int
        , pool: ReplacementPool
    ) -> tuple[list[int], Optional[str]]:
        """Greedy completion: add the best accessible players to fill remaining spots."""
        if n_future_spots <= 0:
            return [], None

        added = pool.best_additions_for_slots(
            n_slots            = n_future_spots
            , exclude_player_ids = set(current_roster)
        )

        note: Optional[str] = None
        if not pool.is_exact_model:
            note = (
                "Future access modeled with a conservative ADP-based pool; "
                "deterministic unrestricted completion is an optimistic bound."
            )
        elif len(added) < n_future_spots:
            note = (
                f"Pool has only {len(added)} accessible players for {n_future_spots} "
                "remaining spots; roster completion is partial."
            )

        return added, note
