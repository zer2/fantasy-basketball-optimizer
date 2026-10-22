"""
Shared response/DTO models — the ones the service tier (evaluate, trading) builds
and the API tier declares as `response_model`. Both tiers import this, so it lives
at the backend top level and imports nothing internal. Request bodies and router-only
responses live in `backend.api.schemas`.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional
from pydantic import BaseModel


# ── /sessions/{id}/evaluate ───────────────────────────────────────────────────

# The candidate expand-view rows below are plain dataclasses rather than Pydantic
# BaseModels. They are built in bulk (hundreds of candidates × several rows each) by
# backend.services.ranking from already-typed numpy .tolist() output, so per-row Pydantic
# validation is pure overhead here — dataclass construction is ~4× faster and Pydantic v2
# still serialises them (nested inside the Candidate/EvaluateResponse models) to identical
# JSON. The outer Candidate/EvaluateResponse stay Pydantic so the API contract is enforced.

@dataclass
class GScoreRow:
    label: str
    values: list[float]
    total: float
    is_total: bool
    is_candidate: bool   # the candidate's own row; the others are team-level sums and differentials


@dataclass
class FlexRow:
    label: str
    values: list[Optional[float]]   # null = ineligible position for this flex slot
    is_total: bool


@dataclass
class FlexAllocations:
    base_positions: list[str]
    rows: list[FlexRow]


@dataclass
class RosterAssignment:
    player_id: int   # display resolution (names, headshots) happens client-side via the registry
    kind: Literal['rostered', 'candidate', 'hypothetical']   # hypothetical: a future pick of a hypothetical team


@dataclass
class Roster:
    slots: list[str]
    assignments: dict[str, Optional[RosterAssignment]]


class AuctionValues(BaseModel):
    your_dollar:   float   # SAVOR on H-scores, team-specific, current cash/picks, plus unusable_cash_share
    unusable_cash_share: float   # the part of your_dollar that is the team's otherwise unusable cash (auction_pricing)
    # The team's spendable cash over what its best options cost (auction_pricing.compute_cash_to_cost_ratio), the
    # ratio the smooth rule acts on; the same for every candidate. None when no available player fits an open slot.
    cash_to_cost_ratio: Optional[float]
    generic_dollar:   float   # SAVOR on H-scores, current cash/picks (generic baseline)
    original_dollar:   float   # SAVOR on H-scores, full original cash/picks
    generic_dollar_g_score: float   # SAVOR on G-scores, current cash/picks
    original_dollar_g_score: float   # SAVOR on G-scores, full original cash/picks


class Candidate(BaseModel):
    player_id: int   # display resolution (name, positions, headshot) via the session registry
    h_score: float
    h_rank: int
    win_rates: list[float]
    category_weights: Optional[list[float]] = None
    g_score_rows: list[GScoreRow]
    flex_allocations: Optional[FlexAllocations] = None   # None when position data absent
    roster: Optional[Roster] = None                       # None when position data absent
    auction_values: Optional[AuctionValues] = None        # None in draft mode


class EvaluateResponse(BaseModel):
    iteration: int
    candidates: list[Candidate]
    has_more: bool = False   # True when more candidate batches remain beyond this slice
    total_candidates: int = 0   # total candidates across all batches; lets the client reserve tail space


# ── /sessions/{id}/hypothetical-team ─────────────────────────────────────────

class HypotheticalPick(BaseModel):
    player_id: int
    pick_round: Optional[int] = None        # draft: the round the pick is planned for
    predicted_price: Optional[int] = None   # auction: whole generic dollars


class HypotheticalTeamResponse(BaseModel):
    roster: Roster                         # the full team: rostered, candidate and hypothetical assignments
    hypothetical_picks: list[HypotheticalPick]


# ── /sessions/{id}/trade/analyze ─────────────────────────────────────────────

class TeamHScore(BaseModel):
    h_score: float
    rates: list[float]   # per-category win rates


class TeamTradeResult(BaseModel):
    pre: TeamHScore
    post: TeamHScore


class TradeAnalyzeResponse(BaseModel):
    your_team: Optional[TeamTradeResult] = None
    their_team: Optional[TeamTradeResult] = None
    error: Optional[str] = None


# ── /sessions/{id}/trade/suggest ─────────────────────────────────────────────

# The lowest H-score gain (a fraction) a trade search may ask about, for either side: -2%.
TRADE_THRESHOLD_FLOOR = -0.02


class ComboParam(BaseModel):
    n_traded: int
    n_received: int
    # The value window searched, in percentage points of summed general value (base H-score) between the two sides:
    # combos further apart than `threshold` are skipped, and None searches every gap. Combos within
    # `searched_threshold` were searched by an earlier batch, so they are skipped too (None: no earlier batch). The
    # frontend searches within 10 for every team first, then beyond 10 -- a gap that wide has not held a good trade in
    # any league measured, but nothing is left out by assumption.
    threshold: Optional[float]
    searched_threshold: Optional[float] = None


class TradeSuggestion(BaseModel):
    their_team: str = ''   # the team the trade is made with: a request may search several
    send: list[int]
    receive: list[int]
    your_score: float
    their_score: float


class TradeSuggestResponse(BaseModel):
    suggestions: list[TradeSuggestion]


# ── /sessions/{id}/evaluate (points mode) ────────────────────────────────────

class PointsAuctionValues(BaseModel):
    reference_price:  float   # generic market reference (pooled budgets)
    personal_value:   float   # this team's willingness-to-pay
    legal_max_bid:    float   # B - m * (r - 1)
    minimum_bid:      int
    is_draftable:     bool    # False when all rosters are complete


class PointsCandidate(BaseModel):
    player_id:              int
    projected_fp_per_game:  Optional[float] = None   # None for zero-GP rows
    projected_fp_total:     Optional[float] = None
    marginal_fp:            float                    # F(R + candidate) - F(R)
    dynamic_value:          Optional[float] = None   # from DynamicPointsAgent
    rank:                   int
    is_exact_model:         bool = True
    approximation_note:     Optional[str] = None
    completion_ids:         list[int] = []
    auction_values:         Optional[PointsAuctionValues] = None   # auction mode only
    drop_player_id:         Optional[int] = None
    delta_fp:               Optional[float] = None
    candidate_games:        Optional[int] = None
    dropped_games:          Optional[int] = None
    daily_deltas:           Optional[dict[str, float]] = None


class PointsEvaluateResponse(BaseModel):
    scoring_system:  str = 'points'
    objective:       str = 'expected_points'
    units:           str = 'fantasy_points'
    horizon:         Optional[str] = None
    source_version:  Optional[str] = None
    candidates:      list[PointsCandidate]
    has_more:        bool = False
    total_candidates: int = 0


# ── /sessions/{id}/trade (points mode) ───────────────────────────────────────

class LeagueTradeImpactModel(BaseModel):
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


class PointsTradeAnalyzeResponse(BaseModel):
    scoring_system: str = 'points'
    team_a_pre_fp: float
    team_a_post_fp: float
    team_a_delta_fp: float
    team_a_net_delta_fp: float
    team_a_streamer_boost: float = 0.0
    team_b_pre_fp: float
    team_b_post_fp: float
    team_b_delta_fp: float
    team_b_net_delta_fp: float
    team_b_streamer_boost: float = 0.0
    is_legal: bool = True
    rejection_reason: Optional[str] = None
    error: Optional[str] = None
    team_a_opened_spots: int = 0
    team_b_opened_spots: int = 0
    league_impact: Optional[LeagueTradeImpactModel] = None
    your_net_delta_fp: Optional[float] = None
    their_net_delta_fp: Optional[float] = None


class PointsTradeSuggestionModel(BaseModel):
    partner_team_id: str
    user_sends: list[int]
    partner_sends: list[int]
    user_gain: float
    partner_gain: float
    total_gain: float = 0.0
    trade_format: str
    send: Optional[list[int]] = None
    receive: Optional[list[int]] = None
    user_delta_fp: Optional[float] = None
    partner_delta_fp: Optional[float] = None
    net_delta_fp: Optional[float] = None
    analysis: Optional[PointsTradeAnalyzeResponse] = None


class PointsTradeSuggestResponse(BaseModel):
    scoring_system: str = 'points'
    suggestions: list[PointsTradeSuggestionModel]

