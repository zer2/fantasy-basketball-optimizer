"""
Pydantic request / response models matching api_spec.md.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


# ── /data/upload ──────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    data_id: str
    file_type: str
    n_players: int
    expires_at: str


# ── /sessions POST ────────────────────────────────────────────────────────────

class LeagueSettings(BaseModel):
    sport: str
    n_drafters: int
    n_picks: int
    scoring_format: str
    categories: list[str] = []
    cash_per_team: Optional[int] = None   # Auction Mode only


class ModelParameters(BaseModel):
    omega: float
    gamma: float
    beth: float
    upsilon: float
    psi: float
    chi: float
    aleph: float
    n_iterations: int
    streaming_noise: float


class DataSource(BaseModel):
    type: str
    season: Optional[str] = None                           # 'historical' type only
    blend_weights: Optional[dict[str, float]] = None          # 'projections' type only
    custom_data_ids: Optional[dict[str, Optional[str]]] = None  # 'csv' / 'projections'


class SessionRequest(BaseModel):
    league: LeagueSettings
    platform: str = 'Enter your own data'
    slot_counts: dict[str, int]
    parameters: ModelParameters
    data_source: DataSource
    injured_players: list[str] = []
    my_team_id: Optional[str] = None


class PlayerGScore(BaseModel):
    name: str
    total: float
    values: list[float]   # per-category G-scores, same order as categories


class SessionResponse(BaseModel):
    session_id: str
    n_players_loaded: int
    categories: list[str]
    g_scores: list[PlayerGScore]
    expires_at: str


# ── /sessions/{id} PATCH ──────────────────────────────────────────────────────

class PatchLeague(BaseModel):
    n_drafters: Optional[int] = None
    n_picks: Optional[int] = None
    scoring_format: Optional[str] = None
    categories: Optional[list[str]] = None
    cash_per_team: Optional[int] = None


class PatchRequest(BaseModel):
    from_step: int
    parameters: Optional[ModelParameters] = None
    league: Optional[PatchLeague] = None
    data_source: Optional[DataSource] = None
    slot_counts: Optional[dict[str, int]] = None
    injured_players: Optional[list[str]] = None


class PatchResponse(BaseModel):
    ok: bool
    steps_rerun: list[int]


# ── /sessions/{id}/g-scores GET ───────────────────────────────────────────────

class GScoresResponse(BaseModel):
    g_scores: list[PlayerGScore]


# ── /sessions/{id}/evaluate ───────────────────────────────────────────────────

class EvaluateRequest(BaseModel):
    player_assignments: dict[str, list[str]]
    my_team_id: str
    remaining_cash: Optional[dict[str, float]] = None   # Auction Mode only
    exclusion_list: list[str] = []


class GScoreRow(BaseModel):
    label: str
    values: list[float]
    total: float
    is_total: bool


class FlexRow(BaseModel):
    label: str
    values: list[Optional[float]]   # null = ineligible position for this flex slot
    is_total: bool


class FlexAllocations(BaseModel):
    base_positions: list[str]
    rows: list[FlexRow]


class RosterAssignment(BaseModel):
    name: str
    is_candidate: bool


class Roster(BaseModel):
    slots: list[str]
    assignments: dict[str, Optional[RosterAssignment]]


class AuctionValues(BaseModel):
    your_dollar:   float   # SAVOR on H-scores, team-specific, current cash/picks
    gnrc_dollar:   float   # SAVOR on H-scores, current cash/picks (generic baseline)
    orig_dollar:   float   # SAVOR on H-scores, full original cash/picks
    gnrc_dollar_g: float   # SAVOR on G-scores, current cash/picks
    orig_dollar_g: float   # SAVOR on G-scores, full original cash/picks


class Candidate(BaseModel):
    name: str
    position: str
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


# ── /sessions/{id}/trade/analyze ─────────────────────────────────────────────

class TradeAnalyzeRequest(BaseModel):
    player_assignments: dict[str, list[str]]
    my_team: str
    their_team: str
    my_trade: list[str]
    their_trade: list[str]
    ignore_position_check: bool = False


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

class ComboParam(BaseModel):
    n_traded: int
    n_received: int
    threshold: float


class TradeSuggestRequest(BaseModel):
    player_assignments: dict[str, list[str]]
    my_team: str
    their_team: str
    combo_params: list[ComboParam]
    your_differential_threshold: float = 0.0
    their_differential_threshold: float = -0.20
    ignore_position_check: bool = False


class TradeSuggestion(BaseModel):
    send: list[str]
    receive: list[str]
    your_score: float
    their_score: float


class TradeSuggestResponse(BaseModel):
    suggestions: list[TradeSuggestion]
