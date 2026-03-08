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
    sport: str = 'NBA'
    n_drafters: int = 10
    n_picks: int = 13
    scoring_format: str = 'Head to Head: Most Categories'
    categories: list[str] = []
    cash_per_team: Optional[int] = None   # Auction Mode only


class ModelParameters(BaseModel):
    omega: float = 0.7     # parameters.yaml punting_defaults.Moderate punting
    gamma: float = 0.25    # parameters.yaml punting_defaults.Moderate punting
    beth: float = 3.0      # parameters.yaml options.beth.default
    upsilon: float = 1.0   # parameters.yaml options.upsilon.default (NBA)
    psi: float = 0.8       # parameters.yaml options.psi.default
    chi: float = 0.6       # parameters.yaml options.chi.default
    aleph: float = 0.2     # parameters.yaml options.aleph.default
    n_iterations: int = 30 # parameters.yaml punting_defaults.Moderate punting
    streaming_noise: float = 10.0  # parameters.yaml options.S.default


class DataSource(BaseModel):
    type: str = 'mock'
    season: Optional[str] = None                           # 'historical' type only
    blend_weights: Optional[dict[str, float]] = None          # 'blended' type only
    custom_data_ids: Optional[dict[str, Optional[str]]] = None  # 'csv' / 'blended'


class SessionRequest(BaseModel):
    league: LeagueSettings
    platform: str = 'Enter your own data'
    slot_counts: dict[str, int] = {}
    parameters: ModelParameters = ModelParameters()
    data_source: DataSource = DataSource()
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


class TradeSuggestion(BaseModel):
    send: list[str]
    receive: list[str]
    your_score: float
    their_score: float


class TradeSuggestResponse(BaseModel):
    suggestions: list[TradeSuggestion]
