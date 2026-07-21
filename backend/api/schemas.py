"""
API request bodies and router-built response models — the half of the old models.py
that only the transport layer touches. Shared response DTOs that a service builds live
in backend.models (this module imports from it, never the reverse).
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel

from backend.models import ComboParam


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
    reg_strength: float = 0.00005
    n_iterations: int
    streaming_noise: float


class DataSource(BaseModel):
    type: str
    season: Optional[str] = None                           # 'historical' type only
    blend_weights: Optional[dict[str, float]] = None          # 'projections' type only
    custom_data_ids: Optional[dict[str, Optional[str]]] = None  # 'csv' / 'projections'


class PlatformConfigRequest(BaseModel):
    league_id: str
    division_id: Optional[str] = None


class SessionRequest(BaseModel):
    league: LeagueSettings
    platform: str = 'Enter your own data'
    slot_counts: dict[str, int]
    parameters: ModelParameters
    data_source: DataSource
    injured_players: list[str] = []
    my_team_id: Optional[str] = None
    platform_config: Optional[PlatformConfigRequest] = None   # live platforms only


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
    platform: Optional[str] = None                            # set when connecting a live platform
    platform_config: Optional[PlatformConfigRequest] = None   # set when connecting a live platform


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
    # Draft/waiver batching: evaluate only a slice of the candidate pool (ordered by the cached
    # default/generic H-score ranking) so the top players can paint before the deep bench is scored.
    # candidate_limit=None evaluates everyone (auction always does; the first eval does too, since it
    # is what establishes the generic ranking).
    candidate_offset: int = 0
    candidate_limit: Optional[int] = None


# ── /sessions/{id}/trade/analyze ─────────────────────────────────────────────

class TradeAnalyzeRequest(BaseModel):
    player_assignments: dict[str, list[str]]
    my_team: str
    their_team: str
    my_trade: list[str]
    their_trade: list[str]
    ignore_position_check: bool = False


# ── /sessions/{id}/trade/suggest ─────────────────────────────────────────────

class TradeSuggestRequest(BaseModel):
    player_assignments: dict[str, list[str]]
    my_team: str
    their_team: str
    combo_params: list[ComboParam]
    your_differential_threshold: float = 0.0
    their_differential_threshold: float = -0.20
    ignore_position_check: bool = False


# ── /platforms/* (live platform integration) ─────────────────────────────────

class DivisionsResponse(BaseModel):
    divisions: list[dict]            # [{name, id}]; empty when the league has none


class LeaguesResponse(BaseModel):
    leagues: list[dict]              # [{id, name, season}]; empty for manual-id platforms (Fantrax)


class ConnectRequest(BaseModel):
    league_id: str
    division_id: Optional[str] = None


class ConnectResponse(BaseModel):
    team_names: list[str]
    n_drafters: int
    n_picks: int
    available_modes: list[str]


class DraftStateResponse(BaseModel):
    player_assignments: dict[str, list[str]]
    injured_players: list[str]
    status: str
    remaining_cash: Optional[dict[str, float]] = None   # Auction Mode only


# Yahoo OAuth (manual code-paste flow)

class YahooAuthUrlResponse(BaseModel):
    auth_url: str


class YahooTokenRequest(BaseModel):
    auth_code: str


# ESPN auth (s2 + SWID cookies)

class EspnCredentialsRequest(BaseModel):
    s2: str
    swid: str
