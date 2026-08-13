"""
Shared response/DTO models — the ones the service tier (evaluate, trading) builds
and the API tier declares as `response_model`. Both tiers import this, so it lives
at the backend top level and imports nothing internal. Request bodies and router-only
responses live in `backend.api.schemas`.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
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
    is_candidate: bool


@dataclass
class Roster:
    slots: list[str]
    assignments: dict[str, Optional[RosterAssignment]]


class AuctionValues(BaseModel):
    your_dollar:   float   # SAVOR on H-scores, team-specific, current cash/picks
    gnrc_dollar:   float   # SAVOR on H-scores, current cash/picks (generic baseline)
    orig_dollar:   float   # SAVOR on H-scores, full original cash/picks
    gnrc_dollar_g: float   # SAVOR on G-scores, current cash/picks
    orig_dollar_g: float   # SAVOR on G-scores, full original cash/picks


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

class ComboParam(BaseModel):
    n_traded: int
    n_received: int
    threshold: float


class TradeSuggestion(BaseModel):
    send: list[int]
    receive: list[int]
    your_score: float
    their_score: float


class TradeSuggestResponse(BaseModel):
    suggestions: list[TradeSuggestion]
