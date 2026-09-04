"""
API request bodies and router-built response models — the half of the old models.py
that only the transport layer touches. Shared response DTOs that a service builds live
in backend.models (this module imports from it, never the reverse).
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, model_validator

from backend.models import ComboParam


# ── /data/upload ──────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    data_id: str
    n_players: int
    expires_at: str
    # Standard stat columns the file does NOT carry (a source that pairs projections with
    # a league exports only that league's categories) — shown in the upload caption so a
    # lighter file is a visible, deliberate state rather than a surprise at
    # category-selection time.
    missing_stats: list[str] = []


# ── /sessions POST ────────────────────────────────────────────────────────────

class LeagueSettings(BaseModel):
    sport: str
    n_drafters: int
    n_picks: int
    scoring_format: str
    # How much of the Head-to-Head objective is winning the majority of categories, the rest
    # being each category on its own: 0 is Each Category, 1 is Most Categories, and anything
    # between blends them. None under Rotisserie, which scores neither way.
    most_categories_weight: Optional[float] = None
    # The category that settles a tied matchup by counting for two. Head to Head with an even
    # number of categories only; None everywhere else, including at weight 0, where every
    # category is scored on its own and nothing can tie.
    tiebreaker_category: Optional[str] = None
    categories: list[str] = []
    cash_per_team: Optional[int] = None   # Auction Mode only

    @model_validator(mode='after')
    def check_most_categories_weight_matches_format(self) -> 'LeagueSettings':
        """Head to Head needs the dial; Rotisserie must not carry one. Rejected here rather than
        defaulted, so a client that forgets it hears about it instead of silently drafting to a
        different objective than the one it meant."""
        if self.scoring_format == 'Rotisserie':
            if self.most_categories_weight is not None:
                raise ValueError('most_categories_weight does not apply to Rotisserie.')
        elif self.most_categories_weight is None:
            raise ValueError('most_categories_weight is required for Head to Head (0 = Each '
                             'Category, 1 = Most Categories).')
        elif not 0.0 <= self.most_categories_weight <= 1.0:
            raise ValueError('most_categories_weight must be between 0 and 1.')
        if self.tiebreaker_category is not None:
            if self.categories and self.tiebreaker_category not in self.categories:
                raise ValueError('tiebreaker_category must be one of the scored categories.')
            if self.categories and len(self.categories) % 2 == 1:
                raise ValueError('A tiebreaker needs an even number of categories.')
        return self


class ModelSettings(BaseModel):
    # Window of the truncated-max future-pick model: how many surviving players a future
    # pick effectively chooses among (the punt-aggressiveness dial). Defaulted so clients
    # predating the model still work.
    pick_pool_size: int = 25
    beth: float
    upsilon: float
    psi: float
    chi: float
    aleph: float
    kappa: float = 0.3
    # Peak L1 pull of category weights toward neutral, as a fraction of the descent's
    # per-iteration category step (see REG_LAMBDA_UNIT): 0.05 shrinks up to 5% of a step.
    # Named reg_lambda rather than lambda, which is a Python keyword.
    reg_lambda: float = 0.05
    # How sharply opponents are expected to pursue their predicted punts, and at 0 whether they are
    # modelled as strategic at all. Pinned to 1.0 under Rotisserie by the agent.
    opponent_model_confidence: float = 0.5
    n_iterations: int
    streaming_noise: float


class DataSource(BaseModel):
    type: str
    season: Optional[str] = None                     # 'historical' type only
    blend_weights: Optional[dict[str, float]] = None    # 'projections' — keys: ESPN, DARKO, and upload data_ids
    custom_data_ids: Optional[list[str]] = None         # 'csv' / 'projections' — uploaded data_ids


class PlatformConfigRequest(BaseModel):
    league_id: str
    division_id: Optional[str] = None


class SessionRequest(BaseModel):
    league: LeagueSettings
    # The session's league type: auction sessions require remaining_cash on every evaluate,
    # non-auction sessions forbid it. Patched whenever the user switches modes.
    is_auction: bool = False
    platform: str = 'Enter your own data'
    slot_counts: dict[str, int]
    model_settings: ModelSettings
    data_source: DataSource
    injured_players: list[str] = []
    my_team_id: Optional[str] = None
    platform_config: Optional[PlatformConfigRequest] = None   # live platforms only


class PlayerGScore(BaseModel):
    player_id: int
    total: float
    values: list[float]   # per-category G-scores, same order as categories


class PlayerRegistryEntry(BaseModel):
    """One session player identity: everything the display layer needs to render a player.
    has_headshot is False for the RP sentinel and synthetic ids (no NBA CDN image exists)."""
    player_id: int
    name: str
    last_name: str
    positions: list[str]
    has_headshot: bool


class SessionResponse(BaseModel):
    session_id: str
    n_players_loaded: int
    categories: list[str]
    players: list[PlayerRegistryEntry]
    g_scores: list[PlayerGScore]
    expires_at: str


# ── /sessions/{id} PATCH ──────────────────────────────────────────────────────

class PatchLeague(BaseModel):
    n_drafters: Optional[int] = None
    n_picks: Optional[int] = None
    scoring_format: Optional[str] = None
    # Omitted = unchanged, like every field here. The resulting format/weight pair is validated
    # after the patch merges, in session_management — only there is the outcome known.
    most_categories_weight: Optional[float] = None
    tiebreaker_category: Optional[str] = None
    categories: Optional[list[str]] = None
    cash_per_team: Optional[int] = None


class PatchRequest(BaseModel):
    from_step: int
    is_auction: Optional[bool] = None   # omitted = unchanged; True/False sets the league type
    model_settings: Optional[ModelSettings] = None
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
    players: list[PlayerRegistryEntry]
    g_scores: list[PlayerGScore]


# ── /sessions/{id}/evaluate ───────────────────────────────────────────────────

class EvaluateRequest(BaseModel):
    player_assignments: dict[str, list[int]]
    my_team_id: str
    remaining_cash: Optional[dict[str, float]] = None   # Auction Mode only
    exclusion_list: list[int] = []
    # Draft/waiver batching: evaluate only a slice of the candidate pool (ordered by the cached
    # default/generic H-score ranking) so the top players can paint before the deep bench is scored.
    # candidate_limit=None evaluates everyone (auction always does; the first eval does too, since it
    # is what establishes the generic ranking).
    candidate_offset: int = 0
    candidate_limit: Optional[int] = None


# ── /sessions/{id}/trade/analyze ─────────────────────────────────────────────

class TradeAnalyzeRequest(BaseModel):
    player_assignments: dict[str, list[int]]
    my_team: str
    their_team: str
    my_trade: list[int]
    their_trade: list[int]
    position_check: bool = True


# ── /sessions/{id}/trade/suggest ─────────────────────────────────────────────

class TradeSuggestRequest(BaseModel):
    player_assignments: dict[str, list[int]]
    my_team: str
    their_team: str
    combo_params: list[ComboParam]
    your_differential_threshold: float = 0.0
    their_differential_threshold: float = -0.20
    position_check: bool = True


# ── /platforms/* (live platform integration) ─────────────────────────────────

class DivisionsResponse(BaseModel):
    divisions: list[dict]            # [{name, id}]; empty when the league has none


class LeaguesResponse(BaseModel):
    leagues: list[dict]              # [{id, name, season}]; empty for manual-id platforms (Fantrax)


class ConnectResponse(BaseModel):
    team_names: list[str]
    n_drafters: int
    n_picks: int
    available_modes: list[str]


class DraftStateResponse(BaseModel):
    player_assignments: dict[str, list[int]]
    injured_players: list[int]
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
