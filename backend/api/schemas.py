"""
API request bodies and router-built response models — the half of the old models.py
that only the transport layer touches. Shared response DTOs that a service builds live
in backend.models (this module imports from it, never the reverse).
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field, model_validator

from backend.models import ComboParam, TRADE_THRESHOLD_FLOOR


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

class PointsConfigRequest(BaseModel):
    """Points-mode scoring configuration block inside LeagueSettings."""
    points_input_mode:  str = 'projected_points'   # 'projected_points' | 'raw_stats'
    projection_basis:   str = 'per_game'           # 'per_game' | 'season_total' | ...
    scoring_rules: dict[str, float] = {}
    projection_source: Optional[str] = None
    season:            Optional[str] = None
    captured_at:       Optional[str] = None
    auction_budget:  Optional[int] = None
    minimum_bid:     int = 1
    bid_increment:   int = 1
    savor_enabled:   bool = False
    savor_noise:     float = 0.0
    active_slots: Optional[dict[str, int]] = None
    bench_slots: Optional[int] = Field(default=None, ge=0)
    ir_slots: Optional[int] = Field(default=None, ge=0)
    lineup_lock_mode: str = 'pregame_daily'
    evaluation_window: Optional[tuple[str, str]] = None
    time_zone: str = 'America/New_York'

    @model_validator(mode='after')
    def validate_points_context(self):
        from datetime import date
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
        if self.lineup_lock_mode not in ('pregame_daily', 'weekly'):
            raise ValueError('Unsupported lineup lock mode.')
        if self.active_slots is not None and any(v < 0 for v in self.active_slots.values()):
            raise ValueError('Active slot counts must be nonnegative.')
        try:
            ZoneInfo(self.time_zone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError('Unknown analysis timezone.') from exc
        if self.evaluation_window:
            start, end = map(date.fromisoformat, self.evaluation_window)
            if end < start or (end - start).days > 365:
                raise ValueError('Analysis date range must contain 1 to 366 days.')
        return self


class LeagueSettings(BaseModel):
    sport: str
    n_drafters: int = Field(ge=2)
    n_picks: int
    scoring_format: str
    scoring_system: str = 'categories'   # 'categories' | 'points'
    most_categories_weight: Optional[float] = None
    tiebreaker_category: Optional[str] = None
    categories: list[str] = []
    cash_per_team: Optional[int] = None   # Auction Mode only
    points_config: Optional[PointsConfigRequest] = None

    @model_validator(mode='after')
    def check_most_categories_weight_matches_format(self) -> 'LeagueSettings':
        """Head to Head needs the dial; Rotisserie must not carry one. Rejected here rather than
        defaulted, so a client that forgets it hears about it instead of silently drafting to a
        different objective than the one it meant."""
        if self.scoring_system == 'points' or self.scoring_format in ('Points', 'Points League'):
            self.scoring_system = 'points'
            if not self.categories:
                self.categories = ['Points']
            return self
        if not self.categories:
            raise ValueError('categories must be a non-empty list. There is no server-side '
                             'default: the client chooses what the league scores.')
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
    # lambda_c pulls the category weights toward balanced, as a fraction of the category
    # optimizer's per-iteration step (0.05 shrinks up to 5% of a step; see REG_LAMBDA_UNIT).
    # lambda_p pulls the flex-position shares toward uniform, as a fraction of the SHARES
    # optimizer's step, derived per agent from the base-position count — 1.0 pins shares at
    # uniform on the first pick, values below are a continuous dial; the default 3.0 pins them
    # through the first five picks. Named lambda_c /
    # lambda_p rather than lambda, which is a Python keyword.
    lambda_c: float = 0.05
    lambda_p: float = 3.0
    # How sharply opponents are expected to pursue their predicted punts, and at 0 whether they are
    # modelled as strategic at all. Pinned to 1.0 under Rotisserie by the agent.
    opponent_model_confidence: float = 0.5
    n_iterations: int
    streaming_noise: float
    # SAVOR's replacement threshold in dollars (algorithm_helpers.savor_calculation): only outcomes above it count, a
    # slot below it being better used for streaming. 0 is vanilla SAVOR.
    replacement_threshold: float
    # How early the smooth cash-share rule starts raising a team's values to use cash it could not otherwise spend
    # (auction_pricing.smooth_spending_scale): 0 waits for an actual surplus, higher starts earlier.
    cash_share_softness: float
    # Force-weighting: when on, a category with a pinned weight is held at it and excluded from the
    # descent, constraining the optimiser into a chosen build. The pins themselves ride on the
    # evaluate request, not here -- they change the served board but nothing the pipeline builds, so
    # sending them as a setting would rebuild the agent for every keystroke.
    allow_force_weighting: bool = False


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
    # Season Mode never shows the base (empty-board) H-scores, it only ranks by them (trade suggestions, the waiver
    # comparison), so its sessions build them in one neutral pass instead of the opponent self-play. Draft and
    # Auction Mode (the other modes, like is_auction's False) keep the self-play.
    is_season_mode: bool = False
    platform: str = 'Enter your own data'
    slot_counts: dict[str, int]
    model_settings: ModelSettings
    data_source: DataSource
    injured_players: list[int] = []   # player ids, picked from the session's registry
    my_team_id: Optional[str] = None
    platform_config: Optional[PlatformConfigRequest] = None   # live platforms only


class PlayerGScore(BaseModel):
    player_id: int
    total: float
    values: list[float]   # per-category G-scores, same order as categories


class PlayerRegistryEntry(BaseModel):
    points_per_game: Optional[float] = None
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
    scoring_system: str = 'categories'   # 'categories' | 'points'
    categories: list[str]
    players: list[PlayerRegistryEntry]
    g_scores: list[PlayerGScore]
    expires_at: str


# ── /sessions/{id} PATCH ──────────────────────────────────────────────────────

class PatchLeague(BaseModel):
    n_drafters: Optional[int] = Field(default=None, ge=2)   # see LeagueSettings.n_drafters
    n_picks: Optional[int] = None
    scoring_format: Optional[str] = None
    scoring_system: Optional[str] = None
    points_config: Optional[PointsConfigRequest] = None
    # Omitted = unchanged, like every field here. The resulting format/weight pair is validated
    # after the patch merges, in session_management — only there is the outcome known.
    most_categories_weight: Optional[float] = None
    tiebreaker_category: Optional[str] = None
    categories: Optional[list[str]] = None
    cash_per_team: Optional[int] = None


class PatchRequest(BaseModel):
    from_step: int
    is_auction: Optional[bool] = None   # omitted = unchanged; True/False sets the league type
    is_season_mode: Optional[bool] = None   # omitted = unchanged
    model_settings: Optional[ModelSettings] = None
    league: Optional[PatchLeague] = None
    data_source: Optional[DataSource] = None
    slot_counts: Optional[dict[str, int]] = None
    injured_players: Optional[list[int]] = None   # player ids, picked from the session's registry
    platform: Optional[str] = None                            # set when connecting a live platform
    platform_config: Optional[PlatformConfigRequest] = None   # set when connecting a live platform


class PatchResponse(BaseModel):
    players: Optional[list[PlayerRegistryEntry]] = None
    ok: bool
    steps_rerun: list[int]


# ── /sessions/{id}/g-scores GET ───────────────────────────────────────────────

class GScoresResponse(BaseModel):
    players: list[PlayerRegistryEntry]
    g_scores: list[PlayerGScore]


# ── /sessions/{id}/evaluate ───────────────────────────────────────────────────

class EvaluateRequest(BaseModel):
    drop_player_id: Optional[int] = None
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
    # Force-weighting pins: category -> a weight on the scale the UI displays (100 = neutral), or
    # None/absent for "the algorithm sets this one". Ignored unless the session's model settings have
    # allow_force_weighting on, so the toggle alone governs the feature and the typed boxes survive
    # being toggled off and back on.
    forced_category_weights: Optional[dict[str, Optional[float]]] = None


# ── /sessions/{id}/hypothetical-team ─────────────────────────────────────────

class HypotheticalTeamRequest(BaseModel):
    # The board exactly as the evaluate that ranked the candidate stated it: the request reads that evaluate's
    # strategy for the candidate, keyed by these fields.
    player_assignments: dict[str, list[int]]
    my_team_id: str
    remaining_cash: Optional[dict[str, float]] = None   # Auction Mode only
    exclusion_list: list[int] = []
    forced_category_weights: Optional[dict[str, Optional[float]]] = None
    candidate_player_id: int


# ── /sessions/{id}/auction-autodraft ──────────────────────────────────────────

class AuctionNominationRequest(BaseModel):
    player_assignments: dict[str, list[int]]
    remaining_cash: dict[str, float]
    nominator_id: str
    # Both None: the nominator is an autodrafter and picks its own player, opening at $1.
    # Both set: a manual nomination.
    nominated_player_id: Optional[int]
    opening_bid: Optional[int]
    valuation_team_ids: list[str]


class AuctionNominationResponse(BaseModel):
    nominated_player_id: int
    nominated_player_name: str
    opening_bid: int
    valuations: dict[str, float]
    max_allowed_bids: dict[str, int]


class AuctionValuationRequest(BaseModel):
    player_assignments: dict[str, list[int]]
    remaining_cash: dict[str, float]
    player_id: int
    valuation_team_ids: list[str]


class AuctionValuationResponse(BaseModel):
    valuations: dict[str, float]


# ── /sessions/{id}/trade/analyze ─────────────────────────────────────────────

class TradeAnalyzeRequest(BaseModel):
    player_assignments: dict[str, list[int]]
    my_team: str
    their_team: str
    my_trade: list[int]
    their_trade: list[int]
    position_check: bool = True
    ignore_position_check: Optional[bool] = None

    @model_validator(mode='after')
    def resolve_position_check(self) -> 'TradeAnalyzeRequest':
        if self.ignore_position_check is not None:
            self.position_check = not self.ignore_position_check
        return self


# ── /sessions/{id}/trade/suggest ─────────────────────────────────────────────

class TradeSuggestRequest(BaseModel):
    player_assignments: dict[str, list[int]]
    my_team: str
    their_teams: Optional[list[str]] = None
    their_team: Optional[str] = None
    combo_params: list[ComboParam]
    your_differential_threshold: float = 0.0
    their_differential_threshold: float = 0.0
    position_check: bool = True
    ignore_position_check: Optional[bool] = None

    @model_validator(mode='after')
    def resolve_fields(self) -> 'TradeSuggestRequest':
        if self.their_teams is None and self.their_team is not None:
            self.their_teams = [self.their_team]
        elif self.their_teams is None:
            self.their_teams = []
        if self.ignore_position_check is not None:
            self.position_check = not self.ignore_position_check
        return self


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
    # Whether this LEAGUE drafts by auction, as opposed to which modes the PLATFORM supports.
    # None when the platform does not report it, which means "no opinion" rather than "no".
    is_auction_draft: Optional[bool] = None
    # The league's own position structure in the app's slot names, when the platform reports it.
    slot_counts: Optional[dict[str, int]] = None
    points_metadata: Optional[dict] = None


class EspnLeagueSeasonsResponse(BaseModel):
    seasons: list[int]               # ESPN's season numbers (2027 is 2026-27), newest first


class DraftStateResponse(BaseModel):
    player_assignments: dict[str, list[int]]
    injured_players: list[int]
    # team -> {position in that team's player_assignments: the platform's name}, for the players the session could not
    # match (carried there as the replacement player, -1). See PlatformSelections.
    unmatched_player_names: dict[str, dict[int, str]]
    # Season Mode: team -> the positions in its player_assignments that were empty on the platform and are filled with
    # the replacement player (-1) so the team is full (helpers.fill_empty_slots_with_replacement). Empty otherwise.
    empty_slots: dict[str, list[int]]
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
