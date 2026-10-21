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

class LeagueSettings(BaseModel):
    sport: str
    # At least two, because a league with one drafter has no opponents — and the whole model is
    # built on the distribution of a matchup. Below two it does not score badly, it fails deep in
    # the pipeline on an empty opponent set. The sidebar input already enforces the same floor;
    # the platform path is what could smuggle a smaller number in, from a draft room that has
    # only one team in it so far.
    n_drafters: int = Field(ge=2)
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
    n_drafters: Optional[int] = Field(default=None, ge=2)   # see LeagueSettings.n_drafters
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
    is_season_mode: Optional[bool] = None   # omitted = unchanged
    model_settings: Optional[ModelSettings] = None
    league: Optional[PatchLeague] = None
    data_source: Optional[DataSource] = None
    slot_counts: Optional[dict[str, int]] = None
    injured_players: Optional[list[int]] = None   # player ids, picked from the session's registry
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


# ── /sessions/{id}/trade/suggest ─────────────────────────────────────────────

class TradeSuggestRequest(BaseModel):
    player_assignments: dict[str, list[int]]
    my_team: str
    # Every team searched, each with the same combos: one request covers one value-gap band (see ComboParam) across the
    # league, so the frontend's batches are bands rather than teams.
    their_teams: list[str]
    combo_params: list[ComboParam]
    # H-score gains as fractions, each side strictly above its threshold. Never below TRADE_THRESHOLD_FLOOR: the
    # frontend searches at the floor and filters stricter thresholds itself, so the floor bounds what one search
    # returns (2026-10-13: 101,480 3-for-3 trades, 16.7 MB, in a 16-team league).
    your_differential_threshold: float = Field(ge=TRADE_THRESHOLD_FLOOR)
    their_differential_threshold: float = Field(ge=TRADE_THRESHOLD_FLOOR)
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
    # Whether this LEAGUE drafts by auction, as opposed to which modes the PLATFORM supports.
    # None when the platform does not report it, which means "no opinion" rather than "no".
    is_auction_draft: Optional[bool] = None
    # The league's own position structure in the app's slot names, when the platform reports it.
    slot_counts: Optional[dict[str, int]] = None


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
