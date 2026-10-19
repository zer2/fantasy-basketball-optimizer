"""
Points scoring configuration.

Typed dataclass that carries the full scoring-system description for a points-mode
session.  The router validates the raw request and constructs one of these; every
downstream function receives it explicitly rather than reading current_params directly.

Incompatible combinations fail at construction time with a descriptive message rather
than at evaluation time with an opaque crash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


# ── Types ─────────────────────────────────────────────────────────────────────

ScoringSystem    = Literal['categories', 'points']
CompetitionFmt   = Literal['head_to_head', 'season_total']
PointsObjective  = Literal['expected_points']
InputMode        = Literal['projected_points', 'raw_stats']
ProjectionBasis  = Literal['per_game', 'per_week', 'season_total', 'rest_of_season_total']
LineupLockMode   = Literal['pregame_daily', 'weekly']


@dataclass(frozen=True)
class PointsScoringConfig:
    """Immutable description of one points-mode session's rules.

    Construct via ``build_points_scoring_config``; never build directly from a
    request dict without calling that function so validation always runs.
    """

    # ── Scoring system identity ────────────────────────────────────────────
    scoring_system:     ScoringSystem   = 'points'
    competition_format: CompetitionFmt  = 'head_to_head'
    points_objective:   PointsObjective = 'expected_points'

    # ── Projection input ───────────────────────────────────────────────────
    points_input_mode:  InputMode       = 'projected_points'
    projection_basis:   ProjectionBasis = 'per_game'

    # {stat_name: weight}; negative weights are valid.
    # Required when points_input_mode == 'raw_stats'.  Must be absent (or empty)
    # when points_input_mode == 'projected_points'.
    scoring_rules: dict[str, float] = field(default_factory=dict)

    # Provenance / audit
    projection_source: Optional[str] = None   # e.g. 'basketball_monster'
    season:            Optional[str] = None   # e.g. '2025-26'
    captured_at:       Optional[str] = None   # ISO-8601 timestamp of the file
    decision_as_of:    Optional[str] = None   # ISO-8601 timestamp of the run

    # ── Roster structure ───────────────────────────────────────────────────
    # active_slots mirrors the league's slot_counts for scoring slots only.
    active_slots: dict[str, int] = field(default_factory=dict)
    bench_slots:  int = 4
    ir_slots:     int = 3

    # ── Schedule / window ─────────────────────────────────────────────────
    lineup_lock_mode: LineupLockMode = 'pregame_daily'
    # (start_date, end_date) as ISO-8601 date strings; None means full season.
    evaluation_window: Optional[tuple[str, str]] = None

    # ── Auction settings ───────────────────────────────────────────────────
    auction_budget:  Optional[int] = None   # per-team starting budget
    minimum_bid:     int = 1
    bid_increment:   int = 1
    # SAVOR: off by default for points; apply only to discretionary premiums.
    savor_enabled:   bool = False
    savor_noise:     float = 0.0


# ── Factory / validation ──────────────────────────────────────────────────────

class PointsConfigError(ValueError):
    """An incompatible or missing field in a points-mode configuration."""


def build_points_scoring_config(
    points_input_mode:  str
    , projection_basis: str
    , scoring_rules:    dict[str, float]
    , competition_format: str = 'head_to_head'
    , active_slots:     dict[str, int] | None = None
    , bench_slots:      int = 4
    , ir_slots:         int = 3
    , lineup_lock_mode: str = 'pregame_daily'
    , evaluation_window: tuple[str, str] | None = None
    , projection_source: str | None = None
    , season:           str | None = None
    , captured_at:      str | None = None
    , decision_as_of:   str | None = None
    , auction_budget:   int | None = None
    , minimum_bid:      int = 1
    , bid_increment:    int = 1
    , savor_enabled:    bool = False
    , savor_noise:      float = 0.0
) -> PointsScoringConfig:
    """Validate inputs and return an immutable ``PointsScoringConfig``.

    Raises ``PointsConfigError`` for any incompatible combination so that
    callers receive a clear message rather than a deep internal crash.
    """
    _validate_mode(points_input_mode, InputMode)
    _validate_mode(projection_basis, ProjectionBasis)
    _validate_mode(competition_format, CompetitionFmt)
    _validate_mode(lineup_lock_mode, LineupLockMode)

    if points_input_mode == 'raw_stats' and not scoring_rules:
        raise PointsConfigError(
            "points_input_mode 'raw_stats' requires a non-empty scoring_rules dict "
            "(stat_name → weight). Provide at least one stat-weight pair."
        )

    if points_input_mode == 'projected_points' and scoring_rules:
        raise PointsConfigError(
            "points_input_mode 'projected_points' means the provider's points are used "
            "directly; scoring_rules must be empty. Remove scoring_rules or switch "
            "input mode to 'raw_stats'."
        )

    if minimum_bid < 1:
        raise PointsConfigError(
            f"minimum_bid must be at least 1; got {minimum_bid}."
        )

    if bid_increment < 1:
        raise PointsConfigError(
            f"bid_increment must be at least 1; got {bid_increment}."
        )

    if projection_basis in ('per_week', 'season_total', 'rest_of_season_total'):
        if points_input_mode == 'raw_stats':
            raise PointsConfigError(
                f"projection_basis '{projection_basis}' with input_mode 'raw_stats' "
                "requires season-total stat columns, not per-game stats. "
                "Divide totals by projected games first, or use basis 'per_game'."
            )

    return PointsScoringConfig(
        scoring_system     = 'points'
        , competition_format = competition_format
        , points_objective   = 'expected_points'
        , points_input_mode  = points_input_mode
        , projection_basis   = projection_basis
        , scoring_rules      = dict(scoring_rules)
        , projection_source  = projection_source
        , season             = season
        , captured_at        = captured_at
        , decision_as_of     = decision_as_of
        , active_slots       = dict(active_slots or {})
        , bench_slots        = bench_slots
        , ir_slots           = ir_slots
        , lineup_lock_mode   = lineup_lock_mode
        , evaluation_window  = evaluation_window
        , auction_budget     = auction_budget
        , minimum_bid        = minimum_bid
        , bid_increment      = bid_increment
        , savor_enabled      = savor_enabled
        , savor_noise        = savor_noise
    )


# ── helpers ───────────────────────────────────────────────────────────────────

def _validate_mode(value: str, expected_type) -> None:
    """Raise PointsConfigError if value is not one of the Literal's valid strings."""
    # Extract the allowed strings from a Literal type.
    allowed = getattr(expected_type, '__args__', None)
    if allowed is None:
        return
    if value not in allowed:
        raise PointsConfigError(
            f"Invalid value '{value}'. Allowed: {', '.join(repr(a) for a in allowed)}."
        )
