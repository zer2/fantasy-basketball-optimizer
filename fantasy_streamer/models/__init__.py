"""Domain models for fantasy basketball schedule, projections, and opportunity penalties."""

from fantasy_streamer.models.penalties import (
    DropPenaltyCalculator,
    DropPenaltyResult,
    calculate_drop_penalty,
    calculate_effective_survival_horizon,
    calculate_matchup_urgency,
    calculate_waiver_snipe_probability,
    calculate_weekly_vorp,
)
from fantasy_streamer.models.projections import (
    Player,
    ProjectionsPool,
    compute_replacement_baseline,
    prune_candidate_pool,
)
from fantasy_streamer.models.schedule import (
    MatchupHorizon,
    ScheduleMatrix,
    create_synthetic_nba_schedule,
)

__all__ = [
    "DropPenaltyCalculator",
    "DropPenaltyResult",
    "calculate_drop_penalty",
    "calculate_effective_survival_horizon",
    "calculate_matchup_urgency",
    "calculate_waiver_snipe_probability",
    "calculate_weekly_vorp",
    "Player",
    "ProjectionsPool",
    "compute_replacement_baseline",
    "prune_candidate_pool",
    "MatchupHorizon",
    "ScheduleMatrix",
    "create_synthetic_nba_schedule",
]
