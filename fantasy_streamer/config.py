"""League configuration and roster settings for the rolling-horizon MILP streamer."""

from dataclasses import dataclass, field
from typing import Dict, List, Set

# Standard ESPN Points League Roster
TOTAL_ROSTER_SIZE: int = 13
ACTIVE_STARTERS_COUNT: int = 10
BENCH_SPOTS_COUNT: int = 3

# Active roster slot names (L)
ACTIVE_SLOTS: List[str] = [
    "PG",
    "SG",
    "SF",
    "PF",
    "C",
    "G",
    "F",
    "UTIL1",
    "UTIL2",
    "UTIL3",
]

# Primary position to slot eligibility mapping (E_p)
POSITION_TO_SLOTS: Dict[str, List[str]] = {
    "PG": ["PG", "G", "UTIL1", "UTIL2", "UTIL3"],
    "SG": ["SG", "G", "UTIL1", "UTIL2", "UTIL3"],
    "SF": ["SF", "F", "UTIL1", "UTIL2", "UTIL3"],
    "PF": ["PF", "F", "UTIL1", "UTIL2", "UTIL3"],
    "C": ["C", "UTIL1", "UTIL2", "UTIL3"],
}

# Transaction & Horizon Defaults
DEFAULT_WEEKLY_ADD_CAP: int = 7
DEFAULT_HORIZON_DAYS: int = 10
ADD_FRICTION_EPSILON: float = 0.01
DAILY_DISCOUNT_FACTOR: float = 0.98

# Penalty Model Defaults
DEFAULT_TOP_K_REPLACEMENT: int = 10
DEFAULT_EFFECTIVE_GAMES_PER_WEEK: float = 3.0
DEFAULT_RETENTION_RATE: float = 0.75
DEFAULT_MATCHUP_SIGMA: float = 65.0
DEFAULT_URGENCY_MAX_CLAMP: float = 3.0


@dataclass
class LeagueConfig:
    roster_size: int = TOTAL_ROSTER_SIZE
    active_starters: int = ACTIVE_STARTERS_COUNT
    bench_spots: int = BENCH_SPOTS_COUNT
    active_slots: List[str] = field(default_factory=lambda: list(ACTIVE_SLOTS))
    weekly_acquisition_cap: int = DEFAULT_WEEKLY_ADD_CAP
    default_horizon_days: int = DEFAULT_HORIZON_DAYS
    add_friction_epsilon: float = ADD_FRICTION_EPSILON
    daily_discount_factor: float = DAILY_DISCOUNT_FACTOR

    # Drop penalty parameters
    top_k_replacement: int = DEFAULT_TOP_K_REPLACEMENT
    effective_games_per_week: float = DEFAULT_EFFECTIVE_GAMES_PER_WEEK
    retention_rate_gamma: float = DEFAULT_RETENTION_RATE
    matchup_sigma: float = DEFAULT_MATCHUP_SIGMA
    urgency_max_clamp: float = DEFAULT_URGENCY_MAX_CLAMP

    def get_eligible_slots(self, player_positions: List[str]) -> Set[str]:
        """Compute the set of eligible slots for a player given their position list."""
        eligible = set()
        for pos in player_positions:
            pos_clean = pos.strip().upper()
            if pos_clean in POSITION_TO_SLOTS:
                eligible.update(POSITION_TO_SLOTS[pos_clean])
        # Ensure only active slots in league are returned
        return eligible.intersection(set(self.active_slots))
