"""Player projections container, replacement baseline calculator, and candidate pruning."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from fantasy_streamer.config import DEFAULT_TOP_K_REPLACEMENT, POSITION_TO_SLOTS


@dataclass
class Player:
    """Represents a fantasy player with projections, positions, and roster status."""

    player_id: int
    name: str
    team: str
    positions: List[str]
    fppg: float  # Projected Fantasy Points Per Game (mu_p)
    is_rostered: bool = False
    is_droppable: bool = True  # Whether player can be considered for streaming drops
    drop_penalty_c_p: float = 0.0

    @property
    def primary_position(self) -> str:
        return self.positions[0] if self.positions else "UTIL"

    @property
    def position_group(self) -> str:
        """Coarse position group: Guard (G), Forward (F), or Center (C)."""
        pos_upper = {p.upper() for p in self.positions}
        if "C" in pos_upper:
            return "C"
        if "SF" in pos_upper or "PF" in pos_upper:
            return "F"
        return "G"


@dataclass
class ProjectionsPool:
    """Collection of rostered players and available free agents."""

    rostered_players: List[Player] = field(default_factory=list)
    free_agents: List[Player] = field(default_factory=list)

    @property
    def all_players(self) -> List[Player]:
        return self.rostered_players + self.free_agents

    def get_player_by_id(self, player_id: int) -> Optional[Player]:
        for p in self.all_players:
            if p.player_id == player_id:
                return p
        return None


def compute_replacement_baseline(
    free_agents: List[Player],
    k: int = DEFAULT_TOP_K_REPLACEMENT,
) -> float:
    """Compute the replacement baseline mu_rep as the mean projected FPPG of top K free agents.

    If fewer than K free agents are available, averages all available free agents.
    If no free agents are available, returns a safe default baseline (e.g., 18.0 FPPG).
    """
    if not free_agents:
        return 18.0

    sorted_fa = sorted(free_agents, key=lambda p: p.fppg, reverse=True)
    top_k = sorted_fa[:k]
    return sum(p.fppg for p in top_k) / float(len(top_k))


def prune_candidate_pool(
    free_agents: List[Player],
    max_per_team_per_pos: int = 3,
    max_total_candidates: Optional[int] = None,
) -> List[Player]:
    """Prune candidate free agents to top N per team per position group.

    Limits P_FA to top candidates (e.g. top 3 per team per G/F/C group) sorted by FPPG.
    This shrinks the binary decision space significantly to keep MILP solve times < 50ms
    while retaining all high-yield streaming targets.
    """
    # Group by (team, position_group)
    grouped: Dict[tuple, List[Player]] = {}
    for p in free_agents:
        key = (p.team.upper(), p.position_group)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(p)

    pruned: List[Player] = []
    for key, players in grouped.items():
        # Sort descending by FPPG
        players_sorted = sorted(players, key=lambda x: x.fppg, reverse=True)
        pruned.extend(players_sorted[:max_per_team_per_pos])

    if max_total_candidates is not None and len(pruned) > max_total_candidates:
        pruned = sorted(pruned, key=lambda x: x.fppg, reverse=True)[:max_total_candidates]

    return pruned
