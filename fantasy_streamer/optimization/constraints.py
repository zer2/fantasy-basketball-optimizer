"""MILP constraints definition and formulation helpers for roster and lineup streaming."""

from dataclasses import dataclass
from typing import Dict, List, Set

from fantasy_streamer.config import LeagueConfig
from fantasy_streamer.models.projections import Player
from fantasy_streamer.models.schedule import MatchupHorizon, ScheduleMatrix


@dataclass
class OptimizationProblemData:
    """Consolidated input parameters for building the MILP problem."""

    config: LeagueConfig
    rostered_players: List[Player]
    candidate_free_agents: List[Player]
    schedule: ScheduleMatrix
    horizon: MatchupHorizon
    rem_acquisitions_w1: int
    cap_acquisitions_w2: int
    week1_urgency_weight: float = 1.0
    week2_urgency_weight: float = 0.85
    daily_discount_gamma: float = 0.98
    add_friction_epsilon: float = 0.01

    @property
    def all_players(self) -> List[Player]:
        return self.rostered_players + self.candidate_free_agents

    @property
    def droppable_roster(self) -> List[Player]:
        return [p for p in self.rostered_players if p.is_droppable]

    @property
    def non_droppable_roster(self) -> List[Player]:
        return [p for p in self.rostered_players if not p.is_droppable]


class ConstraintManager:
    """Pre-calculates eligibility tables, game matrices, and discount coefficients."""

    def __init__(self, data: OptimizationProblemData):
        self.data = data
        self.config = data.config
        self.horizon = data.horizon
        self.schedule = data.schedule

        # Map player_id -> Player object
        self.player_map: Dict[int, Player] = {
            p.player_id: p for p in data.all_players
        }

        # Map player_id -> set of eligible slots
        self.eligible_slots_map: Dict[int, Set[str]] = {
            p.player_id: self.config.get_eligible_slots(p.positions)
            for p in data.all_players
        }

        # Pre-compute game indicator: (player_id, day_idx) -> int (0 or 1)
        self.game_indicator: Dict[tuple, int] = {}
        for p in data.all_players:
            for d in range(self.horizon.total_days):
                has_g = 1 if self.schedule.has_game(p.team, d, p.player_id) else 0
                self.game_indicator[(p.player_id, d)] = has_g

        # Pre-compute objective weight for each day: U_w * gamma^(d - d_0)
        self.day_weights: Dict[int, float] = {}
        for d in range(self.horizon.total_days):
            week = self.horizon.get_week_of_day(d)
            u_w = (
                self.data.week1_urgency_weight
                if week == 1
                else self.data.week2_urgency_weight
            )
            discount = self.data.daily_discount_gamma ** d
            self.day_weights[d] = u_w * discount

    def is_eligible(self, player_id: int, slot: str) -> bool:
        return slot in self.eligible_slots_map.get(player_id, set())

    def has_game(self, player_id: int, day_idx: int) -> bool:
        return self.game_indicator.get((player_id, day_idx), 0) == 1
