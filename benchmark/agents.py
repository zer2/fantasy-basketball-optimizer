# benchmark/agents.py
"""Draft agents. All expose make_pick(player_assignments, seat, rng) -> player."""
import numpy as np

from src.math.algorithm_agents import get_default_h_values
from src.helpers.helper_functions import gen_key
from benchmark.opponent_model import weighted_softmax_pick, _eligible

def _all_taken(player_assignments):
    return [p for v in player_assignments.values() for p in v if p == p]

class RandomAgent:
    def __init__(self, info):
        self.positions = info['Positions']
        self.pool = list(info['G-scores'].index)

    def make_pick(self, player_assignments, seat, rng):
        taken = set(_all_taken(player_assignments))
        avail = _eligible([p for p in self.pool if p not in taken],
                          self.positions, player_assignments[seat])
        return str(rng.choice(avail)) if avail else None

class GScoreAgent:
    def __init__(self, info, temperature=0.0, top_k=15):
        self.ranking = info['G-scores']['Total'].sort_values(ascending=False)
        self.positions = info['Positions']
        self.temperature = temperature
        self.top_k = top_k

    def make_pick(self, player_assignments, seat, rng):
        taken = set(_all_taken(player_assignments))
        avail = [p for p in self.ranking.index if p not in taken]
        return weighted_softmax_pick(self.ranking, avail, self.temperature, rng,
                                     self.positions, player_assignments[seat], self.top_k)

class HScoreAgent:
    """Field/hero agent using a cached static H-score ordering (cheap; no per-pick descent)."""
    def __init__(self, info, cfg, temperature=0.0, top_k=15):
        h = get_default_h_values(
            gen_key(), cfg.omega, cfg.gamma, cfg.n_starters, cfg.n_drafters,
            cfg.n_iterations, cfg.beth, cfg.scoring_format)
        self.ranking = h.set_index('Player')['H-score'].sort_values(ascending=False)
        self.positions = info['Positions']
        self.temperature = temperature
        self.top_k = top_k

    def make_pick(self, player_assignments, seat, rng):
        taken = set(_all_taken(player_assignments))
        avail = [p for p in self.ranking.index if p not in taken]
        return weighted_softmax_pick(self.ranking, avail, self.temperature, rng,
                                     self.positions, player_assignments[seat], self.top_k)
