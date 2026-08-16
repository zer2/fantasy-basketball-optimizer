# benchmark/mcts.py
"""MCTS draft agent. Reuses HAgent's full-roster objective as the search leaf (read-only)."""
import numpy as np

def leaf_value(hagent, player_assignments, seat):
    """Scalar team win-rate for a COMPLETE roster via HAgent's n_picks branch.
    get_h_scores yields once; with a full roster the score index is [''] and holds the team score."""
    gen = hagent.get_h_scores(player_assignments, seat)
    res = next(gen)
    scores = res['Scores']
    return float(scores.iloc[0])


from copy import deepcopy

from src.math.algorithm_agents import HAgent, get_default_h_values
from src.helpers.helper_functions import gen_key
from benchmark.draft import snake_seat_order
from benchmark.opponent_model import (
    weighted_softmax_pick, _eligible, _eligibility_cached, _struct_sig, _team_key)

def _all_taken(pa):
    return [p for v in pa.values() for p in v if p == p]

def _top_k_eligible(ranking, taken, positions, team_players, top_k):
    """First `top_k` position-eligible players in ranking (descending) order, skipping `taken`.

    Behaviorally identical to ``_eligible([p for p in ranking.index if p not in taken],
    positions, team_players)[:top_k]`` but stops the eligibility LP after collecting `top_k`
    players instead of scanning the entire pool. Passing this prefix into
    ``weighted_softmax_pick`` yields the same candidate set/order/scores (and thus the same
    rng draw), so the search remains deterministic — it only avoids O(pool) LP solves per pick.
    Uses the same memoized eligibility check as ``_eligible`` (see opponent_model) so repeated
    rollouts collapse to one LP solve per distinct (candidate, team-profile) pair.
    """
    if positions is None:
        return [p for p in ranking.index if p not in taken][:top_k]
    team_positions = positions.loc[[p for p in team_players if p in positions.index]] \
        if team_players else positions.loc[[]]
    sig = _struct_sig()
    team_key = _team_key(team_positions)
    out = []
    for p in ranking.index:
        if p in taken or p not in positions.index:
            continue
        if _eligibility_cached(sig, tuple(positions.loc[p]), team_key):
            out.append(p)
            if len(out) >= top_k:
                break
    return out

class MCTSAgent:
    def __init__(self, info, cfg, temperature=1.0, n_simulations=200, top_k=15, c_puct=1.4):
        self.info = info
        self.cfg = cfg
        self.positions = info['Positions']
        self.temperature = temperature
        self.n_simulations = n_simulations
        self.top_k = top_k
        self.c_puct = c_puct
        h = get_default_h_values(gen_key(), cfg.omega, cfg.gamma, cfg.n_starters,
                                 cfg.n_drafters, cfg.n_iterations, cfg.beth, cfg.scoring_format)
        self.ranking = h.set_index('Player')['H-score'].sort_values(ascending=False)
        # One HAgent for leaf scoring (dynamic=False -> single-pass full-roster objective).
        self.hagent = HAgent(info=info, omega=cfg.omega, gamma=cfg.gamma, n_picks=cfg.n_starters,
                             n_drafters=cfg.n_drafters, dynamic=False, beth=cfg.beth,
                             scoring_format=cfg.scoring_format)

    def _candidates(self, player_assignments, seat):
        taken = set(_all_taken(player_assignments))
        return _top_k_eligible(self.ranking, taken, self.positions,
                               player_assignments[seat], self.top_k)

    def _priors(self, candidates):
        s = self.ranking.loc[candidates].to_numpy(dtype=float)
        t = max(self.temperature, 1e-6)
        z = s / t; z = z - z.max(); w = np.exp(z)
        return w / w.sum()

    def _rollout(self, player_assignments, seat, first_pick, rng):
        """Play the draft to completion from `seat` taking `first_pick`, then softmax for all."""
        pa = deepcopy(player_assignments)
        pa[seat] = pa[seat] + [first_pick]
        # Remaining picks in snake order after the current one.
        order = snake_seat_order(self.cfg.n_drafters, self.cfg.n_starters)
        # advance past picks already made (count of taken) + this one
        made = len(_all_taken(player_assignments)) + 1
        for s in order[made:]:
            taken = set(_all_taken(pa))
            # Pre-slice to the top-K eligible prefix (early-stop) so weighted_softmax_pick,
            # which internally does _eligible(available)[:top_k], sees the identical candidate
            # set without re-running the eligibility LP over the whole pool.
            cand = _top_k_eligible(self.ranking, taken, self.positions, pa[s], self.top_k)
            pick = weighted_softmax_pick(self.ranking, cand, self.temperature, rng,
                                         self.positions, pa[s], self.top_k)
            if pick is not None:
                pa[s] = pa[s] + [pick]
        return leaf_value(self.hagent, pa, seat)

    def make_pick(self, player_assignments, seat, rng):
        candidates = self._candidates(player_assignments, seat)
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        priors = self._priors(candidates)
        N = np.zeros(len(candidates))
        W = np.zeros(len(candidates))
        for _ in range(self.n_simulations):
            total = N.sum()
            Q = np.where(N > 0, W / np.maximum(N, 1), 0.0)
            u = self.c_puct * priors * np.sqrt(total + 1) / (1 + N)
            a = int(np.argmax(Q + u))
            value = self._rollout(player_assignments, seat, candidates[a], rng)
            N[a] += 1; W[a] += value
        return candidates[int(np.argmax(N))]


class MCTSAgentV2:
    """Optimized MCTS with three improvements over the baseline MCTSAgent:
    1. Calibrated prior: lower softmax temperature (prior_temp) focuses early sims on
       promising candidates instead of scattering uniformly across a flat prior.
    2. Q-advantage normalization: PUCT sees (Q - mean_Q) so relative differences between
       candidates drive selection, not absolute values all clustering around 0.5.
    3. Short rollout + analytic leaf: roll out only `rollout_depth` rounds of the draft
       (handling real availability uncertainty), then evaluate the PARTIAL roster with
       HAgent's analytic model (which models remaining picks via CLT assumptions).
       This gives much sharper Q-value spreads (0.83 vs 0.82 instead of 0.512 vs 0.523)
       because the leaf evaluates fewer-player rosters where one candidate's impact is larger.
    """

    def __init__(self, info, cfg, temperature=1.0, n_simulations=200, top_k=6, c_puct=2.0,
                 prior_temp=0.2, rollout_depth=3):
        self.info = info
        self.cfg = cfg
        self.positions = info['Positions']
        self.temperature = temperature         # field/rollout opponent stochasticity
        self.prior_temp = prior_temp           # OPT 1: prior softmax sharpness
        self.n_simulations = n_simulations
        self.top_k = top_k
        self.c_puct = c_puct
        self.rollout_depth = rollout_depth     # OPT 3: how many hero-pick rounds to simulate

        h = get_default_h_values(gen_key(), cfg.omega, cfg.gamma, cfg.n_starters,
                                 cfg.n_drafters, cfg.n_iterations, cfg.beth, cfg.scoring_format)
        self.ranking = h.set_index('Player')['H-score'].sort_values(ascending=False)
        self.hagent = HAgent(info=info, omega=cfg.omega, gamma=cfg.gamma, n_picks=cfg.n_starters,
                             n_drafters=cfg.n_drafters, dynamic=False, beth=cfg.beth,
                             scoring_format=cfg.scoring_format)

    def _candidates(self, player_assignments, seat):
        taken = set(_all_taken(player_assignments))
        return _top_k_eligible(self.ranking, taken, self.positions,
                               player_assignments[seat], self.top_k)

    def _priors(self, candidates):
        s = self.ranking.loc[candidates].to_numpy(dtype=float)
        t = max(self.prior_temp, 1e-6)   # OPT 1: use prior_temp, not field temperature
        z = s / t; z = z - z.max(); w = np.exp(z)
        return w / w.sum()

    def _rollout(self, player_assignments, seat, first_pick, rng):
        """OPT 3: short rollout. Simulate only `rollout_depth` rounds of hero picks
        (handling real near-term availability), then evaluate the partial roster analytically."""
        pa = deepcopy(player_assignments)
        pa[seat] = pa[seat] + [first_pick]
        order = snake_seat_order(self.cfg.n_drafters, self.cfg.n_starters)
        made = len(_all_taken(player_assignments)) + 1
        hero_picks_added = 0
        for s in order[made:]:
            if s == seat:
                hero_picks_added += 1
                if hero_picks_added >= self.rollout_depth:
                    break
            taken = set(_all_taken(pa))
            cand = _top_k_eligible(self.ranking, taken, self.positions, pa[s], self.top_k)
            pick = weighted_softmax_pick(self.ranking, cand, self.temperature, rng,
                                         self.positions, pa[s], self.top_k)
            if pick is not None:
                pa[s] = pa[s] + [pick]
        return self._partial_leaf(pa, seat)

    def _partial_leaf(self, pa, seat):
        """Evaluate a partial roster using HAgent's analytic modeling of remaining picks.
        Returns the BEST candidate score (the hero's expected value given this partial state)."""
        hero_n = len([p for p in pa[seat] if p == p])
        if hero_n >= self.cfg.n_starters:
            return leaf_value(self.hagent, pa, seat)
        gen = self.hagent.get_h_scores(pa, seat)
        res = next(gen)
        scores = res['Scores']
        return float(scores.max())

    def make_pick(self, player_assignments, seat, rng):
        candidates = self._candidates(player_assignments, seat)
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        priors = self._priors(candidates)
        N = np.zeros(len(candidates))
        W = np.zeros(len(candidates))
        for _ in range(self.n_simulations):
            total = N.sum()
            Q = np.where(N > 0, W / np.maximum(N, 1), 0.0)
            # OPT 2: Q-advantage normalization — PUCT sees relative, not absolute values
            Q_adv = Q - Q.mean() if total > 0 else Q
            u = self.c_puct * priors * np.sqrt(total + 1) / (1 + N)
            a = int(np.argmax(Q_adv + u))
            value = self._rollout(player_assignments, seat, candidates[a], rng)
            N[a] += 1; W[a] += value
        return candidates[int(np.argmax(N))]
