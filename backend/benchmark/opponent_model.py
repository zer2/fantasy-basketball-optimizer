# benchmark/opponent_model.py
"""Shared weighted-softmax opponent policy. Used by the scored field AND MCTS rollouts."""
from functools import lru_cache

import numpy as np

from backend.math.position_optimization import check_single_player_eligibility
from backend.math.position_config import build_position_config
from backend.benchmark.config import get_params, DEFAULT_SLOT_COUNTS, DEFAULT_STRUCT_SIG

# Lazy singleton — built once on first use, reused for every eligibility check.
_pos_cfg = None

def _get_pos_cfg():
    global _pos_cfg
    if _pos_cfg is None:
        _pos_cfg = build_position_config(get_params()['NBA'], dict(DEFAULT_SLOT_COUNTS))
    return _pos_cfg


@lru_cache(maxsize=None)
def _eligibility_cached(struct_sig, cand_key, team_key):
    """Memoized ``check_single_player_eligibility``.

    Eligibility is a pure function of (a) the league's position-slot structure, (b) the
    candidate's eligible-position list, and (c) the *multiset* of teammates' position lists
    — it does not depend on teammate ordering (verified empirically: order-invariant). During
    an MCTS rollout the same (candidate, team-profile) pair is re-checked thousands of times,
    each re-solving a linear-assignment LP; caching on these keys collapses that to one solve
    per distinct profile while returning byte-identical results. ``struct_sig`` guards against
    stale hits if the position structure ever changes within a process.
    """
    return check_single_player_eligibility(list(cand_key), [list(t) for t in team_key], _get_pos_cfg())


def _team_key(team_positions_series):
    """Order-invariant key for a team's roster of position lists."""
    return tuple(sorted(tuple(x) for x in team_positions_series))


def _eligible(candidates, positions, team_players):
    if positions is None:
        return list(candidates)
    team_positions = positions.loc[[p for p in team_players if p in positions.index]] \
        if team_players else positions.loc[[]]
    # Team profile + structure are constant across all candidates in this call, so compute
    # their cache keys once and reuse them per-candidate.
    team_key = _team_key(team_positions)
    return [p for p in candidates
            if p in positions.index
            and _eligibility_cached(DEFAULT_STRUCT_SIG, tuple(positions.loc[p]), team_key)]

def weighted_softmax_pick(ranking, available, temperature, rng, positions, team_players, top_k=15):
    """Return one player from `available`, sampled ∝ exp(score/T) over the top-K eligible."""
    avail = [p for p in ranking.index if p in set(available)]        # ranking order (desc)
    avail = _eligible(avail, positions, team_players)
    if not avail:
        return None
    avail = avail[:top_k]
    scores = ranking.loc[avail].to_numpy(dtype=float)
    if temperature <= 0:
        return avail[int(np.argmax(scores))]
    z = scores / temperature
    z = z - z.max()
    w = np.exp(z)
    w = w / w.sum()
    return str(rng.choice(avail, p=w))
