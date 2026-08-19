# benchmark/tests/test_opponent_model.py
import numpy as np
import pandas as pd
from backend.benchmark.opponent_model import weighted_softmax_pick

def _ranking():
    return pd.Series({'P1': 10.0, 'P2': 9.0, 'P3': 8.0, 'P4': 1.0})

def test_zero_temperature_is_argmax():
    rng = np.random.default_rng(0)
    pick = weighted_softmax_pick(_ranking(), available=['P1','P2','P3','P4'],
                                 temperature=0.0, rng=rng, positions=None, team_players=[])
    assert pick == 'P1'

def test_respects_availability():
    rng = np.random.default_rng(0)
    pick = weighted_softmax_pick(_ranking(), available=['P2','P3','P4'],
                                 temperature=0.0, rng=rng, positions=None, team_players=[])
    assert pick == 'P2'

def test_high_temperature_sometimes_picks_nonmax():
    rng = np.random.default_rng(1)
    picks = {weighted_softmax_pick(_ranking(), available=['P1','P2','P3','P4'],
                                   temperature=5.0, rng=rng, positions=None, team_players=[])
             for _ in range(50)}
    assert len(picks) > 1   # stochastic: not always the same player

def test_top_k_limits_candidates():
    rng = np.random.default_rng(2)
    picks = {weighted_softmax_pick(_ranking(), available=['P1','P2','P3','P4'],
                                   temperature=5.0, rng=rng, positions=None, team_players=[], top_k=2)
             for _ in range(50)}
    assert picks <= {'P1', 'P2'}   # P3/P4 never sampled
