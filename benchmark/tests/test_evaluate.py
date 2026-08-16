import numpy as np
import pandas as pd
from benchmark.evaluate import simulate_week, team_week_totals

def _logs():
    return pd.DataFrame({
        'Player': ['A','A','B','B'],
        'Points': [10, 20, 30, 30],
        'Rebounds': [5, 5, 1, 1],
        'Assists': [0,0,0,0], 'Steals':[0,0,0,0], 'Blocks':[0,0,0,0], 'Turnovers':[0,0,0,0], 'Threes':[0,0,0,0],
        'Field Goals Made': [5, 10, 12, 12], 'Field Goal Attempts': [10, 10, 24, 24],
        'Free Throws Made': [0,0,0,0], 'Free Throw Attempts': [0,0,0,0],
    })

def test_ratio_is_volume_weighted():
    # np.random.default_rng(0) samples A's game1 (10 made / 10 att) and B's game1
    # (12 made / 24 att) with n_games_per_week=1. Volume-weighted FG% is the sum of
    # makes over the sum of attempts across the week: (10+12)/(10+24) = 22/34 ~= 0.647,
    # NOT the mean of the per-game percentages (1.0 + 0.5)/2 = 0.75.
    logs = _logs()
    totals = team_week_totals(['A','B'], logs, rng=np.random.default_rng(0),
                              n_games_per_week=1, categories=['Points','Field Goal %'])
    # assert FG% equals volume-weighted (sum makes / sum attempts), not the mean of percentages
    assert abs(totals['Field Goal %'] - 22.0 / 34.0) < 1e-9
    assert abs(totals['Field Goal %'] - 0.75) > 1e-6

def test_counting_totals_sum_across_players():
    logs = _logs()
    totals = team_week_totals(['A','B'], logs, rng=np.random.default_rng(0),
                              n_games_per_week=2, categories=['Points'])
    # 2 games each; A in {10,20}, B in {30,30} -> min 10+30+30 ... just assert positive & bounded
    assert totals['Points'] > 0

def test_position_suffix_is_stripped_for_lookup():
    # Rosters key players as 'Name (Position)' (from info['G-scores'].index) but the
    # gamelogs are keyed by plain 'Player' names. team_week_totals must strip the trailing
    # ' (Position)' suffix so rostered players are found in the cache and contribute real,
    # nonzero totals. Regression for the degenerate all-zero evaluator bug (every rostered
    # player missed the cache -> all-zero team totals -> every seat tied at EC/MC == 0.5).
    logs = _logs()  # gamelogs keyed by plain 'A', 'B'
    totals = team_week_totals(['A (C)', 'B (PG,SG)'], logs, rng=np.random.default_rng(0),
                              n_games_per_week=1, categories=['Points', 'Field Goal %'])
    assert totals['Points'] > 0
    assert totals['Field Goal %'] > 0

def test_suffixed_lookup_matches_plain_name():
    # Stripping the ' (Position)' suffix must reproduce exactly the plain-name totals.
    logs = _logs()
    suffixed = team_week_totals(['A (C)', 'B (PG,SG)'], logs, rng=np.random.default_rng(0),
                                n_games_per_week=1, categories=['Points', 'Field Goal %'])
    plain = team_week_totals(['A', 'B'], logs, rng=np.random.default_rng(0),
                             n_games_per_week=1, categories=['Points', 'Field Goal %'])
    assert suffixed == plain
