"""
Mock NBA projection data for backend development/testing.

Produces a DataFrame whose structure matches what player_stats_v0 looks like
after loading from ESPN/DARKO/BBM projections: per-game stats as floats,
Games Played % as a 0–1 fraction, and Position as a comma-separated string.
"""

import numpy as np
import pandas as pd

# Ordered base positions for NBA
BASE_POSITIONS = ['PG', 'SG', 'SF', 'PF', 'C']

# Position eligibility profiles:  (list-of-positions, relative-frequency)
_POSITION_PROFILES = [
    (['PG'],        0.16),
    (['SG'],        0.14),
    (['SF'],        0.14),
    (['PF'],        0.14),
    (['C'],         0.13),
    (['PG', 'SG'],  0.07),
    (['SG', 'SF'],  0.07),
    (['SF', 'PF'],  0.07),
    (['PF', 'C'],   0.08),
]


def make_nba_mock_data(n_players: int = 200, seed: int = 42) -> pd.DataFrame:
    """Return a DataFrame of mock NBA per-game projections."""
    rng = np.random.default_rng(seed)

    profiles, weights = zip(*_POSITION_PROFILES)
    weights = np.array(weights) / sum(weights)
    profile_idx = rng.choice(len(profiles), size=n_players, p=weights)
    positions = [','.join(profiles[i]) for i in profile_idx]

    is_guard = np.array([any(p in ('PG', 'SG') for p in profiles[i]) for i in profile_idx])
    is_big   = np.array([any(p in ('C', 'PF')  for p in profiles[i]) for i in profile_idx])
    is_wing  = ~is_guard & ~is_big

    # Skill 0-1 (exponential → approximately pareto-shaped talent distribution)
    skill = rng.exponential(scale=1.4, size=n_players)
    skill = np.clip(skill / np.percentile(skill, 97), 0, 1.0)

    # ── Counting stats (per game) ──────────────────────────────────────────────
    pts  = 6  + skill * 24 + rng.normal(0, 1.5, n_players)

    reb_base = np.where(is_big, 5.0, np.where(is_wing, 3.0, 2.0))
    reb  = reb_base + skill * 8 + rng.normal(0, 1.0, n_players)

    ast_base = np.where(is_guard, 3.0, np.where(is_wing, 1.5, 1.0))
    ast  = ast_base + skill * 6 + rng.normal(0, 0.8, n_players)

    stl  = 0.3 + skill * 1.5 + rng.normal(0, 0.25, n_players)

    blk_base = np.where(is_big, 0.7, np.where(is_wing, 0.3, 0.1))
    blk  = blk_base + skill * 2 + rng.normal(0, 0.2, n_players)

    # 3-pointers
    three_att_base = np.where(is_guard, 4.5, np.where(is_wing, 3.0, 0.8))
    three_att = np.clip(three_att_base + skill * 3 + rng.normal(0, 0.5, n_players), 0.2, 10)
    three_pct = np.clip(0.28 + skill * 0.12 + rng.normal(0, 0.03, n_players), 0.2, 0.48)
    threes = three_att * three_pct

    tov  = 0.4 + skill * 2.5 + rng.normal(0, 0.3, n_players)

    oreb = np.where(is_big, reb * 0.28, reb * 0.12) + rng.normal(0, 0.15, n_players)
    oreb = np.clip(oreb, 0, 5)
    dreb = reb - oreb

    dd   = np.clip(skill * 0.18 * np.where(is_big, 1.6, 1.0) + rng.uniform(0, 0.04, n_players), 0, 0.5)

    # ── Ratio-stat volume columns ─────────────────────────────────────────────
    fga  = np.clip(pts / (2.1 + skill * 0.4) + rng.normal(0, 0.6, n_players), 2, 22)
    fg_pct = np.clip(0.40 + skill * 0.10 + rng.normal(0, 0.025, n_players), 0.35, 0.68)
    fgm  = fga * fg_pct

    # Estimate FTA from scoring that isn't from 3s or 2FGM
    fta  = np.clip((pts - threes * 3 - fgm * 2) * 0.5 + 0.5 + rng.normal(0, 0.3, n_players), 0.2, 10)
    ft_pct = np.clip(0.65 + skill * 0.20 + rng.normal(0, 0.04, n_players), 0.55, 0.97)
    ftm  = fta * ft_pct

    # Games Played % (0–1 fraction)
    gp   = np.clip(rng.beta(7, 2, n_players), 0.55, 1.0)

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    data = {
        'Points':               np.clip(pts,       2, 42),
        'Rebounds':             np.clip(reb,       1, 16),
        'Def Rebounds':         np.clip(dreb,      0.5, 13),
        'Off Rebounds':         np.clip(oreb,      0, 5),
        'Assists':              np.clip(ast,       0.3, 13),
        'Steals':               np.clip(stl,       0.1, 3.5),
        'Blocks':               np.clip(blk,       0, 4.5),
        'Threes':               np.clip(threes,    0, 5),
        'Three Attempts':       np.clip(three_att, 0.2, 11),
        'Three %':              np.clip(three_pct, 0.20, 0.48),
        'Turnovers':            np.clip(tov,       0.3, 5),
        'Double Doubles':       np.clip(dd,        0, 0.5),
        'Field Goals Made':     np.clip(fgm,       1, 16),
        'Field Goal Attempts':  np.clip(fga,       2, 22),
        'Field Goal %':         np.clip(fg_pct,    0.35, 0.68),
        'Free Throws Made':     np.clip(ftm,       0.1, 9),
        'Free Throw Attempts':  np.clip(fta,       0.2, 10),
        'Free Throw %':         np.clip(ft_pct,    0.55, 0.97),
        'Games Played %':       gp,
        'Position':             positions,
    }

    names = [f"Player {i + 1:03d}" for i in range(n_players)]
    df = pd.DataFrame(data, index=pd.Index(names, name='Player'))

    # Sort by a rough skill proxy so index ordering makes sense
    df['_rank'] = pts + reb + ast
    df.sort_values('_rank', ascending=False, inplace=True)
    df.drop(columns=['_rank'], inplace=True)

    return df
