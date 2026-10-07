"""Independent Monte-Carlo of the fantasy season on resampled REAL game logs.
Never calls the H-score objective."""
import re

import numpy as np
import pandas as pd

RATIO_COMPONENTS = {
    'Field Goal %': ('Field Goals Made', 'Field Goal Attempts'),
    'Free Throw %': ('Free Throws Made', 'Free Throw Attempts'),
}
NEGATIVE = {'Turnovers'}

# Rosters key players as 'Name (Position)' (from info['G-scores'].index, appended in
# data.py) while the gamelogs are keyed by the plain 'Player' name. Strip the trailing
# ' (Position)' suffix so cache lookups line up; without this every rostered player misses
# the cache and team totals collapse to all-zeros (degenerate EC/MC == 0.5 for all seats).
_POSITION_SUFFIX = re.compile(r'\s*\([^()]*\)\s*$')


def strip_position_suffix(name):
    """Return the plain player name, dropping any trailing ' (Position)' suffix."""
    return _POSITION_SUFFIX.sub('', name)


def build_player_cache(gamelogs):
    """Group the game logs by player once into per-column numpy arrays.

    ``simulate_week`` is called thousands of times per season sweep; re-running
    ``gamelogs.groupby('Player')`` (a full pass over ~26k rows) each time dominates the
    evaluator runtime. This hoists that grouping to a single pass and stores, per player,
    a dict of ``column -> np.ndarray`` plus the row count. Sampling then does
    ``rng.integers(0, n, size=gpw)`` (identical draw to indexing the DataFrame) followed by
    numpy fancy-indexing + sum, so totals are byte-identical to the pandas path — only far
    faster.
    """
    cache = {}
    for p, g in gamelogs.groupby('Player'):
        cols = {c: g[c].to_numpy() for c in g.columns if c != 'Player'}
        cache[p] = {'n': len(g), 'cols': cols}
    return cache


def team_week_totals(team_players, gamelogs, rng, n_games_per_week, categories,
                     player_cache=None):
    """Sum a team's category totals over one simulated week.

    If ``player_cache`` (from :func:`build_player_cache`) is supplied, the per-player
    numpy arrays are reused instead of re-grouping ``gamelogs`` on every call. Passing it
    is optional and does not change results — only speed.
    """
    if player_cache is None:
        player_cache = build_player_cache(gamelogs)
    counting = [c for c in categories if c not in RATIO_COMPONENTS]
    totals = {c: 0.0 for c in counting}
    made = {c: 0.0 for c in categories if c in RATIO_COMPONENTS}
    att = {c: 0.0 for c in categories if c in RATIO_COMPONENTS}

    for p in team_players:
        entry = player_cache.get(p)
        if entry is None:
            # Rosters carry a ' (Position)' suffix the plain-name gamelog cache lacks.
            entry = player_cache.get(strip_position_suffix(p))
        if entry is None:
            continue
        n = entry['n']
        cols = entry['cols']
        idx = rng.integers(0, n, size=n_games_per_week)
        for c in counting:
            arr = cols.get(c)
            if arr is not None:
                totals[c] += float(arr[idx].sum())
        for c in made:
            m, a = RATIO_COMPONENTS[c]
            made[c] += float(cols[m][idx].sum())
            att[c] += float(cols[a][idx].sum())

    for c in made:
        totals[c] = made[c] / att[c] if att[c] > 0 else 0.0
    return totals

def simulate_week(team_players, gamelogs, rng, n_games_per_week=3,
                  categories=None, player_cache=None):
    return team_week_totals(team_players, gamelogs, rng, n_games_per_week, categories,
                            player_cache=player_cache)

def compare_categories(totals_a, totals_b, categories, negative=NEGATIVE):
    wins_a = wins_b = ties = 0
    for c in categories:
        va, vb = totals_a.get(c, 0.0), totals_b.get(c, 0.0)
        if c in negative:
            va, vb = -va, -vb
        if va > vb:
            wins_a += 1
        elif vb > va:
            wins_b += 1
        else:
            ties += 1
    return wins_a, wins_b, ties

def _round_robin_pairs(n):
    return [(i, j) for i in range(n) for j in range(n) if i < j]

def evaluate_rosters(player_assignments, gamelogs, cfg, exp_cfg, rng, weeks=20):
    """Return {seat: {'EC': rate, 'MC': rate, 'EC_ci': half, 'MC_ci': half}}."""
    seats = list(player_assignments.keys())
    n = len(seats)
    cats = list(cfg.selected_categories)
    gpw = 3  # n_games_per_week (NBA default from parameters.yaml)
    pairs = _round_robin_pairs(n)
    player_cache = build_player_cache(gamelogs)  # group once, reuse across every week/season

    ec = np.zeros((exp_cfg.n_season_sims, n))
    mc = np.zeros((exp_cfg.n_season_sims, n))

    for s in range(exp_cfg.n_season_sims):
        ec_num = np.zeros(n); ec_den = np.zeros(n)
        mc_num = np.zeros(n); mc_den = np.zeros(n)
        for _w in range(weeks):
            week_totals = {seat: simulate_week(player_assignments[seat], gamelogs, rng, gpw, cats,
                                               player_cache=player_cache)
                           for seat in seats}
            for i, j in pairs:
                wa, wb, t = compare_categories(week_totals[i], week_totals[j], cats)
                # EC: share of categories won (ties = half)
                ec_num[i] += wa + 0.5 * t; ec_den[i] += len(cats)
                ec_num[j] += wb + 0.5 * t; ec_den[j] += len(cats)
                # MC: one matchup win to the category majority
                mc_den[i] += 1; mc_den[j] += 1
                if wa > wb: mc_num[i] += 1
                elif wb > wa: mc_num[j] += 1
                else: mc_num[i] += 0.5; mc_num[j] += 0.5
        ec[s] = ec_num / ec_den
        mc[s] = mc_num / mc_den

    out = {}
    for k, seat in enumerate(seats):
        out[seat] = {
            'EC': float(ec[:, k].mean()),
            'MC': float(mc[:, k].mean()),
            'EC_ci': float(1.96 * ec[:, k].std(ddof=1) / np.sqrt(exp_cfg.n_season_sims)),
            'MC_ci': float(1.96 * mc[:, k].std(ddof=1) / np.sqrt(exp_cfg.n_season_sims)),
        }
    return out
