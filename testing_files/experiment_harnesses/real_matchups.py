"""Real head-to-head matchups for drafted rosters: each player brings one real week of the season, drawn at random.

A week is drawn from every week of the season, and a week a player has no games in is a zero week. That is the
assumption of an H-score at upsilon = 1, psi = 0: a missed game counts in full and nobody replaces it. Each team's
week is the sum of its whole roster's weeks (no bench), and percentages are the summed makes over the summed attempts.
A tied category counts as half a win; in Most Categories a matchup tied on categories counts as half a matchup.
"""
import numpy as np
import pandas as pd

from backend.data_retrieval import get_weekly_box_scores
from backend.parameters import load_all_params

SPORT = 'NBA'


def build_weekly_tensor(
    season: str
    , player_ids: list[int]
    , statistics: list[str]
) -> np.ndarray:
    """(player, week, statistic) real weekly totals over every week of the season, zero where a player has no
    games. A player with no games all season is all zeros."""
    weekly = get_weekly_box_scores(season, load_all_params()[SPORT]).reset_index()
    weeks = sorted(weekly['WEEK'].unique())
    by_player_and_week = weekly.set_index(['NBA_PLAYER_ID', 'WEEK'])[statistics]
    full_index = pd.MultiIndex.from_product([player_ids, weeks], names=['NBA_PLAYER_ID', 'WEEK'])
    filled = by_player_and_week.reindex(full_index, fill_value=0.0).astype(float)
    return filled.to_numpy().reshape(len(player_ids), len(weeks), len(statistics))


def list_category_statistics(categories: list[str]) -> tuple[list[str], list[tuple[str, str | None]]]:
    """The weekly statistics the categories need, and each category as (numerator, denominator or None)."""
    sport_params = load_all_params()[SPORT]
    ratio_statistics = sport_params['ratio-statistics']
    parts = []
    for category in categories:
        if category in ratio_statistics:
            parts.append((ratio_statistics[category]['made-statistic'],
                          ratio_statistics[category]['volume-statistic']))
        else:
            parts.append((category, None))
    statistics = sorted({name for pair in parts for name in pair if name is not None})
    return statistics, parts


def simulate_real_win_rates(
    season: str
    , assignments: dict[str, list[int]]
    , team: str
    , categories: list[str]
    , n_simulations: int
    , seed: int
) -> dict[str, float]:
    """`team`'s real win rates against every other team, from n_simulations random weeks per opponent.

    Returns {'category': the mean share of categories won, 'matchup': the share of matchups won (more than half the
    categories)}, each averaged over the opponents."""
    statistics, parts = list_category_statistics(categories)
    statistic_index = {name: position for position, name in enumerate(statistics)}
    negative = set(load_all_params()[SPORT]['negative-statistics'])
    player_ids = sorted({player_id for roster in assignments.values() for player_id in roster})
    row_of = {player_id: position for position, player_id in enumerate(player_ids)}
    tensor = build_weekly_tensor(season, player_ids, statistics)
    n_weeks = tensor.shape[1]
    rng = np.random.default_rng(seed)

    def draw_team_weeks(roster):
        rows = np.array([row_of[player_id] for player_id in roster])
        weeks = rng.integers(n_weeks, size=(n_simulations, len(rows)))
        return tensor[rows[None, :], weeks].sum(axis=1)                       # (simulation, statistic)

    def compute_category_values(totals):
        columns = []
        for numerator, denominator in parts:
            values = totals[:, statistic_index[numerator]]
            if denominator is not None:
                values = values / totals[:, statistic_index[denominator]]
            columns.append(values)
        return np.stack(columns, axis=1)                                      # (simulation, category)

    direction = np.array([-1.0 if category in negative else 1.0 for category in categories])
    category_rates, matchup_rates = [], []
    for opponent, roster in assignments.items():
        if opponent == team:
            continue
        own = compute_category_values(draw_team_weeks(assignments[team])) * direction
        other = compute_category_values(draw_team_weeks(roster)) * direction
        category_wins = (own > other) + 0.5 * (own == other)
        wins_per_matchup = category_wins.sum(axis=1)
        half = len(categories) / 2
        category_rates.append(category_wins.mean())
        matchup_rates.append(np.mean((wins_per_matchup > half) + 0.5 * (wins_per_matchup == half)))
    return {'category': float(np.mean(category_rates)), 'matchup': float(np.mean(matchup_rates))}
