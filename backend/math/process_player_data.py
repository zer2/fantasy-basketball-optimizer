"""
Backend-only copy of src/math/process_player_data.py.

Changes vs original:
- All st.session_state / helper_function calls replaced with explicit parameters.
- process_player_data receives player_stats_v2 DataFrame directly.
- make_upsilon_adjustment receives player_stats_v1 DataFrame directly.
- drop_injured_players receives player_stats_v0 DataFrame directly.
- get_category_level_rv takes `categories` explicitly.
- @st.cache_data decorators removed (caching handled by the Session layer).
The original src/ file is untouched.
"""

from __future__ import annotations

import uuid

import numpy as np
import pandas as pd

from backend.helper_functions import get_counting_stats, get_ratio_stats


# ── public helpers ─────────────────────────────────────────────────────────────

def get_category_level_rv(rv: float,
                          v: pd.Series,
                          categories: list[str]) -> pd.Series:
    rv_multiple = (rv / (len(categories) - 2)
                   if 'Turnovers' in categories
                   else rv / len(categories))
    return pd.Series({
        stat: -rv_multiple / v[stat] if stat == 'Turnovers' else rv_multiple / v[stat]
        for stat in categories
    })


# ── coefficient calculation ────────────────────────────────────────────────────

def calculate_coefficients(player_means: pd.DataFrame,
                            representative_player_set: list,
                            mean_of_variances: pd.Series,
                            counting_stats: list[str],
                            ratio_stats: list[str],
                            params: dict) -> pd.DataFrame:

    var_of_means  = player_means.loc[representative_player_set, counting_stats].var(axis=0)
    mean_of_means = player_means.loc[representative_player_set, counting_stats].mean(axis=0)

    for ratio_stat, ratio_stat_info in params['ratio-statistics'].items():
        if ratio_stat in ratio_stats:
            volume_statistic = ratio_stat_info['volume-statistic']

            volume_mean_of_means = player_means.loc[representative_player_set, volume_statistic].mean()
            mean_of_means.loc[volume_statistic] = volume_mean_of_means

            agg_average = (
                player_means.loc[representative_player_set, ratio_stat]
                * player_means.loc[representative_player_set, volume_statistic]
            ).mean() / volume_mean_of_means
            mean_of_means.loc[ratio_stat] = agg_average

            numerator = (
                player_means.loc[representative_player_set, volume_statistic] / volume_mean_of_means
                * (player_means.loc[representative_player_set, ratio_stat] - agg_average)
            )
            var_of_means.loc[ratio_stat] = numerator.var()

    return pd.DataFrame({
        'Mean of Means':      mean_of_means,
        'Variance of Means':  var_of_means,
        'Mean of Variances':  mean_of_variances.reindex(var_of_means.index),
    })


def calculate_coefficients_historical(weekly_df: pd.DataFrame,
                                       representative_player_set: list,
                                       params: dict,
                                       counting_stats: list[str],
                                       ratio_stats: list[str],
                                       coefficient_exploration_mode: bool = False,
                                       ) -> pd.DataFrame:
    player_stats = weekly_df.groupby(level='Player').agg(['mean', 'var'])

    mean_of_vars  = player_stats.loc[representative_player_set, (counting_stats, 'var')].mean(axis=0)
    var_of_means  = player_stats.loc[representative_player_set, (counting_stats, 'mean')].var(axis=0)
    mean_of_means = player_stats.loc[representative_player_set, (counting_stats, 'mean')].mean(axis=0)

    for ratio_stat, ratio_stat_info in params['ratio-statistics'].items():
        if ratio_stat in ratio_stats:
            volume_statistic = ratio_stat_info['volume-statistic']
            made_statistic   = ratio_stat_info['made-statistic']

            made_mean_of_means   = player_stats.loc[representative_player_set, (made_statistic, 'mean')].mean()
            volume_mean_of_means = player_stats.loc[representative_player_set, (volume_statistic, 'mean')].mean()

            mean_of_means.loc[volume_statistic] = volume_mean_of_means
            ratio_agg_average = made_mean_of_means / volume_mean_of_means
            mean_of_means.loc[ratio_stat] = ratio_agg_average

            ratio       = player_stats.loc[:, (made_statistic, 'mean')] / player_stats.loc[:, (volume_statistic, 'mean')]
            ratio_num   = player_stats.loc[:, (volume_statistic, 'mean')] / volume_mean_of_means * (ratio - ratio_agg_average)
            var_of_means.loc[ratio_stat] = ratio_num.loc[representative_player_set].var()

            weekly_df.loc[:, 'volume_adjusted_' + ratio_stat] = (
                (weekly_df[made_statistic] - weekly_df[volume_statistic] * ratio_agg_average)
                / volume_mean_of_means
            )
            ratio_mean_of_vars = (
                weekly_df['volume_adjusted_' + ratio_stat]
                .loc[representative_player_set]
                .groupby('Player').var().mean()
            )
            mean_of_vars.loc[ratio_stat] = ratio_mean_of_vars

    return pd.DataFrame({
        'Mean of Means':     mean_of_means.droplevel(level=1),
        'Variance of Means': var_of_means.droplevel(level=1),
        'Mean of Variances': mean_of_vars.droplevel(level=1),
    })


def calculate_scores_from_coefficients(player_means: pd.DataFrame,
                                        coefficients: pd.DataFrame,
                                        params: dict,
                                        alpha_weight: float,
                                        beta_weight: float,
                                        counting_stats: list[str],
                                        ratio_stats: list[str],
                                        categories: list[str]) -> pd.DataFrame:

    counting_mean    = coefficients.loc[counting_stats, 'Mean of Means']
    counting_var_m   = coefficients.loc[counting_stats, 'Variance of Means']
    counting_mean_v  = coefficients.loc[counting_stats, 'Mean of Variances']

    denom    = (counting_var_m.values * alpha_weight + counting_mean_v.values * beta_weight) ** 0.5
    num      = player_means.loc[:, counting_stats] - counting_mean
    main_scores = num.divide(denom)

    ratio_scores: dict = {}
    for ratio_stat, ratio_stat_info in params['ratio-statistics'].items():
        if ratio_stat in ratio_stats:
            volume_statistic = ratio_stat_info['volume-statistic']
            denom_r = (
                coefficients.loc[ratio_stat, 'Variance of Means'] * alpha_weight
                + coefficients.loc[ratio_stat, 'Mean of Variances'] * beta_weight
            ) ** 0.5
            num_r = (
                player_means.loc[:, volume_statistic]
                / coefficients.loc[volume_statistic, 'Mean of Means']
                * (player_means[ratio_stat] - coefficients.loc[ratio_stat, 'Mean of Means'])
            )
            ratio_scores[ratio_stat] = num_r.divide(denom_r)

    res = pd.concat(
        [ratio_scores[r] for r in ratio_scores] + [main_scores], axis=1
    )
    res.columns = ratio_stats + counting_stats

    for neg_stat in params['negative-statistics']:
        if neg_stat in res.columns:
            res[neg_stat] = -res[neg_stat]

    return res.fillna(0)[categories]


def _weighted_cov_matrix(df: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    weighted_means = np.average(df, axis=0, weights=weights)
    deviations     = df - weighted_means
    weighted_cov   = np.dot(weights * deviations.T, deviations) / weights.sum()
    return pd.DataFrame(weighted_cov, columns=df.columns, index=df.columns)


def games_played_adjustment(scores: pd.DataFrame,
                             replacement_games_rate: pd.Series,
                             representative_player_set: list[str],
                             params: dict,
                             categories: list[str],
                             v: pd.Series = None) -> pd.DataFrame:

    if v is None:
        v = pd.Series({stat: 1 / len(categories) for stat in categories})

    totals = scores.dot(v)
    n_players = len(representative_player_set)
    rv = totals.sort_values(ascending=False).iloc[n_players]
    category_level_rv = get_category_level_rv(rv, v, categories)

    replacement_player_value = (
        np.array(category_level_rv.T).reshape(1, -1)
        * np.array(replacement_games_rate).reshape(-1, 1)
    )
    adjusted_scores = scores + replacement_player_value
    adjusted_scores = adjusted_scores - adjusted_scores.loc[representative_player_set].mean()
    return adjusted_scores


# ── public pipeline steps ──────────────────────────────────────────────────────

def process_player_data(player_stats_v2: pd.DataFrame,
                        weekly_df,
                        mean_of_variances: pd.Series,
                        psi: float,
                        chi: float,
                        scoring_format: str,
                        n_drafters: int,
                        n_starters: int,
                        params: dict,
                        categories: list[str],
                        sport: str = 'NBA',
                        ) -> tuple[dict, str]:
    """Explicit-parameter version of process_player_data.
    Receives player_stats_v2 directly instead of reading from st.session_state."""

    counting_stats = get_counting_stats(params, categories)
    ratio_stats    = get_ratio_stats(params, categories)
    n_players      = n_drafters * n_starters
    player_means   = player_stats_v2

    if weekly_df is not None:
        all_players = list(pd.unique(weekly_df.index.get_level_values('Player')))
        coeff_first = calculate_coefficients_historical(weekly_df, all_players, params,
                                                        counting_stats, ratio_stats)
    else:
        coeff_first = calculate_coefficients(player_means, player_means.index,
                                             mean_of_variances, counting_stats,
                                             ratio_stats, params)

    beta_weight = chi if scoring_format == 'Rotisserie' else 1

    g_first = calculate_scores_from_coefficients(player_means, coeff_first, params, 1, beta_weight,
                                                  counting_stats, ratio_stats, categories)
    representative_player_set = (
        g_first.sum(axis=1).sort_values(ascending=False).index[:n_starters * n_drafters]
    )

    if weekly_df is not None:
        coefficients = calculate_coefficients_historical(weekly_df, representative_player_set,
                                                         params, counting_stats, ratio_stats)
    else:
        coefficients = calculate_coefficients(player_means, representative_player_set,
                                             mean_of_variances, counting_stats, ratio_stats, params)

    mov = coefficients.loc[categories, 'Mean of Variances']
    vom = coefficients.loc[categories, 'Variance of Means']
    v_original = np.sqrt(mov / (mov + vom))
    v = v_original / v_original.sum()
    w = vom / mov

    g_scores = calculate_scores_from_coefficients(player_means, coefficients, params, 1, beta_weight,
                                                   counting_stats, ratio_stats, categories)
    x_scores = calculate_scores_from_coefficients(player_means, coefficients, params, 0, beta_weight,
                                                   counting_stats, ratio_stats, categories)

    replacement_games_rate = (1 - player_means['Games Played %'] / 100) * psi
    g_scores = games_played_adjustment(g_scores, replacement_games_rate, representative_player_set,
                                        params, categories)
    x_scores = games_played_adjustment(x_scores, replacement_games_rate, representative_player_set,
                                        params, categories, v=v)

    x_scores.loc['RP', :] = -1
    g_scores.loc['RP', :] = -1

    g_scores.insert(loc=0, column='Total', value=g_scores.sum(axis=1))
    g_scores.sort_values('Total', ascending=False, inplace=True)
    x_scores = x_scores.loc[g_scores.index]

    positions = player_means['Position'].str.split(',')
    position_structure = params['position_structure']
    base_position_list = position_structure['base_list']

    try:
        players_and_positions = pd.merge(x_scores, positions, left_index=True, right_index=True)
        players_and_positions_limited = players_and_positions.iloc[:n_players]
        players_and_positions_limited = players_and_positions_limited.copy()
        players_and_positions_limited[categories] = (
            players_and_positions_limited[categories]
            .sub(players_and_positions_limited[categories].mean(axis=0))
        )
        positions_exploded = (
            players_and_positions_limited
            .explode('Position')
            .reset_index()
            .set_index(['Player', 'Position'])
        )
        position_mean_weights = 1 / positions_exploded.groupby('Player').transform('count')
        position_means_weighted = positions_exploded.mul(position_mean_weights)

        position_means = (
            position_means_weighted.groupby('Position').sum()
            / position_mean_weights.groupby('Position').sum()
        )
        positions_exploded = positions_exploded.sub(positions_exploded.mean(axis=0))
        position_means = position_means.loc[base_position_list, :]
        position_means_g = position_means * v

        if sport == 'MLB':
            pitching_positions = ['SP', 'RP']
            batting_positions  = [p for p in position_means.index if p not in pitching_positions]
            batting_stats  = [x for x in params['batter_stats']  if x in position_means.columns]
            pitching_stats = [x for x in params['pitcher_stats'] if x in position_means.columns]

            from functools import reduce as _reduce
            from backend.session import build_position_config
            # derive position_numbers from params (no slot_counts here; use balanced defaults)
            # this path is only reached for MLB historical mode — not our primary use case
            position_numbers = {p: 1 for p in base_position_list + position_structure['flex_list']}
            pitching_numbers = {p: v_n for p, v_n in position_numbers.items() if p in pitching_positions}
            batting_numbers  = {p: v_n for p, v_n in position_numbers.items() if p in batting_positions}
            pitching_series  = pd.Series(pitching_numbers)
            batting_series   = pd.Series(batting_numbers)
            pitching_series  = pitching_series / pitching_series.sum()
            batting_series   = batting_series / batting_series.sum()

            position_means_g.loc[pitching_positions, batting_stats] = 0
            fix_factor = position_means_g.loc[pitching_positions, pitching_stats].mean(axis=1).values.reshape(-1, 1)
            position_means_g.loc[pitching_positions, pitching_stats] -= fix_factor
            fix_factor_2 = (position_means_g.loc[pitching_positions, pitching_stats]
                            * pitching_series.values.reshape(-1, 1)).sum(axis=0).values.reshape(1, -1)
            position_means_g.loc[pitching_positions, pitching_stats] -= fix_factor_2

            position_means_g.loc[batting_positions, pitching_stats] = 0
            fix_factor = position_means_g.loc[batting_positions, batting_stats].mean(axis=1).values.reshape(-1, 1)
            position_means_g.loc[batting_positions, batting_stats] -= fix_factor
            fix_factor_2 = (position_means_g.loc[batting_positions, batting_stats]
                            * batting_series.values.reshape(-1, 1)).sum(axis=0).values.reshape(1, -1)
            position_means_g.loc[batting_positions, batting_stats] -= fix_factor_2

            total_value = g_scores.loc[representative_player_set].sum(axis=1).sort_values(ascending=False)
            relative_value = total_value - total_value.min()
            helper_df = pd.DataFrame({
                'Round': [i // n_drafters for i in range(n_drafters * n_starters)],
                'Value': relative_value,
            })
            average_round_value = helper_df.groupby('Round')['Value'].mean()

        else:  # NBA (default)
            position_means_g = position_means_g.sub(position_means_g.mean(axis=1), axis=0)
            position_means_g = position_means_g.sub(position_means_g.mean(axis=0), axis=1)
            average_round_value = None

        position_means = position_means_g / v

        L_by_position = pd.concat({
            position: _weighted_cov_matrix(
                positions_exploded.loc[pd.IndexSlice[:, position], :],
                position_mean_weights.loc[pd.IndexSlice[:, position],
                                          position_mean_weights.columns[0]],
            )
            for position in base_position_list
        })

    except Exception:
        position_means      = None
        L_by_position       = np.array([x_scores.cov()])
        average_round_value = None

    info = {
        'G-scores':            g_scores,
        'X-scores':            x_scores,
        'w':                   w,
        'Positions':           positions,
        'Mov':                 mov,
        'Vom':                 vom,
        'Position-Means':      position_means,
        'L-by-Position':       L_by_position,
        'Average-Round-Value': average_round_value,
    }

    return info, str(uuid.uuid4())


def make_upsilon_adjustment(player_stats_v1: pd.DataFrame,
                             upsilon: float,
                             params: dict) -> tuple[pd.DataFrame, str]:
    """Explicit-parameter version: receives the DataFrame directly."""
    df = player_stats_v1.copy()
    df['Games Played %'] = 1 - (1 - df['Games Played %']) * upsilon

    counting_stats   = params['counting-statistics']
    volume_stats     = [info['volume-statistic'] for info in params['ratio-statistics'].values()]
    games_per_week   = params['n_games_per_week']

    for col in set(counting_stats + volume_stats):
        if col in df.columns:
            df[col] = df[col].astype(float) * df['Games Played %'] * games_per_week

    return df, str(uuid.uuid4())


def drop_injured_players(player_stats_v0: pd.DataFrame,
                          injured_players: tuple | list) -> tuple[pd.DataFrame, str]:
    """Explicit-parameter version: receives the DataFrame directly."""
    res = player_stats_v0.drop(list(injured_players), errors='ignore')
    return res, str(uuid.uuid4())
