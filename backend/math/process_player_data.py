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

import os
import uuid

import numpy as np
import pandas as pd

# ── public helpers ─────────────────────────────────────────────────────────────

def get_counting_stats(params: dict, categories: list[str]) -> list[str]:
    """Return counting statistics from params that are in the active categories."""
    return [c for c in params['counting-statistics'] if c in categories]


def get_ratio_stats(params: dict, categories: list[str]) -> list[str]:
    """Return ratio statistics from params that are in the active categories."""
    return [c for c in params['ratio-statistics'] if c in categories]


def get_category_level_rv(rv: float
                          , v: pd.Series
                          , categories: list[str]) -> pd.Series:
    rv_multiple = (rv / (len(categories) - 2)
                   if 'Turnovers' in categories
                   else rv / len(categories))
    return pd.Series({
        stat: -rv_multiple / v[stat] if stat == 'Turnovers' else rv_multiple / v[stat]
        for stat in categories
    })


# ── coefficient calculation ────────────────────────────────────────────────────

def calculate_coefficients(player_means: pd.DataFrame
                            , representative_player_set: list
                            , mean_of_variances: pd.Series
                            , counting_stats: list[str]
                            , ratio_stats: list[str]
                            , params: dict) -> pd.DataFrame:

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


def calculate_coefficients_historical(weekly_df: pd.DataFrame
                                       , representative_player_set: list
                                       , params: dict
                                       , counting_stats: list[str]
                                       , ratio_stats: list[str]
                                       , coefficient_exploration_mode: bool = False
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


def scale_tiebreaker_value(category_values, tiebreaker_position: int, most_categories_weight: float):
    """Give the tiebreaker category the extra value the scoring rules give it, in place-safe form.

    v is the value of a category per unit of x-score (g = x * v exactly), so a category that
    counts twice in the majority half of the objective and once in the per-category half is worth
    (1 + most_categories_weight) of an ordinary one. Everything that reads v inherits that: the
    neutral weight vector a balanced team drafts to, the reference the anti-crowded-punt penalty
    measures punt depth against, the field weights inside get_x_mu, and the x <-> g conversion.

    Shared because v is built in two places — here and again in HAgent, which recomputes it from
    the same coefficients — and the two must not drift apart. Takes and returns a plain array, so
    the caller keeps whatever index it had.
    """
    scaled = np.asarray(category_values, dtype=float).copy()
    scaled[tiebreaker_position] = scaled[tiebreaker_position] * (1 + most_categories_weight)
    return scaled


def calculate_scores_from_coefficients(player_means: pd.DataFrame
                                        , coefficients: pd.DataFrame
                                        , params: dict
                                        , alpha_weight: float
                                        , beta_weight: float
                                        , counting_stats: list[str]
                                        , ratio_stats: list[str]
                                        , categories: list[str]
                                        , n_starters: int) -> pd.DataFrame:

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
            volume_average = coefficients.loc[volume_statistic, 'Mean of Means']
            player_volume  = player_means.loc[:, volume_statistic]
            # Team-denominator correction (the unpublished revision of the G-score paper):
            # the paper's equation 4 approximates a team's attempt volume as
            # n_starters * V-bar regardless of who the player is. Without that
            # approximation, a team fielding player p attempts
            # (n_starters - 1) * V-bar + V_p, so p's percentage impact is the original
            # numerator times n_starters * V-bar / ((n_starters - 1) * V-bar + V_p) --
            # a player's own volume slightly dampens their own percentage impact.
            team_volume_correction = (
                n_starters * volume_average
                / ((n_starters - 1) * volume_average + player_volume)
            )
            num_r = (
                player_volume / volume_average
                * (player_means[ratio_stat] - coefficients.loc[ratio_stat, 'Mean of Means'])
                * team_volume_correction
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


def games_played_adjustment(scores: pd.DataFrame
                             , replacement_games_rate: pd.Series
                             , representative_player_set: list[str]
                             , params: dict
                             , categories: list[str]
                             , v: pd.Series = None) -> pd.DataFrame:

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


# Fraction of the top of the draftable (top-n_players) G-score pool to EXCLUDE when building the
# position means. The remaining (weaker) players better represent the replacement-tier talent that
# actually fills flex/late slots; the star-dominated full pool over-states how much value a stacked
# position delivers, which pushed the optimiser to over-commit to a single position. Default 0.25
# trims the top quartile (a mild shrink that stays net-positive without over-flattening the tilts).
# 0.0 recovers the prior full-pool behaviour; overridable via the POSITION_MEAN_POOL_TOP_TRIM env var.
_POSITION_MEAN_POOL_TOP_TRIM = float(os.environ.get('POSITION_MEAN_POOL_TOP_TRIM', '0.25'))


# ── public pipeline steps ──────────────────────────────────────────────────────

def process_player_data(player_stats_v2: pd.DataFrame
                        , weekly_df
                        , mean_of_variances: pd.Series
                        , psi: float
                        , chi: float
                        , scoring_format: str
                        , n_drafters: int
                        , n_starters: int
                        , params: dict
                        , categories: list[str]
                        , sport: str = 'NBA'
                        , tiebreaker_category: str | None = None
                        , most_categories_weight: float | None = None
                        ) -> tuple[dict, str]:
    """Explicit-parameter version of process_player_data.
    Receives player_stats_v2 directly instead of reading from st.session_state.

    A tiebreaker category is worth more than the others, so the G-scores below carry that: see
    the scaling near the Total column. X-scores and the coefficients are deliberately left alone —
    they describe how categories are DISTRIBUTED, which the scoring rules do not change."""

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

    # Bake the chi factor into the noise term (Mean of Variances) up front -- Rotisserie only -- so it
    # propagates consistently to the scores AND to v (the x->g weighting) and w (the opponent-variance
    # term). Previously chi was applied only in the score denominators, leaving v and w computed as if
    # chi = 1, which over-weighted the high-signal categories and under-stated opponent variance.
    chi_factor = chi if scoring_format == 'Rotisserie' else 1
    coeff_first['Mean of Variances'] = coeff_first['Mean of Variances'] * chi_factor

    g_first = calculate_scores_from_coefficients(player_means, coeff_first, params, 1, 1,
                                                  counting_stats, ratio_stats, categories,
                                                  n_starters)
    representative_player_set = (
        g_first.sum(axis=1).sort_values(ascending=False).index[:n_starters * n_drafters]
    )

    if weekly_df is not None:
        coefficients = calculate_coefficients_historical(weekly_df, representative_player_set,
                                                         params, counting_stats, ratio_stats)
    else:
        coefficients = calculate_coefficients(player_means, representative_player_set,
                                             mean_of_variances, counting_stats, ratio_stats, params)

    coefficients['Mean of Variances'] = coefficients['Mean of Variances'] * chi_factor

    mov = coefficients.loc[categories, 'Mean of Variances']
    vom = coefficients.loc[categories, 'Variance of Means']
    # v is the x -> g conversion: g = x * v_original, exactly (the two score calls below differ
    # only in whether Variance of Means joins the denominator). It is therefore also the statement
    # of what a category is WORTH per unit of x, which is where a tiebreaker belongs — worth twice
    # in the majority half of the objective and once in the per-category half, so (1 + weight).
    #
    # Putting it here rather than on the G-scores alone keeps every consumer consistent, because v
    # wears several hats: the neutral weight vector a balanced team drafts to, the reference the
    # anti-crowded-punt penalty measures punt depth against, the field weights inside get_x_mu, and
    # the x <-> g conversion for position means. Scaling the G-scores alone would have left all of
    # those describing a category that counts once.
    v_original = np.sqrt(mov / (mov + vom))
    if tiebreaker_category is not None:
        if tiebreaker_category not in list(v_original.index):
            raise ValueError(f'Tiebreaker category {tiebreaker_category!r} is not among the scored '
                             f'categories {list(v_original.index)}.')
        if most_categories_weight is None:
            raise ValueError('A tiebreaker category needs most_categories_weight to price it.')
        v_original = pd.Series(
            scale_tiebreaker_value(v_original,
                                   list(v_original.index).index(tiebreaker_category),
                                   most_categories_weight),
            index=v_original.index,
        )
    v = v_original / v_original.sum()
    w = vom / mov

    g_scores = calculate_scores_from_coefficients(player_means, coefficients, params, 1, 1,
                                                   counting_stats, ratio_stats, categories,
                                                   n_starters)
    x_scores = calculate_scores_from_coefficients(player_means, coefficients, params, 0, 1,
                                                   counting_stats, ratio_stats, categories,
                                                   n_starters)

    replacement_games_rate = (1 - player_means['Games Played %'] / 100) * psi
    g_scores = games_played_adjustment(g_scores, replacement_games_rate, representative_player_set,
                                        params, categories)
    x_scores = games_played_adjustment(x_scores, replacement_games_rate, representative_player_set,
                                        params, categories, v=v)

    from backend.player_identity import RP_PLAYER_ID
    x_scores.loc[RP_PLAYER_ID, :] = -1
    g_scores.loc[RP_PLAYER_ID, :] = -1

    # A tiebreaker category is worth (1 + most_categories_weight) of an ordinary one: twice in the
    # majority half of the objective, once in the per-category half. G-scores are the pipeline's
    # statement of what a player is WORTH, so that belongs here — a shot-blocker really is more
    # valuable when blocks settle tied matchups, and nothing downstream knew it.
    #
    # Applied to the finished table, so every row scales together (the replacement sentinel
    # included: missing a doubled category costs double too), and before the Total below, so the
    # ranking carries it. That ranking is load-bearing rather than cosmetic — the agent orders its
    # X-scores by G-total, which decides the draftable pool, the position means and covariance
    # drawn from it, and the anchors the opponent field is built out of. The field therefore ends
    # up holding more of the doubled category because those players are worth more, rather than
    # because every opponent was assumed to chase it.
    #
    # Deliberately NOT scaled: x_scores and the coefficients (mov, vom, v, w). Those describe how
    # categories are distributed, which is a fact about basketball, not about scoring rules — and
    # the win-count DP already gives the tiebreaker its doubled weight in the objective, so scaling
    # them would state the same rule twice, in a form ("you win it by twice as much") that is not
    # what the rule says. The representative player set is left alone for the same reason: it
    # anchors the statistical baseline.
    # The same factor already applied to v above, so g = x * v_original still holds exactly.
    if tiebreaker_category is not None:
        g_scores[tiebreaker_category] = g_scores[tiebreaker_category] * (1 + most_categories_weight)

    g_scores.insert(loc=0, column='Total', value=g_scores.sum(axis=1))
    g_scores.sort_values('Total', ascending=False, inplace=True)
    x_scores = x_scores.loc[g_scores.index]

    positions = player_means['Position'].str.split(',')
    position_structure = params['position_structure']
    base_position_list = position_structure['base_list']

    players_and_positions = pd.merge(x_scores, positions, left_index=True, right_index=True)

    # Position means (the positional *tilt*) are built from the weaker slice of the draftable pool —
    # the top _POSITION_MEAN_POOL_TOP_TRIM fraction is skipped, since the replacement-tier players that
    # actually fill flex/late slots have milder tilts than the star-heavy full pool. This trim applies
    # ONLY to the means: the covariance L below is estimated from the full pool, because it is a
    # distributional property that a biased weak subsample would badly mis-estimate.
    pool_start     = int(n_players * _POSITION_MEAN_POOL_TOP_TRIM)
    means_pool     = players_and_positions.iloc[pool_start:n_players].copy()
    means_pool[categories] = means_pool[categories].sub(means_pool[categories].mean(axis=0))
    means_exploded = (
        means_pool.explode('Position').reset_index().set_index(['Player', 'Position'])
    )
    means_weights  = 1 / means_exploded.groupby('Player').transform('count')
    position_means = (
        means_exploded.mul(means_weights).groupby('Position').sum()
        / means_weights.groupby('Position').sum()
    )

    # Covariance inputs use the full draftable pool (unbiased) — matching the untrimmed behaviour, so
    # the trim never perturbs L.
    full_pool = players_and_positions.iloc[:n_players].copy()
    full_pool[categories] = full_pool[categories].sub(full_pool[categories].mean(axis=0))
    positions_exploded = (
        full_pool.explode('Position').reset_index().set_index(['Player', 'Position'])
    )
    position_mean_weights = 1 / positions_exploded.groupby('Player').transform('count')
    positions_exploded = positions_exploded.sub(positions_exploded.mean(axis=0))
    # Some historical seasons carry no position data — every player is 'NP'. There are then no
    # base-position rows to build means/covariances from (position_means.loc[base_position_list]
    # would raise KeyError). Fall back to "no position data": position_means=None makes the agents
    # skip roster-slot assignment and evaluate.py show every candidate instead of dropping them all.
    # Without this the candidate table comes back empty for all-'NP' seasons (e.g. 1984-85).
    if position_means.index.intersection(base_position_list).empty:
        position_means      = None
        L_by_position       = np.array([x_scores[categories].cov()])
        average_round_value = None
    else:
        position_means = position_means.loc[base_position_list, :]
        position_means_g = position_means * v

        if sport == 'MLB':
            pitching_positions = ['SP', 'RP']
            batting_positions  = [p for p in position_means.index if p not in pitching_positions]
            batting_stats  = [x for x in params['batter_stats']  if x in position_means.columns]
            pitching_stats = [x for x in params['pitcher_stats'] if x in position_means.columns]

            # Balanced default since slot_counts is not available in this path; MLB
            # historical mode is not the primary use case.
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


def make_upsilon_adjustment(player_stats_v1: pd.DataFrame
                             , upsilon: float
                             , params: dict) -> tuple[pd.DataFrame, str]:
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


def drop_injured_players(player_stats_v0: pd.DataFrame
                          , injured_players: tuple | list) -> tuple[pd.DataFrame, str]:
    """Explicit-parameter version: receives the DataFrame directly."""
    res = player_stats_v0.drop(list(injured_players), errors='ignore')
    return res, str(uuid.uuid4())
