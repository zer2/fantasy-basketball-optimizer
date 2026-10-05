"""
Player-stat processing: G-scores, X-scores, covariance, and the info dict the agent runs on.

Ported from the original Streamlit implementation (whose src/ tree is retired); every
function receives its DataFrame and parameters explicitly, and caching moved to the
Session layer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from backend.player_identity import RP_PLAYER_ID


# ── public pipeline steps ──────────────────────────────────────────────────────

def drop_injured_players(player_stats_v0: pd.DataFrame
                          , injured_players: tuple | list) -> pd.DataFrame:
    """Pipeline step 2: drop the players marked injured or excluded. An id not in this pool is skipped:
    a player the pool does not have needs no excluding."""
    return player_stats_v0.drop(list(injured_players), errors='ignore')


def make_upsilon_adjustment(player_stats_v1: pd.DataFrame
                             , upsilon: float
                             , sport_params: dict) -> pd.DataFrame:
    """Pipeline step 3: scale per-game stats to weekly totals, discounting each player's missed games by
    upsilon (0 ignores them, 1 counts them in full)."""
    df = player_stats_v1.copy()
    df['Games Played %'] = 1 - (1 - df['Games Played %']) * upsilon

    counting_stats   = sport_params['counting-statistics']
    volume_stats     = [info['volume-statistic'] for info in sport_params['ratio-statistics'].values()]
    games_per_week   = sport_params['n_games_per_week']

    for col in set(counting_stats + volume_stats):
        if col in df.columns:
            df[col] = df[col].astype(float) * df['Games Played %'] * games_per_week

    return df


def process_player_data(player_stats_v2: pd.DataFrame
                        , weekly_df
                        , mean_of_variances: pd.Series
                        , psi: float
                        , chi: float
                        , scoring_format: str
                        , n_drafters: int
                        , n_active: int
                        , sport_params: dict
                        , categories: list[str]
                        , sport: str = 'NBA'
                        , tiebreaker_category: str | None = None
                        , most_categories_weight: float | None = None
                        ) -> dict:
    """Explicit-parameter version of process_player_data.
    Receives player_stats_v2 directly instead of reading from st.session_state.

    A tiebreaker category is worth more than the others, so the G-scores below carry that: see
    the scaling near the Total column. X-scores and the coefficients are deliberately left alone —
    they describe how categories are DISTRIBUTED, which the scoring rules do not change."""

    counting_stats = _list_counting_stats(sport_params, categories)
    ratio_stats    = _list_ratio_stats(sport_params, categories)
    n_players      = n_drafters * n_active

    if weekly_df is not None:
        all_players = list(pd.unique(weekly_df.index.get_level_values('Player')))
        coeff_first = calculate_coefficients_historical(weekly_df, all_players, sport_params,
                                                        counting_stats, ratio_stats)
    else:
        coeff_first = calculate_coefficients(player_stats_v2, player_stats_v2.index,
                                             mean_of_variances, counting_stats,
                                             ratio_stats, sport_params)

    # Bake the chi factor into the noise term (Mean of Variances) up front -- Rotisserie only -- so it
    # propagates consistently to the scores AND to v (the x->g weighting) and w (the opponent-variance
    # term). Previously chi was applied only in the score denominators, leaving v and w computed as if
    # chi = 1, which over-weighted the high-signal categories and under-stated opponent variance.
    chi_factor = chi if scoring_format == 'Rotisserie' else 1
    coeff_first['Mean of Variances'] = coeff_first['Mean of Variances'] * chi_factor

    g_first = calculate_scores_from_coefficients(player_stats_v2, coeff_first, sport_params, 1, 1,
                                                  counting_stats, ratio_stats, categories)
    g_first = apply_team_volume_correction(g_first, player_stats_v2, coeff_first, sport_params,
                                           ratio_stats, n_active)
    representative_player_set = (
        g_first.sum(axis=1).sort_values(ascending=False).index[:n_active * n_drafters]
    )

    if weekly_df is not None:
        coefficients = calculate_coefficients_historical(weekly_df, representative_player_set,
                                                         sport_params, counting_stats, ratio_stats)
    else:
        coefficients = calculate_coefficients(player_stats_v2, representative_player_set,
                                             mean_of_variances, counting_stats, ratio_stats, sport_params)

    coefficients['Mean of Variances'] = coefficients['Mean of Variances'] * chi_factor

    mov = coefficients.loc[categories, 'Mean of Variances']
    vom = coefficients.loc[categories, 'Variance of Means']
    # v is the x -> g conversion: g = x * v_original before the per-player volume
    # correction G-scores take for percentage categories (the two score calls below differ
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

    g_scores = calculate_scores_from_coefficients(player_stats_v2, coefficients, sport_params, 1, 1,
                                                   counting_stats, ratio_stats, categories)
    # G-scores alone carry the per-player team-volume correction; X-scores stay uncorrected because the
    # H-score agent models each team's volume itself (see apply_team_volume_correction). So g = x *
    # v_original holds for counting categories only: a percentage G-score is corrected before the
    # games-played adjustment, whose re-centring on the representative set then includes it, so it is
    # not a fixed multiple of x * v_original. Anything adding x-space rows to G-scores (the H-score
    # breakdown in ranking.py) converts with x * v_original instead of reading this table.
    g_scores = apply_team_volume_correction(g_scores, player_stats_v2, coefficients, sport_params,
                                            ratio_stats, n_active)
    x_scores = calculate_scores_from_coefficients(player_stats_v2, coefficients, sport_params, 0, 1,
                                                   counting_stats, ratio_stats, categories)

    replacement_games_rate = (1 - player_stats_v2['Games Played %'] / 100) * psi
    g_scores = games_played_adjustment(g_scores, replacement_games_rate, representative_player_set,
                                        sport_params, categories)
    x_scores = games_played_adjustment(x_scores, replacement_games_rate, representative_player_set,
                                        sport_params, categories, v=v)

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
    # The same factor already applied to v above, so the x -> g identity is unchanged by it.
    if tiebreaker_category is not None:
        g_scores[tiebreaker_category] = g_scores[tiebreaker_category] * (1 + most_categories_weight)

    g_scores.insert(loc=0, column='Total', value=g_scores.sum(axis=1))
    g_scores.sort_values('Total', ascending=False, inplace=True)
    x_scores = x_scores.loc[g_scores.index]

    volume_units = calculate_volume_units(player_stats_v2, coefficients, sport_params, ratio_stats)
    volume_units = volume_units.reindex(x_scores.index)
    # The replacement player (D3 of percentage_volume_plan.md): a drafted player nobody could place
    # stands for replacement level, so his volume is that of the players just past the draftable set
    # -- the next round's worth after the top n_drafters * n_active by G-score.
    replacement_tier = [player for player in g_scores.index[n_players:n_players + n_drafters]
                        if player != RP_PLAYER_ID]
    if not replacement_tier:
        replacement_tier = [player for player in g_scores.index[-n_drafters:] if player != RP_PLAYER_ID]
    volume_units.loc[RP_PLAYER_ID] = volume_units.loc[replacement_tier].mean()

    positions = player_stats_v2['Position'].str.split(',')
    position_structure = sport_params['position_structure']
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
    # Volume beside the categories, for the volume-by-category covariance the future-pick model uses to
    # project volume (Phase 2 of percentage_volume_plan.md). Renamed: the volume columns are keyed by
    # their percentage category, which would collide with that category's X column.
    volume_column_names = {ratio_stat: f'{ratio_stat} volume' for ratio_stat in volume_units.columns}
    volume_beside_x = volume_units.rename(columns=volume_column_names)
    if position_means.index.intersection(base_position_list).empty:
        position_means      = None
        L_by_position       = np.array([x_scores[categories].cov()])
        volume_covariance_by_position = np.array([
            pd.concat([x_scores[categories], volume_beside_x], axis=1).cov()
            .loc[list(volume_column_names.values()), categories]
        ])
        average_round_value = None
    else:
        position_means = position_means.loc[base_position_list, :]
        position_means_g = position_means * v

        # MLB: unsupported and unreachable — no ingestion path produces MLB data
        # (see the MLB note in algorithm_agents.HAgent.__init__).
        if sport == 'MLB':
            pitching_positions = ['SP', 'RP']
            batting_positions  = [p for p in position_means.index if p not in pitching_positions]
            batting_stats  = [x for x in sport_params['batter_stats']  if x in position_means.columns]
            pitching_stats = [x for x in sport_params['pitcher_stats'] if x in position_means.columns]

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
                'Round': [i // n_drafters for i in range(n_drafters * n_active)],
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
        volume_covariance_by_position = calculate_volume_covariance_by_position(
            full_pool, volume_beside_x, position_mean_weights, base_position_list, categories)

    # The replacement player gets a position row too, eligible for every base slot.
    #
    # He already has X- and G-score rows (above) but was left out of `positions`, which is built
    # from player_stats_v2 and joined inwards — so a roster holding him crashed the position-aware
    # solve on a lookup he had no row for. He stands for a drafted player who did not resolve to
    # anyone in the pool, and such a player HAS taken a roster spot, so the least-wrong assumption
    # is that he can fill any of them; his -1 scores already make him worthless to field.
    #
    # Added here, after every statistical pool has been sliced out of players_and_positions, so he
    # cannot skew a position mean or a covariance estimate. He is kept out of the candidate list
    # explicitly in algorithm_agents (he used to be excluded only as a side effect of missing from
    # this table, which would have made him draftable the moment he was added).
    positions = pd.concat([positions, pd.Series({RP_PLAYER_ID: list(base_position_list)})])

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
        'Volume-Units':        volume_units,
        'Volume-Covariance-by-Position': volume_covariance_by_position,
    }

    return info


# ── coefficient calculation ────────────────────────────────────────────────────

# Fraction of the top of the draftable (top-n_players) G-score pool to EXCLUDE when building the
# position means. The remaining (weaker) players better represent the replacement-tier talent that
# actually fills flex/late slots; the star-dominated full pool over-states how much value a stacked
# position delivers, which pushed the optimiser to over-commit to a single position. Default 0.25
# trims the top quartile (a mild shrink that stays net-positive without over-flattening the tilts).
# 0.0 recovers the prior full-pool behaviour.
_POSITION_MEAN_POOL_TOP_TRIM = 0.25


def calculate_coefficients(player_means: pd.DataFrame
                            , representative_player_set: list
                            , mean_of_variances: pd.Series
                            , counting_stats: list[str]
                            , ratio_stats: list[str]
                            , sport_params: dict) -> pd.DataFrame:

    var_of_means  = player_means.loc[representative_player_set, counting_stats].var(axis=0)
    mean_of_means = player_means.loc[representative_player_set, counting_stats].mean(axis=0)

    for ratio_stat, ratio_stat_info in sport_params['ratio-statistics'].items():
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
                                       , sport_params: dict
                                       , counting_stats: list[str]
                                       , ratio_stats: list[str]
                                       ) -> pd.DataFrame:
    player_stats = weekly_df.groupby(level='Player').agg(['mean', 'var'])

    mean_of_vars  = player_stats.loc[representative_player_set, (counting_stats, 'var')].mean(axis=0)
    var_of_means  = player_stats.loc[representative_player_set, (counting_stats, 'mean')].var(axis=0)
    mean_of_means = player_stats.loc[representative_player_set, (counting_stats, 'mean')].mean(axis=0)

    for ratio_stat, ratio_stat_info in sport_params['ratio-statistics'].items():
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


def calculate_volume_units(player_means: pd.DataFrame
                           , coefficients: pd.DataFrame
                           , sport_params: dict
                           , ratio_stats: list[str]) -> pd.DataFrame:
    """Each player's VOLUME in every percentage category, relative to average: V / V-bar, the same V-bar
    (the volume statistic's Mean of Means over the representative set) the percentage X-scores divide by.
    A team of n average players totals n.

    Relative volume rather than a volume X-score, (V - V-bar) / sigma_V: the model only ever uses
    V / V-bar, and sigma_V would add a failure mode for nothing -- a pool whose volumes are all equal
    (an upload listing one FGA figure for everyone) makes it zero and every X-score 0/0. Columns are
    keyed by the percentage category, not the volume statistic: Turnovers is both a scored category
    and Assist to TO's volume, and keying by the ratio category keeps the two apart."""
    return pd.DataFrame({
        ratio_stat: player_means[sport_params['ratio-statistics'][ratio_stat]['volume-statistic']]
                    / coefficients.loc[sport_params['ratio-statistics'][ratio_stat]['volume-statistic'],
                                       'Mean of Means']
        for ratio_stat in ratio_stats
    }, index=player_means.index, columns=ratio_stats)


def calculate_scores_from_coefficients(player_means: pd.DataFrame
                                        , coefficients: pd.DataFrame
                                        , sport_params: dict
                                        , alpha_weight: float
                                        , beta_weight: float
                                        , counting_stats: list[str]
                                        , ratio_stats: list[str]
                                        , categories: list[str]) -> pd.DataFrame:

    counting_mean    = coefficients.loc[counting_stats, 'Mean of Means']
    counting_var_m   = coefficients.loc[counting_stats, 'Variance of Means']
    counting_mean_v  = coefficients.loc[counting_stats, 'Mean of Variances']

    denom    = (counting_var_m.values * alpha_weight + counting_mean_v.values * beta_weight) ** 0.5
    num      = player_means.loc[:, counting_stats] - counting_mean
    main_scores = num.divide(denom)

    ratio_scores: dict = {}
    for ratio_stat, ratio_stat_info in sport_params['ratio-statistics'].items():
        if ratio_stat in ratio_stats:
            volume_statistic = ratio_stat_info['volume-statistic']
            denom_r = (
                coefficients.loc[ratio_stat, 'Variance of Means'] * alpha_weight
                + coefficients.loc[ratio_stat, 'Mean of Variances'] * beta_weight
            ) ** 0.5
            volume_average = coefficients.loc[volume_statistic, 'Mean of Means']
            player_volume  = player_means.loc[:, volume_statistic]
            # Volume-weighted excess over the league rate ("excess makes"), per average volume. How a
            # team's total volume dilutes this, and how it scales the weekly noise, is handled at the
            # team level by the agent (algorithm_agents.apply_volume_awareness); G-scores take the
            # per-player version afterwards (apply_team_volume_correction).
            num_r = (
                player_volume / volume_average
                * (player_means[ratio_stat] - coefficients.loc[ratio_stat, 'Mean of Means'])
            )
            ratio_scores[ratio_stat] = num_r.divide(denom_r)

    res = pd.concat(
        [ratio_scores[r] for r in ratio_scores] + [main_scores], axis=1
    )
    res.columns = ratio_stats + counting_stats

    for neg_stat in sport_params['negative-statistics']:
        if neg_stat in res.columns:
            res[neg_stat] = -res[neg_stat]

    return res.fillna(0)[categories]


def calculate_volume_covariance_by_position(full_pool: pd.DataFrame
                                            , volume_beside_x: pd.DataFrame
                                            , position_mean_weights: pd.DataFrame
                                            , base_position_list: list[str]
                                            , categories: list[str]) -> np.ndarray:
    """Per base position, the covariance of each player's volume (V / V-bar, one row per percentage
    category) with his category X-scores, shape (n_positions, n_volume, n_categories).

    Built exactly as L-by-Position is -- the same draftable pool, exploded over positions with the same
    1 / n_positions weights, the same within-position weighted covariance -- so that combined with the
    same slot weights it is the volume-by-category block of the joint covariance whose category block is
    the pick model's L. Computed separately rather than by widening L's frame, so L stays untouched."""
    volume_columns = list(volume_beside_x.columns)
    joint_pool = full_pool.join(volume_beside_x)
    joint_exploded = joint_pool.explode('Position').reset_index().set_index(['Player', 'Position'])
    joint_exploded = joint_exploded[categories + volume_columns].astype(float)
    joint_exploded = joint_exploded.sub(joint_exploded.mean(axis=0))
    return np.array([
        _weighted_cov_matrix(
            joint_exploded.loc[pd.IndexSlice[:, position], :],
            position_mean_weights.loc[pd.IndexSlice[:, position], position_mean_weights.columns[0]],
        ).loc[volume_columns, categories].to_numpy()
        for position in base_position_list
    ])


def _weighted_cov_matrix(df: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    weighted_means = np.average(df, axis=0, weights=weights)
    deviations     = df - weighted_means
    weighted_cov   = np.dot(weights * deviations.T, deviations) / weights.sum()
    return pd.DataFrame(weighted_cov, columns=df.columns, index=df.columns)


def apply_team_volume_correction(scores: pd.DataFrame
                                 , player_means: pd.DataFrame
                                 , coefficients: pd.DataFrame
                                 , sport_params: dict
                                 , ratio_stats: list[str]
                                 , n_active: int) -> pd.DataFrame:
    """Scale each percentage score by the team-volume correction (docs/gscores.md, addendum):
    n_active * V-bar / ((n_active - 1) * V-bar + V_p), the dilution a player's own volume causes on an
    otherwise-average team, in place of equation 4's assumption that team volume is n_active * V-bar
    whoever the player is.

    For G-scores only. It is the H-score agent's volume model (apply_volume_awareness) evaluated on a
    team of average players, which is the team a G-score assumes; the agent tracks real teams' volume
    itself, so its X-scores must stay uncorrected or the dilution would count twice.

    A player with no volume figure already scores 0 in the category (calculate_scores_from_coefficients
    fills it), and any factor leaves 0 at 0, so his factor is taken as 1 rather than letting the NaN
    back in."""
    corrected = scores.copy()
    for ratio_stat in ratio_stats:
        volume_statistic = sport_params['ratio-statistics'][ratio_stat]['volume-statistic']
        volume_average = coefficients.loc[volume_statistic, 'Mean of Means']
        correction_factor = (
            n_active * volume_average
            / ((n_active - 1) * volume_average + player_means[volume_statistic])
        ).reindex(scores.index)
        corrected[ratio_stat] = scores[ratio_stat] * correction_factor.where(scores[ratio_stat] != 0, 1)
    return corrected


def games_played_adjustment(scores: pd.DataFrame
                             , replacement_games_rate: pd.Series
                             , representative_player_set: list[str]
                             , sport_params: dict
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


# ── helpers ────────────────────────────────────────────────────────────────────
# The two category filters serve only process_player_data; get_category_level_rv is public
# (algorithm_agents imports it).

def _list_counting_stats(sport_params: dict, categories: list[str]) -> list[str]:
    """Return counting statistics from sport_params that are in the active categories."""
    return [c for c in sport_params['counting-statistics'] if c in categories]


def _list_ratio_stats(sport_params: dict, categories: list[str]) -> list[str]:
    """Return ratio statistics from sport_params that are in the active categories."""
    return [c for c in sport_params['ratio-statistics'] if c in categories]


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
