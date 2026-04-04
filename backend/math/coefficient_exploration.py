"""
Standalone coefficient exploration for historical seasons.

Computes coefficients (Mean of Means, Variance of Means, Mean of Variances),
the tau/sigma ratio, and the inter-category correlation matrix for each season.
Uses all statistics defined in params, not just the currently selected categories.

Intended to be run locally as a script to inform coefficient and covariance
calibration. Not part of the production server path.

Usage:
    python -m backend.math.coefficient_exploration
"""

from __future__ import annotations

import pandas as pd

from backend.data_retrieval import get_available_seasons, get_weekly_box_scores
from backend.math.process_player_data import (
    calculate_coefficients_historical
    , calculate_scores_from_coefficients
)


def compute_season_coefficients(
    weekly_df: pd.DataFrame
    , params: dict
    , n_drafters: int
    , n_starters: int
) -> dict:
    """Compute coefficients and correlation matrix for one season.

    Args:
        weekly_df:   DataFrame indexed by ('Player', 'Week'), one summed-stat
                     row per player per week, as returned by get_weekly_box_scores.
        params:      Full params dict for the sport (e.g. NBA entry from parameters.yaml).
        n_drafters:  Number of drafters in the representative league.
        n_starters:  Number of picks per drafter.

    Returns:
        Dict with keys:
          'coefficients'            — DataFrame (stats × Mean of Means /
                                      Variance of Means / Mean of Variances)
          'tau_sigma_ratio'         — Series (stat → Mean of Variances /
                                      Variance of Means)
          'correlations'            — DataFrame (category × category average
                                      within-player weekly correlation)
          'representative_player_set' — list of players used for final pass
    """
    all_counting_stats = params['counting-statistics']
    all_ratio_stats    = list(params['ratio-statistics'].keys())
    all_categories     = all_counting_stats + all_ratio_stats

    all_players = pd.unique(weekly_df.index.get_level_values('Player'))

    # Player means for score calculation — includes made/volume columns for ratio stats
    extra_columns = {
        col
        for info in params['ratio-statistics'].values()
        for col in (info['made-statistic'], info['volume-statistic'])
        if col in weekly_df.columns
    }
    player_means_columns = list(set(all_counting_stats) | extra_columns)
    player_means = weekly_df[player_means_columns].groupby(level='Player').mean()

    # Ratio stats are not in weekly_df; compute them from player-level made/volume means
    for ratio_stat, ratio_stat_info in params['ratio-statistics'].items():
        made_statistic   = ratio_stat_info['made-statistic']
        volume_statistic = ratio_stat_info['volume-statistic']
        if made_statistic in player_means.columns and volume_statistic in player_means.columns:
            player_means[ratio_stat] = player_means[made_statistic] / player_means[volume_statistic]

    # First pass: use all players to get first-order scores for representative set selection
    first_order_coefficients = calculate_coefficients_historical(
        weekly_df.copy()
        , all_players
        , params
        , all_counting_stats
        , all_ratio_stats
    )

    first_order_scores = calculate_scores_from_coefficients(
        player_means
        , first_order_coefficients
        , params
        , alpha_weight  = 1
        , beta_weight   = 1
        , counting_stats = all_counting_stats
        , ratio_stats    = all_ratio_stats
        , categories     = all_categories
    )
    n_players = n_drafters * n_starters
    representative_player_set = (
        first_order_scores.sum(axis=1).nlargest(n_players).index.tolist()
    )

    # Second pass: final coefficients with representative set.
    # calculate_coefficients_historical writes volume_adjusted_ columns onto
    # the DataFrame in-place, which we need for the correlation matrix.
    weekly_df_copy = weekly_df.copy()
    coefficients = calculate_coefficients_historical(
        weekly_df_copy
        , representative_player_set
        , params
        , all_counting_stats
        , all_ratio_stats
    )

    # Correlation matrix: average per-player weekly correlation across categories.
    # These only get used for Roto 
    # Ratio stats use their volume-adjusted weekly values (computed by the second pass above).
    adjusted_column_names = (
        all_counting_stats
        + ['volume_adjusted_' + ratio_stat for ratio_stat in all_ratio_stats]
    )
    wdf_adjusted = weekly_df_copy[adjusted_column_names].copy()
    wdf_adjusted.columns = all_categories

    per_player_corr = (
        wdf_adjusted
        .loc[representative_player_set]
        .groupby(level='Player')
        .corr()
    )
    per_player_corr.index = per_player_corr.index.rename(['Player', 'Category'])
    correlations = per_player_corr.groupby(level='Category').mean()

    return {
        'coefficients':               coefficients,
        'tau_sigma_ratio':            coefficients['Mean of Variances'] / coefficients['Variance of Means'],
        'correlations':               correlations,
        'representative_player_set':  representative_player_set,
    }


def explore_all_seasons(
    params: dict
    , n_drafters: int
    , n_starters: int
    , seasons: list[str] | None = None
) -> dict[str, dict]:
    """Run coefficient exploration across all (or specified) historical seasons.

    Args:
        params:     Full params dict for the sport.
        n_drafters: Number of drafters in the representative league.
        n_starters: Number of picks per drafter.
        seasons:    Explicit list of season strings to run. If None, all
                    available seasons from Snowflake are used.

    Returns:
        Dict mapping season string → result dict from compute_season_coefficients.
    """
    if seasons is None:
        seasons = get_available_seasons()

    return {
        season: compute_season_coefficients(
            get_weekly_box_scores(season, params)
            , params
            , n_drafters
            , n_starters
        )
        for season in seasons
    }


def tabulate_results(results: dict[str, dict]) -> dict[str, pd.DataFrame]:
    """Pivot explore_all_seasons results into cross-season summary tables.

    Args:
        results: dict mapping season string → result dict from
                 compute_season_coefficients.

    Returns:
        Dict with keys:
          'tau_sigma_ratio'  — DataFrame (stat × season)
          'mean_of_means'    — DataFrame (stat × season)
          'variance_of_means'— DataFrame (stat × season)
          'mean_of_variances'— DataFrame (stat × season)
    """
    seasons = list(results.keys())

    tau_sigma_ratio   = pd.DataFrame({s: results[s]['tau_sigma_ratio'] for s in seasons})
    mean_of_means     = pd.DataFrame({s: results[s]['coefficients']['Mean of Means'] for s in seasons})
    variance_of_means = pd.DataFrame({s: results[s]['coefficients']['Variance of Means'] for s in seasons})
    mean_of_variances = pd.DataFrame({s: results[s]['coefficients']['Mean of Variances'] for s in seasons})

    return {
        'tau_sigma_ratio':    tau_sigma_ratio,
        'mean_of_means':      mean_of_means,
        'variance_of_means':  variance_of_means,
        'mean_of_variances':  mean_of_variances,
    }


# ── Script entry point ────────────────────────────────────────────────────────

if __name__ == '__main__':
    import yaml
    from pathlib import Path

    params_path  = Path(__file__).parents[2] / 'parameters.yaml'
    with open(params_path) as f:
        all_params = yaml.safe_load(f)
    params = all_params['NBA']

    output_dir = Path(__file__).parents[2] / 'coefficient_exploration_output'
    output_dir.mkdir(exist_ok=True)

    results = explore_all_seasons(params, n_drafters=12, n_starters=13)
    tables  = tabulate_results(results)

    tables['tau_sigma_ratio'].to_csv(output_dir / 'tau_sigma_ratio.csv')
    tables['mean_of_means'].to_csv(output_dir / 'mean_of_means.csv')
    tables['variance_of_means'].to_csv(output_dir / 'variance_of_means.csv')
    tables['mean_of_variances'].to_csv(output_dir / 'mean_of_variances.csv')

    for season, data in results.items():
        data['correlations'].to_csv(output_dir / f'correlations_{season}.csv')

    print(f'Results saved to {output_dir}')
