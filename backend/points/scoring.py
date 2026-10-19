"""
Deterministic points scoring from normalised per-game projections.

Applies scoring rules to per-game stats and computes expected fantasy points.
Handles negative weights, overlapping components (e.g. Pts + FGM + 3PM together),
and missed-shot penalty derivation.

Threshold/stacking bonuses require explicit projected event-count distributions
and are rejected as unsupported rather than silently approximated.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from backend.points.config import PointsScoringConfig


class UnsupportedScoringRuleError(ValueError):
    """A scoring rule cannot be computed from the available projection data."""


def compute_player_fp_per_game(
    per_game_stats: pd.DataFrame
    , scoring_rules: dict[str, float]
) -> pd.Series:
    """Compute expected fantasy points per game for each player.

    Args:
        per_game_stats: DataFrame indexed by player_id; columns are stat names.
                        Players with zero projected games should already have NaN stats.
        scoring_rules:  {stat_name: weight}; negative weights are valid.

    Returns:
        Series indexed like per_game_stats; NaN where stats are NaN.

    Raises:
        UnsupportedScoringRuleError: if a rule's stat is absent from per_game_stats
            AND cannot be derived from available columns.
    """
    stats = _expand_derived_stats(per_game_stats, scoring_rules)
    missing = [stat for stat in scoring_rules if stat not in stats.columns]
    if missing:
        raise UnsupportedScoringRuleError(
            f"Scoring rules reference stats not available in the projection data: "
            f"{missing}. Check your scoring_rules configuration and ensure the "
            f"projection file contains the required columns."
        )

    contributions = [
        scoring_rules[stat] * stats[stat]
        for stat in scoring_rules
    ]
    if not contributions:
        raise UnsupportedScoringRuleError("scoring_rules is empty; cannot compute FP.")

    return pd.concat(contributions, axis=1).sum(axis=1, skipna=False)


def compute_player_fp_total(
    per_game_fp: pd.Series
    , projected_games: pd.Series
) -> pd.Series:
    """Multiply per-game FP by projected games; players with zero games → NaN."""
    safe_games = projected_games.where(projected_games > 0, np.nan)
    return per_game_fp * safe_games


def build_sample_game_calculation(
    stat_values: dict[str, float]
    , scoring_rules: dict[str, float]
) -> dict:
    """Return a step-by-step breakdown for a single example game.

    Useful for the import preview / candidate explanation.  Always returns a
    complete calculation dict so the UI can display it without extra logic.

    Args:
        stat_values:   {stat_name: value} for one sample game.
        scoring_rules: {stat_name: weight}.

    Returns:
        {
          'components': [{stat, weight, value, contribution}, ...],
          'total': float,
          'unsupported': [stat names missing from stat_values],
        }
    """
    # Expand derived stats in the sample values
    expanded = dict(stat_values)
    if 'FGMiss' in scoring_rules and 'FGMiss' not in expanded:
        fgm = expanded.get('FGM', 0)
        fga = expanded.get('FGA', 0)
        expanded['FGMiss'] = max(0.0, fga - fgm)
    if 'FTMiss' in scoring_rules and 'FTMiss' not in expanded:
        ftm = expanded.get('FTM', 0)
        fta = expanded.get('FTA', 0)
        expanded['FTMiss'] = max(0.0, fta - ftm)

    components = []
    total = 0.0
    unsupported_in_sample = []

    for stat, weight in scoring_rules.items():
        if stat in expanded:
            value = expanded[stat]
            contribution = weight * value
            total += contribution
            components.append({
                'stat':         stat,
                'weight':       weight,
                'value':        value,
                'contribution': contribution,
            })
        else:
            unsupported_in_sample.append(stat)

    return {
        'components':   components,
        'total':        total,
        'unsupported':  unsupported_in_sample,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _expand_derived_stats(
    stats: pd.DataFrame
    , scoring_rules: dict[str, float]
) -> pd.DataFrame:
    """Add FGMiss / FTMiss columns if the rules require them."""
    derived: dict[str, pd.Series] = {}

    if 'FGMiss' in scoring_rules and 'FGMiss' not in stats.columns:
        if 'FGA' in stats.columns and 'FGM' in stats.columns:
            derived['FGMiss'] = (stats['FGA'] - stats['FGM']).clip(lower=0)

    if 'FTMiss' in scoring_rules and 'FTMiss' not in stats.columns:
        if 'FTA' in stats.columns and 'FTM' in stats.columns:
            derived['FTMiss'] = (stats['FTA'] - stats['FTM']).clip(lower=0)

    if derived:
        return pd.concat([stats, pd.DataFrame(derived, index=stats.index)], axis=1)
    return stats
