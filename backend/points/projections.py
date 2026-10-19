"""
Points projection normalisation.

Two equally valid entry paths:

  A. Already-scored fantasy points (points_input_mode == 'projected_points'):
     Accept provider fantasy points as-is.  Never multiply by stat weights.

  B. Raw projected statistics (points_input_mode == 'raw_stats'):
     Compute expected_points_per_game = Σ(weight[stat] × projected_stat_per_game[stat]).
     Negative weights are valid.  Missing stat columns are an explicit error.

Basketball Monster season-total CSV path is handled here: season totals are divided by
projected games to produce per-game rates, zero-GP rows are kept with NaN per-game values,
and provider IDs are left in place for the caller to crosswalk via the player registry.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Optional

import numpy as np
import pandas as pd

from backend.points.config import PointsScoringConfig

logger = logging.getLogger('fbbo.points')


# ── Column name constants for the Basketball Monster export format ─────────────
# These describe ONLY the specific BM season-total export that was audited.
# Inspect the actual file before choosing a mapping; these are a starting reference.
_BM_PLAYER_ID_COL  = 'player_id'
_BM_FIRST_NAME_COL = 'first_name'
_BM_LAST_NAME_COL  = 'last_name'
_BM_GAMES_COL      = 'games'

# Stats available in the BM season-total export
_BM_STAT_COLS = (
    'field_goals', 'field_goals_attempted',
    'free_throws', 'free_throws_attempted',
    'threes', 'threes_attempted',
    'offensive_rebounds', 'defensive_rebounds',
    'assists', 'steals', 'blocks', 'turnovers',
)


# ── Public normalisation entry point ──────────────────────────────────────────

@dataclass_like = None  # see NormalisedProjections below


class NormalisedProjections:
    """Result of normalising a projection file for a points session.

    Attributes:
        per_game_fp:   Series (player_id → fp_per_game); NaN for zero-GP rows.
        total_fp:      Series (player_id → fp_season_total); NaN for zero-GP rows.
        projected_games: Series (player_id → projected_games).
        raw_stats_per_game: DataFrame (player_id × stat_name) if input_mode=='raw_stats'.
        display_name:  Series (player_id → 'First Last') if available.
        audit:         JSON-serialisable dict summarising provenance.
        unsupported_rules: list of stat names that were in scoring_rules but not in data.
        zero_gp_ids:   list of player_ids with zero projected games (no per-game value).
    """

    def __init__(
        self
        , per_game_fp:       pd.Series
        , total_fp:          pd.Series
        , projected_games:   pd.Series
        , raw_stats_per_game: Optional[pd.DataFrame]
        , display_name:      Optional[pd.Series]
        , audit:             dict
        , unsupported_rules: list[str]
        , zero_gp_ids:       list
    ):
        self.per_game_fp       = per_game_fp
        self.total_fp          = total_fp
        self.projected_games   = projected_games
        self.raw_stats_per_game = raw_stats_per_game
        self.display_name      = display_name
        self.audit             = audit
        self.unsupported_rules = unsupported_rules
        self.zero_gp_ids       = zero_gp_ids


def normalise_projection_file(
    raw_df: pd.DataFrame
    , config: PointsScoringConfig
    , file_sha256: Optional[str] = None
    , provider_id_col: Optional[str] = None
    , direct_fp_col: Optional[str] = None
) -> NormalisedProjections:
    """Parse and normalise a projection DataFrame into per-game fantasy points.

    Args:
        raw_df:         Parsed projection DataFrame indexed by player row. Must
                        contain a column for projected games (either 'games' for
                        season-total BM format, or 'Games Played %' / 'Games Played').
        config:         Points session configuration.
        file_sha256:    Optional hash of the original file bytes for provenance.
        provider_id_col: Column name containing provider player IDs (BM: 'player_id').
        direct_fp_col:  Column name carrying already-scored fantasy points
                        (used when points_input_mode == 'projected_points').

    Returns:
        NormalisedProjections with per-game and total FP, audit metadata,
        and information on unsupported rules / zero-GP rows.
    """
    df = raw_df.copy()

    # ── Resolve projected games ────────────────────────────────────────────
    projected_games = _resolve_projected_games(df)
    zero_gp_mask = projected_games == 0

    if config.points_input_mode == 'projected_points':
        return _normalise_direct_points(
            df           = df
            , config     = config
            , fp_col     = direct_fp_col
            , projected_games = projected_games
            , zero_gp_mask    = zero_gp_mask
            , file_sha256     = file_sha256
            , provider_id_col = provider_id_col
        )

    # points_input_mode == 'raw_stats'
    return _normalise_raw_stats(
        df              = df
        , config        = config
        , projected_games = projected_games
        , zero_gp_mask  = zero_gp_mask
        , file_sha256   = file_sha256
        , provider_id_col = provider_id_col
    )


# ── Direct-points path ────────────────────────────────────────────────────────

def _normalise_direct_points(
    df:              pd.DataFrame
    , config:        PointsScoringConfig
    , fp_col:        Optional[str]
    , projected_games: pd.Series
    , zero_gp_mask:  pd.Series
    , file_sha256:   Optional[str]
    , provider_id_col: Optional[str]
) -> NormalisedProjections:
    """Accept the provider's fantasy points as authoritative; no re-weighting."""
    if fp_col is None:
        # Try common column name spellings
        for candidate in ('FantasyPts', 'Fantasy Points', 'FP', 'fantasy_pts', 'fantasy_points'):
            if candidate in df.columns:
                fp_col = candidate
                break
    if fp_col is None or fp_col not in df.columns:
        raise ValueError(
            f"points_input_mode is 'projected_points' but no fantasy-points column was found. "
            f"Available columns: {list(df.columns)}. "
            f"Set direct_fp_col to the correct column name."
        )

    basis = config.projection_basis
    raw_fp = pd.to_numeric(df[fp_col], errors='coerce')

    if basis == 'per_game':
        per_game_fp = raw_fp.copy()
        per_game_fp[zero_gp_mask] = np.nan
        total_fp = per_game_fp * projected_games
    elif basis in ('season_total', 'rest_of_season_total'):
        total_fp = raw_fp.copy()
        total_fp[zero_gp_mask] = np.nan
        per_game_fp = total_fp / projected_games.replace(0, np.nan)
    elif basis == 'per_week':
        raise ValueError(
            "projection_basis 'per_week' requires explicit season-week count to convert "
            "to per-game; supply season-total or per-game projections instead."
        )
    else:
        raise ValueError(f"Unsupported projection_basis: {basis!r}")

    display_name = _extract_display_name(df)
    audit = _build_audit(
        input_mode  = 'projected_points'
        , basis     = basis
        , fp_col    = fp_col
        , rules     = {}
        , n_players = len(df)
        , n_zero_gp = int(zero_gp_mask.sum())
        , file_sha256 = file_sha256
        , config    = config
    )

    return NormalisedProjections(
        per_game_fp       = per_game_fp
        , total_fp        = total_fp
        , projected_games = projected_games
        , raw_stats_per_game = None
        , display_name    = display_name
        , audit           = audit
        , unsupported_rules = []
        , zero_gp_ids     = list(df.index[zero_gp_mask])
    )


# ── Raw-stats path ────────────────────────────────────────────────────────────

def _normalise_raw_stats(
    df:              pd.DataFrame
    , config:        PointsScoringConfig
    , projected_games: pd.Series
    , zero_gp_mask:  pd.Series
    , file_sha256:   Optional[str]
    , provider_id_col: Optional[str]
) -> NormalisedProjections:
    """Derive expected points from raw stat projections and the configured weights."""
    rules = config.scoring_rules
    basis = config.projection_basis

    per_game_stats = _derive_per_game_stats(df, projected_games, zero_gp_mask, basis)

    # Expand derived stats (missed shots) required by scoring rules
    per_game_stats = _expand_derived_stats(per_game_stats, rules)

    unsupported: list[str] = []
    stat_contributions: dict[str, pd.Series] = {}

    for stat, weight in rules.items():
        if stat not in per_game_stats.columns:
            unsupported.append(stat)
        else:
            stat_contributions[stat] = weight * per_game_stats[stat]

    if unsupported:
        raise ValueError(
            f"Scoring rules reference stats not present in the projection file: "
            f"{unsupported}. Either add those columns to the file or remove the "
            f"corresponding rules from scoring_rules."
        )

    per_game_fp = pd.concat(list(stat_contributions.values()), axis=1).sum(axis=1)
    per_game_fp[zero_gp_mask] = np.nan

    # Season total on the raw-stat basis
    if basis in ('season_total', 'rest_of_season_total'):
        # stats are already totals; fp total = direct application of rules to totals
        total_contributions = {
            stat: weight * df[stat].fillna(0)
            for stat, weight in rules.items()
            if stat in df.columns or (stat in per_game_stats.columns)
        }
        total_fp = pd.concat(list(total_contributions.values()), axis=1).sum(axis=1)
        total_fp[zero_gp_mask] = np.nan
    else:
        total_fp = per_game_fp * projected_games

    display_name = _extract_display_name(df)
    audit = _build_audit(
        input_mode  = 'raw_stats'
        , basis     = basis
        , fp_col    = None
        , rules     = rules
        , n_players = len(df)
        , n_zero_gp = int(zero_gp_mask.sum())
        , file_sha256 = file_sha256
        , config    = config
    )

    return NormalisedProjections(
        per_game_fp        = per_game_fp
        , total_fp         = total_fp
        , projected_games  = projected_games
        , raw_stats_per_game = per_game_stats
        , display_name     = display_name
        , audit            = audit
        , unsupported_rules = unsupported
        , zero_gp_ids      = list(df.index[zero_gp_mask])
    )


# ── Basketball Monster season-total helpers ───────────────────────────────────

def normalise_bm_season_totals(
    raw_df: pd.DataFrame
    , config: PointsScoringConfig
    , file_sha256: Optional[str] = None
) -> NormalisedProjections:
    """Normalise a Basketball Monster season-total CSV export.

    Derives NBA points from field_goals, threes, free_throws when not already
    present.  Renames BM columns to canonical names before calling
    normalise_projection_file so the standard pipeline handles the rest.
    """
    df = raw_df.copy()

    # Derive NBA points total if not present (2*FGM + 3PM + FTM)
    if 'Points' not in df.columns and all(
        col in df.columns for col in ('field_goals', 'threes', 'free_throws')
    ):
        df['Points'] = 2 * df['field_goals'] + df['threes'] + df['free_throws']

    # Combine rebounds
    if 'Rebounds' not in df.columns and all(
        col in df.columns for col in ('offensive_rebounds', 'defensive_rebounds')
    ):
        df['Rebounds'] = df['offensive_rebounds'] + df['defensive_rebounds']

    # Map BM column names → canonical projection column names
    rename = {
        'assists':   'Assists',
        'steals':    'Steals',
        'blocks':    'Blocks',
        'turnovers': 'Turnovers',
        'threes':    'Threes',
        'field_goals':           'FGM',
        'field_goals_attempted': 'FGA',
        'free_throws':           'FTM',
        'free_throws_attempted': 'FTA',
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Build display name from first_name / last_name if present
    if _BM_FIRST_NAME_COL in df.columns and _BM_LAST_NAME_COL in df.columns:
        df['Player'] = (
            df[_BM_FIRST_NAME_COL].fillna('').str.strip()
            + ' '
            + df[_BM_LAST_NAME_COL].fillna('').str.strip()
        ).str.strip()

    return normalise_projection_file(
        raw_df         = df
        , config       = config
        , file_sha256  = file_sha256
        , provider_id_col = _BM_PLAYER_ID_COL
        , direct_fp_col   = None  # BM total CSV carries raw stats, not scored FP
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _resolve_projected_games(df: pd.DataFrame) -> pd.Series:
    """Return a numeric Series of projected games, index aligned to df."""
    if _BM_GAMES_COL in df.columns:
        return pd.to_numeric(df[_BM_GAMES_COL], errors='coerce').fillna(0)
    if 'Games Played' in df.columns:
        return pd.to_numeric(df['Games Played'], errors='coerce').fillna(0)
    if 'Games Played %' in df.columns:
        gp_pct = pd.to_numeric(df['Games Played %'], errors='coerce').fillna(0)
        return (gp_pct * 82.0).clip(0, 82)
    raise ValueError(
        "Cannot determine projected games: no 'games', 'Games Played', or "
        "'Games Played %' column found in the projection file."
    )


def _derive_per_game_stats(
    df: pd.DataFrame
    , projected_games: pd.Series
    , zero_gp_mask: pd.Series
    , basis: str
) -> pd.DataFrame:
    """Convert stat columns to per-game rates based on projection_basis."""
    stat_cols = [
        col for col in df.columns
        if col not in ('Player', 'Position', 'Games Played', 'Games Played %',
                       _BM_GAMES_COL, _BM_FIRST_NAME_COL, _BM_LAST_NAME_COL,
                       _BM_PLAYER_ID_COL)
    ]
    numeric = df[stat_cols].apply(pd.to_numeric, errors='coerce')

    if basis == 'per_game':
        result = numeric.copy()
    elif basis in ('season_total', 'rest_of_season_total'):
        # Divide season totals by projected games; produce NaN for zero-GP rows
        safe_games = projected_games.replace(0, np.nan)
        result = numeric.div(safe_games, axis=0)
    else:
        raise ValueError(f"Unsupported projection_basis for raw-stats path: {basis!r}")

    result.loc[zero_gp_mask] = np.nan
    return result


def _expand_derived_stats(per_game: pd.DataFrame, rules: dict[str, float]) -> pd.DataFrame:
    """Add derived stat columns (missed shots) required by the scoring rules."""
    if 'FGMiss' in rules and 'FGMiss' not in per_game.columns:
        if 'FGA' in per_game.columns and 'FGM' in per_game.columns:
            per_game = per_game.copy()
            per_game['FGMiss'] = (per_game['FGA'] - per_game['FGM']).clip(lower=0)
        else:
            # Will be reported as unsupported in the caller
            pass

    if 'FTMiss' in rules and 'FTMiss' not in per_game.columns:
        if 'FTA' in per_game.columns and 'FTM' in per_game.columns:
            per_game = per_game.copy()
            per_game['FTMiss'] = (per_game['FTA'] - per_game['FTM']).clip(lower=0)

    return per_game


def _extract_display_name(df: pd.DataFrame) -> Optional[pd.Series]:
    """Return a Series of display names if the data carries them."""
    if 'Player' in df.columns:
        return df['Player'].astype(str)
    if _BM_FIRST_NAME_COL in df.columns and _BM_LAST_NAME_COL in df.columns:
        return (
            df[_BM_FIRST_NAME_COL].fillna('').str.strip()
            + ' '
            + df[_BM_LAST_NAME_COL].fillna('').str.strip()
        ).str.strip()
    return None


def _build_audit(
    input_mode: str
    , basis: str
    , fp_col: Optional[str]
    , rules: dict
    , n_players: int
    , n_zero_gp: int
    , file_sha256: Optional[str]
    , config: PointsScoringConfig
) -> dict:
    """JSON-serialisable provenance record."""
    return {
        'input_mode':        input_mode,
        'projection_basis':  basis,
        'direct_fp_col':     fp_col,
        'scoring_rules':     rules,
        'n_players':         n_players,
        'n_zero_gp':         n_zero_gp,
        'projection_source': config.projection_source,
        'season':            config.season,
        'captured_at':       config.captured_at,
        'decision_as_of':    config.decision_as_of,
        'file_sha256':       file_sha256,
    }


def hash_projection_bytes(file_bytes: bytes) -> str:
    """Return the SHA-256 hex digest of a projection file's bytes."""
    return hashlib.sha256(file_bytes).hexdigest()
