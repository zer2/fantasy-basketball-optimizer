"""
Pipeline orchestration for the 5-step initialization chain.

Steps:
  1. Load player_stats_v0 from CSV / Snowflake → session.v0_clean
  2. drop_injured_players → session.v1_clean
  3. make_upsilon_adjustment → session.v2
  4. process_player_data → session.info
  5. build HAgent → session.H

No session_context or _SessionState: each step reads/writes session fields directly.
"""

from __future__ import annotations

import io
import time
import threading
import yaml
import pandas as pd
from pathlib import Path

from backend.session import Session


_MEAN_OF_VARIANCES_PATH = Path(__file__).parents[1] / 'coefficient_exploration_output' / 'mean_of_variances.csv'


def _load_mean_of_variances(sport: str) -> pd.Series:
    """Load empirical mean-of-variances for the most recent season from the
    coefficient exploration output CSV.  The CSV has stats as rows and seasons
    as columns (newest first); the first data column is the most recent season.
    """
    df = pd.read_csv(_MEAN_OF_VARIANCES_PATH, index_col=0)
    return df.iloc[:, 0]


# ── v0_clean cache ────────────────────────────────────────────────────────────
# Caches the output of run_step1 (loaded + processed player stats) independently
# of session ID, keyed by data source parameters.  This avoids re-querying
# Snowflake and re-processing the DataFrame every time a new session is created
# with the same data source.

_v0_cache: dict[tuple, tuple[float, pd.DataFrame]] = {}
_v0_cache_lock = threading.Lock()
_V0_CACHE_TTL = 24 * 3600  # 24 hours

# Parameters file path (relative to the project root)
_PARAMS_PATH = 'parameters.yaml'

# ── helpers ───────────────────────────────────────────────────────────────────

def _load_params() -> dict:
    with open(_PARAMS_PATH, 'r') as f:
        return yaml.safe_load(f)


def _sport_params(session: Session) -> tuple[dict, dict, str]:
    """Return (all_params, sport_params, sport) for the current session."""
    all_params = _load_params()
    sport = session.current_params['sport']
    return all_params, all_params[sport], sport


# ── Step 1: load player data ──────────────────────────────────────────────────

def _v0_cache_key(cp: dict) -> tuple | None:
    """Return a hashable cache key for v0_clean based on data source params.

    Returns None for CSV uploads (not cacheable by params alone).
    """
    source_type = cp['data_source_type']
    sport = cp.get('sport', '')
    if source_type == 'historical':
        return (sport, 'historical', cp.get('season') or '2024-25')
    if source_type == 'blended':
        blend_weights = cp.get('blend_weights', {})
        # Only Snowflake-backed sources are cacheable; uploaded DFs are session-specific
        snowflake_keys = tuple(sorted(
            (k, v) for k, v in blend_weights.items() if k not in ('HTB', 'BBM')
        ))
        return (sport, 'blended', snowflake_keys)
    return None


def run_step1(
    session: Session,
    csv_bytes: bytes | None = None,
    file_type: str | None = None,
    uploaded_dfs: dict | None = None,
) -> None:
    """Load player_stats_v0 into session.v0_clean.

    Branches on current_params['data_source_type']:
      'csv'        — single uploaded CSV (csv_bytes + file_type required)
      'historical' — Snowflake historical stats for current_params['season']
      'blended'    — weighted blend of Snowflake sources + any uploaded_dfs

    Results for Snowflake-backed sources are cached at the module level for
    24 hours so repeated session creations with the same data source skip the
    Snowflake round-trip entirely.
    """
    _, params, _ = _sport_params(session)
    cp = session.current_params
    source_type = cp['data_source_type']
    cache_key = _v0_cache_key(cp)

    if cache_key is not None:
        with _v0_cache_lock:
            entry = _v0_cache.get(cache_key)
            if entry is not None and time.time() - entry[0] < _V0_CACHE_TTL:
                session.v0_clean = entry[1].copy()
                return

    if source_type == 'csv':
        v0 = _parse_projection_csv(csv_bytes, file_type, params)

    elif source_type == 'historical':
        from backend.data_retrieval import get_specified_historical_stats

        v0 = get_specified_historical_stats(cp.get('season') or '2024-25', params)

    elif source_type == 'blended':
        from backend.data_retrieval import combine_projections
        v0 = combine_projections(
            blend_weights = cp.get('blend_weights', {}),
            params        = params,
            uploaded_dfs  = uploaded_dfs,
        )

    else:
        raise ValueError(f"Unknown data_source_type: {source_type!r}")

    if cache_key is not None:
        with _v0_cache_lock:
            _v0_cache[cache_key] = (time.time(), v0.copy())

    session.v0_clean = v0.copy()


def _parse_projection_csv(csv_bytes: bytes, file_type: str, params: dict) -> pd.DataFrame:
    """Parse an uploaded CSV (HTB or BBM format) into the canonical column set."""
    df = pd.read_csv(io.BytesIO(csv_bytes))

    renamer_key = {
        'HTB': 'htb-renamer',
        'BBM': 'bbm-renamer',
    }.get(file_type.upper(), 'htb-renamer')

    renamer = params.get(renamer_key, {})
    df = df.rename(columns=renamer)

    # HTB/BBM provide per-game stats directly; ensure Position + Games Played %
    if 'Position' not in df.columns:
        raise ValueError("CSV missing Position column after rename")

    if 'Games Played %' not in df.columns:
        if 'Games Played' in df.columns:
            df['Games Played %'] = df['Games Played'] / 82.0
        else:
            df['Games Played %'] = 0.85  # default

    # Clamp GP to 0–1
    df['Games Played %'] = df['Games Played %'].clip(0, 1)

    if 'Player' in df.columns:
        df = df.set_index('Player')

    return df


# ── Step 2: drop injured players ──────────────────────────────────────────────

def run_step2(session: Session) -> None:
    """Run drop_injured_players and store in session.v1_clean."""
    from backend.math.process_player_data import drop_injured_players

    injured = session.current_params.get('injured_players', [])
    v1, _ = drop_injured_players(session.v0_clean, tuple(injured))
    session.v1_clean = v1.copy()


# ── Step 3: upsilon adjustment ────────────────────────────────────────────────

def run_step3(session: Session) -> None:
    """Run make_upsilon_adjustment using a fresh copy of v1_clean."""
    from backend.math.process_player_data import make_upsilon_adjustment

    _, params, _ = _sport_params(session)
    upsilon = session.current_params['upsilon']
    # Always start from the clean v1 so repeated PATCH calls don't stack adjustments
    v2, _ = make_upsilon_adjustment(session.v1_clean.copy(), upsilon, params)
    session.v2 = v2


# ── Step 4: process_player_data ───────────────────────────────────────────────

def run_step4(session: Session) -> None:
    """Build the info dict (G-scores, X-scores, covariance, etc.)."""
    from backend.math.process_player_data import process_player_data

    _, params, sport = _sport_params(session)
    cp = session.current_params

    scoring_format = cp['scoring_format']
    n_drafters  = cp['n_drafters']
    n_picks     = cp['n_picks']
    slot_counts = cp['slot_counts']
    n_starters  = sum(slot_counts.values()) if slot_counts else n_picks
    categories  = cp['categories']

    info, _ = process_player_data(
        player_stats_v2   = session.v2,
        weekly_df         = None,
        mean_of_variances = _load_mean_of_variances(sport),
        psi               = cp['psi'],
        chi               = cp['chi'],
        scoring_format    = scoring_format,
        n_drafters        = n_drafters,
        n_starters        = n_starters,
        params            = params,
        categories        = categories,
        sport             = sport,
    )
    session.info = info


# ── Step 5: build HAgent ──────────────────────────────────────────────────────

def run_step5(session: Session) -> None:
    """Construct HAgent and store in session.H."""
    from backend.math.algorithm_agents import HAgent

    _, params, sport = _sport_params(session)
    cp = session.current_params

    scoring_format = cp['scoring_format']
    n_picks     = cp['n_picks']
    slot_counts = cp['slot_counts']
    n_starters  = sum(slot_counts.values()) if slot_counts else n_picks
    n_drafters  = cp['n_drafters']

    session.generic_h_scores = None  # invalidate cached baseline whenever HAgent rebuilds

    session.H = HAgent(
        info           = session.info,
        omega          = cp['omega'],
        gamma          = cp['gamma'],
        n_picks        = n_starters,
        n_drafters     = n_drafters,
        dynamic        = cp['n_iterations'] > 0,
        scoring_format = scoring_format,
        sport          = sport,
        params         = params,
        slot_counts    = slot_counts,
        aleph          = cp['aleph'],
        beth           = cp['beth'],
    )


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    session: Session,
    from_step: int = 1,
    csv_bytes: bytes | None = None,
    file_type: str | None = None,
    uploaded_dfs: dict | None = None,
) -> None:
    """Re-run the pipeline starting from the given step number (1–5)."""
    if from_step <= 1:
        run_step1(session, csv_bytes=csv_bytes, file_type=file_type, uploaded_dfs=uploaded_dfs)
    if from_step <= 2:
        run_step2(session)
    if from_step <= 3:
        run_step3(session)
    if from_step <= 4:
        run_step4(session)
    if from_step <= 5:
        run_step5(session)

