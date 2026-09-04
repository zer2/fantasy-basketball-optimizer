"""
Pipeline orchestration for the 5-step initialization chain.

Steps (build_agent dispatches by number; each function is one step):
  1. load_player_pool         — CSV / Snowflake → session.v0_clean + player_registry
  2. remove_injured_players   — → session.v1_clean
  3. apply_upsilon_adjustment — → session.v2
  4. build_scoring_info       — process_player_data → session.info
  5. build_session_agent      — HAgent + primed baseline → session.agent

No session_context or _SessionState: each step reads/writes session fields directly.
The projection-file parser this module orchestrates around lives in projection_parsing.
"""

from __future__ import annotations

import time
import threading

import pandas as pd
from pathlib import Path

from backend.parameters import load_all_params
from backend.services.projection_parsing import parse_projection_upload
from backend.data_retrieval import attach_player_ids_by_name, combine_projections, get_specified_historical_stats
from backend.math.algorithm_agents import HAgent
from backend.math.process_player_data import drop_injured_players, make_upsilon_adjustment, process_player_data
from backend.player_identity import (
    RP_PLAYER_ID, allocate_synthetic_player_ids, make_player_identity,
    make_replacement_player_identity, resolve_typed_player_names,
)
from backend.state.session import Session


class InsufficientPlayerPoolError(Exception):
    """The available player pool is too small to fill every roster in the league. This is a bad
    league configuration (too few players for the number of teams x roster spots), not a server
    fault, so the routes surface it as a 400."""


# build_agent.py is backend/services/build_agent.py, so the project root is parents[2].
_MEAN_OF_VARIANCES_PATH = Path(__file__).parents[2] / 'coefficient_exploration_output' / 'mean_of_variances.csv'


def _load_mean_of_variances() -> pd.Series:
    """Load empirical mean-of-variances for the most recent season from the
    coefficient exploration output CSV.  The CSV has stats as rows and seasons
    as columns (newest first); the first data column is the most recent season.
    """
    df = pd.read_csv(_MEAN_OF_VARIANCES_PATH, index_col=0)
    return df.iloc[:, 0]


# ── v0_clean cache ────────────────────────────────────────────────────────────
# Caches the output of load_player_pool (loaded + processed player stats) independently
# of session ID, keyed by data source parameters.  This avoids re-querying
# Snowflake and re-processing the DataFrame every time a new session is created
# with the same data source.

_v0_cache: dict[tuple, tuple[float, pd.DataFrame]] = {}
_v0_cache_lock = threading.Lock()
_V0_CACHE_TTL = 24 * 3600  # 24 hours


def clear_v0_cache() -> None:
    """Evict all entries from the v0_clean in-memory cache."""
    with _v0_cache_lock:
        _v0_cache.clear()


# ── helpers ───────────────────────────────────────────────────────────────────

def _resolve_sport_params(session: Session) -> tuple[dict, dict, str]:
    """Return (all_params, sport_params, sport) for the current session."""
    all_params = load_all_params()
    sport = session.current_settings['sport']
    return all_params, all_params[sport], sport


# ── Step 1: load player data ──────────────────────────────────────────────────

def _build_v0_cache_key(current_settings: dict) -> tuple | None:
    """Return a hashable cache key for v0_clean based on data source params.

    A projections blend is fully described by every source weight plus the ids of any
    uploaded tables feeding it: uploads are stored immutably under their data_id, so the
    id doubles as a content key. (Leaving the uploaded-source weights or the upload ids out of
    the key — as an earlier version did — served stale blends when an upload's weight
    changed, and could leak an uploaded blend into sessions that never uploaded anything.)
    Returns None only for single-CSV mode, whose bytes arrive outside current_settings.
    """
    source_type = current_settings['data_source_type']
    sport = current_settings['sport']
    if source_type == 'historical':
        return (sport, 'historical', current_settings['season'])
    if source_type == 'projections':
        blend_weights = current_settings['blend_weights']
        custom_data_ids = current_settings.get('custom_data_ids') or []
        weight_keys = tuple(sorted(blend_weights.items()))
        upload_keys = tuple(sorted(custom_data_ids))
        return (sport, 'projections', weight_keys, upload_keys)
    return None


def _resolve_single_csv_player_ids(parsed_csv: pd.DataFrame) -> pd.DataFrame:
    """Bring a single uploaded CSV (name-indexed by parse_projection_upload) to the id-keyed
    contract: resolve names via the unified table, keep unresolved rows under synthetic
    ids, and return an id-indexed frame with the display 'Player' column retained."""

    frame = attach_player_ids_by_name(parsed_csv.reset_index())
    synthetic_ids = allocate_synthetic_player_ids(
        frame.loc[frame['player_id'].isna(), 'Player'])
    resolved = frame['player_id'].astype('object').fillna(frame['Player'].map(synthetic_ids))
    frame = frame.drop(columns=['player_id'])
    frame.index = pd.Index(resolved.astype(int), name='Player')
    if frame.index.has_duplicates:
        duplicated = frame.index[frame.index.duplicated()].tolist()
        raise ValueError(f'Uploaded CSV: multiple rows resolve to the same player id(s): {duplicated}')
    return frame


def _build_player_registry(v0_with_names: pd.DataFrame) -> dict:
    """One PlayerIdentity per pool row (from the id index + 'Player'/'Position' columns),
    plus the replacement-player sentinel."""

    registry = {
        int(player_id): make_player_identity(int(player_id), str(name), position_value)
        for player_id, name, position_value in zip(
            v0_with_names.index, v0_with_names['Player'], v0_with_names['Position'])
    }
    registry[RP_PLAYER_ID] = make_replacement_player_identity()
    return registry


def _count_starters(slot_counts: dict, n_picks: int) -> int:
    """Starters fielded per scoring period: the slot total when a structure is set, else every pick."""
    return sum(slot_counts.values()) if slot_counts else n_picks


def derive_effective_objective(session: Session) -> tuple[list[str], str | None]:
    """The categories and tiebreaker the build can actually score, derived from the
    REQUESTED objective in current_settings and the columns v2 actually carries.

    A category needs its own column, and a ratio category also needs the volume column
    that weights it (a percentage cannot be scored without the attempts behind it).
    The narrowing can drop the very category chosen to break ties, or leave an odd
    count where no tie can arise — either way the tiebreaker no longer refers to
    anything the agent could resolve, so it comes back as None.

    Deliberately a pure derivation, recomputed by every consumer (both build steps, the
    evaluate path, the response serializers) rather than written back into
    current_settings: the request snapshot keeps saying what the user REQUESTED, the
    pipeline cache keys stay coherent with the request, and a category dropped for one
    data source comes back by itself when a later patch restores its columns.
    """
    _, sport_params, _ = _resolve_sport_params(session)
    available_columns = set(session.v2.columns)
    ratio_statistics  = sport_params['ratio-statistics']
    categories = [
        category for category in session.current_settings['categories']
        if category in available_columns
        and ratio_statistics.get(category, {}).get('volume-statistic', category) in available_columns
    ]
    tiebreaker = session.current_settings['tiebreaker_category']
    if tiebreaker not in categories or len(categories) % 2 == 1:
        tiebreaker = None
    return categories, tiebreaker


def load_player_pool(
    session: Session,
    csv_bytes: bytes | None = None,
    uploaded_dfs: dict | None = None,
) -> None:
    """Load player_stats_v0 into session.v0_clean and build session.player_registry.

    Branches on current_settings['data_source_type']:
      'csv'        — single uploaded CSV (csv_bytes required; format auto-detected)
      'historical' — Snowflake historical stats for current_settings['season']
      'projections' — weighted blend of Snowflake sources + any uploaded_dfs

    Every branch produces a frame INDEXED BY PLAYER ID with the display name in a
    'Player' column; the name is popped into the registry so v0_clean carries exactly
    the stats + Position columns the pipeline has always seen. Results for
    Snowflake-backed sources are cached (with names) at the module level for 24 hours
    so repeated session creations with the same data source skip the round-trip and
    rebuild an identical registry.
    """

    #also why re-declare current_settings? we can just call session.current_settings every time, its not too wordy IMO
    _, sport_params, _ = _resolve_sport_params(session)
    current_settings = session.current_settings
    source_type = current_settings['data_source_type']
    cache_key = _build_v0_cache_key(current_settings)

    v0_with_names = None
    if cache_key is not None:
        with _v0_cache_lock:
            entry = _v0_cache.get(cache_key)
            if entry is not None and time.time() - entry[0] < _V0_CACHE_TTL:
                v0_with_names = entry[1].copy()

    if v0_with_names is None:
        if source_type == 'csv':
            v0_with_names = _resolve_single_csv_player_ids(parse_projection_upload(csv_bytes, sport_params))

        elif source_type == 'historical':

            season = current_settings['season']
            if not season:
                raise ValueError(
                    "data_source.season is required when data_source.type == 'historical'"
                )
            v0_with_names = get_specified_historical_stats(season, sport_params)

        elif source_type == 'projections':

            blend_weights = current_settings['blend_weights']
            # All-zero weights would blend nothing and crash deep in the pipeline with an
            # opaque 500 — reject it up front with an actionable message instead.
            if not any(weight > 0 for weight in blend_weights.values()):
                raise InsufficientPlayerPoolError(
                    'All projection blend weights are zero. Set at least one source weight above zero.'
                )
            v0_with_names = combine_projections(
                blend_weights = blend_weights,
                sport_params  = sport_params,
                uploaded_dfs  = uploaded_dfs,
            )

        else:
            raise ValueError(f"Unknown data_source_type: {source_type!r}")

        if not v0_with_names.index.is_unique:
            duplicated = v0_with_names.index[v0_with_names.index.duplicated()].tolist()
            raise ValueError(f'Player pool has duplicate player id(s): {duplicated}')

        if cache_key is not None:
            with _v0_cache_lock:
                _v0_cache[cache_key] = (time.time(), v0_with_names.copy())

    session.player_registry = _build_player_registry(v0_with_names)
    session.v0_clean = v0_with_names.drop(columns=['Player'])


# ── Step 2: remove injured players ────────────────────────────────────────────

def remove_injured_players(session: Session) -> None:
    """Resolve the free-typed injured list to player ids and drop them into v1_clean."""

    injured_names = session.current_settings['injured_players']
    injured_player_ids = resolve_typed_player_names(session.player_registry, injured_names)
    v1 = drop_injured_players(session.v0_clean, tuple(injured_player_ids))
    session.v1_clean = v1.copy()


# ── Step 3: upsilon adjustment ────────────────────────────────────────────────

def apply_upsilon_adjustment(session: Session) -> None:
    """Run make_upsilon_adjustment using a fresh copy of v1_clean."""

    _, sport_params, _ = _resolve_sport_params(session)
    upsilon = session.current_settings['upsilon']
    # Always start from the clean v1 so repeated PATCH calls don't stack adjustments
    v2 = make_upsilon_adjustment(session.v1_clean.copy(), upsilon, sport_params)
    session.v2 = v2


# ── Step 4: build the scoring info ────────────────────────────────────────────

def build_scoring_info(session: Session) -> None:
    """Build the info dict (G-scores, X-scores, covariance, etc.) onto session.info."""

    _, sport_params, sport = _resolve_sport_params(session)
    current_settings = session.current_settings

    scoring_format = current_settings['scoring_format']
    n_drafters  = current_settings['n_drafters']
    n_picks     = current_settings['n_picks']
    slot_counts = current_settings['slot_counts']
    n_starters  = _count_starters(slot_counts, n_picks)

    # The pool must be able to fill every roster; otherwise the whole model is ill-posed (there is
    # no replacement-level player to anchor auction values, and managers could not complete teams).
    # Reject it here rather than letting process_player_data or the auction math fail obscurely later.
    n_available = len(session.v2)
    n_required  = n_drafters * n_picks
    if n_available < n_required:
        raise InsufficientPlayerPoolError(
            f'Only {n_available} players are available, but this league needs at least {n_required} '
            f'({n_drafters} teams x {n_picks} roster spots) to fill every roster.'
        )

    # The requested objective, narrowed to what this data source can score. Derived, never
    # written back: current_settings stays the request (see derive_effective_objective).
    effective_categories, effective_tiebreaker = derive_effective_objective(session)

    info = process_player_data(
        player_stats_v2   = session.v2,
        weekly_df         = None,
        mean_of_variances = _load_mean_of_variances(),
        psi               = current_settings['psi'],
        chi               = current_settings['chi'],
        scoring_format    = scoring_format,
        n_drafters        = n_drafters,
        n_starters        = n_starters,
        sport_params      = sport_params,
        categories        = effective_categories,
        sport             = sport,
        tiebreaker_category    = effective_tiebreaker,   # always a live category, by derivation
        most_categories_weight = current_settings.get('most_categories_weight'),
    )
    # session.info is the pipeline's step-4 intermediate; step 5 builds the agent from it (and the
    # agent retains it, so consumers read G-scores via session.agent.info). On a from_step==5 patch
    # this step is skipped and the existing session.info is reused.
    session.info = info


# ── Step 5: build the session's agent ─────────────────────────────────────────

def build_session_agent(session: Session) -> None:
    """Build the HAgent from the scored data and prime its neutral baseline — the whole agent."""

    _, sport_params, sport = _resolve_sport_params(session)
    current_settings = session.current_settings

    scoring_format = current_settings['scoring_format']
    n_picks     = current_settings['n_picks']
    slot_counts = current_settings['slot_counts']
    n_starters  = _count_starters(slot_counts, n_picks)
    n_drafters  = current_settings['n_drafters']
    _, effective_tiebreaker = derive_effective_objective(session)

    session.agent = HAgent(
        info           = session.info,   # step-4 output (unchanged on a from_step==5 patch)
        pick_pool_size = current_settings['pick_pool_size'],
        n_picks        = n_starters,
        n_drafters     = n_drafters,
        dynamic        = current_settings['n_iterations'] > 0,
        scoring_format = scoring_format,
        most_categories_weight = current_settings['most_categories_weight'],
        tiebreaker_category    = effective_tiebreaker,
        sport          = sport,
        sport_params   = sport_params,
        slot_counts    = slot_counts,
        aleph          = current_settings['aleph'],
        reg_lambda     = current_settings['reg_lambda'],
        opponent_model_confidence = current_settings['opponent_model_confidence'],
        beth           = current_settings['beth'],
    )

    # Prime the neutral (empty-board) baseline as part of the build — this workflow always evaluates,
    # so the throttle ranking + auction anchor are always needed. Auction sessions pass full cash
    # (every team at cash_per_team); everything else passes None — the is_auction gate matters
    # because a cash_per_team value can linger on the session after the user leaves Auction Mode.
    cash_per_team = current_settings['cash_per_team'] if current_settings['is_auction'] else None
    default_cash = (
        {f'Team {i + 1}': cash_per_team for i in range(n_drafters)}
        if cash_per_team is not None else None
    )
    session.agent.populate_default_h_scores(current_settings['n_iterations'], default_cash)


# ── Full pipeline ─────────────────────────────────────────────────────────────

def build_agent(
    session: Session,
    from_step: int = 1,
    csv_bytes: bytes | None = None,
    uploaded_dfs: dict | None = None,
) -> None:
    """Re-run the pipeline starting from the given step number (1–5), leaving session.agent built."""
    if from_step <= 1:
        load_player_pool(session, csv_bytes=csv_bytes, uploaded_dfs=uploaded_dfs)
    if from_step <= 2:
        remove_injured_players(session)
    if from_step <= 3:
        apply_upsilon_adjustment(session)
    if from_step <= 4:
        build_scoring_info(session)
    if from_step <= 5:
        build_session_agent(session)

