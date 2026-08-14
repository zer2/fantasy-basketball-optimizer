"""
Pipeline orchestration for the 5-step initialization chain.

Steps:
  1. Load player_stats_v0 from CSV / Snowflake → session.v0_clean
  2. drop_injured_players → session.v1_clean
  3. make_upsilon_adjustment → session.v2
  4. process_player_data → session.info
  5. build HAgent (retains info; populates the default baseline) → session.agent

No session_context or _SessionState: each step reads/writes session fields directly.
"""

from __future__ import annotations

import io
import time
import threading
import pandas as pd
from pathlib import Path

from backend.parameters import load_all_params
from backend.state.session import Session


class InsufficientPlayerPoolError(Exception):
    """The available player pool is too small to fill every roster in the league. This is a bad
    league configuration (too few players for the number of teams x roster spots), not a server
    fault, so the routes surface it as a 400."""


# build_agent.py is backend/services/build_agent.py, so the project root is parents[2].
_MEAN_OF_VARIANCES_PATH = Path(__file__).parents[2] / 'coefficient_exploration_output' / 'mean_of_variances.csv'


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


def clear_v0_cache() -> None:
    """Evict all entries from the v0_clean in-memory cache."""
    with _v0_cache_lock:
        _v0_cache.clear()


# ── helpers ───────────────────────────────────────────────────────────────────

def _sport_params(session: Session) -> tuple[dict, dict, str]:
    """Return (all_params, sport_params, sport) for the current session."""
    all_params = load_all_params()
    sport = session.current_params['sport']
    return all_params, all_params[sport], sport


# ── Step 1: load player data ──────────────────────────────────────────────────

def _v0_cache_key(cp: dict) -> tuple | None:
    """Return a hashable cache key for v0_clean based on data source params.

    A projections blend is fully described by every source weight plus the ids of any
    uploaded tables feeding it: uploads are stored immutably under their data_id, so the
    id doubles as a content key. (Leaving the HTB/BBM weights or the upload ids out of
    the key — as an earlier version did — served stale blends when an upload's weight
    changed, and could leak an uploaded blend into sessions that never uploaded anything.)
    Returns None only for single-CSV mode, whose bytes arrive outside current_params.
    """
    source_type = cp['data_source_type']
    sport = cp.get('sport', '')
    if source_type == 'historical':
        return (sport, 'historical', cp['season'])
    if source_type == 'projections':
        blend_weights = cp.get('blend_weights', {})
        custom_data_ids = cp.get('custom_data_ids') or []
        weight_keys = tuple(sorted(blend_weights.items()))
        upload_keys = tuple(sorted(custom_data_ids))
        return (sport, 'projections', weight_keys, upload_keys)
    return None


def _resolve_single_csv_player_ids(parsed_csv: pd.DataFrame) -> pd.DataFrame:
    """Bring a single uploaded CSV (name-indexed by parse_projection_csv) to the id-keyed
    contract: resolve names via the unified table, keep unresolved rows under synthetic
    ids, and return an id-indexed frame with the display 'Player' column retained."""
    from backend.data_retrieval import attach_player_ids_by_name
    from backend.player_identity import allocate_synthetic_player_ids

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
    from backend.player_identity import (
        RP_PLAYER_ID, make_player_identity, make_replacement_player_identity,
    )

    registry = {
        int(player_id): make_player_identity(int(player_id), str(name), position_value)
        for player_id, name, position_value in zip(
            v0_with_names.index, v0_with_names['Player'], v0_with_names['Position'])
    }
    registry[RP_PLAYER_ID] = make_replacement_player_identity()
    return registry


def run_step1(
    session: Session,
    csv_bytes: bytes | None = None,
    uploaded_dfs: dict | None = None,
) -> None:
    """Load player_stats_v0 into session.v0_clean and build session.player_registry.

    Branches on current_params['data_source_type']:
      'csv'        — single uploaded CSV (csv_bytes required; format auto-detected)
      'historical' — Snowflake historical stats for current_params['season']
      'projections' — weighted blend of Snowflake sources + any uploaded_dfs

    Every branch produces a frame INDEXED BY PLAYER ID with the display name in a
    'Player' column; the name is popped into the registry so v0_clean carries exactly
    the stats + Position columns the pipeline has always seen. Results for
    Snowflake-backed sources are cached (with names) at the module level for 24 hours
    so repeated session creations with the same data source skip the round-trip and
    rebuild an identical registry.
    """
    _, params, _ = _sport_params(session)
    cp = session.current_params
    source_type = cp['data_source_type']
    cache_key = _v0_cache_key(cp)

    v0_with_names = None
    if cache_key is not None:
        with _v0_cache_lock:
            entry = _v0_cache.get(cache_key)
            if entry is not None and time.time() - entry[0] < _V0_CACHE_TTL:
                v0_with_names = entry[1].copy()

    if v0_with_names is None:
        if source_type == 'csv':
            parsed_csv, _detected_format = parse_projection_csv(csv_bytes, params)
            v0_with_names = _resolve_single_csv_player_ids(parsed_csv)

        elif source_type == 'historical':
            from backend.data_retrieval import get_specified_historical_stats

            season = cp['season']
            if not season:
                raise ValueError(
                    "data_source.season is required when data_source.type == 'historical'"
                )
            v0_with_names = get_specified_historical_stats(season, params)

        elif source_type == 'projections':
            from backend.data_retrieval import combine_projections

            blend_weights = cp.get('blend_weights', {})
            # All-zero weights would blend nothing and crash deep in the pipeline with an
            # opaque 500 — reject it up front with an actionable message instead.
            if not any(weight > 0 for weight in blend_weights.values()):
                raise InsufficientPlayerPoolError(
                    'All projection blend weights are zero. Set at least one source weight above zero.'
                )
            v0_with_names = combine_projections(
                blend_weights = blend_weights,
                params        = params,
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


# The per-game stats a projection export CAN map to. League-paired exports (Basketball
# Monster pairs each projection set with a league and emits only that league's active
# categories) may legitimately lack several — so recognition requires Player, Position,
# and only _MIN_MATCHED_CORE_COLUMNS of these. That is still unmistakably a projection
# file, while a format-drifted export (renamed headers) maps ~none and is rejected with
# a clear message instead of failing much later, deep in the blend, as a baffling
# "0 players available" error.
_CORE_PROJECTION_COLUMNS = ('Points', 'Rebounds', 'Assists', 'Steals', 'Blocks', 'Turnovers', 'Threes')
_MIN_MATCHED_CORE_COLUMNS = 3

# Known export formats, in detection order. A file already using canonical column names
# passes under the first mapping (renaming is a no-op), so "any common projection file"
# includes plain canonical CSVs for free.
_PROJECTION_FORMATS = (
    ('HTB', 'htb-renamer'),
    ('BBM', 'bbm-renamer'),
)


def parse_projection_csv(csv_bytes: bytes, params: dict) -> tuple[pd.DataFrame, str]:
    """Parse an uploaded projection CSV into the canonical column set, auto-detecting the
    export format. Returns (parsed frame, detected format name). Raises ValueError with an
    aggregated, per-format diagnosis when no known format fits."""
    df_raw = pd.read_csv(io.BytesIO(csv_bytes))

    failures: list[str] = []
    for format_name, renamer_key in _PROJECTION_FORMATS:
        renamer = params.get(renamer_key, {})
        renamed_columns = set(df_raw.rename(columns=renamer).columns)
        missing_identity = [column for column in ('Player', 'Position')
                            if column not in renamed_columns]
        matched_cores = [column for column in _CORE_PROJECTION_COLUMNS
                         if column in renamed_columns]
        if not missing_identity and len(matched_cores) >= _MIN_MATCHED_CORE_COLUMNS:
            return _parse_with_renamer(df_raw, renamer), format_name
        missing_cores = [column for column in _CORE_PROJECTION_COLUMNS
                         if column not in renamed_columns]
        problem = (
            f"missing {', '.join(missing_identity)}" if missing_identity
            else f'recognized only {len(matched_cores)} of {len(_CORE_PROJECTION_COLUMNS)} '
                 f"core stats (missing {', '.join(missing_cores)})"
        )
        failures.append(f'{format_name}: {problem}')

    def count_matched_cores(renamer: dict) -> int:
        return len(set(df_raw.rename(columns=renamer).columns) & set(_CORE_PROJECTION_COLUMNS))

    best_renamer = max(
        (params.get(renamer_key, {}) for _, renamer_key in _PROJECTION_FORMATS),
        key=count_matched_cores,
    )
    unmapped = [column for column in df_raw.columns if column not in best_renamer]
    raise ValueError(
        f"File does not match any known projection format ({'; '.join(failures)}). "
        f"Headers in the file that were not recognized: {', '.join(map(str, unmapped))}."
    )


def _parse_with_renamer(df_raw: pd.DataFrame, renamer: dict) -> pd.DataFrame:
    """The format-specific parse: rename to canonical columns, drop junk, coerce stats."""
    df = df_raw.rename(columns=renamer)

    # Keep only canonical columns. Unmapped extras (ranks, dollar values, minutes, ...)
    # would otherwise join the blend's column union, where every player from the OTHER
    # sources is "missing" them — and the blend drops any player missing any column
    # across all sources, so a single junk column can wipe out the entire pool.
    canonical_columns = set(renamer.values()) | {'Games Played %'}
    df = df[[column for column in df.columns if column in canonical_columns]]

    # Exports carry non-numeric stat values: hashtagbasketball.com repeats its header row
    # inside the table body (every stat cell a string), and formats ratio stats as
    # "0.474 (5.2/11.0)". Extract the leading number where a stat column holds strings,
    # then drop rows with no numeric stats at all — those are the embedded header/junk rows.
    def coerce_stat_column(column: pd.Series) -> pd.Series:
        if column.dtype == object:
            # Leading number only, and not one embedded in a word — the repeated header
            # rows contain cells like "3PM", which must NOT read as the number 3.
            column = column.astype(str).str.extract(
                r'^\s*(-?(?:\d+\.?\d*|\.\d+))(?![A-Za-z])', expand=False)
        return pd.to_numeric(column, errors='coerce')

    stat_columns = [column for column in df.columns if column not in ('Player', 'Position')]
    df[stat_columns] = df[stat_columns].apply(coerce_stat_column)
    df = df.dropna(subset=stat_columns, how='all')

    if 'Games Played %' not in df.columns:
        if 'Games Played' in df.columns:
            df['Games Played %'] = df['Games Played'] / 82.0
        else:
            raise ValueError(
                "CSV missing both 'Games Played %' and 'Games Played' columns after rename"
            )

    # Clamp GP to 0–1
    df['Games Played %'] = df['Games Played %'].clip(0, 1)

    # Raw games played is only an intermediate for the % above. Left in, it becomes an
    # upload-only column in the blend's union, and the blend's coverage rule (a player
    # must have every column covered by some source that carries them) would then drop
    # every player the upload doesn't cover — a partial upload would gut the pool.
    df = df.drop(columns=['Games Played'], errors='ignore')

    if 'Player' in df.columns:
        df = df.set_index('Player')

    return df


# ── Step 2: drop injured players ──────────────────────────────────────────────

def run_step2(session: Session) -> None:
    """Resolve the free-typed injured list to player ids and drop them into v1_clean."""
    from backend.math.process_player_data import drop_injured_players
    from backend.player_identity import resolve_typed_player_names

    injured_names = session.current_params.get('injured_players', [])
    injured_player_ids = resolve_typed_player_names(session.player_registry, injured_names)
    v1, _ = drop_injured_players(session.v0_clean, tuple(injured_player_ids))
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
    """Build the info dict (G-scores, X-scores, covariance, etc.) onto session.info."""
    from backend.math.process_player_data import process_player_data

    _, params, sport = _sport_params(session)
    cp = session.current_params

    scoring_format = cp['scoring_format']
    n_drafters  = cp['n_drafters']
    n_picks     = cp['n_picks']
    slot_counts = cp['slot_counts']
    n_starters  = sum(slot_counts.values()) if slot_counts else n_picks

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

    available_columns = set(session.v2.columns)
    categories = [c for c in cp['categories'] if c in available_columns]
    cp['categories'] = categories

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
    # session.info is the pipeline's step-4 intermediate; step 5 builds the agent from it (and the
    # agent retains it, so consumers read G-scores via session.agent.info). On a from_step==5 patch
    # this step is skipped and the existing session.info is reused.
    session.info = info


# ── Step 5: build HAgent ──────────────────────────────────────────────────────

def run_step5(session: Session) -> None:
    """Build the HAgent from the scored data and prime its neutral baseline — the whole agent."""
    from backend.math.algorithm_agents import HAgent

    _, params, sport = _sport_params(session)
    cp = session.current_params

    scoring_format = cp['scoring_format']
    n_picks     = cp['n_picks']
    slot_counts = cp['slot_counts']
    n_starters  = sum(slot_counts.values()) if slot_counts else n_picks
    n_drafters  = cp['n_drafters']

    session.agent = HAgent(
        info           = session.info,   # step-4 output (unchanged on a from_step==5 patch)
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
        kappa          = cp['kappa'],
        # .get: mirrors the schema default for sessions persisted before the parameter existed.
        opponent_sophistication = cp.get('opponent_sophistication', True),
        behavior_model_confidence   = cp.get('behavior_model_confidence', 1.0),
        beth           = cp['beth'],
    )

    # Prime the neutral (empty-board) baseline as part of the build — this workflow always evaluates,
    # so the throttle ranking + auction anchor are always needed. Auction sessions pass full cash
    # (every team at cash_per_team); everything else passes None — the is_auction gate matters
    # because a cash_per_team value can linger on the session after the user leaves Auction Mode.
    cash_per_team = cp.get('cash_per_team') if cp.get('is_auction') else None
    default_cash = (
        {f'Team {i + 1}': cash_per_team for i in range(n_drafters)}
        if cash_per_team is not None else None
    )
    session.agent.populate_default_h_scores(cp['n_iterations'], default_cash)


# ── Full pipeline ─────────────────────────────────────────────────────────────

def build_agent(
    session: Session,
    from_step: int = 1,
    csv_bytes: bytes | None = None,
    uploaded_dfs: dict | None = None,
) -> None:
    """Re-run the pipeline starting from the given step number (1–5), leaving session.agent built."""
    if from_step <= 1:
        run_step1(session, csv_bytes=csv_bytes, uploaded_dfs=uploaded_dfs)
    if from_step <= 2:
        run_step2(session)
    if from_step <= 3:
        run_step3(session)
    if from_step <= 4:
        run_step4(session)
    if from_step <= 5:
        run_step5(session)

