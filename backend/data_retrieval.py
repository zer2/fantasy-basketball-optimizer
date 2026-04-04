"""
Backend data retrieval from Snowflake.

Equivalent to src/data_retrieval/get_data.py but:
- No Streamlit dependencies
- Explicit `params: dict` instead of get_params()
- Player names always mapped to the canonical 'Player' column
- TTL cache via module-level dict (replacing @st.cache_data(ttl='1d'))
- Snowflake connection managed with a simple TTL (replacing @st.cache_resource)
"""

from __future__ import annotations

import os
import time
import threading
import tomllib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import snowflake.connector


# ── Snowflake connection ───────────────────────────────────────────────────────

_con: Optional[snowflake.connector.SnowflakeConnection] = None
_con_expires_at: float = 0.0
_con_lock = threading.Lock()
_CON_TTL = 3600  # 1 hour


_SECRETS_TOML = Path(__file__).parent.parent / '.streamlit' / 'secrets.toml'
_SNOWFLAKE_KEYS = ('SNOWFLAKE_ACCOUNT', 'SNOWFLAKE_USER', 'SNOWFLAKE_PASSWORD',
                   'SNOWFLAKE_DATABASE', 'SNOWFLAKE_SCHEMA')


def _snowflake_creds() -> dict[str, str | None]:
    """Read Snowflake credentials from env vars, falling back to .streamlit/secrets.toml."""
    creds: dict[str, str | None] = {k: os.getenv(k) for k in _SNOWFLAKE_KEYS}
    if not all(creds.values()) and _SECRETS_TOML.exists():
        with open(_SECRETS_TOML, 'rb') as f:
            secrets = tomllib.load(f)
        for k in _SNOWFLAKE_KEYS:
            if not creds[k]:
                creds[k] = secrets.get(k)
    return creds


def _get_connection() -> snowflake.connector.SnowflakeConnection:
    global _con, _con_expires_at
    with _con_lock:
        if _con is None or time.time() > _con_expires_at:
            creds = _snowflake_creds()
            _con = snowflake.connector.connect(
                account  = creds['SNOWFLAKE_ACCOUNT'],
                user     = creds['SNOWFLAKE_USER'],
                password = creds['SNOWFLAKE_PASSWORD'],
                database = creds['SNOWFLAKE_DATABASE'],
                schema   = creds['SNOWFLAKE_SCHEMA'],
            )
            _con_expires_at = time.time() + _CON_TTL
        return _con


# ── Query cache ───────────────────────────────────────────────────────────────

_cache: dict[str, tuple[float, pd.DataFrame]] = {}
_cache_lock = threading.Lock()
_CACHE_TTL = 24 * 3600  # 24 hours

# If set, Parquet files are read/written here before falling back to Snowflake.
# In production this is the GCS bucket mount path (e.g. /cache).
_DISK_CACHE_DIR: Path | None = (
    Path(os.environ['DISK_CACHE_DIR']) if 'DISK_CACHE_DIR' in os.environ else None
)


def _disk_cache_path(view_name: str) -> Path | None:
    if _DISK_CACHE_DIR is None:
        return None
    return _DISK_CACHE_DIR / f'{view_name}.parquet'


def _query(view_name: str) -> pd.DataFrame:
    """Fetch a view from Snowflake with a 24-hour in-memory and disk cache."""
    with _cache_lock:
        entry = _cache.get(view_name)
        if entry is not None and time.time() - entry[0] < _CACHE_TTL:
            return entry[1].copy()

    disk_path = _disk_cache_path(view_name)
    if disk_path is not None and disk_path.exists():
        age = time.time() - disk_path.stat().st_mtime
        if age < _CACHE_TTL:
            df = pd.read_parquet(disk_path)
            with _cache_lock:
                _cache[view_name] = (time.time() - age, df)
            return df.copy()

    df = _get_connection().cursor().execute(f'SELECT * FROM {view_name}').fetch_pandas_all()

    if disk_path is not None:
        disk_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(disk_path, index=False)

    with _cache_lock:
        _cache[view_name] = (time.time(), df)

    return df.copy()


# ── Player name mapping ───────────────────────────────────────────────────────

def _map_player_names(df: pd.DataFrame, source_col: str) -> pd.DataFrame:
    """Map source-specific player names to the canonical 'Player' column.

    source_col is the column name in PLAYER_MAPPING_VIEW whose values match
    what's currently in df['Player']. The canonical name column in the view
    is 'PLAYER_NAME' (not 'Player').
    """
    mapping_view = _query('PLAYER_MAPPING_VIEW')
    mapper = (
        mapping_view.dropna(subset=[source_col])
        .set_index(source_col)['PLAYER_NAME']
    )
    df = df.copy()
    df['Player'] = df['Player'].map(mapper).fillna(df['Player'])
    return df


# ── Weekly box scores ─────────────────────────────────────────────────────────

def get_weekly_box_scores(season: str, params: dict) -> pd.DataFrame:
    """Fetch weekly box score totals for a season from WEEKLY_NUMBERS_VIEW.

    Column names are mapped using params['stat-df-renamer'].

    Returns a DataFrame indexed by ('Player', 'Week') with one summed-stat
    row per player per week — the format expected by
    calculate_coefficients_historical.
    """
    df = _get_connection().cursor().execute(
        f"SELECT * FROM WEEKLY_NUMBERS_VIEW WHERE SEASON = '{season}'"
    ).fetch_pandas_all()
    df = df.rename(columns=params['stat-df-renamer'])
    df = df.apply(pd.to_numeric, errors='ignore')
    df = df.set_index(['Player', 'WEEK']).sort_index()
    return df.select_dtypes(include='number')


# ── Available seasons ─────────────────────────────────────────────────────────

_seasons_cache: tuple[float, list[str]] | None = None
_seasons_lock = threading.Lock()


def get_available_seasons() -> list[str]:
    """Return distinct historical seasons from Snowflake, newest first.

    Result is cached for 24 hours (same TTL as the full data cache).
    """
    global _seasons_cache
    with _seasons_lock:
        if _seasons_cache is not None and time.time() - _seasons_cache[0] < _CACHE_TTL:
            return _seasons_cache[1]
        df = _get_connection().cursor().execute(
            'SELECT DISTINCT SEASON FROM AVERAGE_NUMBERS_VIEW_2 ORDER BY SEASON DESC'
        ).fetch_pandas_all()
        seasons = [str(s) for s in df['SEASON'].tolist()]
        _seasons_cache = (time.time(), seasons)
        return seasons


# ── Historical data ───────────────────────────────────────────────────────────

def get_historical_data(params: dict) -> pd.DataFrame:
    """Fetch full historical player data from Snowflake.

    Returns a DataFrame indexed by (Season, Player).
    """
    df = _query('AVERAGE_NUMBERS_VIEW_2')
    df = df.rename(columns=params['stat-df-renamer'])
    df = df.apply(pd.to_numeric, errors='ignore')

    df['Free Throw %']  = df['Free Throws Made'] / df['Free Throw Attempts']
    df['Field Goal %']  = df['Field Goals Made'] / df['Field Goal Attempts']
    df['Three %']       = df['Threes'] / df['Three Attempts']
    df['Assist to TO']  = df['Assists'] / df['Turnovers']
    df['Position']      = df['Position'].fillna('NP')

    # Known duplicate in the reference table
    df = df[df['Player'] != 'OG Anunoby']

    return df.set_index(['Season', 'Player']).sort_index().fillna(0)


def get_specified_historical_stats(season: str, params: dict) -> pd.DataFrame:
    """Return player stats for a specific season.

    Returns a DataFrame indexed by 'Player (Position)' — the canonical format
    for player_stats_v0 (same as what process_player_data expects).
    """
    df = get_historical_data(params).loc[season].copy()
    df.index = df.index + ' (' + df['Position'] + ')'
    df.index.name = 'Player'
    return df


# ── Projection data ───────────────────────────────────────────────────────────

def get_espn_projections(params: dict) -> pd.DataFrame:
    """Fetch ESPN projections from Snowflake."""
    n_games = params['n_games']
    df = _query('ESPN_PROJECTION_VIEW')
    df = df.rename(columns=params['espn-renamer'])
    df = _map_player_names(df, 'ESPN_NAME')
    df['Games Played %'] = df['Games Played'] / n_games
    return df.set_index('Player')


def get_darko_data(params: dict) -> pd.DataFrame:
    """Fetch DARKO projections from Snowflake, scaled from per-100 to per-game."""
    n_games = params['n_games']

    df = _query('DARKO_VIEW')
    df = df.rename(columns=params['darko-renamer'])
    df = df.apply(pd.to_numeric, errors='ignore')
    df = _map_player_names(df, 'DARKO_NAME')
    df = df.set_index('Player').sort_index().fillna(0)

    # Fetch position / minutes / games from ESPN table
    extra = _query('ESPN_PROJECTION_TABLE')[['ESPN_NAME', 'MINUTES_PLAYED', 'GAMES_PLAYED', 'POSITION']]
    extra.columns = ['Player', 'Minutes', 'Games Played %', 'Position']
    extra['Games Played %'] = extra['Games Played %'].astype(float) / n_games
    extra = _map_player_names(extra, 'ESPN_NAME').set_index('Player')

    df = df.merge(extra, left_index=True, right_index=True)
    possessions_per_game = df['Pace'] / 100 * df['Minutes'] / 48

    per_100_cols = {
        'Points':               'Points/100',
        'Rebounds':             'Rebounds/100',
        'Assists':              'Assists/100',
        'Steals':               'Steals/100',
        'Blocks':               'Blocks/100',
        'Threes':               'Threes/100',
        'Turnovers':            'Turnovers/100',
        'Free Throw Attempts':  'Free Throw Attempts/100',
        'Field Goal Attempts':  'Field Goal Attempts/100',
        'Three Attempts':       'Three Attempts/100',
    }
    for col, per100 in per_100_cols.items():
        df[col] = df[per100] * possessions_per_game

    df['Field Goals Made'] = (
        df['Field Goal Attempts/100'] * df['Field Goal %'] * possessions_per_game
    )
    df['Assist to TO'] = df['Assists'] / df['Turnovers']

    required = (
        params['counting-statistics']
        + list(params['ratio-statistics'].keys())
        + [info['volume-statistic'] for info in params['ratio-statistics'].values()]
        + params['other-columns']
    )
    required = [c for c in required if c in df.columns]
    return df[list(set(required))]


# ── Blended projections ───────────────────────────────────────────────────────

def combine_projections(
    blend_weights: dict[str, float],
    params: dict,
    uploaded_dfs: dict[str, Optional[pd.DataFrame]] | None = None,
) -> pd.DataFrame:
    """Blend multiple projection sources using provided weights.

    blend_weights keys: 'HTB', 'BBM', 'DARKO', 'ESPN'
    uploaded_dfs: pre-parsed DataFrames for user-uploaded sources (HTB, BBM).
    Snowflake sources (DARKO, ESPN) are fetched automatically if weight > 0.
    """
    uploaded = uploaded_dfs or {}
    source_keys = ['HTB', 'BBM', 'DARKO', 'ESPN']

    sources: dict[str, Optional[pd.DataFrame]] = {
        'HTB':  uploaded.get('HTB'),
        'BBM':  uploaded.get('BBM'),
        'DARKO': get_darko_data(params)   if blend_weights.get('DARKO', 0) > 0 else None,
        'ESPN':  get_espn_projections(params) if blend_weights.get('ESPN', 0)  > 0 else None,
    }

    weights = [blend_weights.get(k, 0.0) for k in source_keys]

    all_players = {
        p
        for k in source_keys
        if sources[k] is not None
        for p in sources[k].index
    }

    df = pd.concat({k: sources[k] for k in source_keys}, names=['Source'])
    new_index = pd.MultiIndex.from_product(
        [source_keys, sorted(all_players)], names=['Source', 'Player']
    )
    df = df.reindex(new_index)

    # Drop players missing from all sources
    ineligible = (df.isna().groupby('Player').sum() == 4).sum(axis=1) > 0
    df = df[~df.index.get_level_values('Player').isin(ineligible.index[ineligible])]

    df = df.groupby('Player').agg(
        lambda x: (
            np.ma.average(np.ma.MaskedArray(x, mask=np.isnan(x)), weights=weights)
            if np.issubdtype(x.dtype, np.number)
            else x.dropna().iloc[0]
        )
    )

    if 'Double Doubles' in df.columns:
        df['Double Doubles'] = df['Double Doubles'].astype(float)

    df['Position'] = df['Position'].fillna('NP')
    df = df.fillna(0)

    df.index = df.index + ' (' + df['Position'] + ')'
    df.index.name = 'Player'
    return df
