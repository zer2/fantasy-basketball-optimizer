"""Snowflake access layer: connection management + query caching.

Generic, domain-free plumbing — it knows how to connect to Snowflake and cache
results, but nothing about players, stats, or projections (that's data_retrieval).

Replaces the Streamlit decorators: @st.cache_resource for the connection (kept
alive on a simple TTL) and @st.cache_data(ttl='1d') for query results (an
in-memory dict plus an optional Parquet disk cache).

Public API:
  - query(view_name)            cached SELECT * FROM <view>
  - run_query(sql)              uncached arbitrary SQL
  - view_cache_timestamp(name)  load-time of a cached view, for lockstep derivations
"""

from __future__ import annotations

import os
import time
import threading
from pathlib import Path
from typing import Optional

import pandas as pd
import snowflake.connector

from backend.infra.secret_config import get_secret


# ── Connection ─────────────────────────────────────────────────────────────────

_con: Optional[snowflake.connector.SnowflakeConnection] = None
_con_expires_at: float = 0.0
_con_lock = threading.Lock()
_CON_TTL = 3600  # 1 hour

_SNOWFLAKE_KEYS = ('SNOWFLAKE_ACCOUNT', 'SNOWFLAKE_USER', 'SNOWFLAKE_PASSWORD',
                   'SNOWFLAKE_DATABASE', 'SNOWFLAKE_SCHEMA')


def _snowflake_creds() -> dict[str, str | None]:
    """Read Snowflake credentials via the shared resolver (env first, then secrets.toml)."""
    return {key: get_secret(key) for key in _SNOWFLAKE_KEYS}


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


def run_query(sql: str) -> pd.DataFrame:
    """Execute arbitrary SQL on the shared connection and return the result (uncached)."""
    return _get_connection().cursor().execute(sql).fetch_pandas_all()


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


def query(view_name: str) -> pd.DataFrame:
    """Fetch a full view from Snowflake with a 24-hour in-memory and disk cache."""
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

    df = run_query(f'SELECT * FROM {view_name}')

    if disk_path is not None:
        disk_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(disk_path, index=False)

    with _cache_lock:
        _cache[view_name] = (time.time(), df)

    return df.copy()


def view_cache_timestamp(view_name: str) -> float | None:
    """Load-time of the cached entry for view_name if present and unexpired, else None.

    Lets a caller derive something from a cached view and stay in lockstep with the
    cache's refreshes without re-fetching or re-iterating (see get_available_seasons).
    """
    with _cache_lock:
        entry = _cache.get(view_name)
        if entry is not None and time.time() - entry[0] < _CACHE_TTL:
            return entry[0]
    return None
