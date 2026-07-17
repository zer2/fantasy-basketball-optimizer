"""
Backend data retrieval from Snowflake.

Domain data-access layer: player mapping, historical stats, projections, and
seasons — all the fantasy-basketball-specific reads. The generic Snowflake
connection + query caching lives in backend.snowflake_connection.

Equivalent to src/data_retrieval/get_data.py but:
- No Streamlit dependencies
- Explicit `params: dict` instead of get_params()
- Player names always mapped to the canonical 'Player' column
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from backend.infra.snowflake_connection import query, run_query, view_cache_timestamp


# ── Player name mapping ───────────────────────────────────────────────────────

def _map_player_names(df: pd.DataFrame, source_col: str) -> pd.DataFrame:
    """Map source-specific player names to the canonical 'Player' column.

    source_col is the column name in UNIFIED_PLAYER_TABLE whose values match
    what's currently in df['Player']. The canonical name column in that table
    is 'MASTER_PLAYER_NAME' (not 'Player').
    """
    unified_player_table = query('UNIFIED_PLAYER_TABLE')
    mapper = (
        unified_player_table.dropna(subset=[source_col])
        .set_index(source_col)['MASTER_PLAYER_NAME']
    )
    df = df.copy()
    df['Player'] = df['Player'].map(mapper).fillna(df['Player'])
    return df


def get_unified_player_table() -> pd.DataFrame:
    """The single cross-platform player table (UNIFIED_PLAYER_TABLE), whose columns
    include the canonical MASTER_PLAYER_NAME plus each bridge id/name (NBA_PLAYER_ID,
    YAHOO_PLAYER_ID, FANTRAX_ID, ESPN_NAME, DARKO_NAME, ...). Used by platform
    integrations to map roster ids/names to canonical."""
    return query('UNIFIED_PLAYER_TABLE')


# ── Weekly box scores ─────────────────────────────────────────────────────────

def get_weekly_box_scores(season: str, params: dict) -> pd.DataFrame:
    """Fetch weekly box score totals for a season from WEEKLY_NUMBERS_VIEW.

    Column names are mapped using params['stat-df-renamer'].

    Returns a DataFrame indexed by ('Player', 'Week') with one summed-stat
    row per player per week — the format expected by
    calculate_coefficients_historical.
    """
    df = run_query(f"SELECT * FROM WEEKLY_NUMBERS_VIEW WHERE SEASON = '{season}'")
    df = df.rename(columns=params['stat-df-renamer'])
    df = df.apply(pd.to_numeric, errors='ignore')
    df = df.set_index(['Player', 'WEEK']).sort_index()
    return df.select_dtypes(include='number')


# ── Available seasons ─────────────────────────────────────────────────────────

# (view_load_time, seasons). The timestamp is the load time of the cached
# HISTORICAL_SEASONAL_AVERAGES_VIEW entry — when that entry is refreshed the
# timestamp changes and we re-derive. This keeps the two in lockstep without
# re-iterating the DataFrame on every call.
_seasons_cache: tuple[float, list[str]] | None = None

_HISTORICAL_VIEW = 'HISTORICAL_SEASONAL_AVERAGES_VIEW'


def get_available_seasons() -> list[str]:
    """Return distinct historical seasons from Snowflake, newest first.

    Derived from the same cached HISTORICAL_SEASONAL_AVERAGES_VIEW DataFrame
    that get_historical_data uses, so the season list and the season data
    cannot drift apart. Re-derived only when the view cache is refreshed.
    """
    global _seasons_cache
    load_time = view_cache_timestamp(_HISTORICAL_VIEW)
    if load_time is not None and _seasons_cache is not None and _seasons_cache[0] == load_time:
        return _seasons_cache[1]

    df = query(_HISTORICAL_VIEW)
    load_time = view_cache_timestamp(_HISTORICAL_VIEW)
    _seasons_cache = (load_time, sorted({str(s) for s in df['SEASON'].tolist()}, reverse=True))
    return _seasons_cache[1]


# ── Historical data ───────────────────────────────────────────────────────────

def get_historical_data(params: dict) -> pd.DataFrame:
    """Fetch full historical player data from Snowflake.

    Returns a DataFrame indexed by (Season, Player).
    """
    df = query('HISTORICAL_SEASONAL_AVERAGES_VIEW')
    df = df.rename(columns=params['stat-df-renamer'])
    df = df.apply(pd.to_numeric, errors='ignore')

    df['Free Throw %']  = df['Free Throws Made'] / df['Free Throw Attempts']
    df['Field Goal %']  = df['Field Goals Made'] / df['Field Goal Attempts']
    df['Three %']       = df['Threes'] / df['Three Attempts']
    df['Assist to TO']  = df['Assists'] / df['Turnovers']
    df['Position']      = df['Position'].fillna('NP')

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
    df = query('ESPN_PROJECTION_VIEW')
    df = df.rename(columns=params['espn-renamer'])
    df = _map_player_names(df, 'ESPN_NAME')
    df['Games Played %'] = df['Games Played'] / n_games
    return df.set_index('Player')


def get_darko_data(params: dict) -> pd.DataFrame:
    """Fetch DARKO projections from Snowflake, scaled from per-100 to per-game."""
    n_games = params['n_games']

    df = query('DARKO_VIEW')
    df = df.rename(columns=params['darko-renamer'])
    df = df.apply(pd.to_numeric, errors='ignore')
    df = _map_player_names(df, 'DARKO_NAME')
    df = df.set_index('Player').sort_index().fillna(0)

    # Fetch position / minutes / games from ESPN table
    extra = query('ESPN_PROJECTION_TABLE')[['ESPN_NAME', 'MINUTES_PLAYED', 'GAMES_PLAYED', 'POSITION']]
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
