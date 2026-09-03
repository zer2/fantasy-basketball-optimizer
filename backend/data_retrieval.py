"""
Backend data retrieval from Snowflake.

Domain data-access layer: player mapping, historical stats, projections, and
seasons — all the fantasy-basketball-specific reads. The generic Snowflake
connection + query caching lives in backend.infra.snowflake_connection.

Ported from the original Streamlit data-retrieval module, but:
- No Streamlit dependencies
- Explicit `sport_params: dict` instead of get_params()
- Player names always mapped to the canonical 'Player' column
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from backend.infra.snowflake_connection import query, run_query
# One-way edge: player_identity deliberately imports nothing from this module (the resolver
# takes the unified table as an argument), so these module-level imports cannot cycle.
from backend.player_identity import allocate_synthetic_player_ids, build_name_to_player_id_resolver


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

def get_weekly_box_scores(season: str, sport_params: dict) -> pd.DataFrame:
    """Fetch weekly box score totals for a season from WEEKLY_NUMBERS_VIEW.

    Column names are mapped using sport_params['stat-df-renamer'].

    Returns a DataFrame indexed by ('Player', 'Week') with one summed-stat
    row per player per week — the format expected by
    calculate_coefficients_historical.
    """
    df = run_query(f"SELECT * FROM WEEKLY_NUMBERS_VIEW WHERE SEASON = '{season}'")
    df = df.rename(columns=sport_params['stat-df-renamer'])
    df = df.apply(pd.to_numeric, errors='ignore')
    df = df.set_index(['Player', 'WEEK']).sort_index()
    return df.select_dtypes(include='number')


# ── Available seasons ─────────────────────────────────────────────────────────

_HISTORICAL_VIEW = 'HISTORICAL_SEASONAL_AVERAGES_VIEW'


def get_available_seasons() -> list[str]:
    """Return distinct historical seasons from Snowflake, newest first.

    Derived on every call from the same cached HISTORICAL_SEASONAL_AVERAGES_VIEW
    frame that get_historical_data uses, so the season list and the season data
    cannot drift apart. Deliberately not memoised: past the first load this is one
    cached-frame copy and a sort, on a route hit about once per page load.
    """
    df = query(_HISTORICAL_VIEW)
    return sorted({str(s) for s in df['SEASON'].tolist()}, reverse=True)


# ── Historical data ───────────────────────────────────────────────────────────

def get_historical_data(sport_params: dict) -> pd.DataFrame:
    """Fetch full historical player data from Snowflake.

    Returns a DataFrame indexed by (Season, player id), with the season row's native
    display name kept as the 'Player' column — the view carries NBA_PLAYER_ID for every
    row back to 1984-85, so historical identity never depends on the unified table.
    """
    df = query('HISTORICAL_SEASONAL_AVERAGES_VIEW')
    df = df.rename(columns=sport_params['stat-df-renamer'])
    df = df.apply(pd.to_numeric, errors='ignore')

    df['Free Throw %']  = df['Free Throws Made'] / df['Free Throw Attempts']
    df['Field Goal %']  = df['Field Goals Made'] / df['Field Goal Attempts']
    df['Three %']       = df['Threes'] / df['Three Attempts']
    df['Assist to TO']  = df['Assists'] / df['Turnovers']
    df['Position']      = df['Position'].fillna('NP')

    df['NBA_PLAYER_ID'] = df['NBA_PLAYER_ID'].astype(int)
    # ROW ORDER IS LOAD-BEARING: the pool has always been name-sorted, and stable sorts
    # downstream (G-score ties, anchor ordering, top-N selections) inherit it — id-sorting
    # here would change served H-scores. Sort by name, then key by id.
    # The id index level keeps the legacy 'Player' level name: pipeline internals
    # reference the level by that name.
    df = df.sort_values(['Season', 'Player']).fillna(0)
    df = df.set_index(['Season', 'NBA_PLAYER_ID'])
    df.index = df.index.set_names(['Season', 'Player'])
    return df


def get_specified_historical_stats(season: str, sport_params: dict) -> pd.DataFrame:
    """Return player stats for a specific season, indexed by player id.

    The 'Player' column carries the season row's native display name (registry material
    popped off by load_player_pool); stats and 'Position' are the pipeline's v0 columns.
    """
    return get_historical_data(sport_params).loc[season].copy()


# ── Projection data ───────────────────────────────────────────────────────────

def attach_player_ids_by_name(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve df['Player'] (a source's own spellings) to NBA player ids in a nullable
    'player_id' column — the ingestion edge for name-keyed sources. Unresolved rows keep
    a null id; the caller decides between synthetic allocation (uploads) and loud
    warnings (curated sources)."""
    resolver = build_name_to_player_id_resolver(get_unified_player_table())
    df = df.copy()
    df['player_id'] = df['Player'].map(resolver).astype('Int64')
    return df


def get_espn_projections(sport_params: dict) -> pd.DataFrame:
    """Fetch ESPN projections from Snowflake, with 'player_id' resolved from ESPN names."""
    n_games = sport_params['n_games']
    df = query('ESPN_PROJECTION_VIEW')
    df = df.rename(columns=sport_params['espn-renamer'])
    df = attach_player_ids_by_name(df)      # resolve the raw ESPN spellings
    df = _map_player_names(df, 'ESPN_NAME')  # display names stay master-mapped as today
    df['Games Played %'] = df['Games Played'] / n_games
    return df


def get_darko_data(sport_params: dict) -> pd.DataFrame:
    """Fetch DARKO projections from Snowflake, scaled from per-100 to per-game.
    DARKO carries NBA_PLAYER_ID natively; position/minutes ride in from the ESPN table
    joined by id (its names resolved through the unified table)."""
    n_games = sport_params['n_games']

    df = query('DARKO_VIEW')
    df = df.rename(columns=sport_params['darko-renamer'])
    df = df.apply(pd.to_numeric, errors='ignore')
    df = _map_player_names(df, 'DARKO_NAME')
    df['player_id'] = df['NBA_PLAYER_ID'].astype('Int64')
    df = df.drop(columns=['NBA_PLAYER_ID']).sort_values('Player').fillna(0)

    # Fetch position / minutes / games from ESPN table, joined by resolved id
    extra = query('ESPN_PROJECTION_TABLE')[['ESPN_NAME', 'MINUTES_PLAYED', 'GAMES_PLAYED', 'POSITION']]
    extra.columns = ['Player', 'Minutes', 'Games Played %', 'Position']
    extra['Games Played %'] = extra['Games Played %'].astype(float) / n_games
    extra = attach_player_ids_by_name(extra).dropna(subset=['player_id'])
    extra = extra.drop(columns=['Player'])

    df = df.merge(extra, on='player_id')
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
        sport_params['counting-statistics']
        + list(sport_params['ratio-statistics'].keys())
        + [info['volume-statistic'] for info in sport_params['ratio-statistics'].values()]
        + sport_params['other-columns']
        + ['Player', 'player_id']
    )
    required = [c for c in required if c in df.columns]
    return df[list(set(required))]


# ── Canonical position eligibility ────────────────────────────────────────────

def get_canonical_position_eligibility(sport_params: dict) -> pd.Series:
    """Per-player position eligibility from Yahoo — the single authority for the blend.

    Positions are never blended across projection sources: every source contributes
    stats only, and eligibility comes from the latest season in
    YAHOO_PLAYER_POSITION_ELIGIBILITY_TABLE. Composite slots (Util/G/F) are dropped
    and base positions are emitted in base_list order, so a player's position identity
    cannot depend on which sources are active, on a file's comma spacing, or on the
    order a file lists positions in.

    Returns a Series indexed by NBA player id with 'PG,SG'-style values — the join
    path is pool id -> (unified table) -> YAHOO_PLAYER_ID -> eligibility rows.
    """
    base_order = {
        position: rank
        for rank, position in enumerate(sport_params['position_structure']['base_list'])
    }

    eligibility = query('YAHOO_PLAYER_POSITION_ELIGIBILITY_TABLE')
    eligibility = eligibility[
        eligibility['ELIGIBLE'] & eligibility['POSITION'].isin(base_order)
    ]
    eligibility = eligibility[eligibility['SEASON_ID'] == eligibility['SEASON_ID'].max()]

    unified = get_unified_player_table()[['YAHOO_PLAYER_ID', 'NBA_PLAYER_ID']].dropna()
    unified = unified.astype({'YAHOO_PLAYER_ID': 'int64', 'NBA_PLAYER_ID': 'int64'})
    merged = eligibility.astype({'YAHOO_PLAYER_ID': 'int64'}).merge(unified, on='YAHOO_PLAYER_ID')

    return (
        merged.sort_values('POSITION', key=lambda column: column.map(base_order))
              .groupby('NBA_PLAYER_ID')['POSITION']
              .agg(','.join)
    )


# ── Blended projections ───────────────────────────────────────────────────────

def combine_projections(
    blend_weights: dict[str, float],
    sport_params: dict,
    uploaded_dfs: dict[str, Optional[pd.DataFrame]] | None = None,
) -> pd.DataFrame:
    """Blend multiple projection sources using provided weights.

    blend_weights keys: 'DARKO', 'ESPN', plus one key per uploaded source. Uploaded
    sources are keyed by their upload data_id (the id is the source's identity);
    uploaded_dfs maps those same ids to their pre-parsed DataFrames. Snowflake
    sources (DARKO, ESPN) are fetched automatically if weight > 0.
    """
    uploaded = uploaded_dfs or {}

    # Every source is gated on its weight — an uploaded file at weight zero must not
    # participate: its columns would still join the blend's column union, where players
    # from other sources are "missing" them, and the eligibility filter below would then
    # drop those players even though the source contributes nothing.
    sources: dict[str, Optional[pd.DataFrame]] = {
        key: uploaded_df if blend_weights.get(key, 0) > 0 else None
        for key, uploaded_df in sorted(uploaded.items())
    }
    sources['DARKO'] = get_darko_data(sport_params)       if blend_weights.get('DARKO', 0) > 0 else None
    sources['ESPN']  = get_espn_projections(sport_params) if blend_weights.get('ESPN', 0)  > 0 else None
    source_keys = list(sources.keys())

    weights = [blend_weights.get(k, 0.0) for k in source_keys]

    # Uploaded frames arrive name-indexed from parse_projection_upload; bring them to the
    # id-column contract the Snowflake loaders already follow.
    for key in source_keys:
        source_frame = sources[key]
        if source_frame is not None and 'player_id' not in source_frame.columns:
            source_frame = source_frame.reset_index()
            sources[key] = attach_player_ids_by_name(source_frame)

    # Synthetic ids are allocated ONCE over the union of unresolved names across every
    # active source, so the same unknown spelling merges across sources and rebuilds of
    # the same data produce the same ids. The player is kept, never dropped — a changed
    # pool would silently change every downstream number.
    unresolved_names = sorted({
        name
        for key in source_keys
        if sources[key] is not None
        for name in sources[key].loc[sources[key]['player_id'].isna(), 'Player']
    })
    if unresolved_names:
        logging.getLogger('fbbo').warning(
            'combine_projections: %d source name(s) resolve to no NBA id; keeping them '
            'under synthetic ids: %s', len(unresolved_names), ', '.join(unresolved_names[:10]))
    synthetic_ids = allocate_synthetic_player_ids(unresolved_names)

    for key in source_keys:
        source_frame = sources[key]
        if source_frame is None:
            continue
        resolved = source_frame['player_id'].astype('object')
        resolved = resolved.fillna(source_frame['Player'].map(synthetic_ids))
        source_frame = source_frame.drop(columns=['player_id'])
        # The display name travels as '_display_name' through the blend so the id index
        # can keep the legacy 'Player' level name without an index/column name collision.
        source_frame = source_frame.rename(columns={'Player': '_display_name'})
        source_frame.index = pd.Index(resolved.astype(int), name='Player')
        # Two source rows landing on one id (e.g. a file listing a player twice) would
        # silently double-count in the blend — refuse instead.
        if source_frame.index.has_duplicates:
            duplicated = source_frame.index[source_frame.index.duplicated()].tolist()
            raise ValueError(f'{key}: multiple rows resolve to the same player id(s): {duplicated}')
        sources[key] = source_frame

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

    # Drop players missing a column from every source. '_display_name' is exempt: it is
    # registry material, not a blended stat, and a source that carries a player always
    # names them.
    blend_columns = [c for c in df.columns if c != '_display_name']
    ineligible = (df[blend_columns].isna().groupby('Player').sum() == len(source_keys)).sum(axis=1) > 0
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

    # Positions are not blended: overwrite with the canonical eligibility wherever the
    # player is known, so identity is stable across any combination of active sources.
    # A source's own Position survives only for players the canonical table lacks —
    # who, in practice, exist in that source alone, so no identity conflict is possible.
    canonical_positions = get_canonical_position_eligibility(sport_params)
    mapped_positions = pd.Series(df.index.map(canonical_positions), index=df.index)
    df['Position'] = mapped_positions.fillna(df['Position'])

    df['Position'] = df['Position'].fillna('NP')
    stat_columns = [c for c in df.columns if c != '_display_name']
    df[stat_columns] = df[stat_columns].fillna(0)
    # ROW ORDER IS LOAD-BEARING (see get_historical_data): the blend has always emitted a
    # name-sorted pool; groupby ordered it by id above, so restore name order here — while
    # the display column still has its collision-free blend name (renaming first would make
    # sort_values('Player') ambiguous against the id level of the same name).
    df = df.sort_values('_display_name')
    return df.rename(columns={'_display_name': 'Player'})
