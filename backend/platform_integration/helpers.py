"""
Shared, framework-agnostic helpers used across platform integrations:
team-name deduplication, the roster-DataFrame → assignments converter, and the
platform → canonical player-name lookup.

Pure functions only (no network / no Streamlit): callers pass in any data fetched
from external services, which keeps these unit-testable.
"""

from __future__ import annotations

import pandas as pd


def deduplicate_team_names(team_pairs: list[tuple[str, str]]) -> dict[str, str]:
    """Build a {team_name: team_id} map, disambiguating duplicate display names
    with ' 2', ' 3', ...

    Dedupes on the NAME, working from raw (name, id) pairs before they collapse
    into a dict. The Streamlit Fantrax path instead built {name: id} first (losing
    duplicate-named teams in the comprehension) and ran its dedup helper on the
    ids, so duplicate names silently dropped a team. Shared by every platform
    (Yahoo needs the same dedup).
    """
    teams_dict: dict[str, str] = {}
    used_names: list[str] = []
    for name, team_id in team_pairs:
        unique_name = name
        counter = 1
        while unique_name in used_names:
            counter += 1
            unique_name = f'{name} {counter}'
        used_names.append(unique_name)
        teams_dict[unique_name] = team_id
    return teams_dict


def build_platform_player_id_lookup(
    player_registry: dict
    , player_name_column: str
    , unified_player_table: pd.DataFrame
) -> dict[str, int]:
    """Map a platform's player ids/names to the session's player ids.

    Composes UNIFIED_PLAYER_TABLE (the platform's `player_name_column` -> the row's
    NBA_PLAYER_ID), filtered to ids the session's registry actually holds — the pool
    is the authority on who exists in this session. Keys absent from either side are
    omitted, so a later `lookup.get(key, RP_PLAYER_ID)` yields the replacement-player
    fallback.

    Replaces the name-composing lookup (platform key -> canonical 'Name (POS)') the
    string-identity era used: the unified row's id is now kept instead of being
    discarded after formatting a name.
    """
    rows = unified_player_table.dropna(subset=[player_name_column, 'NBA_PLAYER_ID'])
    return {
        platform_name: int(nba_player_id)
        for platform_name, nba_player_id in zip(rows[player_name_column], rows['NBA_PLAYER_ID'])
        if int(nba_player_id) in player_registry
    }
