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


def build_platform_name_lookup(
    info: dict
    , player_name_column: str
    , mapping_view: pd.DataFrame
) -> dict[str, str]:
    """Map a platform's player names to canonical 'Name (POS1,POS2)' identifiers.

    Composes PLAYER_MAPPING_VIEW (the platform's `player_name_column` -> canonical
    'PLAYER_NAME') with info['Positions'] (canonical base name -> position codes),
    so a platform-specific spelling such as 'Nik Jokic' resolves to 'Nikola Jokic
    (C)'. Names absent from either table are simply omitted from the lookup, so a
    later `lookup.get(name, 'RP')` yields the 'RP' replacement-player fallback.

    Replaces the Streamlit get_fixed_player_name suffix-strip shortcut, which
    assumed the platform name already equalled the canonical name; this consults
    the mapping view instead (which is why each integration declares
    player_name_column).
    """
    # Canonical base name (suffix stripped) -> position codes. keep='first' guards
    # the rare case of two players sharing a base name but different positions.
    positions = info['Positions']
    positions = positions[~positions.index.duplicated(keep='first')]
    positions_by_base = {name.split(' (')[0]: codes for name, codes in positions.items()}

    # Platform name -> canonical base 'PLAYER_NAME'.
    base_by_platform_name = (
        mapping_view.dropna(subset=[player_name_column])
        .set_index(player_name_column)['PLAYER_NAME']
    )

    lookup: dict[str, str] = {}
    for platform_name, canonical_base in base_by_platform_name.items():
        codes = positions_by_base.get(canonical_base)
        if codes is not None:
            lookup[platform_name] = f"{canonical_base} ({','.join(codes)})"
    return lookup
