"""
Shared, framework-agnostic helpers used across platform integrations:
team-name deduplication, the roster-DataFrame → assignments converter, and the
platform → canonical player-name lookup.

Pure functions only (no network / no Streamlit): callers pass in any data fetched
from external services, which keeps these unit-testable.
"""

from __future__ import annotations

import re
import unicodedata
import pandas as pd


def normalize_player_name(name: str) -> str:
    """Normalize a player name for fuzzy/flexible matching:
    - Removes unicode diacritics/accents (e.g. Dončić -> Doncic, Jokić -> Jokic)
    - Lowercases & strips whitespace
    - Removes punctuation (periods, apostrophes, etc.)
    - Removes common generational suffixes (Jr, Sr, II, III, IV)
    - Normalizes internal hyphens and spaces
    """
    if not name or not isinstance(name, str):
        return ''
    decomposed = unicodedata.normalize('NFKD', name)
    s = ''.join(c for c in decomposed if not unicodedata.combining(c))
    s = s.strip().lower()
    s = re.sub(r"['\.\`\"]", "", s)
    s = re.sub(r"[-_/]", " ", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


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
    """
    rows = unified_player_table.dropna(subset=[player_name_column, 'NBA_PLAYER_ID'])
    lookup: dict[str, int] = {
        platform_name: int(nba_player_id)
        for platform_name, nba_player_id in zip(rows[player_name_column], rows['NBA_PLAYER_ID'])
        if int(nba_player_id) in player_registry
    }

    # If this is a name-based platform column (e.g. ESPN_NAME), populate all cross-source
    # name variants from Snowflake's UNIFIED_PLAYER_TABLE (DARKO_NAME, ROTOWIRE_NAME,
    # BBM_NAME, HTB_NAME, MASTER_PLAYER_NAME) along with accent/punctuation-normalized
    # variants so players match seamlessly without hardcoded alias dictionaries.
    is_name_column = player_name_column in (
        'ESPN_NAME', 'MASTER_PLAYER_NAME', 'DARKO_NAME', 'ROTOWIRE_NAME', 'HTB_NAME', 'BBM_NAME'
    )
    if is_name_column:
        candidate_cols = [
            'DARKO_NAME', 'ESPN_NAME', 'ROTOWIRE_NAME', 'HTB_NAME', 'BBM_NAME', 'MASTER_PLAYER_NAME'
        ]
        available_cols = [c for c in candidate_cols if c in unified_player_table.columns]
        valid_rows = unified_player_table.dropna(subset=['NBA_PLAYER_ID'])
        for _, row in valid_rows.iterrows():
            nba_id = int(row['NBA_PLAYER_ID'])
            if nba_id not in player_registry:
                continue
            for col in available_cols:
                if pd.notna(row[col]):
                    raw_val = str(row[col]).strip()
                    if raw_val and raw_val not in lookup:
                        lookup[raw_val] = nba_id
                    norm = normalize_player_name(raw_val)
                    if norm and norm not in lookup:
                        lookup[norm] = nba_id

        # Also map all names registered in the session registry
        for p_id, identity in player_registry.items():
            if hasattr(identity, 'name') and identity.name:
                raw_name = identity.name
                if raw_name not in lookup:
                    lookup[raw_name] = p_id
                norm = normalize_player_name(raw_name)
                if norm and norm not in lookup:
                    lookup[norm] = p_id

    return lookup
