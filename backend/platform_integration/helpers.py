"""
Shared, framework-agnostic helpers used across platform integrations:
team-name deduplication, the roster-DataFrame → assignments converter, and the
platform player → session player lookup.

Pure functions only (no network / no Streamlit): callers pass in any data fetched
from external services, which keeps these unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from backend.data_retrieval import strip_accents
from backend.player_identity import RP_PLAYER_ID


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


@dataclass(frozen=True)
class PlatformPlayerLookup:
    """How a player on a platform's roster is found among the session's players.

    First by the platform's own key -- its player id, or for ESPN its name -- through UNIFIED_PLAYER_TABLE. Then, when
    the platform gives the player's name, by that name as projections are matched: every spelling the unified table
    knows (PLAYER_NAME_RESOLVER_VIEW), and the names the session's own projections use, exactly and then without
    accents. So a player whose unified row lacks this platform's key (a rookie the table has not caught up with) is
    still found, and so is a player only a projection names (kept under a synthetic id). Neither: the replacement
    player.
    """
    player_ids_by_platform_key: dict
    player_ids_by_name: dict[str, int]
    player_ids_by_accent_free_name: dict[str, int]

    def resolve_player_id(
        self
        , platform_key
        , platform_name: Optional[str]
    ) -> int:
        """The session player id for one rostered player; platform_name is None where the platform gives no name."""
        if platform_key in self.player_ids_by_platform_key:
            return self.player_ids_by_platform_key[platform_key]
        if platform_name is not None:
            if platform_name in self.player_ids_by_name:
                return self.player_ids_by_name[platform_name]
            accent_free_name = strip_accents(platform_name)
            if accent_free_name in self.player_ids_by_accent_free_name:
                return self.player_ids_by_accent_free_name[accent_free_name]
        return RP_PLAYER_ID


def build_platform_player_id_lookup(
    player_registry: dict
    , player_name_column: str
    , unified_player_table: pd.DataFrame
    , name_resolvers: tuple[dict[str, int], dict[str, int]]
) -> PlatformPlayerLookup:
    """The session's PlatformPlayerLookup.

    The platform keys compose UNIFIED_PLAYER_TABLE (the platform's `player_name_column` -> the row's NBA_PLAYER_ID),
    filtered to ids the session's registry actually holds -- the pool is the authority on who exists in this session.
    name_resolvers are get_player_name_resolvers()'s (exact, accent-free) spellings -> NBA id, filtered the same way;
    the registry's own names fill in after them, so the unified table settles a spelling it knows. A registry name two
    players share is left out: matching it would be a guess.
    """
    rows = unified_player_table.dropna(subset=[player_name_column, 'NBA_PLAYER_ID'])
    player_ids_by_platform_key = {
        platform_key: int(nba_player_id)
        for platform_key, nba_player_id in zip(rows[player_name_column], rows['NBA_PLAYER_ID'])
        if int(nba_player_id) in player_registry
    }
    exact_resolver, accent_free_resolver = name_resolvers
    player_ids_by_name = {name: nba_player_id for name, nba_player_id in exact_resolver.items()
                          if nba_player_id in player_registry}
    player_ids_by_accent_free_name = {name: nba_player_id for name, nba_player_id in accent_free_resolver.items()
                                      if nba_player_id in player_registry}

    def collect_unique_names(names_by_player_id: dict[int, str]) -> dict[str, int]:
        players_by_name: dict[str, set[int]] = {}
        for player_id, name in names_by_player_id.items():
            players_by_name.setdefault(name, set()).add(player_id)
        return {name: next(iter(player_ids)) for name, player_ids in players_by_name.items() if len(player_ids) == 1}

    registry_names = {player_id: identity.name for player_id, identity in player_registry.items()
                      if player_id != RP_PLAYER_ID}
    for name, player_id in collect_unique_names(registry_names).items():
        player_ids_by_name.setdefault(name, player_id)
    accent_free_registry_names = {player_id: strip_accents(name) for player_id, name in registry_names.items()}
    for name, player_id in collect_unique_names(accent_free_registry_names).items():
        player_ids_by_accent_free_name.setdefault(name, player_id)

    return PlatformPlayerLookup(
        player_ids_by_platform_key     = player_ids_by_platform_key,
        player_ids_by_name             = player_ids_by_name,
        player_ids_by_accent_free_name = player_ids_by_accent_free_name,
    )
