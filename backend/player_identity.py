"""Player identity: the id-keyed registry and the name→id resolution edges.

The app keys every player by an integer id — NBA_PLAYER_ID for NBA. The stats sources
carry it natively (the historical view and DARKO both ship an NBA_PLAYER_ID column), so
ids flow through ingestion, the math engine, session state, and the API without ever
passing through a name. Names exist only in the per-session registry built at ingestion,
and are rendered only at display time.

Name→id conversion therefore happens ONLY at the edges where names enter the system:
  - ingesting name-keyed sources (ESPN projections, uploaded HTB/BBM CSVs), via
    build_name_to_player_id_resolver — every name-variant column of UNIFIED_PLAYER_TABLE
    mapped to the row's NBA id, with MASTER_PLAYER_NAME applied last so the canonical
    spelling wins collisions;
  - human text inputs (the injured-players list, season-roster clipboard paste).

Two reserved id ranges:
  - RP_PLAYER_ID (-1): the replacement-player sentinel the pipeline injects.
  - synthetic ids (-2, -3, ...): user-uploaded rows whose names resolve to nothing.
    The player is KEPT — dropping or failing would change the player pool — and the
    state is modeled: no headshot, source-native display name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

RP_PLAYER_ID = -1
_FIRST_SYNTHETIC_PLAYER_ID = -2

# Every name column of UNIFIED_PLAYER_TABLE that a source or user might spell a player
# with. MASTER_PLAYER_NAME is applied last so the canonical spelling wins collisions.
PLAYER_NAME_COLUMNS = ['DARKO_NAME', 'ESPN_NAME', 'ROTOWIRE_NAME', 'HTB_NAME', 'BBM_NAME',
                       'MASTER_PLAYER_NAME']


@dataclass
class PlayerIdentity:
    player_id: int
    name: str              # display name (see the display-name precedence in the refactor plan)
    last_name: str
    positions: list[str]   # base position codes, e.g. ['PG', 'SG']; [] for RP/synthetic
    has_headshot: bool     # False for RP and synthetic ids (no NBA CDN image exists)


def extract_last_name(full_name: str) -> str:
    """'Nikola Jokic' -> 'Jokic'; single-word names return themselves ('RP' -> 'RP')."""
    parts = full_name.split(' ')
    return ' '.join(parts[1:]) if len(parts) > 1 else full_name


def make_player_identity(
    player_id: int
    , name: str
    , position_value: str
) -> PlayerIdentity:
    """Build one registry entry from a display name and a 'PG,SG'-style position string."""
    positions = [p for p in str(position_value).split(',') if p and p != 'NP'] \
        if position_value == position_value else []
    return PlayerIdentity(
        player_id    = player_id,
        name         = name,
        last_name    = extract_last_name(name),
        positions    = positions,
        has_headshot = player_id > 0,
    )


def make_replacement_player_identity() -> PlayerIdentity:
    """The registry entry for the pipeline's replacement-player sentinel."""
    return PlayerIdentity(
        player_id    = RP_PLAYER_ID,
        name         = 'RP',
        last_name    = 'RP',
        positions    = [],
        has_headshot = False,
    )


def build_name_to_player_id_resolver() -> dict[str, int]:
    """Every known spelling of every unified-table player -> NBA player id.

    The ingestion edge for name-keyed sources. Rows without an NBA id are omitted —
    a name that only maps to an id-less row is treated as unresolvable, exactly like
    a name the table has never seen (the synthetic-id path handles both).
    """
    from backend.data_retrieval import get_unified_player_table

    players = get_unified_player_table().dropna(subset=['NBA_PLAYER_ID'])
    resolver: dict[str, int] = {}
    for name_column in PLAYER_NAME_COLUMNS:
        named = players.dropna(subset=[name_column])
        resolver.update({
            name: int(nba_player_id)
            for name, nba_player_id in zip(named[name_column], named['NBA_PLAYER_ID'])
        })
    return resolver


def resolve_typed_player_names(
    player_registry: dict[int, 'PlayerIdentity']
    , typed_names: Iterable[str]
) -> list[int]:
    """Resolve free-typed names (the injured-players list) against a session registry:
    exact match on the display name or on the legacy 'Name (POS)' form. Unmatched
    entries are ignored — the pipeline's long-standing errors='ignore' semantics for
    this one human-text input."""
    lookup: dict[str, int] = {}
    for identity in player_registry.values():
        lookup[identity.name] = identity.player_id
        if identity.positions:
            lookup[f"{identity.name} ({','.join(identity.positions)})"] = identity.player_id
    return [lookup[name.strip()] for name in typed_names if name.strip() in lookup]


def allocate_synthetic_player_ids(unresolved_names: Iterable[str]) -> dict[str, int]:
    """Deterministic session-scoped ids for names nothing resolves: sorted names get
    -2, -3, ... so rebuilding the same data always produces the same ids."""
    return {
        name: _FIRST_SYNTHETIC_PLAYER_ID - offset
        for offset, name in enumerate(sorted(set(unresolved_names)))
    }
