"""Player identity: the id-keyed registry and the name→id resolution edges.

The app keys every player by an integer id — NBA_PLAYER_ID for NBA. The stats sources
carry it natively (the historical view and DARKO both ship an NBA_PLAYER_ID column), so
ids flow through ingestion, the math engine, session state, and the API without ever
passing through a name. Names exist only in the per-session registry built at ingestion,
and are rendered only at display time.

Name→id conversion therefore happens ONLY at the edges where names enter the system:
  - ingesting name-keyed sources (ESPN projections, uploaded projection CSVs), via
    PLAYER_NAME_RESOLVER_VIEW (data_retrieval.get_player_name_resolver): every spelling
    UNIFIED_PLAYER_TABLE knows, in any of its name columns, mapped to one NBA id -- a
    spelling shared by two players goes to the most recently active, rookies first (the
    rule is in scripts/player_name_resolver_view.sql);
  - human text input: the season-roster clipboard paste (resolved in the frontend).

Reserved ids (NBA ids are positive, so none of these can collide with a real player):
  - FULL_ROSTER_SCORE_PLAYER_ID (0): the single team-score row a full-roster evaluate
    returns. Not a player and has no registry entry — clients read only its scores.
  - RP_PLAYER_ID (-1): ONE stand-in for "a replacement-level player". It is not a player
    from any projection: the pipeline injects it (scores of -1 everywhere), and the live
    platform integrations map any rostered player they cannot match to it.
  - synthetic ids (-2, -3, ...): REAL players from a projection source -- with their own
    stats -- whose name matches no spelling UNIFIED_PLAYER_TABLE knows. Each distinct
    unmatched name gets its own id, handed out at runtime. The player is KEPT, because
    dropping him would silently change the player pool and every number derived from it;
    he just has no headshot and shows the source's own spelling.
    RP and synthetic ids never meet: RP stands in for players the app does not have,
    synthetic ids carry players it has under names it cannot place. One consequence: a
    synthetic player has no NBA id, so a live platform cannot match him either -- if he
    is drafted there, the board records RP and he stays available in the rankings.
    This is not rare. In the 2026 files, 8 of 428 Hashtag Basketball names and 28 of 532
    Basketball Monster names went synthetic: accents (Diabaté), suffixes (Bronny James
    Jr.), initials (L.J. Cryer), nicknames (Lu Dort), and fringe players and rookies the
    table has not been given yet.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

# The one evaluate with nothing to rank: a full roster's single team-score row (see
# _build_candidates). Id 0 can never collide — NBA ids are positive, RP is -1, and
# synthetic ids descend from -2.
FULL_ROSTER_SCORE_PLAYER_ID = 0

RP_PLAYER_ID = -1

# Where the synthetic ids start: the first unmatched name gets -2, the next -3, and so on
# downward (see allocate_synthetic_player_ids). Synthetic ids have no constants of their own;
# they are handed out at runtime, and this starting point is the only fixed value. It is
# private (underscored) because nothing outside this module needs it, unlike RP_PLAYER_ID
# and FULL_ROSTER_SCORE_PLAYER_ID, which other modules compare ids against.
_FIRST_SYNTHETIC_PLAYER_ID = -2


@dataclass
class PlayerIdentity:
    player_id: int
    name: str              # display name (see the display-name precedence in the refactor plan)
    last_name: str
    positions: list[str]   # base position codes, e.g. ['PG', 'SG']; [] for RP/synthetic
    has_headshot: bool     # False for RP and synthetic ids (no NBA CDN image exists)


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


def extract_last_name(full_name: str) -> str:
    """'Nikola Jokic' -> 'Jokic'; single-word names return themselves ('RP' -> 'RP')."""
    parts = full_name.split(' ')
    return ' '.join(parts[1:]) if len(parts) > 1 else full_name


def make_replacement_player_identity() -> PlayerIdentity:
    """The registry entry for the pipeline's replacement-player sentinel."""
    return PlayerIdentity(
        player_id    = RP_PLAYER_ID,
        name         = 'RP',
        last_name    = 'RP',
        positions    = [],
        has_headshot = False,
    )


def allocate_synthetic_player_ids(unresolved_names: Iterable[str]) -> dict[str, int]:
    """Deterministic session-scoped ids for names nothing resolves: sorted names get
    -2, -3, ... so rebuilding the same data always produces the same ids. These are real
    players kept under names the table cannot place -- not RP; see the module docstring."""
    return {
        name: _FIRST_SYNTHETIC_PLAYER_ID - offset
        for offset, name in enumerate(sorted(set(unresolved_names)))
    }
