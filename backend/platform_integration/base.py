"""
Abstract base for live fantasy-platform integrations (ESPN, Yahoo, Fantrax).

Ported from the original Streamlit platform-integration module, but
framework-agnostic: these classes perform only data retrieval and contain no UI
or HTTP-framework code. The credential collection, league selection, and polling
control flow that lived in the Streamlit setup() method are handled by the API
routes (backend/api/routers) and the frontend.

This module is the dependency-free root of the package: concrete integrations and
the registry import it, never the reverse.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

from backend.platform_integration.helpers import PlatformPlayerLookup
from backend.team_identity import TeamIdentity


@dataclass
class LeagueShape:
    """League metadata needed to configure a session, fetched before any roster
    data and therefore without requiring session.agent.info."""
    # Every seat in the league, occupied or not, in the platform's order (team_identity): each joined team under its
    # platform id, and a seat nobody has taken under "empty-N". The seat selector, the board columns and every
    # team-keyed field index by these ids; the drafter count is their number.
    teams:      list[TeamIdentity]
    n_picks:    int
    # True for an auction league, False for a snake draft, None when the platform does not say.
    # None is a real third state, not a default standing in for False: it means "do not judge the
    # user's chosen mode against this", which is what platforms that never report it need.
    is_auction_draft: Optional[bool] = None
    # The league's own position structure, in the app's slot names ({'PG': 1, ..., 'Util': 3}), when the platform
    # reports one; None when it does not, and the app falls back to its default for the roster size.
    slot_counts: Optional[dict[str, int]] = None


@dataclass
class PlatformConfig:
    """Per-session platform connection, stored on the Session after connect so the
    draft-state poll can refetch rosters."""
    platform:           str
    league_id:          str
    division_id:        Optional[str]
    player_name_column: str
    # Every seat, as LeagueShape.teams. Boards are seeded from these, so a seat nobody has taken yet is still a team
    # with an empty roster rather than missing from the league.
    teams:              list[TeamIdentity]


@dataclass
class PlatformSelections:
    """Current selections pulled from a platform — draft board, season rosters, or
    auction.

    Every field is keyed by team id (team_identity). `player_assignments` maps each team to its list of player ids (the shape /evaluate
    expects). `injured_players` lists the ids of players the platform flags as out
    (Season Mode only); empty otherwise. `unmatched_player_names` names the rostered players the session could not
    match, which player_assignments carries as RP_PLAYER_ID: unmatched_player_names[team][i] is the platform's name for
    player_assignments[team][i], so the roster can show who it is. Empty where the platform gives no names (a draft
    board read by id). `costs` is populated only for auctions: costs[team][i] is the price paid for
    player_assignments[team][i].
    """
    player_assignments:     dict[str, list[int]]           # team id -> player ids
    status:                 str
    injured_players:        list[int]
    unmatched_player_names: dict[str, dict[int, str]]      # team id -> {roster position: the platform's name}
    costs:                  Optional[dict[str, list[float]]] = None   # team id -> prices


class PlatformIntegration(abc.ABC):
    # No __init__ here on purpose: the ABC constrains behavior, not construction.
    # Each integration declares its own explicit constructor credentials (Fantrax none;
    # Yahoo auth_dir), and the registry spreads the credentials bag into them via
    # get_integration's cls(**(credentials or {})).

    @property
    @abc.abstractmethod
    def available_modes(self) -> list[str]:
        """Modes this platform supports, e.g. ['Draft Mode', 'Season Mode']."""

    @property
    @abc.abstractmethod
    def description_string(self) -> str:
        """Human-readable label, e.g. 'Retrieve from Fantrax'."""

    @property
    @abc.abstractmethod
    def player_name_column(self) -> str:
        """The platform's player id/name column in UNIFIED_PLAYER_TABLE, e.g. 'FANTRAX_ID'."""

    def list_leagues(self) -> list[dict]:
        """The user's leagues as [{'id', 'name', ...}] for auth-based platforms (Yahoo,
        ESPN). Default [] for platforms that take a manually-entered league id (Fantrax)."""
        return []

    @abc.abstractmethod
    def list_divisions(self, league_id: str) -> list[dict]:
        """Return [{'name': ..., 'id': ...}] per division; empty when the league has none."""

    @abc.abstractmethod
    def fetch_league_shape(
        self
        , league_id: str
        , division_id: Optional[str]
    ) -> LeagueShape:
        """Fetch the teams (with their ids), and the pick count, for the league/division."""

    @abc.abstractmethod
    def get_draft_results(
        self
        , config: PlatformConfig
        , mode: str
        , player_id_lookup: PlatformPlayerLookup
    ) -> PlatformSelections:
        """Fetch the current draft board / rosters, mapping each rostered player to a session
        player id via player_id_lookup (built once upstream from the session registry): by
        platform key, then by name where the platform gives one, else RP_PLAYER_ID."""

    @abc.abstractmethod
    def get_auction_results(
        self
        , config: PlatformConfig
        , mode: str
        , player_id_lookup: PlatformPlayerLookup
    ) -> Optional[PlatformSelections]:
        """Fetch the current auction state (with `costs`), or None when the platform has
        no auction support."""
