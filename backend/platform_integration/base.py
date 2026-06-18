"""
Abstract base for live fantasy-platform integrations (ESPN, Yahoo, Fantrax).

Ported from the Streamlit src/platform_integration/platform_integration.py, but
framework-agnostic: these classes perform only data retrieval and contain no UI
or HTTP-framework code. The credential collection, league selection, and polling
control flow that lived in the Streamlit setup() method are handled by the API
routes (backend/main.py) and the frontend.

This module is the dependency-free root of the package: concrete integrations and
the registry import it, never the reverse.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional


@dataclass
class LeagueShape:
    """League metadata needed to configure a session, fetched before any roster
    data and therefore without requiring session.info."""
    team_names: list[str]
    n_drafters: int
    n_picks:    int
    teams_dict: dict[str, str]   # team_name -> platform team_id


@dataclass
class PlatformConfig:
    """Per-session platform connection, stored on the Session after connect so the
    draft-state poll can refetch rosters."""
    platform:           str
    league_id:          str
    division_id:        Optional[str]
    teams_dict:         dict[str, str]   # team_name -> platform team_id
    player_name_column: str


@dataclass
class PlatformDraftState:
    """Current state pulled from a platform.

    `player_assignments` maps each team to its list of canonical 'Name (POS)'
    players (the shape /evaluate expects). `injured_players` lists players the
    platform flags as out (Season Mode only); empty otherwise.
    """
    player_assignments: dict[str, list[str]]
    status:             str
    injured_players:    list[str]


class PlatformIntegration(abc.ABC):

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
        """The platform's player-name column in PLAYER_MAPPING_VIEW, e.g. 'FANTRAX_PLAYER_NAME'."""

    @abc.abstractmethod
    def list_divisions(self, league_id: str) -> list[dict]:
        """Return [{'name': ..., 'id': ...}] per division; empty when the league has none."""

    @abc.abstractmethod
    def fetch_league_shape(
        self
        , league_id: str
        , division_id: Optional[str]
    ) -> LeagueShape:
        """Fetch team names, drafter count, and pick count for the league/division."""

    @abc.abstractmethod
    def get_draft_results(
        self
        , config: PlatformConfig
        , mode: str
        , name_lookup: dict[str, str]
    ) -> PlatformDraftState:
        """Fetch the current draft board / rosters, mapping platform names to canonical
        via name_lookup (built once upstream from session.info)."""

    @abc.abstractmethod
    def get_auction_results(
        self
        , config: PlatformConfig
        , mode: str
        , name_lookup: dict[str, str]
    ) -> Optional[PlatformDraftState]:
        """Fetch the current auction state, or None when the platform has no auction support."""
