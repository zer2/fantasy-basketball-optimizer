"""
Fantrax live integration. Ported from the Streamlit FantraxIntegration with all
Streamlit UI and caching removed. Fantrax needs only a league ID (no auth).

Roster data is read through the fantraxapi package's private `_request` (as the
Streamlit code did); see the spec for the public-method fallback if it breaks.
"""

from __future__ import annotations

import logging
from typing import Optional

from fantraxapi import FantraxAPI

logger = logging.getLogger('fbbo')

from backend.platform_integration.base import (
    PlatformIntegration, LeagueShape, PlatformConfig, PlatformSelections,
)
from backend.platform_integration.helpers import deduplicate_team_names


# Standings tabs that are not divisions.
_NON_DIVISION_TAB_NAMES = {'All', 'Combined', 'Results', 'Season Stats', 'Playoffs'}

# Roster-slot cap the optimizer respects (mirrors the Streamlit min(..., 16)).
_MAX_ROSTER_SLOTS = 16

# Fantrax statusId for injured-reserve players, excluded in Season Mode.
_INJURED_RESERVE_STATUS_ID = '3'


class FantraxIntegration(PlatformIntegration):
    # Organized by workflow, not by visibility: each public operation is grouped
    # with the private _fetch_* helpers it uses (metadata → connection → roster).

    # ── Platform metadata ──────────────────────────────────────────────────────

    @property
    def available_modes(self) -> list[str]:
        return ['Draft Mode', 'Season Mode']

    @property
    def description_string(self) -> str:
        return 'Retrieve from Fantrax'

    @property
    def player_name_column(self) -> str:
        # Fantrax players are matched to canonical by their stable Fantrax id
        # (UNIFIED_PLAYER_TABLE.FANTRAX_ID == the roster row's scorer 'scorerId'),
        # not by name — so no Fantrax name needs storing in the unified table.
        return 'FANTRAX_ID'

    # ── External dependency (wrapped so tests can substitute it) ───────────────

    def _make_api(self, league_id: str) -> FantraxAPI:
        return FantraxAPI(league_id)

    # ── Connection (list_divisions / fetch_league_shape + their fetchers) ──────

    def list_divisions(self, league_id: str) -> list[dict]:
        """Return the league's divisions as [{'name', 'id'}] (empty if it has none)."""
        api = self._make_api(league_id)
        tabs = api._request('getStandings', view='SCHEDULE')['displayedLists']['tabs']
        return [
            {'name': tab['name'], 'id': tab['id']}
            for tab in tabs
            if tab['name'] not in _NON_DIVISION_TAB_NAMES
        ]

    def _fetch_team_pairs(
        self
        , api: FantraxAPI
        , division_id: Optional[str]
    ) -> list[tuple[str, str]]:
        """Return raw (team_name, team_id) pairs for the league or a single division."""
        if division_id is None:
            fantasy_teams = api._request('getFantasyTeams')['fantasyTeams']
            return [(team['name'], team['id']) for team in fantasy_teams]
        standings_rows = api._request('getStandings', view=division_id)['tableList'][0]['rows']
        if len(standings_rows) == 0:
            standings_rows = api._request('getStandings', view=division_id)['tableList'][1]['rows']
        return [
            (row['fixedCells'][1]['content'], row['fixedCells'][1]['teamId'])
            for row in standings_rows
        ]

    def fetch_league_shape(
        self
        , league_id: str
        , division_id: Optional[str]
    ) -> LeagueShape:
        """Return the league/division's team names, drafter count, and pick count."""
        api = self._make_api(league_id)
        teams_dict = deduplicate_team_names(self._fetch_team_pairs(api, division_id))
        team_names = list(teams_dict.keys())
        n_picks = self._fetch_n_picks(api, next(iter(teams_dict.values())))
        return LeagueShape(
            team_names = team_names,
            n_drafters = len(team_names),
            n_picks    = n_picks,
            teams_dict = teams_dict,
        )

    def _fetch_n_picks(self, api: FantraxAPI, team_id: str) -> int:
        """Number of active roster slots (excluding Injured Reserve), capped at 16."""
        status_totals = api._request('getTeamRosterInfo', teamId=team_id)['miscData']['statusTotals']
        active_slots = sum(entry['max'] for entry in status_totals if entry['name'] != 'Inj Res')
        return min(active_slots, _MAX_ROSTER_SLOTS)

    # ── Roster / draft state ──────────────────────────────────────────────────

    def _fetch_team_roster_rows(self, api: FantraxAPI, team_id: str) -> list[dict]:
        """Return a team's raw roster rows (each row may carry a 'scorer')."""
        return api._request('getTeamRosterInfo', teamId=team_id)['tables'][0]['rows']

    def get_draft_results(
        self
        , config: PlatformConfig
        , mode: str
        , player_id_lookup: dict[str, int]
    ) -> PlatformSelections:
        """Read each team's current roster, mapping each player's Fantrax scorer id to a
        session player id (RP_PLAYER_ID for any player missing from the lookup, counted
        and logged — a whole-roster fallback means the mapping is broken, not the roster).
        In Season Mode, players flagged injured-reserve are moved to injured_players
        instead of the roster."""
        from backend.player_identity import RP_PLAYER_ID

        api = self._make_api(config.league_id)
        exclude_injured = mode == 'Season Mode'
        injured_players: list[int] = []
        unmatched_count = 0

        player_assignments: dict[str, list[int]] = {}
        for team_name, team_id in config.teams_dict.items():
            roster: list[int] = []
            for row in self._fetch_team_roster_rows(api, team_id):
                if 'scorer' not in row:
                    continue
                player_id = player_id_lookup.get(row['scorer']['scorerId'], RP_PLAYER_ID)
                unmatched_count += player_id == RP_PLAYER_ID
                if exclude_injured and row['statusId'] == _INJURED_RESERVE_STATUS_ID:
                    injured_players.append(player_id)
                else:
                    roster.append(player_id)
            player_assignments[team_name] = roster

        if unmatched_count:
            logger.warning('Fantrax roster mapping: %d player(s) fell back to RP', unmatched_count)
        return PlatformSelections(
            player_assignments = player_assignments,
            status             = 'Success',
            injured_players    = injured_players,
        )

    def get_auction_results(
        self
        , config: PlatformConfig
        , mode: str
        , player_id_lookup: dict[str, int]
    ) -> Optional[PlatformSelections]:
        """Fantrax has no auction support, so this always returns None (matches Streamlit)."""
        return None
