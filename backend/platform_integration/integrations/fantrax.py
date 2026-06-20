"""
Fantrax live integration. Ported from the Streamlit FantraxIntegration with all
Streamlit UI and caching removed. Fantrax needs only a league ID (no auth).

Roster data is read through the fantraxapi package's private `_request` (as the
Streamlit code did); see the spec for the public-method fallback if it breaks.
"""

from __future__ import annotations

from typing import Optional

from fantraxapi import FantraxAPI

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
        return 'FANTRAX_PLAYER_NAME'

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
        , name_lookup: dict[str, str]
    ) -> PlatformSelections:
        """Read each team's current roster, mapping platform names to canonical via
        name_lookup ('RP' for any player missing from it). In Season Mode, players
        flagged injured-reserve are moved to injured_players instead of the roster."""
        api = self._make_api(config.league_id)
        exclude_injured = mode == 'Season Mode'
        injured_players: list[str] = []

        player_assignments: dict[str, list[str]] = {}
        for team_name, team_id in config.teams_dict.items():
            roster: list[str] = []
            for row in self._fetch_team_roster_rows(api, team_id):
                if 'scorer' not in row:
                    continue
                player = name_lookup.get(row['scorer']['name'], 'RP')
                if exclude_injured and row['statusId'] == _INJURED_RESERVE_STATUS_ID:
                    injured_players.append(player)
                else:
                    roster.append(player)
            player_assignments[team_name] = roster

        return PlatformSelections(
            player_assignments = player_assignments,
            status             = 'Success',
            injured_players    = injured_players,
        )

    def get_auction_results(
        self
        , config: PlatformConfig
        , mode: str
        , name_lookup: dict[str, str]
    ) -> Optional[PlatformSelections]:
        """Fantrax has no auction support, so this always returns None (matches Streamlit)."""
        return None
