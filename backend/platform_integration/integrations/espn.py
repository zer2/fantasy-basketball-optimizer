"""
ESPN live integration — UNTESTED THEORETICAL PORT.

Ported from the Streamlit src/platform_integration/espn_integration.py, framework-agnostic
(no Streamlit). Never run against a real ESPN league; treat every path as provisional.

Auth is two cookies — `espn_s2` and `SWID` — which the user finds via a browser plugin and
pastes (no OAuth). They're persisted per-client by credential_store and spread into this
constructor as `s2` / `swid`. Modes: Season only (the Streamlit ESPN never implemented draft).

Risks / gaps to check first at E2E:
  - The league id from the fan API is composite; we carry the season as
    "<fan-api id>::<year>" through league_id, then `League(league_id=<id>.split(':')[1],
    year=<year>, ...)` (mirrors Streamlit's split(':')[1]). Verify that id shape.
  - player_name_column is 'ESPN_NAME' (the UNIFIED_PLAYER_TABLE column), so the prebuilt
    name_lookup maps espn_api's player.name -> canonical 'Name (POS)'. Verify those names
    match ESPN_NAME at E2E.
  - SWID is stored with braces stripped (as Streamlit did); espn_api may want them — untested.
  - KEY GAP: ESPN is Season-only, and **live Season-mode roster population is not wired** —
    `get_draft_results` returns the current rosters as {team: [players]}, but nothing yet fills
    the season roster grid (sr-player-*) from a poll, so the Waiver/Trading tabs won't see them.
    This gap affects Fantrax/Yahoo Season mode too; it is the main thing left for ESPN to function.
"""

from __future__ import annotations

from typing import Optional

import requests
from espn_api.basketball import League

from backend.platform_integration.base import (
    PlatformIntegration, LeagueShape, PlatformConfig, PlatformSelections,
)
from backend.platform_integration.helpers import deduplicate_team_names
from backend.player_identity import RP_PLAYER_ID


_ESPN_FAN_API = 'https://fan.api.espn.com/apis/v2/fans/'
_LEAGUE_ID_YEAR_SEP = '::'   # league_id is "<fan-api id>::<year>"


class ESPNIntegration(PlatformIntegration):

    def __init__(self, s2: Optional[str] = None, swid: Optional[str] = None):
        # s2 + SWID cookies, spread from the creds bag by get_integration.
        self._s2 = s2
        self._swid = swid

    # ── Platform metadata ──────────────────────────────────────────────────────

    @property
    def available_modes(self) -> list[str]:
        return ['Season Mode']

    @property
    def description_string(self) -> str:
        return 'Retrieve from ESPN'

    @property
    def player_name_column(self) -> str:
        return 'ESPN_NAME'

    # ── League id <-> (espn id, year) ──────────────────────────────────────────

    @staticmethod
    def _split_league_id(league_id: str) -> tuple[str, int]:
        """Our composite league_id carries the season: '<fan-api id>::<year>'."""
        fan_api_id, year = league_id.rsplit(_LEAGUE_ID_YEAR_SEP, 1)
        return fan_api_id.split(':')[1], int(year)   # mirrors Streamlit split(':')[1]

    def _make_league(self, league_id: str) -> League:
        espn_id, year = self._split_league_id(league_id)
        return League(league_id=espn_id, year=year, espn_s2=self._s2, swid=self._swid)

    # ── Connection ─────────────────────────────────────────────────────────────

    def list_divisions(self, league_id: str) -> list[dict]:
        """ESPN has no divisions."""
        return []

    def list_leagues(self) -> list[dict]:
        """The user's ESPN leagues via the fan API (keyed by SWID), reverse-chronological.
        `id` is composite ('<fan-api id>::<year>') so connect/draft-state carry the season."""
        query = (
            f'{_ESPN_FAN_API}%7B{self._swid}%7D?displayHiddenPrefs=true&context=fantasy'
            '&useCookieAuth=true&source=fantasyapp-ios&featureFlags=challengeEntries'
        )
        response = requests.get(query)
        response.raise_for_status()
        leagues = response.json()['preferences']
        leagues = sorted(leagues, key=lambda league: league['metaData']['entry']['seasonId'], reverse=True)
        result = []
        for league in leagues:
            entry = league['metaData']['entry']
            season = entry['seasonId']
            name = entry['groups'][0]['groupName']
            result.append({
                'id':     f"{league['id']}{_LEAGUE_ID_YEAR_SEP}{season}",
                'name':   f'{name} ({season - 1}-{season})',
                'season': season,
            })
        return result

    def _team_pairs(self, league: League) -> list[tuple[str, str]]:
        return [(team.team_name, str(team.team_id)) for team in league.teams]

    def fetch_league_shape(
        self
        , league_id: str
        , division_id: Optional[str]
    ) -> LeagueShape:
        league = self._make_league(league_id)
        teams_dict = deduplicate_team_names(self._team_pairs(league))
        n_picks = max((len(team.roster) for team in league.teams), default=0)
        return LeagueShape(
            team_names = list(teams_dict.keys()),
            n_drafters = len(teams_dict),
            n_picks    = n_picks,
            teams_dict = teams_dict,
        )

    # ── Roster state (Season only) ─────────────────────────────────────────────

    def get_draft_results(
        self
        , config: PlatformConfig
        , mode: str
        , player_id_lookup: dict[str, int]
    ) -> PlatformSelections:
        """Current rosters per team (ESPN supports only Season Mode)."""

        league = self._make_league(config.league_id)
        team_by_id = {str(team.team_id): team for team in league.teams}

        player_assignments: dict[str, list[int]] = {}
        for team_name, team_id in config.teams_dict.items():
            team = team_by_id.get(team_id)
            roster = [] if team is None else [
                player_id_lookup.get(player.name, RP_PLAYER_ID) for player in team.roster
            ]
            player_assignments[team_name] = roster

        return PlatformSelections(
            player_assignments = player_assignments,
            status             = 'Success',
            injured_players    = [],
        )

    def get_auction_results(
        self
        , config: PlatformConfig
        , mode: str
        , player_id_lookup: dict[str, int]
    ) -> Optional[PlatformSelections]:
        # ESPN auctions are not supported.
        return None
