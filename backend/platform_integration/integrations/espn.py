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

import logging
from typing import Optional

import requests
from espn_api.basketball import League as NBALeague
from espn_api.wbasketball import League as WNBALeague

from backend.platform_integration.base import (
    PlatformIntegration, LeagueShape, PlatformConfig, PlatformSelections,
)
from backend.platform_integration.helpers import PlatformPlayerLookup, deduplicate_team_names
from backend.player_identity import RP_PLAYER_ID


logger = logging.getLogger('fbbo')

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
    def _split_league_id(league_id: str) -> tuple[str, str, int]:
        """Our composite league_id carries the season: '<fan-api id>::<year>'."""
        fan_api_id, year_str = league_id.rsplit(_LEAGUE_ID_YEAR_SEP, 1)
        parts = fan_api_id.split(':')
        if len(parts) >= 4:
            # Live ESPN Fan API format: '<entryId>:<groupId>:<sportId>:<seasonId>'
            numeric_id = parts[1]
            prefix = 'wfba' if parts[2] == '5' else 'fba'
        elif len(parts) == 2:
            # Prefixed format: 'fba:12345' or 'wfba:12345'
            numeric_id = parts[1]
            prefix = 'wfba' if fan_api_id.startswith('wfba:') else 'fba'
        else:
            numeric_id = parts[0]
            prefix = 'fba'
        return prefix, numeric_id, int(year_str)

    def _format_swid(self) -> Optional[str]:
        """ESPN cookie authentication requires SWID enclosed in curly braces."""
        if not self._swid:
            return None
        raw = self._swid.strip().strip('{}')
        return f"{{{raw}}}"

    def _make_league(self, league_id: str):
        prefix, espn_id, year = self._split_league_id(league_id)
        swid = self._format_swid()
        if prefix == 'wfba':
            return WNBALeague(league_id=int(espn_id), year=year, espn_s2=self._s2, swid=swid)
        else:
            return NBALeague(league_id=int(espn_id), year=year, espn_s2=self._s2, swid=swid)

    # ── Connection ─────────────────────────────────────────────────────────────

    def list_divisions(self, league_id: str) -> list[dict]:
        """ESPN has no divisions."""
        return []

    def list_leagues(self) -> list[dict]:
        """The user's ESPN leagues via the fan API (keyed by SWID), reverse-chronological.
        `id` is composite ('<fan-api id>::<year>') so connect/draft-state carry the season."""
        raw_swid = self._swid.strip().strip('{}') if self._swid else ''
        query = (
            f'{_ESPN_FAN_API}%7B{raw_swid}%7D?displayHiddenPrefs=true&context=fantasy'
            '&useCookieAuth=true&source=fantasyapp-ios&featureFlags=challengeEntries'
        )
        cookies = {}
        if self._s2:
            cookies['espn_s2'] = self._s2
        if self._swid:
            cookies['SWID'] = self._format_swid() or ''

        response = requests.get(query, cookies=cookies)
        response.raise_for_status()
        leagues = response.json()['preferences']
        leagues = sorted(leagues, key=lambda league: league['metaData']['entry']['seasonId'], reverse=True)
        result = []
        for league in leagues:
            league_api_id = league.get('id', '')
            entry = league.get('metaData', {}).get('entry', {})
            groups = entry.get('groups', [{}])
            group = groups[0] if groups else {}
            href = group.get('href', '')

            # Identify basketball leagues (NBA or WNBA) vs other sports (NFL, etc.)
            parts = league_api_id.split(':') if isinstance(league_api_id, str) else []
            sport_id = parts[2] if len(parts) >= 3 else None

            # Skip NFL football (sport_id == '1' or '/football/' in href)
            if sport_id == '1' or '/football/' in href:
                continue

            is_basketball = (
                sport_id in ('3', '5')
                or '/basketball/' in href
                or '/womens-basketball/' in href
                or (isinstance(league_api_id, str) and (league_api_id.startswith('fba:') or league_api_id.startswith('wfba:')))
            )
            if not is_basketball:
                continue

            season = entry.get('seasonId', 0)
            name = group.get('groupName') or entry.get('name') or 'Unnamed League'
            result.append({
                'id':     f"{league_api_id}{_LEAGUE_ID_YEAR_SEP}{season}",
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
        n_picks = max((len(team.roster) for team in league.teams), default=13)
        if n_picks == 0:
            n_picks = 13
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
        , player_id_lookup: PlatformPlayerLookup
    ) -> PlatformSelections:
        """Current rosters per team (ESPN supports only Season Mode). ESPN's key is the player's name, so the name
        fallback (PlatformPlayerLookup) tries the same name against every spelling the session knows."""

        league = self._make_league(config.league_id)
        team_by_id = {}
        team_by_name = {}
        for team in league.teams:
            team_by_id[str(team.team_id)] = team
            team_by_id[team.team_id] = team
            team_by_name[team.team_name] = team
            team_by_name[team.team_name.strip()] = team

        player_assignments: dict[str, list[int]] = {}
        injured_players: list[int] = []

        team_list = list(league.teams)
        for i, (team_name, team_id) in enumerate(config.teams_dict.items()):
            team = (
                team_by_id.get(str(team_id))
                or team_by_id.get(team_id)
                or team_by_name.get(team_name)
                or team_by_name.get(team_name.strip())
            )
            # Fallback by positional index if ID and name both failed to match. Whole ESPN teams have been seen to go
            # unmatched, for a reason not yet understood; the warning names the case so it can be found.
            if team is None and i < len(team_list):
                team = team_list[i]
                logger.warning('ESPN: team %r (id %r) matched no team by id or name; using ESPN team %d, %r (id %r), '
                               'by position', team_name, team_id, i, team.team_name, team.team_id)

            roster = []
            if team is not None and getattr(team, 'roster', None):
                for player in team.roster:
                    p_id = player_id_lookup.resolve_player_id(player.name, player.name)
                    roster.append(p_id)
                    is_injured = getattr(player, 'injured', False) or (
                        getattr(player, 'lineupSlot', '') in ('IR', 'IL')
                    )
                    if is_injured and p_id != RP_PLAYER_ID and p_id not in injured_players:
                        injured_players.append(p_id)

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
        , player_id_lookup: PlatformPlayerLookup
    ) -> Optional[PlatformSelections]:
        # ESPN auctions are not supported.
        return None

