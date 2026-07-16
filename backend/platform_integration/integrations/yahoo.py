"""
Yahoo live integration — UNTESTED THEORETICAL PORT.

Ported from the Streamlit src/platform_integration/yahoo_integration.py (+ yahoo_helper.py),
framework-agnostic (no Streamlit). This file has NOT been run against a real Yahoo league
or OAuth app — it is our best guess at the right structure and is expected to need fixes
once exercised. Treat every method as provisional.

As of Phase 2 it IS wired into registry.py, and the routes + credential store exist — but none
of it has been exercised against a real Yahoo league or OAuth app, so every code path below is
unproven. Known risks to check first at E2E:

  - OAuth has never run. `build_auth_url` / `exchange_auth_code` port the Streamlit code-paste
    flow; the routes (GET /platforms/yahoo/auth-url, POST /platforms/yahoo/token) and the
    per-client token store exist, but the full handshake plus yfpy's token refresh is untested.
    Credentials live on the instance as `auth_dir` (the dir yfpy reads token.json/private.json
    from), supplied by get_integration(platform, {'auth_dir': ...}).

  - Name mapping. player_name_column is 'YAHOO_PLAYER_ID' (the UNIFIED_PLAYER_TABLE column for
    Yahoo is an id, not a name), so the prebuilt name_lookup maps Yahoo player ids -> canonical
    'Name (POS)'. VERIFY the YAHOO_PLAYER_ID dtype (int vs str) matches yfpy's player_id — a
    mismatch would silently map everyone to 'RP'.

  - n_picks is hard-coded to 13 (as in Streamlit — a known TODO there).

  - yfpy quirks (mirrored from Streamlit): it sometimes returns a list of dicts, errors when a
    user has no leagues, and shuts off API access after a mock draft. The try/except shims here
    copy that behavior; exact yfpy method names may differ by version (pinned yfpy==15.0.3).
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

import requests
from requests.auth import HTTPBasicAuth
from yfpy.query import YahooFantasySportsQuery

from backend.platform_integration.base import (
    PlatformIntegration, LeagueShape, PlatformConfig, PlatformSelections,
)
from backend.platform_integration.helpers import deduplicate_team_names


_GAME_CODE = 'nba'
_YAHOO_AUTH_URL = 'https://api.login.yahoo.com/oauth2/request_auth'
_YAHOO_TOKEN_URL = 'https://api.login.yahoo.com/oauth2/get_token'
_REDIRECT_URI = 'oob'           # out-of-band: the user pastes the code (manual flow)
_DEFAULT_N_PICKS = 13           # Streamlit hard-codes this (ZR there: fix)

# yfpy roster slot positions meaning "injured", excluded in Season Mode.
_INJURED_POSITIONS = {'IL', 'IL+'}


class YahooIntegration(PlatformIntegration):
    # Organized by workflow, like fantrax.py: metadata → auth → connection → roster/draft.

    def __init__(self, auth_dir: Optional[str] = None):
        # auth_dir holds yfpy's token.json + private.json (written by exchange_auth_code).
        # The registry spreads the creds bag into this param: get_integration(
        # 'Retrieve from Yahoo', {'auth_dir': ...}) -> YahooIntegration(auth_dir=...).
        self._auth_dir = auth_dir

    # ── Platform metadata ──────────────────────────────────────────────────────

    @property
    def available_modes(self) -> list[str]:
        return ['Draft Mode', 'Season Mode', 'Auction Mode']

    @property
    def description_string(self) -> str:
        return 'Retrieve from Yahoo'

    @property
    def player_name_column(self) -> str:
        # The UNIFIED_PLAYER_TABLE column for Yahoo is an id, not a name (see banner).
        return 'YAHOO_PLAYER_ID'

    # ── OAuth2 (manual code-paste flow; needs route + token-store wiring) ───────

    @staticmethod
    def build_auth_url(client_id: str) -> str:
        """The Yahoo page the user visits to obtain an authorization code to paste back."""
        return f'{_YAHOO_AUTH_URL}?client_id={client_id}&redirect_uri={_REDIRECT_URI}&response_type=code'

    @staticmethod
    def exchange_auth_code(
        client_id: str
        , client_secret: str
        , auth_code: str
        , auth_dir: str
    ) -> None:
        """Exchange a pasted authorization code for access/refresh tokens and write yfpy's
        token.json + private.json into auth_dir. Raises on a bad code / token error or a
        write failure; returning normally means success.

        Framework-agnostic port of the Streamlit get_yahoo_access_token dialog. The route
        layer supplies client id/secret (env) and the user-pasted code, and is responsible
        for choosing/persisting auth_dir per client.
        """
        response = requests.post(
            _YAHOO_TOKEN_URL,
            data={
                'redirect_uri': _REDIRECT_URI,
                'code':         auth_code,
                'grant_type':   'authorization_code',
            },
            auth=HTTPBasicAuth(client_id, client_secret),
        )
        response.raise_for_status()
        token_data = response.json()

        os.makedirs(auth_dir, exist_ok=True)
        with open(os.path.join(auth_dir, 'token.json'), 'w') as token_file:
            json.dump({
                'access_token':    token_data.get('access_token', ''),
                'consumer_key':    client_id,
                'consumer_secret': client_secret,
                'guid':            None,
                'refresh_token':   token_data.get('refresh_token', ''),
                'expires_in':      3600,
                'token_time':      time.time(),
                'token_type':      'bearer',
            }, token_file)
        with open(os.path.join(auth_dir, 'private.json'), 'w') as private_file:
            json.dump({'consumer_key': client_id, 'consumer_secret': client_secret}, private_file)

    def _make_query(self, league_id: str = '') -> YahooFantasySportsQuery:
        """Build a yfpy query bound to this integration's auth_dir."""
        if self._auth_dir is None:
            raise RuntimeError('Yahoo integration has no auth_dir; authenticate first (exchange_auth_code).')
        return YahooFantasySportsQuery(auth_dir=self._auth_dir, league_id=league_id, game_code=_GAME_CODE)

    # ── Connection ─────────────────────────────────────────────────────────────

    def list_leagues(self) -> list[dict]:
        """The user's NBA leagues as [{'id', 'name', 'season'}], reverse-chronological.

        Yahoo-specific (NOT in the ABC): the connect flow needs a league-pick step for
        auth-based platforms. Returns [] when the user has no leagues (yfpy errors there).
        """
        try:
            query = self._make_query(league_id='')
            leagues = query.get_user_leagues_by_game_key(game_key=_GAME_CODE)
            if leagues and isinstance(leagues[0], dict):
                # yfpy sometimes returns a list of dicts rather than League objects.
                leagues = [entry['league'] for entry in leagues]
            leagues = sorted(leagues, key=lambda league: league.season, reverse=True)
        except Exception:
            return []
        return [
            {'id': league.league_id, 'name': league.name.decode('UTF-8'), 'season': league.season}
            for league in leagues
        ]

    def list_divisions(self, league_id: str) -> list[dict]:
        """Yahoo has no divisions."""
        return []

    def _fetch_team_pairs(self, league_id: str) -> list[tuple[str, str]]:
        """Raw (team_name, team_id) pairs for the league."""
        query = self._make_query(league_id)
        teams = query.get_league_teams()
        if isinstance(teams, dict):
            teams = list(teams.values())
        return [(team.name.decode('UTF-8'), str(team.team_id)) for team in teams]

    def fetch_league_shape(
        self
        , league_id: str
        , division_id: Optional[str]
    ) -> LeagueShape:
        """Return the league's team names and drafter count (n_picks hard-coded to 13)."""
        teams_dict = deduplicate_team_names(self._fetch_team_pairs(league_id))
        team_names = list(teams_dict.keys())
        return LeagueShape(
            team_names = team_names,
            n_drafters = len(team_names),
            n_picks    = _DEFAULT_N_PICKS,
            teams_dict = teams_dict,
        )

    # ── Roster / draft state ───────────────────────────────────────────────────

    def get_draft_results(
        self
        , config: PlatformConfig
        , mode: str
        , name_lookup: dict[str, str]
    ) -> PlatformSelections:
        """Season Mode → current rosters; otherwise → the live draft board."""
        if mode == 'Season Mode':
            return self._get_season_rosters(config, name_lookup)
        return self._get_draft_board(config, name_lookup)

    def _get_season_rosters(
        self
        , config: PlatformConfig
        , name_lookup: dict[str, str]
    ) -> PlatformSelections:
        query = self._make_query(config.league_id)
        injured_players: list[str] = []
        player_assignments: dict[str, list[str]] = {}

        for team_name, team_id in config.teams_dict.items():
            roster = query.get_team_roster_by_week(team_id=int(team_id))
            players: list[str] = []
            for player in roster.players:
                position = player.selected_position.position
                canonical = name_lookup.get(player.player_id, 'RP')
                if position in _INJURED_POSITIONS:
                    injured_players.append(canonical)
                elif position is not None:
                    players.append(canonical)
            player_assignments[team_name] = players

        return PlatformSelections(
            player_assignments = player_assignments,
            status             = 'Success',
            injured_players    = injured_players,
        )

    def _get_draft_board(
        self
        , config: PlatformConfig
        , name_lookup: dict[str, str]
    ) -> PlatformSelections:
        query = self._make_query(config.league_id)
        try:
            draft_results = query.get_league_draft_results()
        except Exception:
            # yfpy errors before the draft starts (and after it shuts API access off).
            return PlatformSelections(player_assignments={}, status='Draft has not started yet', injured_players=[])

        if draft_results and getattr(draft_results[0], 'cost', None) is not None:
            return PlatformSelections(
                player_assignments={},
                status='This is an auction, not a draft! Change the game mode',
                injured_players=[],
            )

        player_assignments, _ = self._assignments_from_draft(draft_results, config, name_lookup)
        return PlatformSelections(
            player_assignments = player_assignments,
            status             = 'Success',
            injured_players    = [],
        )

    def get_auction_results(
        self
        , config: PlatformConfig
        , mode: str
        , name_lookup: dict[str, str]
    ) -> Optional[PlatformSelections]:
        """Auction board as team -> players plus per-player costs (costs[team][i] is what
        player_assignments[team][i] sold for). The route turns costs into remaining_cash."""
        query = self._make_query(config.league_id)
        try:
            draft_results = query.get_league_draft_results()
        except Exception:
            return PlatformSelections(player_assignments={}, status='Auction has not started', injured_players=[])

        if draft_results and getattr(draft_results[0], 'cost', None) is None:
            return PlatformSelections(
                player_assignments={},
                status='This is a draft, not an auction! Change the game mode',
                injured_players=[],
            )

        player_assignments, costs = self._assignments_from_draft(draft_results, config, name_lookup)
        return PlatformSelections(
            player_assignments = player_assignments,
            status             = 'Success',
            injured_players    = [],
            costs              = costs,
        )

    def _assignments_from_draft(
        self
        , draft_results: list
        , config: PlatformConfig
        , name_lookup: dict[str, str]
    ) -> tuple[dict[str, list[str]], dict[str, list]]:
        """Group draft/auction picks into ({team: [player, ...]}, {team: [cost, ...]}).

        cost entries are None for a draft and floats for an auction; the caller passes
        costs through only for auctions. Yahoo keys are 'game.player.<id>' /
        'game.l.<league>.t.<team>'; we take the trailing id. Unknown teams fall back to
        'Drafter <id>' (mirrors Streamlit).
        """
        team_name_by_id = {team_id: team_name for team_name, team_id in config.teams_dict.items()}
        player_assignments: dict[str, list[str]] = {team_name: [] for team_name in config.teams_dict}
        costs: dict[str, list] = {team_name: [] for team_name in config.teams_dict}

        for draft_obj in draft_results:
            if not draft_obj.player_key:
                continue
            player_id = int(draft_obj.player_key.split('.')[-1])
            team_id = draft_obj.team_key.split('.')[-1]
            team_name = team_name_by_id.get(team_id, f'Drafter {team_id}')
            player_assignments.setdefault(team_name, []).append(name_lookup.get(player_id, 'RP'))
            raw_cost = getattr(draft_obj, 'cost', None)
            costs.setdefault(team_name, []).append(float(raw_cost) if raw_cost is not None else None)

        return player_assignments, costs
