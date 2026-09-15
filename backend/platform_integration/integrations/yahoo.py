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

  - Player mapping. player_name_column is 'YAHOO_PLAYER_ID' (the UNIFIED_PLAYER_TABLE column for
    Yahoo is an id, not a name), so the prebuilt player_id_lookup maps Yahoo player ids -> session
    player ids. VERIFY the YAHOO_PLAYER_ID dtype (int vs str) matches yfpy's player_id — a
    mismatch would silently map everyone to the RP fallback.

  - n_picks is hard-coded to 13 (as in Streamlit — a known TODO there).

  - yfpy quirks (mirrored from Streamlit): it sometimes returns a list of dicts, errors when a
    user has no leagues, and shuts off API access after a mock draft. The try/except shims here
    copy that behavior; exact yfpy method names may differ by version (pinned yfpy==15.0.3).
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

import requests
from requests.auth import HTTPBasicAuth
from yfpy.query import YahooFantasySportsQuery

from backend.platform_integration.base import (
    PlatformIntegration, LeagueShape, PlatformConfig, PlatformSelections,
)
from backend.infra.secret_config import get_secret
from backend.platform_integration.helpers import deduplicate_team_names
from backend.player_identity import RP_PLAYER_ID


_logger = logging.getLogger('fbbo.yahoo')

_GAME_CODE = 'nba'
_YAHOO_AUTH_URL = 'https://api.login.yahoo.com/oauth2/request_auth'
_YAHOO_TOKEN_URL = 'https://api.login.yahoo.com/oauth2/get_token'
# Out-of-band ('oob') is Yahoo's documented redirect for applications that cannot receive a
# callback, and it is what this app WANTS: consent ends on a Yahoo page showing a short code to
# paste back, rather than dumping the user on a dead callback URL to dig a long code out of the
# address bar. It is the deliberate default in every environment, not a fallback for missing
# configuration. Setting YAHOO_REDIRECT_URI overrides it with a real callback, which requires that
# exact string to be registered on the Yahoo app -- Yahoo matches it at both the authorize and the
# token step, which is why it is read once here rather than per call.
_REDIRECT_URI = get_secret('YAHOO_REDIRECT_URI') or 'oob'
_DEFAULT_N_PICKS = 13           # Streamlit hard-codes this (ZR there: fix)

# What the authorization probe reads: the cheapest fantasy endpoint there is, so a grant that
# cannot read fantasy data says so at authorization time instead of at connect time.
_FANTASY_PROBE_URL = 'https://fantasysports.yahooapis.com/fantasy/v2/game/nba/metadata?format=json'
# Read only when that probe fails, to say how wide the refusal is. The two fantasy reads are the
# ones the integration itself depends on -- the league list the connect flow opens with, and the
# bare game resource underneath every other call -- and an application Yahoo has not entitled is
# refused on both alike, where one refusing while the other answers would indict the probe rather
# than the grant. The non-fantasy read is the control: Yahoo honouring the same bearer token there
# proves the handshake produced a working token, leaving the fantasy entitlement as the only thing
# missing. It is only evidence in that direction, since this app never requests the OpenID scopes
# that endpoint wants -- a refusal there says nothing either way.
_CORROBORATION_URLS = {
    'fantasy_user_leagues': 'https://fantasysports.yahooapis.com/fantasy/v2/users;use_login=1/games?format=json',
    'fantasy_game_nba':     'https://fantasysports.yahooapis.com/fantasy/v2/game/nba?format=json',
    'non_fantasy_userinfo': 'https://api.login.yahoo.com/openid/v1/userinfo',
}

# yfpy roster slot positions meaning "injured", excluded in Season Mode.
_INJURED_POSITIONS = {'IL', 'IL+'}


def _read_corroborating_endpoints(access_token: str) -> dict[str, str]:
    """{label: HTTP status or error} for each corroborating endpoint, for the failure log only.

    Never raises: this runs while reporting a failure, so a second failure here must not replace
    the reason being reported.
    """
    statuses: dict[str, str] = {}
    for label, url in _CORROBORATION_URLS.items():
        try:
            response = requests.get(
                url,
                headers={'Authorization': f'Bearer {access_token}'},
                timeout=20,
            )
            # The WWW-Authenticate challenge is the part that says WHY a bearer token was refused:
            # 'insufficient_scope' means Yahoo accepted the token and withheld a permission, while
            # 'invalid_token' means it rejected the token itself. The status code alone conflates
            # the two. It is a standard error header, and carries no secret.
            challenge = response.headers.get('WWW-Authenticate', '')
            statuses[label] = f'{response.status_code} {challenge}'.strip()
        except requests.RequestException as request_error:
            statuses[label] = f'{type(request_error).__name__}'
    return statuses


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
        """The Yahoo page the user visits to obtain an authorization code to paste back.

        No scope parameter: Yahoo grants what the application's own registration was approved for,
        and yahoo_oauth -- the library yfpy drives this handshake with, and the reference every
        working Python integration uses -- asks for none either. Requesting one explicitly can only
        differ from the path known to work.
        """
        return (f'{_YAHOO_AUTH_URL}?client_id={client_id}&redirect_uri={_REDIRECT_URI}'
                f'&response_type=code')

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
        if response.status_code != 200:
            # Yahoo's own reason, rather than a bare "400 Client Error" that says nothing about
            # which of the several possible faults this is. The reused-code case gets named
            # explicitly because its consequence is invisible and severe: per RFC 6749 4.1.2 an
            # authorization server SHOULD revoke every token already issued from a code that is
            # presented twice, so a duplicate exchange does not merely fail -- it kills the
            # working token the FIRST exchange just stored, and every later call then reports an
            # invalid cookie with nothing to connect it back to this moment.
            reason = response.text[:400]
            reused = 'invalid_grant' in reason or 'invalid_code' in reason
            raise RuntimeError(
                f'Yahoo rejected the authorization code (HTTP {response.status_code}): {reason}'
                + ('. This code has already been used. A code works once: request a fresh one '
                   'from the authorization page -- and note that reusing one also revokes the '
                   'token the first exchange produced, so you must authorize again.'
                   if reused else '')
            )
        token_data = response.json()

        # Verify the grant can actually read fantasy data before persisting it. A scope-less grant
        # exchanges and refreshes cleanly and only fails later, which surfaced as an empty league
        # list and a 502 on connect -- far from the cause. Probing the cheapest fantasy endpoint
        # turns that into an error at the moment of authorization, quoting Yahoo's own reason.
        # Tested by use rather than by inspecting the response shape, which Yahoo has changed before.
        access_token = token_data.get('access_token', '')
        probe = requests.get(
            _FANTASY_PROBE_URL,
            headers={'Authorization': f'Bearer {access_token}'},
            timeout=20,
        )
        if probe.status_code != 200:
            # The redirect URI in force is logged because it is the one input to the handshake
            # that varies by environment, and a mismatch with what the Yahoo app has registered
            # fails here rather than at the exchange. The token keys go with it: they say whether
            # Yahoo returned anything unusual alongside the access token. The corroborating reads
            # separate an unentitled application, which is refused everywhere, from a probe that
            # happens to have picked an endpoint with rules of its own.
            _logger.error('Yahoo authorization probe failed (%s); token response carried keys %s, '
                          'redirect_uri %s, corroborating endpoints %s',
                          probe.status_code, sorted(token_data.keys()), _REDIRECT_URI,
                          _read_corroborating_endpoints(access_token))
            raise RuntimeError(
                f'Yahoo accepted the code but the token cannot read fantasy data '
                f"(HTTP {probe.status_code}: {probe.text[:600]}). Either the application's "
                f'Fantasy Sports API access is not active yet (approval pending or incomplete) '
                f"or the grant lacks the fantasy scope -- Yahoo's own message above is the "
                f'authoritative reason. Nothing was stored; re-authorize freshly once access is active.'
            )
        # Yahoo has deprecated xoauth_yahoo_guid on this flow (their docs point at OpenID Connect's
        # id_token for a GUID), so its absence says nothing about the grant and must not be read as
        # a broken handshake. Carried through when present because yahoo_oauth reads it back out of
        # token.json, with the same None default it applies to a token response that omits it.
        guid = token_data.get('xoauth_yahoo_guid')

        os.makedirs(auth_dir, exist_ok=True)
        with open(os.path.join(auth_dir, 'token.json'), 'w') as token_file:
            json.dump({
                'access_token':    token_data.get('access_token', ''),
                'consumer_key':    client_id,
                'consumer_secret': client_secret,
                'guid':            guid,
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
        , player_id_lookup: dict[str, int]
    ) -> PlatformSelections:
        """Season Mode → current rosters; otherwise → the live draft board."""
        if mode == 'Season Mode':
            return self._get_season_rosters(config, player_id_lookup)
        return self._get_draft_board(config, player_id_lookup)

    def _get_season_rosters(
        self
        , config: PlatformConfig
        , player_id_lookup: dict[str, int]
    ) -> PlatformSelections:
        query = self._make_query(config.league_id)
        injured_players: list[int] = []
        player_assignments: dict[str, list[int]] = {}

        for team_name, team_id in config.teams_dict.items():
            roster = query.get_team_roster_by_week(team_id=int(team_id))
            players: list[int] = []
            for player in roster.players:
                position = player.selected_position.position
                canonical = player_id_lookup.get(player.player_id, RP_PLAYER_ID)
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

    def _empty_board(
        self
        , config: PlatformConfig
        , status: str
        , with_costs: bool = False
    ) -> PlatformSelections:
        """Every team in the league, nobody drafted yet.

        An empty board is a real board, not the absence of one. Before a draft starts the league
        still has its teams, and the app must be able to evaluate against them -- pre-draft is
        exactly when a ranking is most wanted. Returning an empty dict instead put a board on the
        wire with no teams in it at all, which fails the moment anything looks up the seat being
        evaluated for: a 500 out of the H-score solve, reading 'Evaluation failed.'

        `with_costs` seeds the per-team cost lists an auction needs, so remaining_cash comes back
        as the full budget for everyone rather than None (which the evaluate route rejects for an
        auction league).
        """
        return PlatformSelections(
            player_assignments = {team_name: [] for team_name in config.teams_dict},
            status             = status,
            injured_players    = [],
            costs              = {team_name: [] for team_name in config.teams_dict} if with_costs
                                 else None,
        )

    def _get_draft_board(
        self
        , config: PlatformConfig
        , player_id_lookup: dict[str, int]
    ) -> PlatformSelections:
        query = self._make_query(config.league_id)
        try:
            draft_results = query.get_league_draft_results()
        except Exception:
            # yfpy errors before the draft starts (and after it shuts API access off).
            return self._empty_board(config, 'Draft has not started yet')

        if draft_results and getattr(draft_results[0], 'cost', None) is not None:
            return self._empty_board(
                config, 'This is an auction, not a draft! Change the game mode')

        player_assignments, _ = self._assignments_from_draft(draft_results, config, player_id_lookup)
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
        """Auction board as team -> players plus per-player costs (costs[team][i] is what
        player_assignments[team][i] sold for). The route turns costs into remaining_cash."""
        query = self._make_query(config.league_id)
        try:
            draft_results = query.get_league_draft_results()
        except Exception:
            return self._empty_board(config, 'Auction has not started', with_costs=True)

        if draft_results and getattr(draft_results[0], 'cost', None) is None:
            return self._empty_board(
                config, 'This is a draft, not an auction! Change the game mode', with_costs=True)

        player_assignments, costs = self._assignments_from_draft(draft_results, config, player_id_lookup)
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
        , player_id_lookup: dict[str, int]
    ) -> tuple[dict[str, list[int]], dict[str, list]]:
        """Group draft/auction picks into ({team: [player, ...]}, {team: [cost, ...]}).

        cost entries are None for a draft and floats for an auction; the caller passes
        costs through only for auctions. Yahoo keys are 'game.player.<id>' /
        'game.l.<league>.t.<team>'; we take the trailing id. Unknown teams fall back to
        'Drafter <id>' (mirrors Streamlit).
        """
        team_name_by_id = {team_id: team_name for team_name, team_id in config.teams_dict.items()}
        player_assignments: dict[str, list[int]] = {team_name: [] for team_name in config.teams_dict}
        costs: dict[str, list] = {team_name: [] for team_name in config.teams_dict}

        for draft_obj in draft_results:
            if not draft_obj.player_key:
                continue
            player_id = int(draft_obj.player_key.split('.')[-1])
            team_id = draft_obj.team_key.split('.')[-1]
            team_name = team_name_by_id.get(team_id, f'Drafter {team_id}')
            player_assignments.setdefault(team_name, []).append(player_id_lookup.get(player_id, RP_PLAYER_ID))
            raw_cost = getattr(draft_obj, 'cost', None)
            costs.setdefault(team_name, []).append(float(raw_cost) if raw_cost is not None else None)

        return player_assignments, costs
