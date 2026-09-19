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
from yfpy.exceptions import YahooFantasySportsDataNotFound

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


def _pad_with_open_seats(joined_names: list[str], n_drafters: int) -> list[str]:
    """One name per seat: the teams that have joined, then a placeholder for each empty seat.

    Yahoo names unclaimed teams "Team N", so these are deliberately worded differently — a
    placeholder here means "nobody has taken this seat yet", and it is replaced by the real name
    on the next connect once someone does.
    """
    seats = list(joined_names)
    taken = set(seats)
    for seat_number in range(len(seats) + 1, n_drafters + 1):
        label = f'Open seat {seat_number}'
        collision = 2
        while label in taken:
            label = f'Open seat {seat_number} ({collision})'
            collision += 1
        seats.append(label)
        taken.add(label)
    return seats


class YahooIntegration(PlatformIntegration):
    # Organized by workflow, like fantrax.py: metadata → auth → connection → roster/draft.

    def __init__(self, auth_dir: Optional[str] = None):
        # auth_dir holds yfpy's token.json + private.json (written by exchange_auth_code).
        # The registry spreads the creds bag into this param: get_integration(
        # 'Retrieve from Yahoo', {'auth_dir': ...}) -> YahooIntegration(auth_dir=...).
        self._auth_dir = auth_dir
        # Resolved lazily and kept: the league -> season map costs one call per season, and
        # every request for a league needs it.
        self._games = None
        self._game_id_by_league: dict[str, Optional[int]] = {}

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
        """Build a yfpy query bound to this integration's auth_dir, for the league's own season.

        A Yahoo league is addressed as `<game key>.l.<league id>`, and the game key is the
        SEASON. Without one yfpy fills in whatever season is current, so every call for a league
        from any other season asks for a league key that does not exist -- Yahoo answers "there
        was a temporary problem with the server" and yfpy raises data-not-found, which reads
        from here as "the draft has not started". Measured: league 56576 under the current
        default returned nothing, and under its own key returned its ten teams.
        """
        if self._auth_dir is None:
            raise RuntimeError('Yahoo integration has no auth_dir; authenticate first (exchange_auth_code).')
        return YahooFantasySportsQuery(auth_dir=self._auth_dir, league_id=league_id,
                                       game_code=_GAME_CODE, game_id=self._game_id_for(league_id))

    def _user_games(self) -> list:
        """The seasons this account has played, newest first, fetched once per integration."""
        if self._games is None:
            games = self._bare_query().get_user_games()
            self._games = sorted(games, key=lambda game: int(game.season), reverse=True)
        return self._games

    def _leagues_in(self, game) -> list:
        """The account's leagues in one season. `get_user_leagues_by_game_key` needs the NUMERIC
        key -- passing the game CODE ('nba') matches nothing and returns no data."""
        try:
            leagues = self._bare_query().get_user_leagues_by_game_key(game_key=str(game.game_key))
        except YahooFantasySportsDataNotFound:
            return []
        if leagues and isinstance(leagues[0], dict):
            # yfpy sometimes returns a list of dicts rather than League objects.
            leagues = [entry['league'] for entry in leagues]
        return leagues

    def _game_id_for(self, league_id: str) -> Optional[int]:
        """Which season's game key holds this league, or None when it is not one of the user's.

        None is the right answer for a league the account cannot see in its own list -- a mock
        draft room, most of all, which Yahoo does not publish as a league. yfpy then falls back
        to the current season, which is where a mock lives anyway.
        """
        if not league_id:
            return None
        if league_id not in self._game_id_by_league:
            for game in self._user_games():
                for league in self._leagues_in(game):
                    self._game_id_by_league.setdefault(str(league.league_id), int(game.game_id))
            self._game_id_by_league.setdefault(league_id, None)
        return self._game_id_by_league[league_id]

    def _bare_query(self) -> YahooFantasySportsQuery:
        """A query for account-level calls, which belong to no league and so need no season."""
        return YahooFantasySportsQuery(auth_dir=self._auth_dir, league_id='', game_code=_GAME_CODE)

    # ── Connection ─────────────────────────────────────────────────────────────

    def list_leagues(self) -> list[dict]:
        """The user's NBA leagues as [{'id', 'name', 'season'}], reverse-chronological.

        Yahoo-specific (NOT in the ABC): the connect flow needs a league-pick step for
        auth-based platforms.

        Every season the account has played, newest first, because a league is only addressable
        under its own season's game key and the picker is where that key is learned. Returns []
        when the user genuinely has no leagues, which yfpy reports as data-not-found. Anything
        else -- an expired token, a rate limit, a network drop -- is raised: an empty dropdown
        saying "(no leagues found)" is the same picture for a new account and a dead token, and
        only the second is fixed by reconnecting.
        """
        listed = []
        for game in self._user_games():
            for league in self._leagues_in(game):
                self._game_id_by_league.setdefault(str(league.league_id), int(game.game_id))
                name = league.name.decode('UTF-8') if isinstance(league.name, bytes) else league.name
                listed.append({'id': league.league_id, 'name': name, 'season': league.season})
        return listed

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

    def _read_league_settings(self, league_id: str):
        """The league's own settings object, or None when Yahoo will not give it up.

        Fetched once because several answers come out of it — how many seats the room holds and
        whether it drafts by auction — and each is a round trip to Yahoo otherwise. Returns None
        rather than raising: a league that cannot describe itself is still worth connecting to on
        whatever the team list says, and the callers below each say what they do without it.
        """
        try:
            return self._make_query(league_id).get_league_settings()
        except Exception as error:
            _logger.warning('Yahoo league %s did not return its settings (%s); the drafter count '
                            'falls back to the teams that have joined and the draft type is '
                            'unknown', league_id, error)
            return None

    @staticmethod
    def _read_seat_count(settings, joined: int) -> int:
        """How many seats the room was created with, however many are occupied right now.

        `max_teams` is fixed when the league is created; the team list only reports seats that
        have been taken. Connecting to a draft room before it fills therefore used to describe a
        four-team — or one-team — league, and a one-team league has no opponents at all, which
        crashed the pipeline rebuild several frames deep. Measured on a real room: four teams
        joined, `max_teams` 14, and thirteen teams present three minutes later.
        """
        if settings is None:
            return joined
        try:
            # A league can never have fewer seats than teams sitting in them; if Yahoo says
            # otherwise, the teams are the thing we can see.
            return max(int(settings.max_teams), joined)
        except (TypeError, ValueError) as error:
            _logger.warning('Yahoo reported an unreadable max_teams (%s); falling back to the '
                            '%d team(s) that have joined', error, joined)
            return joined

    @staticmethod
    def _read_is_auction(settings) -> Optional[bool]:
        """Whether the league drafts by auction, or None when Yahoo will not say.

        Known before a single pick exists, which is the point: the board itself only reveals the
        draft type once picks carry (or lack) a cost, by which time the user has been working in
        the wrong mode for the whole draft.
        """
        if settings is None:
            return None
        try:
            return bool(int(settings.is_auction_draft))
        except (TypeError, ValueError) as error:
            _logger.warning('Yahoo reported an unreadable is_auction_draft (%s); the draft type '
                            'stays unknown', error)
            return None

    def fetch_league_shape(
        self
        , league_id: str
        , division_id: Optional[str]
    ) -> LeagueShape:
        """The league's seats, drafter count and roster size (n_picks still hard-coded to 13).

        team_names covers EVERY seat, padding the joined teams with placeholders; teams_dict
        keeps only the teams that exist on Yahoo's side, since it is the platform-id map.
        """
        teams_dict = deduplicate_team_names(self._fetch_team_pairs(league_id))
        joined_names = list(teams_dict.keys())
        settings   = self._read_league_settings(league_id)
        n_drafters = self._read_seat_count(settings, len(joined_names))
        return LeagueShape(
            team_names        = _pad_with_open_seats(joined_names, n_drafters),
            n_drafters        = n_drafters,
            n_picks           = _DEFAULT_N_PICKS,
            teams_dict        = teams_dict,
            is_auction_draft  = self._read_is_auction(settings),
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
        # Every seat starts empty; the joined teams then fill theirs. A season league is always
        # full, so this differs from teams_dict only in the half-filled draft-room case.
        player_assignments: dict[str, list[int]] = {seat: [] for seat in config.seat_names}

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
            player_assignments = {seat: [] for seat in config.seat_names},
            status             = status,
            injured_players    = [],
            costs              = {seat: [] for seat in config.seat_names} if with_costs else None,
        )

    def _get_draft_board(
        self
        , config: PlatformConfig
        , player_id_lookup: dict[str, int]
    ) -> PlatformSelections:
        query = self._make_query(config.league_id)
        try:
            draft_results = query.get_league_draft_results()
        except YahooFantasySportsDataNotFound:
            # Yahoo answered, and what it said was that there are no draft results. That is what
            # a room looks like before the draft starts, so an empty board is the truth.
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
        except YahooFantasySportsDataNotFound:
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
        # Seeded from the SEATS, not the joined teams: an unfilled seat is a drafter with an empty
        # roster, not a team missing from the league. Picks are still keyed back to real names
        # through teams_dict, so a seat that filled after connect lands under the 'Drafter <id>'
        # fallback below until the next connect picks up its name.
        player_assignments: dict[str, list[int]] = {seat: [] for seat in config.seat_names}
        costs: dict[str, list] = {seat: [] for seat in config.seat_names}

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
