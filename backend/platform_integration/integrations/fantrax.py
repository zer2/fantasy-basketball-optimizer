"""
Fantrax live integration, against Fantrax's official Beta API.

This used to call the fantraxapi package's private `_request`, which posts to
`https://www.fantrax.com/fxpa/req` -- the logged-in web app's own endpoint. That is gated on a
browser session, so any league that is not publicly readable (a mock draft, for one) answered
`Unauthorized: Not Logged in` and the connect route turned it into an opaque 502.

The Beta API at `https://www.fantrax.com/fxea/general/` serves the same leagues anonymously:
only `getLeagues` (listing a user's own leagues) wants the `userSecretId` from their profile
screen, and this integration takes the league id from the user instead, so it never needs one.
That removes the authentication problem rather than solving it, and drops the dependency on a
private endpoint that the previous docstring already flagged as liable to break.
"""

from __future__ import annotations

import logging
from typing import Optional

import requests

logger = logging.getLogger('fbbo')

from backend.platform_integration.base import (
    PlatformIntegration, LeagueShape, PlatformConfig, PlatformSelections,
)
from backend.platform_integration.helpers import deduplicate_team_names
from backend.player_identity import RP_PLAYER_ID


_BETA_API_ROOT = 'https://www.fantrax.com/fxea/general/'
_REQUEST_TIMEOUT_SECONDS = 30

# Roster-slot cap the optimizer respects (mirrors the Streamlit min(..., 16)).
_MAX_ROSTER_SLOTS = 16

# Which roster statuses mean "not on the active roster" in Season Mode. Fantrax reports these
# as words on the Beta API ('IR'), where the old private endpoint used a numeric statusId.
_INJURED_RESERVE_STATUSES = {'IR', 'INJURED_RESERVE'}

# The Beta API names these fields differently across sports and versions, so each is read by
# trying the spellings Fantrax is known to use. A roster item matching none of them is a shape
# change rather than a missing player, and read_roster_field says so instead of guessing.
_PLAYER_ID_FIELDS = ('id', 'scorerId', 'playerId')
_STATUS_FIELDS    = ('status', 'statusId', 'rosterStatus')


def to_unified_fantrax_id(raw_id: str) -> str:
    """The id as UNIFIED_PLAYER_TABLE.FANTRAX_ID spells it, which is wrapped in asterisks.

    The Beta API returns a bare id ('03e75'); the unified table stores '*03e75*'. Of this
    league's 1806 pool ids, none matched as-is and 947 matched once wrapped -- so without this
    every player falls through to the replacement-level id and no roster is ever recognised.
    Already-wrapped input is left alone, so the two spellings can never be double-wrapped.
    """
    return raw_id if raw_id.startswith('*') and raw_id.endswith('*') else f'*{raw_id}*'


def read_roster_field(item: dict, candidates: tuple[str, ...], what: str) -> Optional[str]:
    """The first of `candidates` this roster item carries, or None when it carries none.

    Returning None for a missing STATUS is meaningful -- plenty of leagues have no reserve
    concept -- so the caller decides whether absence is tolerable. For a player id it is not,
    and fetch_roster_player_id raises rather than silently dropping the player.
    """
    for field in candidates:
        if field in item:
            return str(item[field])
    return None


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
        # (UNIFIED_PLAYER_TABLE.FANTRAX_ID == the roster row's scorer id), not by
        # name — so no Fantrax name needs storing in the unified table.
        return 'FANTRAX_ID'

    # ── External dependency (wrapped so tests can substitute it) ───────────────

    def fetch_from_beta_api(self, endpoint: str, league_id: str, **params) -> dict | list:
        """One Beta API call, raising on anything that is not a usable JSON body.

        Fantrax answers an unknown league with a 200 and an error payload rather than a 4xx,
        so a bare status check is not enough to tell a real league from a typo.
        """
        response = requests.get(
            _BETA_API_ROOT + endpoint
            , params  = {'leagueId': league_id, **params}
            , headers = {'User-Agent': 'Mozilla/5.0'}
            , timeout = _REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and 'error' in payload:
            raise ValueError(f'Fantrax {endpoint} rejected league {league_id!r}: {payload["error"]}')
        return payload

    # ── Connection (list_divisions / fetch_league_shape + their fetchers) ──────

    def list_divisions(self, league_id: str) -> list[dict]:
        """Always empty: the Beta API does not report divisions.

        The old private endpoint read them off the standings tabs. Nothing in getLeagueInfo
        carries them, so a divisioned league now connects whole. Returning [] is the interface's
        own way of saying "this league has no divisions to choose between", which is what the
        connector UI already handles.
        """
        return []

    def fetch_league_shape(
        self
        , league_id: str
        , division_id: Optional[str]
    ) -> LeagueShape:
        """Return the league's team names, drafter count, and pick count.

        `division_id` is accepted and ignored — see list_divisions.
        """
        info = self.fetch_from_beta_api('getLeagueInfo', league_id)
        teams_dict = deduplicate_team_names([
            (team['name'], team['id']) for team in info['teamInfo'].values()
        ])
        team_names = list(teams_dict.keys())
        return LeagueShape(
            team_names       = team_names,
            n_drafters       = len(team_names),
            n_picks          = self._roster_slot_count(info),
            teams_dict       = teams_dict,
            is_auction_draft = self._is_auction_draft(info),
        )

    def _roster_slot_count(self, info: dict) -> int:
        """Every roster slot a team can fill, capped at the optimizer's limit.

        maxTotalPlayers counts active and reserve together, which is what the previous
        implementation arrived at by summing the status totals and dropping Injured Reserve.
        """
        return min(info['rosterInfo']['maxTotalPlayers'], _MAX_ROSTER_SLOTS)

    def _is_auction_draft(self, info: dict) -> Optional[bool]:
        """True for an auction, False for a snake, None when Fantrax does not say.

        None is a real answer here rather than a stand-in for False: LeagueShape uses it to
        mean "do not judge the user's chosen mode against this".
        """
        draft_type = info.get('draftSettings', {}).get('draftType') or info.get('draftType')
        if draft_type is None:
            return None
        return draft_type.lower() == 'auction'

    # ── Roster / draft state ──────────────────────────────────────────────────

    def _fetch_rosters_by_team_id(self, league_id: str) -> dict[str, dict]:
        """Every team's roster, keyed by Fantrax team id."""
        return self.fetch_from_beta_api('getTeamRosters', league_id)['rosters']

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

        rosters_by_team_id = self._fetch_rosters_by_team_id(config.league_id)
        exclude_injured = mode == 'Season Mode'
        injured_players: list[int] = []
        unmatched_count = 0

        player_assignments: dict[str, list[int]] = {}
        for team_name, team_id in config.teams_dict.items():
            roster: list[int] = []
            # A team with no entry has not drafted yet, which is the normal state of a draft
            # room that is still filling — not an error.
            items = rosters_by_team_id.get(team_id, {}).get('rosterItems', [])
            unreadable, matched = [], 0
            for item in items:
                scorer_id = read_roster_field(item, _PLAYER_ID_FIELDS, 'player id')
                if scorer_id is None:
                    # One unreadable entry among readable ones is an empty slot, which the old
                    # endpoint also skipped. EVERY entry unreadable is a shape change, and is
                    # raised below -- silently returning an empty roster would tell the app that
                    # nobody has been drafted, which is worse than failing.
                    unreadable.append(sorted(item))
                    continue
                matched += 1
                player_id = player_id_lookup.get(to_unified_fantrax_id(scorer_id), RP_PLAYER_ID)
                unmatched_count += player_id == RP_PLAYER_ID
                status = read_roster_field(item, _STATUS_FIELDS, 'status')
                if exclude_injured and status is not None and status.upper() in _INJURED_RESERVE_STATUSES:
                    injured_players.append(player_id)
                else:
                    roster.append(player_id)
            if items and matched == 0:
                raise ValueError(
                    f'No player id found in any of the {len(items)} Fantrax roster entries '
                    f'for {team_name}. '
                    f'Tried {_PLAYER_ID_FIELDS}; the entries carry {unreadable[0]}. '
                    f'The Beta API shape has changed.')
            if unreadable:
                logger.warning('Fantrax: skipped %d roster entry/entries with no player id for %s',
                               len(unreadable), team_name)
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
