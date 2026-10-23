"""Real NBA calendar inputs and session-local Points analysis context.

ESPN's public professional schedule/player views work independently of the user's
fantasy platform. Cache successful raw responses, never a session's resolved IDs.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import threading
import time
from zoneinfo import ZoneInfo

import requests

from backend.points.schedule import ScheduleCalendar, NBAGame, normalize_team, date_range
from backend.player_identity import RP_PLAYER_ID


class PointsDataUnavailable(RuntimeError):
    """Real schedule inputs are unavailable; do not substitute simulated games."""


_CACHE = {}
_LOCK = threading.Lock()
_TTL = 24 * 3600
_BASE = 'https://lm-api-reads.fantasy.espn.com/apis/v3/games'


def fetch_professional_data(sport: str, season: int):
    """Fetch one season's schedule and complete active player-team directory."""
    game = 'wfba' if sport == 'WNBA' else 'fba'
    key = (game, season)
    with _LOCK:
        entry = _CACHE.get(key)
        if entry and time.monotonic() - entry[0] < _TTL:
            return entry[1], entry[2]
        try:
            base = f'{_BASE}/{game}/seasons/{season}'
            schedule = requests.get(base, params={'view': 'proTeamSchedules_wl'}, timeout=20)
            schedule.raise_for_status()
            players = requests.get(base + '/players', params={'view': 'players_wl'},
                headers={'x-fantasy-filter': json.dumps({'filterActive': {'value': True}})}, timeout=20)
            players.raise_for_status()
            teams, directory = schedule.json()['settings']['proTeams'], players.json()
            if not teams or not isinstance(directory, list) or not directory:
                raise ValueError('Empty professional schedule or player directory')
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            raise PointsDataUnavailable('NBA schedule/player data could not be loaded. Retry analysis shortly.') from exc
        _CACHE[key] = (time.monotonic(), teams, directory)
        return teams, directory


def resolve_season(session):
    """The connected league's season wins over a projection source's season."""
    if session.platform_config and session.platform_config.platform == 'Retrieve from ESPN':
        from backend.platform_integration.integrations.espn import ESPNIntegration
        return ESPNIntegration._split_league_id(session.platform_config.league_id)[2]
    config = session.current_settings.get('points_config') or {}
    if config.get('season'):
        value = str(config['season'])
        return int(value.split('-')[0]) + (1 if '-' in value else 0)
    today = datetime.now(timezone.utc)
    return today.year + (today.month >= 7)


def calendar_from_espn(teams, directory, lookup, time_zone):
    """Resolve ESPN team/player IDs and deduplicate games using local game dates."""
    tz = ZoneInfo(time_zone)
    team_names = {int(t['id']): normalize_team(t['abbrev']) for t in teams if int(t['id']) != 0}
    games = {}
    for team in teams:
        for day_games in team.get('proGamesByScoringPeriod', {}).values():
            for game in day_games:
                if game.get('cancelled') or game.get('canceled'):
                    continue
                home, away = game.get('homeProTeamId'), game.get('awayProTeamId')
                if home not in team_names or away not in team_names or not game.get('date'):
                    continue
                date = datetime.fromtimestamp(game['date'] / 1000, timezone.utc).astimezone(tz).date().isoformat()
                games[str(game['id'])] = NBAGame(date, team_names[home], team_names[away], str(game['id']))
    if not games:
        raise PointsDataUnavailable('NBA schedule contains no usable games for this season.')
    player_teams = {}
    for player in directory:
        pid = lookup.resolve_player_id(player.get('fullName'), player.get('fullName'))
        team_id = player.get('proTeamId')
        if pid != RP_PLAYER_ID and team_id is not None:
            # An unsigned player is known to have no NBA games, unlike a missing mapping.
            if int(team_id) == 0:
                player_teams[pid] = 'FA'
            elif int(team_id) in team_names:
                player_teams[pid] = team_names[int(team_id)]
    return ScheduleCalendar.from_games(games.values(), player_teams=player_teams)


def resolve_points_calendar(session, roster_ids=()):
    """Resolve one calendar/window for waivers and trades; fixtures stay injectable."""
    config = session.current_settings.get('points_config') or {}
    zone = config.get('time_zone', 'America/New_York')
    window = (config.get('evaluation_window') or session.current_settings.get('evaluation_window')
              or session.current_settings.get('window') or session.current_settings.get('evaluation_dates'))
    calendar = getattr(session, 'schedule_calendar', None)
    live_key = getattr(session, 'points_calendar_key', None)
    # An explicitly injected calendar is the offline test/custom calendar contract.
    if calendar is None or live_key is not None:
        season = resolve_season(session)
        key = (session.current_settings.get('sport', 'NBA'), season, zone, session.player_pool_hash)
        if calendar is None or live_key != key or time.monotonic() - session.points_calendar_loaded_at >= _TTL:
            from backend.data_retrieval import get_unified_player_table, get_player_name_resolvers
            from backend.platform_integration.helpers import build_platform_player_id_lookup
            lookup = build_platform_player_id_lookup(session.player_registry, 'ESPN_NAME',
                get_unified_player_table(), get_player_name_resolvers())
            teams, directory = fetch_professional_data(key[0], season)
            calendar = calendar_from_espn(teams, directory, lookup, zone)
            session.schedule_calendar = calendar
            session.points_calendar_key = key
            session.points_calendar_loaded_at = time.monotonic()
    calendar.lineup_lock_mode = config.get('lineup_lock_mode', 'pregame_daily')
    session.agent.schedule = calendar
    if window:
        dates = calendar.normalize_window(tuple(window) if len(window) == 2 else window)
    else:
        start = datetime.now(ZoneInfo(zone)).date()
        dates = date_range(start.isoformat(), (start + timedelta(days=6)).isoformat())
    if session.points_calendar_key is not None:
        missing = sorted({int(pid) for pid in roster_ids if pid != RP_PLAYER_ID
                          and calendar.get_player_team(pid) is None})
        if missing:
            names = [session.player_registry[p].name if p in session.player_registry else str(p) for p in missing]
            raise PointsDataUnavailable('NBA team mapping is missing for rostered players: ' + ', '.join(names))
    return calendar, dates
