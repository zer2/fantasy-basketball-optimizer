"""
Schedule Engine and Multi-Day Calendar for Points Mode.

Provides ScheduleCalendar for NBA team schedules, player game lookups, multi-day
evaluation window handling, back-to-back detection, and deterministic offline
synthetic schedule generation with zero network dependencies.
"""

from __future__ import annotations

import datetime
import json
import logging
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterable, Optional, Union

import pandas as pd

logger = logging.getLogger('fbbo.points.schedule')

# ── Canonical NBA Teams ────────────────────────────────────────────────────────

NBA_TEAMS: tuple[str, ...] = (
    'ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW',
    'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK',
    'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS',
)

TEAM_ALIASES: dict[str, str] = {
    'BRK': 'BKN',
    'CHO': 'CHA',
    'PHO': 'PHX',
    'GS':  'GSW',
    'NO':  'NOP',
    'NOH': 'NOP',
    'NOK': 'NOP',
    'NY':  'NYK',
    'SA':  'SAS',
    'UTAH': 'UTA',
    'WSH': 'WAS',
}


def normalize_team(team: str) -> str:
    """Normalize NBA team abbreviation to canonical 3-letter code."""
    cleaned = team.strip().upper()
    return TEAM_ALIASES.get(cleaned, cleaned)


def normalize_date(date_val: Union[str, datetime.date, datetime.datetime]) -> str:
    """Convert input date into ISO-8601 'YYYY-MM-DD' string."""
    if isinstance(date_val, (datetime.date, datetime.datetime)):
        return date_val.strftime('%Y-%m-%d')
    return str(date_val).strip()[:10]


def date_range(start_date: str, end_date: str) -> list[str]:
    """Return sorted list of ISO date strings from start_date to end_date inclusive."""
    start_dt = datetime.date.fromisoformat(normalize_date(start_date))
    end_dt = datetime.date.fromisoformat(normalize_date(end_date))
    if start_dt > end_dt:
        return []
    days = (end_dt - start_dt).days + 1
    return [(start_dt + datetime.timedelta(days=i)).isoformat() for i in range(days)]


@dataclass(frozen=True)
class NBAGame:
    """One scheduled NBA matchup."""
    date: str
    home_team: str
    away_team: str
    game_id: Optional[str] = None


# ── Synthetic Schedule Generator ──────────────────────────────────────────────

def generate_synthetic_nba_schedule(
    start_date: str = '2024-10-21',
    weeks: int = 25,
    seed: int = 42,
) -> tuple[dict[str, list[str]], list[NBAGame]]:
    """Generate a realistic, deterministic NBA schedule.

    Adheres to NBA scheduling constraints:
    - 30 teams, ~3.4 games per team per week.
    - No 3-in-3 nights.
    - Realistic heavy (Wed/Fri) and light (Tue/Thu) slates.
    - Pure in-memory deterministic generation.
    """
    rng = random.Random(seed)
    start_dt = datetime.date.fromisoformat(normalize_date(start_date))
    team_games: dict[str, set[str]] = {t: set() for t in NBA_TEAMS}
    games_list: list[NBAGame] = []

    # Day-of-week target game counts (0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun)
    day_targets = [7, 5, 11, 4, 11, 6, 7]

    for w in range(weeks):
        for dow in range(7):
            curr_dt = start_dt + datetime.timedelta(days=w * 7 + dow)
            dt_str = curr_dt.isoformat()
            yesterday_str = (curr_dt - datetime.timedelta(days=1)).isoformat()
            day_before_str = (curr_dt - datetime.timedelta(days=2)).isoformat()

            candidates = []
            for t in NBA_TEAMS:
                # Disallow 3 consecutive nights
                if yesterday_str in team_games[t] and day_before_str in team_games[t]:
                    continue
                candidates.append(t)

            rng.shuffle(candidates)
            candidates.sort(key=lambda t: len(team_games[t]))

            target_g = day_targets[dow]
            num_teams = min(len(candidates) - (len(candidates) % 2), 2 * target_g)
            selected = candidates[:num_teams]

            for i in range(0, num_teams, 2):
                t1, t2 = selected[i], selected[i + 1]
                team_games[t1].add(dt_str)
                team_games[t2].add(dt_str)
                games_list.append(NBAGame(date=dt_str, home_team=t1, away_team=t2))

    result_schedules = {t: sorted(team_games[t]) for t in NBA_TEAMS}
    return result_schedules, games_list


# ── ScheduleCalendar Class ───────────────────────────────────────────────────

class ScheduleCalendar:
    """Schedule and calendar service for Points Mode in-season optimization."""

    def __init__(
        self,
        team_schedules: Optional[dict[str, Iterable[str]]] = None,
        player_teams: Optional[dict[int, str]] = None,
        games: Optional[Iterable[NBAGame]] = None,
        season: Optional[str] = None,
    ) -> None:
        self._season = season
        self._player_teams: dict[int, str] = {}
        if player_teams:
            self.register_player_teams(player_teams)

        self._team_schedules: dict[str, set[str]] = {t: set() for t in NBA_TEAMS}
        self._date_to_teams: dict[str, set[str]] = {}
        self._games: list[NBAGame] = []

        if team_schedules is None and games is None:
            # Deterministic offline fallback
            syn_sched, syn_games = generate_synthetic_nba_schedule()
            self._load_from_dict(syn_sched)
            self._games = syn_games
        else:
            if team_schedules is not None:
                self._load_from_dict(team_schedules)
            if games is not None:
                for g in games:
                    self._games.append(g)
                    h, a, d = normalize_team(g.home_team), normalize_team(g.away_team), normalize_date(g.date)
                    self._team_schedules.setdefault(h, set()).add(d)
                    self._team_schedules.setdefault(a, set()).add(d)
                    self._date_to_teams.setdefault(d, set()).update([h, a])

    def _load_from_dict(self, schedules: dict[str, Iterable[str]]) -> None:
        for raw_team, dates in schedules.items():
            team = normalize_team(raw_team)
            date_set = {normalize_date(d) for d in dates}
            self._team_schedules.setdefault(team, set()).update(date_set)
            for d in date_set:
                self._date_to_teams.setdefault(d, set()).add(team)

    # ── Core Interface Contracts (PROJECT.md) ─────────────────────────────────

    def get_team_game_dates(self, team: str, start_date: str, end_date: str) -> list[str]:
        """Return sorted list of ISO date strings where team plays within [start_date, end_date]."""
        norm_team = normalize_team(team)
        if norm_team not in self._team_schedules:
            return []
        s_date = normalize_date(start_date)
        e_date = normalize_date(end_date)
        if s_date > e_date:
            return []
        all_dates = self._team_schedules[norm_team]
        return sorted(d for d in all_dates if s_date <= d <= e_date)

    def get_player_game_dates(self, player_id: int, start_date: str, end_date: str) -> list[str]:
        """Return sorted game dates for a player within [start_date, end_date]."""
        team = self.get_player_team(player_id)
        if not team:
            return []
        return self.get_team_game_dates(team, start_date, end_date)

    # ── High-Efficiency Lineup & Simulation Helpers ───────────────────────────

    def get_team_game_count(self, team: str, start_date: str, end_date: str) -> int:
        """Count of games for a team in [start_date, end_date]."""
        return len(self.get_team_game_dates(team, start_date, end_date))

    def get_player_game_count(self, player_id: int, start_date: str, end_date: str) -> int:
        """Count of games for a player in [start_date, end_date]."""
        return len(self.get_player_game_dates(player_id, start_date, end_date))

    def is_team_playing_on_date(self, team: str, date_str: str) -> bool:
        """Return True if team plays on date_str."""
        norm_team = normalize_team(team)
        d = normalize_date(date_str)
        return d in self._team_schedules.get(norm_team, set())

    def is_player_playing_on_date(self, player_id: int, date_str: str) -> bool:
        """Return True if player has a game on date_str."""
        team = self.get_player_team(player_id)
        if not team:
            return False
        return self.is_team_playing_on_date(team, date_str)

    def get_players_playing_on_date(self, player_ids: Iterable[int], date_str: str) -> set[int]:
        """Filter player_ids to the subset playing on date_str."""
        d = normalize_date(date_str)
        teams_today = self._date_to_teams.get(d, set())
        if not teams_today:
            return set()
        return {
            int(pid) for pid in player_ids
            if self._player_teams.get(int(pid)) in teams_today
        }

    def get_roster_games_in_window(
        self,
        player_ids: Iterable[int],
        start_date: str,
        end_date: str,
    ) -> dict[int, int]:
        """Return {player_id: count_of_games} for weekly lineup optimization."""
        return {
            int(pid): self.get_player_game_count(int(pid), start_date, end_date)
            for pid in player_ids
        }

    def get_teams_playing_on_date(self, date_str: str) -> set[str]:
        """Set of NBA teams playing on date_str."""
        return set(self._date_to_teams.get(normalize_date(date_str), set()))

    # ── Evaluation Window Handling ────────────────────────────────────────────

    def normalize_window(
        self,
        window: Union[list[str], tuple[str, str], str, None],
        default_days: int = 7,
    ) -> list[str]:
        """Normalize various window specs to a sorted list of unique ISO date strings."""
        if window is None:
            all_dates = sorted(self._date_to_teams.keys())
            start = all_dates[0] if all_dates else datetime.date.today().isoformat()
            start_dt = datetime.date.fromisoformat(start)
            return [(start_dt + datetime.timedelta(days=i)).isoformat() for i in range(default_days)]
        if isinstance(window, tuple) and len(window) == 2:
            return date_range(window[0], window[1])
        if isinstance(window, str):
            return [normalize_date(window)]
        if isinstance(window, (list, set)):
            return sorted({normalize_date(d) for d in window})
        raise ValueError(f"Unsupported evaluation window specification: {window}")

    def get_matchup_week_dates(
        self,
        week_number: int,
        season_start_date: str = '2024-10-21',
    ) -> list[str]:
        """Return 7-day Monday-Sunday date list for given matchup week (1-indexed)."""
        if week_number < 1:
            raise ValueError(f"week_number must be >= 1, got {week_number}")
        start_dt = datetime.date.fromisoformat(normalize_date(season_start_date))
        # Ensure start is aligned to Monday
        monday = start_dt - datetime.timedelta(days=start_dt.weekday())
        week_start = monday + datetime.timedelta(weeks=week_number - 1)
        return [(week_start + datetime.timedelta(days=i)).isoformat() for i in range(7)]

    def get_dates_in_window(self, start_date: str, end_date: str) -> list[str]:
        """Return all contiguous dates in [start_date, end_date]."""
        return date_range(start_date, end_date)

    # ── Advanced Streamer & Density Analytics (R1/R4) ─────────────────────────

    def get_team_back_to_backs(
        self,
        team: str,
        start_date: str,
        end_date: str,
    ) -> list[tuple[str, str]]:
        """Return list of consecutive game dates (day1, day2) for team in window."""
        dates = self.get_team_game_dates(team, start_date, end_date)
        b2b: list[tuple[str, str]] = []
        d_objs = [datetime.date.fromisoformat(d) for d in dates]
        for i in range(len(d_objs) - 1):
            if (d_objs[i + 1] - d_objs[i]).days == 1:
                b2b.append((dates[i], dates[i + 1]))
        return b2b

    def get_player_back_to_backs(
        self,
        player_id: int,
        start_date: str,
        end_date: str,
    ) -> list[tuple[str, str]]:
        """Return back-to-backs for player's team."""
        team = self.get_player_team(player_id)
        if not team:
            return []
        return self.get_team_back_to_backs(team, start_date, end_date)

    def get_games_density(
        self,
        team: str,
        start_date: str,
        end_date: str,
        span_days: int = 4,
    ) -> int:
        """Maximum games played in any rolling span_days window (e.g. 3-in-4 nights)."""
        dates = self.get_team_game_dates(team, start_date, end_date)
        if not dates:
            return 0
        d_objs = [datetime.date.fromisoformat(d) for d in dates]
        max_density = 0
        for i, d1 in enumerate(d_objs):
            count = sum(1 for d2 in d_objs[i:] if (d2 - d1).days < span_days)
            if count > max_density:
                max_density = count
        return max_density

    def get_light_slate_dates(
        self,
        window: Union[list[str], tuple[str, str]],
        max_teams: int = 16,
    ) -> list[str]:
        """Return dates in window where active teams <= max_teams (quality off-nights)."""
        return [
            d for d in self.normalize_window(window)
            if len(self.get_teams_playing_on_date(d)) <= max_teams
        ]

    def get_team_schedule_matrix(
        self,
        teams: Iterable[str],
        window: Union[list[str], tuple[str, str]],
    ) -> dict[str, dict[str, bool]]:
        """Matrix {team: {date: bool}} indicating game participation."""
        dates = self.normalize_window(window)
        return {
            normalize_team(t): {
                d: self.is_team_playing_on_date(t, d) for d in dates
            }
            for t in teams
        }

    # ── Player Registry Management ────────────────────────────────────────────

    def register_player_team(self, player_id: int, team: str) -> None:
        """Assign or update player's NBA team."""
        self._player_teams[int(player_id)] = normalize_team(team)

    def register_player_teams(self, player_teams: dict[Union[int, str], str]) -> None:
        """Bulk update player team assignments."""
        for pid, t in player_teams.items():
            self._player_teams[int(pid)] = normalize_team(t)

    def get_player_team(self, player_id: int) -> Optional[str]:
        """Return assigned NBA team code or None."""
        return self._player_teams.get(int(player_id))

    def remove_player(self, player_id: int) -> None:
        """Remove player from registry."""
        self._player_teams.pop(int(player_id), None)

    # ── Factory Constructors ──────────────────────────────────────────────────

    @classmethod
    def from_dict(
        cls,
        schedule_dict: dict[str, Iterable[str]],
        player_teams: Optional[dict[Union[int, str], str]] = None,
    ) -> ScheduleCalendar:
        coerced = {int(k): v for k, v in player_teams.items()} if player_teams else None
        return cls(team_schedules=schedule_dict, player_teams=coerced)

    @classmethod
    def from_games(
        cls,
        games: Iterable[NBAGame],
        player_teams: Optional[dict[Union[int, str], str]] = None,
    ) -> ScheduleCalendar:
        coerced = {int(k): v for k, v in player_teams.items()} if player_teams else None
        return cls(games=games, player_teams=coerced)

    @classmethod
    def from_fixture_file(
        cls,
        file_path: Union[str, Path],
        player_teams: Optional[dict[Union[int, str], str]] = None,
    ) -> ScheduleCalendar:
        path = Path(file_path)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        teams_data = data.get('teams', data) if isinstance(data, dict) else {}
        p_teams = data.get('player_teams', player_teams) if isinstance(data, dict) else player_teams
        return cls.from_dict(teams_data, player_teams=p_teams)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        player_teams: Optional[dict[Union[int, str], str]] = None,
    ) -> ScheduleCalendar:
        sched: dict[str, set[str]] = {}
        cols = {str(c).lower(): c for c in df.columns}

        if 'date' in cols and 'home_team' in cols and 'away_team' in cols:
            d_col, h_col, a_col = cols['date'], cols['home_team'], cols['away_team']
            for _, row in df.iterrows():
                d = normalize_date(row[d_col])
                h = normalize_team(str(row[h_col]))
                a = normalize_team(str(row[a_col]))
                sched.setdefault(h, set()).add(d)
                sched.setdefault(a, set()).add(d)
        elif 'date' in cols and 'team' in cols:
            d_col, t_col = cols['date'], cols['team']
            for _, row in df.iterrows():
                d = normalize_date(row[d_col])
                t = normalize_team(str(row[t_col]))
                sched.setdefault(t, set()).add(d)
        else:
            raise ValueError(f"Unrecognized DataFrame columns for schedule: {list(df.columns)}")

        return cls.from_dict({t: sorted(dates) for t, dates in sched.items()}, player_teams=player_teams)

    @classmethod
    def from_synthetic(
        cls,
        start_date: str = '2024-10-21',
        weeks: int = 25,
        seed: int = 42,
        player_teams: Optional[dict[Union[int, str], str]] = None,
    ) -> ScheduleCalendar:
        sched, games = generate_synthetic_nba_schedule(start_date=start_date, weeks=weeks, seed=seed)
        coerced = {int(k): v for k, v in player_teams.items()} if player_teams else None
        return cls(team_schedules=sched, games=games, player_teams=coerced)

    @classmethod
    def from_snowflake(
        cls,
        season: str = '2024-25',
        player_teams: Optional[dict[Union[int, str], str]] = None,
        fallback_to_synthetic: bool = True,
    ) -> ScheduleCalendar:
        """Query Snowflake schedule table or fall back to synthetic schedule if offline."""
        try:
            from backend.infra.snowflake_connection import query
            df = query(f"SCHEDULE_{season.replace('-', '_')}")
            return cls.from_dataframe(df, player_teams=player_teams)
        except Exception as exc:
            if fallback_to_synthetic:
                logger.warning("Snowflake schedule query failed (%s); falling back to synthetic schedule.", exc)
                return cls.from_synthetic(player_teams=player_teams)
            raise
