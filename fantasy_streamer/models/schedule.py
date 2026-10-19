"""NBA schedule matrix, rolling horizon partitions, and B2B chaining detectors."""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class MatchupHorizon:
    """Manages the rolling-horizon time window spanning current and lookahead matchup weeks."""

    start_date: date
    horizon_days: int = 10
    # Day indices belonging to week 1 (current matchup) and week 2 (next matchup)
    week1_days: List[int] = field(default_factory=list)
    week2_days: List[int] = field(default_factory=list)
    day_dates: List[date] = field(default_factory=list)

    def __post_init__(self):
        if not self.day_dates:
            self.day_dates = [
                self.start_date + timedelta(days=i) for i in range(self.horizon_days)
            ]
        if not self.week1_days and not self.week2_days:
            self._partition_weeks()

    def _partition_weeks(self):
        """Partition days into Week 1 (through Sunday) and Week 2 (starting Monday).

        In NBA / ESPN fantasy, weeks run Monday through Sunday (weekday() 0 to 6).
        """
        w1: List[int] = []
        w2: List[int] = []
        in_week_2 = False

        for i, dt in enumerate(self.day_dates):
            # If we cross from Sunday (weekday 6) to Monday (weekday 0), or if day 0 was already Monday
            # but we reach next Monday:
            if i > 0 and dt.weekday() == 0:
                in_week_2 = True

            if in_week_2:
                w2.append(i)
            else:
                w1.append(i)

        self.week1_days = w1
        self.week2_days = w2

    @property
    def total_days(self) -> int:
        return len(self.day_dates)

    def get_date(self, day_idx: int) -> date:
        return self.day_dates[day_idx]

    def get_week_of_day(self, day_idx: int) -> int:
        return 1 if day_idx in self.week1_days else 2


@dataclass
class ScheduleMatrix:
    """Stores team and player game schedules across the rolling horizon."""

    horizon: MatchupHorizon
    # Map from team abbreviation to list of day indices with games
    team_games: Dict[str, Set[int]] = field(default_factory=dict)
    # Map from player_id to list of day indices with games (can override team schedule)
    player_games: Dict[int, Set[int]] = field(default_factory=dict)

    def has_game(self, team: str, day_idx: int, player_id: Optional[int] = None) -> bool:
        """Returns True if player (or their team) has a game on day_idx."""
        if player_id is not None and player_id in self.player_games:
            return day_idx in self.player_games[player_id]
        team_key = team.upper()
        if team_key in self.team_games:
            return day_idx in self.team_games[team_key]
        return False

    def is_b2b(self, team: str, day_idx: int, player_id: Optional[int] = None) -> bool:
        """Returns True if player plays on day_idx and day_idx + 1."""
        if day_idx + 1 >= self.horizon.total_days:
            return False
        return self.has_game(team, day_idx, player_id) and self.has_game(
            team, day_idx + 1, player_id
        )

    def is_3_in_4(self, team: str, start_day: int, player_id: Optional[int] = None) -> bool:
        """Returns True if player plays at least 3 games in days [start_day .. start_day + 3]."""
        if start_day + 3 >= self.horizon.total_days:
            return False
        count = sum(
            1
            for d in range(start_day, start_day + 4)
            if self.has_game(team, d, player_id)
        )
        return count >= 3

    def get_team_game_count(self, team: str, days: List[int]) -> int:
        return sum(1 for d in days if self.has_game(team, d))


def create_synthetic_nba_schedule(
    horizon: MatchupHorizon,
    teams: Optional[List[str]] = None,
    b2b_teams: Optional[List[str]] = None,
) -> ScheduleMatrix:
    """Create a realistic schedule matrix for testing and benchmarking."""
    if teams is None:
        teams = [
            "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
            "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
            "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
        ]

    team_games: Dict[str, Set[int]] = {t: set() for t in teams}

    # Standard alternating pattern: teams typically play 3-4 games a week
    # e.g., Mon/Wed/Fri/Sun or Tue/Thu/Sat
    for i, t in enumerate(teams):
        # Default distribution based on parity
        if i % 2 == 0:
            pattern = [0, 2, 4, 6, 7, 9]  # Mon, Wed, Fri, Sun, Mon, Wed
        else:
            pattern = [1, 3, 5, 8]  # Tue, Thu, Sat, Tue

        for d in pattern:
            if d < horizon.total_days:
                team_games[t].add(d)

    # Inject specific B2B chains if requested
    if b2b_teams:
        for t in b2b_teams:
            if t in team_games:
                # Add B2B on days 2 and 3 (e.g., Wed/Thu) or 5 and 6 (Sat/Sun)
                team_games[t].add(2)
                team_games[t].add(3)

    return ScheduleMatrix(horizon=horizon, team_games=team_games)
