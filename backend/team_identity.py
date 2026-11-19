"""Team identity: the id every team is keyed by, mirroring player_identity.

The app keys every team by a string id, never by its name: player_assignments, my_team_id,
remaining_cash, the unmatched-player names and the empty slots all index by it. Names exist
only on TeamIdentity and are rendered only at display time, so a rename, a trailing space or
two teams sharing a name cannot move a roster between teams. (Keyed by name, two ESPN teams
whose names ended in a space lost their whole rosters without an error, 2026-11-18.)

The ids:
  - a connected league: the platform's own team id (ESPN's team id, Yahoo's team id, Fantrax's
    team id), so an integration reads a roster by the same id the rest of the app uses;
  - a Yahoo draft room still filling: "empty-1", "empty-2", ... for the seats nobody has taken,
    which have no Yahoo team and so no id (Yahoo lists only the teams that have joined). The
    next connect after someone takes the seat gives it the real id;
  - the user's own data: "seat-1" ... "seat-N".
The prefixes keep the made-up ids from colliding with a platform's: no platform id carries them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TeamIdentity:
    team_id: str
    name: str      # display only


def make_own_data_team_identities(n_drafters: int) -> list[TeamIdentity]:
    """The user's own league: seats "seat-1" ... "seat-N", named "Team 1" ... "Team N"."""
    return [TeamIdentity(team_id=f'seat-{seat}', name=f'Team {seat}') for seat in range(1, n_drafters + 1)]


def make_platform_team_identities(
    platform_teams: list[tuple[str, str]]
    , n_seats: int
) -> list[TeamIdentity]:
    """One identity per seat: each (platform team id, name) the platform lists, in its order, then "empty-1",
    "empty-2", ... for the seats no team has taken. A platform listing one id twice, or more teams than seats, is a
    broken response, not a league, and fails here rather than merging or dropping a team."""
    team_ids = [team_id for team_id, _ in platform_teams]
    if len(set(team_ids)) != len(team_ids):
        raise ValueError(f'The platform listed a team id more than once: {team_ids}.')
    if len(platform_teams) > n_seats:
        raise ValueError(f'The platform listed {len(platform_teams)} teams for a league of {n_seats} seats.')
    teams = [TeamIdentity(team_id=team_id, name=name) for team_id, name in platform_teams]
    for empty_seat in range(1, n_seats - len(platform_teams) + 1):
        teams.append(TeamIdentity(team_id=f'empty-{empty_seat}', name=f'Open seat {empty_seat}'))
    return teams
