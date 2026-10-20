"""
Auction autodraft: nominations and per-team valuations for the browser's round-robin bidding.

The browser runs the bidding itself (manual drafters bid interactively, turn by turn), so this
module supplies only what the browser cannot compute: who an autodrafter nominates, the opening
bid, each autodrafter's valuation of the nominated player, and every team's maximum allowable bid.

Valuations are each team's "Your $" from the same full evaluate the H-score table runs
(rank_candidates over the whole board, no exclusions), so an autodrafter bids up to exactly the
dollar value the UI shows for that team.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from backend.state.session import Session
from backend.models import EvaluateResponse
from backend.player_identity import FULL_ROSTER_SCORE_PLAYER_ID, RP_PLAYER_ID
from backend.services.ranking import rank_candidates, UnknownTeamError
from backend.services.auction_pricing import compute_max_allowed_bids


@dataclass
class AuctionNomination:
    nominated_player_id: int
    nominated_player_name: str
    opening_bid: int
    valuations: dict[str, float]
    max_allowed_bids: dict[str, int]


# ── Public entry points ───────────────────────────────────────────────────────

def nominate_auction_player(
    session: Session
    , player_assignments: dict[str, list[int]]
    , remaining_cash: dict[str, float]
    , nominator_id: str
    , nominated_player_id: Optional[int]
    , opening_bid: Optional[int]
    , valuation_team_ids: list[str]
) -> AuctionNomination:
    """Opens bidding on one player.

    An autodrafter nominator passes nominated_player_id=None and opening_bid=None: it nominates its
    highest-H-score available player and opens at $1. A manual nominator passes both. Valuations are
    returned for every team in valuation_team_ids (the autodrafters)."""
    validate_auction_board(player_assignments, remaining_cash, [nominator_id] + list(valuation_team_ids))

    total_roster_picks = session.current_settings['n_picks']
    if len(player_assignments[nominator_id]) >= total_roster_picks:
        raise ValueError(f'Nominator {nominator_id} already has a full roster ({total_roster_picks} picks).')

    max_allowed_bids = compute_max_allowed_bids(player_assignments, remaining_cash, total_roster_picks)
    nominator_max_bid = max_allowed_bids[nominator_id]
    if nominator_max_bid < 1:
        raise ValueError(
            f'Nominator {nominator_id} cannot open a bid: ${remaining_cash[nominator_id]:g} remaining with '
            f'{total_roster_picks - len(player_assignments[nominator_id])} empty slots to fill at $1 each.')

    evaluations: dict[str, EvaluateResponse] = {}

    if nominated_player_id is None:
        if opening_bid is not None:
            raise ValueError('opening_bid was given without nominated_player_id; an autodrafter nomination opens at $1.')
        if nominator_id not in valuation_team_ids:
            raise ValueError(
                f'Nominator {nominator_id} must be an autodrafter (in valuation_team_ids) to nominate automatically.')
        nominated_player_id = select_autodrafter_nominee(session, player_assignments, remaining_cash, nominator_id,
                                                          evaluations)
        opening_bid = 1
    else:
        if opening_bid is None:
            raise ValueError('A manual nomination must carry its opening_bid.')
        if not 1 <= opening_bid <= nominator_max_bid:
            raise ValueError(
                f'Opening bid ${opening_bid} for {nominator_id} must be between $1 and its maximum allowable bid '
                f'${nominator_max_bid}.')
        validate_nominee_available(player_assignments, nominated_player_id)

    valuations = {
        team: value_player_for_team(session, player_assignments, remaining_cash, team, nominated_player_id,
                                    evaluations)
        for team in valuation_team_ids
    }
    return AuctionNomination(
        nominated_player_id   = nominated_player_id,
        nominated_player_name = read_player_name(session, nominated_player_id),
        opening_bid           = opening_bid,
        valuations            = valuations,
        max_allowed_bids      = max_allowed_bids,
    )


def value_auction_player(
    session: Session
    , player_assignments: dict[str, list[int]]
    , remaining_cash: dict[str, float]
    , player_id: int
    , valuation_team_ids: list[str]
) -> dict[str, float]:
    """Each listed team's valuation of a player already up for bid. The browser asks for this when a
    team is switched to autodraft in the middle of a nomination, so it can bid from then on."""
    validate_auction_board(player_assignments, remaining_cash, list(valuation_team_ids))
    validate_nominee_available(player_assignments, player_id)
    evaluations: dict[str, EvaluateResponse] = {}
    return {
        team: value_player_for_team(session, player_assignments, remaining_cash, team, player_id, evaluations)
        for team in valuation_team_ids
    }


# ── Rules ─────────────────────────────────────────────────────────────────────

def select_autodrafter_nominee(
    session: Session
    , player_assignments: dict[str, list[int]]
    , remaining_cash: dict[str, float]
    , nominator_id: str
    , evaluations: dict[str, EvaluateResponse]
) -> int:
    """The autodrafter's highest-H-score available player. A nominator whose active slots are full is
    filling perma-bench slots, which its evaluate does not rank, so it nominates the top available
    player on the neutral board instead."""
    if len(player_assignments[nominator_id]) < session.agent.n_picks:
        candidates = evaluate_team(session, player_assignments, remaining_cash, nominator_id, evaluations).candidates
        nominable = [candidate.player_id for candidate in candidates if candidate.player_id != FULL_ROSTER_SCORE_PLAYER_ID]
    else:
        rostered = {player for roster in player_assignments.values() for player in roster}
        nominable = [player for player in session.agent.default_h_scores.index
                     if player not in rostered and player not in (FULL_ROSTER_SCORE_PLAYER_ID, RP_PLAYER_ID)]
    if not nominable:
        raise ValueError(f'{nominator_id} has no available player left to nominate.')
    return int(nominable[0])


def value_player_for_team(
    session: Session
    , player_assignments: dict[str, list[int]]
    , remaining_cash: dict[str, float]
    , team: str
    , player_id: int
    , evaluations: dict[str, EvaluateResponse]
) -> float:
    """A team's dollar valuation of a player: its "Your $" for that player while it has active slots to
    fill, $1 once only perma-bench slots remain, and $0 once its roster is full. A pool player the team's
    evaluate leaves out is one whose positions fit none of its open active slots (_build_candidates drops
    those); the team cannot roster him, so he is worth $0 to it, exactly as the H-score table omits him."""
    roster_size = len(player_assignments[team])
    if roster_size >= session.current_settings['n_picks']:
        return 0.0
    if roster_size >= session.agent.n_picks:
        return 1.0
    candidates = evaluate_team(session, player_assignments, remaining_cash, team, evaluations).candidates
    matched = next((candidate for candidate in candidates if candidate.player_id == player_id), None)
    if matched is None:
        if is_in_scored_pool(session, player_id):
            return 0.0
        raise ValueError(f'Player {player_id} is not in the scored player pool, so {team} cannot value him.')
    if matched.auction_values is None:
        raise ValueError(f'The auction evaluate for {team} returned no dollar values.')
    return round(float(matched.auction_values.your_dollar), 2)


def evaluate_team(
    session: Session
    , player_assignments: dict[str, list[int]]
    , remaining_cash: dict[str, float]
    , team: str
    , evaluations: dict[str, EvaluateResponse]
) -> EvaluateResponse:
    """The full auction evaluate from one team's seat, run once per team per request."""
    if team not in evaluations:
        evaluations[team] = rank_candidates(
            session            = session,
            player_assignments = player_assignments,
            my_team_id         = team,
            exclusion_list     = [],
            remaining_cash     = remaining_cash,
        )
    return evaluations[team]


# ── Validation ────────────────────────────────────────────────────────────────

def validate_auction_board(
    player_assignments: dict[str, list[int]]
    , remaining_cash: dict[str, float]
    , named_team_ids: list[str]
) -> None:
    """Every team on the board has a cash entry (and no cash entry lacks a team), and every team the
    request names is on the board."""
    board_teams = set(player_assignments)
    if set(remaining_cash) != board_teams:
        raise UnknownTeamError(
            f'remaining_cash teams {sorted(remaining_cash)} do not match the board teams {sorted(board_teams)}.')
    unknown = [team for team in named_team_ids if team not in board_teams]
    if unknown:
        raise UnknownTeamError(f'Teams {unknown} are not on the board: {sorted(board_teams)}.')


def validate_nominee_available(
    player_assignments: dict[str, list[int]]
    , player_id: int
) -> None:
    if any(player_id in roster for roster in player_assignments.values()):
        raise ValueError(f'Player {player_id} is already rostered.')


def is_in_scored_pool(
    session: Session
    , player_id: int
) -> bool:
    """Whether the evaluate scores this player at all: the same membership test get_h_scores applies to
    its available pool (projected, with a position row, and not the replacement placeholder)."""
    agent = session.agent
    return (player_id != RP_PLAYER_ID
            and player_id in agent.x_scores.index
            and player_id in agent.positions.index)


def read_player_name(
    session: Session
    , player_id: int
) -> str:
    if player_id not in session.player_registry:
        raise ValueError(f'Player {player_id} is not in the session player registry.')
    return session.player_registry[player_id].name
