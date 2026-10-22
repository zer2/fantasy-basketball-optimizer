"""
Auction autodraft: nominations and per-team valuations for the browser's round-robin bidding.

The browser runs the bidding itself (manual drafters bid interactively, turn by turn), so this
module supplies only what the browser cannot compute: who an autodrafter nominates, the opening
bid, each autodrafter's valuation of the nominated player, and every team's maximum allowable bid.

Valuations are each team's "Your $": its H-scores put through the auction valuation (rescaled against the
replacement-level player and through SAVOR, as the H-score table does). Solving every team's H-scores over the
whole pool on every nomination made an all-autodraft auction take minutes per round, so each team's raw H-scores
are kept between nominations (read_team_h_scores) and only the valuation is redone for everyone after every sale:
  - a team that has just bought re-solves its H-scores, for its top (players still to be bought + 40)
    candidates by its previous H-scores (REFRESH_CANDIDATE_MARGIN); the rest are below replacement level anyway;
  - every other team keeps its H-scores, losing the players taken. They are not re-solved for the other teams'
    purchases or spending, an approximation of the full evaluate accepted for speed (the user, 2026-10-21);
  - a board that is not the cached one plus purchases (an undo, a new board) or a rebuilt agent solves in full,
    except an empty one, which starts from the neutral board with only each team's top players re-solved.

Autodrafters do not all see the players alike: each solves with its own noisy view of the X-scores
(read_team_x_scores; drawn once per agent, smaller for lesser players: compute_noise_standard_deviations), so their
valuations differ the way different drafters' would (the user, 2026-10-21). The H-score table is not affected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import zlib

import numpy as np
import pandas as pd

from backend.state.session import Session
from backend.math.algorithm_helpers import auction_value_adjuster
from backend.player_identity import FULL_ROSTER_SCORE_PLAYER_ID, RP_PLAYER_ID
from backend.services.build_agent import count_unscorable_players_as_replacement
from backend.services.ranking import UnknownTeamError, check_rostered_players_in_pool, split_off_bench
from backend.services.auction_pricing import (
    APP_CASH_SHARE_RULE, compute_max_allowed_bids, compute_unusable_cash_shares,
    sum_spendable_league_cash,
)

# How many candidates past the players still to be bought a team re-solves after a purchase (read_team_h_scores).
REFRESH_CANDIDATE_MARGIN = 40

# How many of the top players each team re-solves through its own noise on an empty board (read_team_h_scores).
INITIAL_NOISY_CANDIDATES = 20

# The standard deviation of the noise in each autodrafter's view of the top player's X-score in every category; it
# halves every (players drafted) ranks down the neutral board (compute_noise_standard_deviations).
X_SCORE_NOISE_STANDARD_DEVIATION = 0.2


@dataclass
class TeamHScores:
    """One team's raw H-scores (before the auction valuation) for the players available on `rosters`, and whether
    each fits the team's open active slots there. Kept on the session between nominations (read_team_h_scores)."""
    agent: object
    rosters: dict[str, tuple[int, ...]]
    h_scores: pd.Series
    fits_roster: pd.Series


@dataclass
class AuctionNomination:
    nominated_player_id: int
    nominated_player_name: str
    opening_bid: int
    valuations: dict[str, float]
    # The part of each valuation that is its team's otherwise unusable cash (auction_pricing), so the effect of that
    # adjustment on the bidding can be measured (testing_files/experiment_harnesses/auction_simulation.py).
    cash_shares: dict[str, float]
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

    dollars_by_team: dict[str, tuple[pd.Series, pd.Series]] = {}

    if nominated_player_id is None:
        if opening_bid is not None:
            raise ValueError('opening_bid was given without nominated_player_id; an autodrafter nomination opens at $1.')
        if nominator_id not in valuation_team_ids:
            raise ValueError(
                f'Nominator {nominator_id} must be an autodrafter (in valuation_team_ids) to nominate automatically.')
        nominated_player_id = select_autodrafter_nominee(session, player_assignments, remaining_cash, nominator_id)
        opening_bid = 1
    else:
        if opening_bid is None:
            raise ValueError('A manual nomination must carry its opening_bid.')
        if not 1 <= opening_bid <= nominator_max_bid:
            raise ValueError(
                f'Opening bid ${opening_bid} for {nominator_id} must be between $1 and its maximum allowable bid '
                f'${nominator_max_bid}.')
        validate_nominee_available(player_assignments, nominated_player_id)

    valuations_and_shares = {
        team: value_player_for_team(session, player_assignments, remaining_cash, team, nominated_player_id,
                                    dollars_by_team)
        for team in valuation_team_ids
    }
    return AuctionNomination(
        nominated_player_id   = nominated_player_id,
        nominated_player_name = read_player_name(session, nominated_player_id),
        opening_bid           = opening_bid,
        valuations            = {team: value for team, (value, _) in valuations_and_shares.items()},
        cash_shares           = {team: share for team, (_, share) in valuations_and_shares.items()},
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
    dollars_by_team: dict[str, tuple[pd.Series, pd.Series]] = {}
    return {
        team: value_player_for_team(session, player_assignments, remaining_cash, team, player_id, dollars_by_team)[0]
        for team in valuation_team_ids
    }


# ── Rules ─────────────────────────────────────────────────────────────────────

def select_autodrafter_nominee(
    session: Session
    , player_assignments: dict[str, list[int]]
    , remaining_cash: dict[str, float]
    , nominator_id: str
) -> int:
    """The autodrafter's highest-H-score available player that fits its open slots. A nominator whose active slots
    are full is filling perma-bench slots, which its H-scores do not rank, so it nominates the top available player
    on the neutral board instead."""
    if len(player_assignments[nominator_id]) < session.agent.n_picks:
        team_h_scores = read_team_h_scores(session, player_assignments, remaining_cash, nominator_id)
        fitting = team_h_scores.h_scores[team_h_scores.fits_roster.reindex(team_h_scores.h_scores.index)]
        nominable = list(fitting.sort_values(ascending=False).index)
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
    , dollars_by_team: dict[str, tuple[pd.Series, pd.Series]]
) -> tuple[float, float]:
    """A team's dollar valuation of a player, and the part of it that is the team's unusable-cash share: its "Your $"
    for that player while it has active slots to fill, $1 once only perma-bench slots remain, and $0 once its roster
    is full. A pool player whose positions fit none of the team's open active slots is worth $0 to it: it cannot
    roster him, exactly as the H-score table omits him."""
    roster_size = len(player_assignments[team])
    if roster_size >= session.current_settings['n_picks']:
        return 0.0, 0.0
    if roster_size >= session.agent.n_picks:
        return 1.0, 0.0
    if not is_in_scored_pool(session, player_id):
        raise ValueError(f'Player {player_id} is not in the scored player pool, so {team} cannot value him.')
    team_h_scores = read_team_h_scores(session, player_assignments, remaining_cash, team)
    if player_id not in team_h_scores.h_scores.index:
        raise ValueError(f"Player {player_id} is in the scored pool but not in {team}'s H-scores.")
    if not bool(team_h_scores.fits_roster[player_id]):
        return 0.0, 0.0
    if team not in dollars_by_team:
        dollars_by_team[team] = compute_team_dollars(session, player_assignments, remaining_cash, team,
                                                     team_h_scores)
    dollars, cash_shares = dollars_by_team[team]
    return round(float(dollars[player_id]), 2), round(float(cash_shares[player_id]), 2)


# ── H-scores kept between nominations ─────────────────────────────────────────

def read_team_h_scores(
    session: Session
    , player_assignments: dict[str, list[int]]
    , remaining_cash: dict[str, float]
    , team: str
) -> TeamHScores:
    """The team's H-scores for this board, from the session's kept ones where they still apply (see the module
    docstring): solved in full when none are kept for this agent or the board is not the kept one plus purchases,
    re-solved for the top candidates when the team itself has bought since, and otherwise the kept ones less the
    players taken. The result is kept for the next nomination."""
    board = {board_team: tuple(roster) for board_team, roster in player_assignments.items()}
    rostered = {player for roster in board.values() for player in roster}
    kept = session.auction_team_h_scores.get(team)
    if ((kept is None or kept.agent is not session.agent or not extends_board(kept.rosters, board))
            and all(len(roster) == 0 for roster in board.values())):
        # An empty board starts from the neutral board every team shares (the agent's stored result), with only the
        # team's top INITIAL_NOISY_CANDIDATES re-solved through its own noise: the opening nominations are of top
        # players, and a full noisy solve for every team made the first nomination take ~5 s (2026-10-21). The
        # rest keep the neutral scores until the team's first purchase re-solves its top candidates.
        neutral_h_scores = session.agent.default_h_scores.copy()
        top_candidates = list(neutral_h_scores.sort_values(ascending=False).index[:INITIAL_NOISY_CANDIDATES])
        solved_h_scores, solved_fits = solve_team_h_scores(session, player_assignments, remaining_cash, team,
                                                           top_candidates)
        neutral_h_scores[solved_h_scores.index] = solved_h_scores
        fits_roster = pd.Series(True, index=neutral_h_scores.index)   # every open slot is open on an empty board
        fits_roster[solved_fits.index] = solved_fits
        team_h_scores = TeamHScores(session.agent, board, neutral_h_scores, fits_roster)
    elif kept is None or kept.agent is not session.agent or not extends_board(kept.rosters, board):
        solved_h_scores, solved_fits = solve_team_h_scores(session, player_assignments, remaining_cash, team, None)
        team_h_scores = TeamHScores(session.agent, board, solved_h_scores, solved_fits)
    else:
        h_scores = kept.h_scores.drop(list(rostered), errors='ignore')
        fits_roster = kept.fits_roster.drop(list(rostered), errors='ignore')
        if board[team] != kept.rosters[team]:
            refresh_count = count_remaining_picks(session, player_assignments) + REFRESH_CANDIDATE_MARGIN
            candidate_subset = list(h_scores.sort_values(ascending=False).index[:refresh_count])
            solved_h_scores, solved_fits = solve_team_h_scores(session, player_assignments, remaining_cash, team,
                                                               candidate_subset)
            h_scores = h_scores.copy()
            fits_roster = fits_roster.copy()
            h_scores[solved_h_scores.index] = solved_h_scores
            fits_roster[solved_fits.index] = solved_fits
        team_h_scores = TeamHScores(session.agent, board, h_scores, fits_roster)
    session.auction_team_h_scores[team] = team_h_scores
    return team_h_scores


def extends_board(
    kept_rosters: dict[str, tuple[int, ...]]
    , board: dict[str, tuple[int, ...]]
) -> bool:
    """Whether `board` is `kept_rosters` plus purchases: the same teams, each roster starting with its kept one."""
    return (set(kept_rosters) == set(board)
            and all(board[team][:len(roster)] == roster for team, roster in kept_rosters.items()))


def solve_team_h_scores(
    session: Session
    , player_assignments: dict[str, list[int]]
    , remaining_cash: dict[str, float]
    , team: str
    , candidate_subset: Optional[list[int]]
) -> tuple[pd.Series, pd.Series]:
    """The team's raw H-scores and roster fit for the available players (all of them, or `candidate_subset`), from
    the same solve rank_candidates runs for the H-score table: unscorable players counted as replacement, the board
    checked against the pool, perma-bench players set aside and excluded."""
    agent = session.agent
    board = count_unscorable_players_as_replacement(session, player_assignments)
    check_rostered_players_in_pool(session, board)
    active_board, benched_players = split_off_bench(board, agent.n_picks)
    # The whole pool is named rather than left as None: on an empty board None returns the stored noise-free
    # result, and the team's own view of the players has to apply from the first nomination.
    if candidate_subset is None:
        rostered = {player for roster in active_board.values() for player in roster}
        candidate_subset = [player for player in agent.x_scores.index
                            if player not in rostered and player not in benched_players
                            and is_in_scored_pool(session, player)]
    # The opponents' stored builds are brought up to date with the real X-scores first: they are shared by every
    # team, so they must not be inferred through one team's noise (the solve below finds nothing stale).
    if agent.models_opponents:
        agent.refresh_stale_team_states(active_board, team, remaining_cash)
    with agent.substituting_x_scores(read_team_x_scores(session, team)):
        result = agent.get_h_scores(
            player_assignments      = active_board,
            drafter                 = team,
            n_iterations            = session.current_settings['n_iterations'],
            cash_remaining_per_team = remaining_cash,
            exclusion_list          = benched_players,
            candidate_subset        = candidate_subset,
            candidate_offset        = 0,
        )
    if result is None:
        raise ValueError(f'The H-score solve for {team} returned nothing.')
    h_scores = result['Scores']
    # Rosters column 0 is the candidate's slot, or < 0 when he fits none of the team's open slots; all -1 when the
    # data carries no positions, and then everyone fits (as in _build_candidates).
    first_slot = result['Rosters'].iloc[:, 0]
    has_position_data = not bool((first_slot == -1).all())
    fits_roster = (first_slot >= 0) if has_position_data else pd.Series(True, index=h_scores.index)
    return h_scores, fits_roster.reindex(h_scores.index)


def read_team_x_scores(
    session: Session
    , team: str
) -> pd.DataFrame:
    """The team's own view of the players: the agent's X-scores plus independent normal noise for every player and
    category, of the player's standard deviation (compute_noise_standard_deviations), the replacement player excepted.
    Drawn once per agent, from a seed fixed by the team's name and the session's auction_noise_seed, so the view holds
    between nominations and a rerun of the same auction (in a new session) repeats it -- what lets a change to the
    autodrafter be compared run against run; an experiment sets other noise seeds for independent auctions."""
    agent = session.agent
    kept = session.auction_team_x_scores.get(team)
    if kept is not None and kept[0] is agent:
        return kept[1]
    generator = np.random.default_rng((session.auction_noise_seed, zlib.crc32(team.encode())))
    standard_deviations = compute_noise_standard_deviations(session).to_numpy()[:, np.newaxis]
    noise = generator.normal(0.0, 1.0, size=agent.x_scores.shape) * standard_deviations
    noisy = agent.x_scores + noise
    if RP_PLAYER_ID in noisy.index:
        noisy.loc[RP_PLAYER_ID] = agent.x_scores.loc[RP_PLAYER_ID]
    session.auction_team_x_scores[team] = (agent, noisy)
    return noisy


def compute_noise_standard_deviations(
    session: Session
) -> pd.Series:
    """Each player's noise standard deviation, by his rank on the neutral board (the agent's default H-scores, rank 0
    the best): X_SCORE_NOISE_STANDARD_DEVIATION halving every (players drafted) ranks, so 0.2 for the stars, 0.1 at
    replacement level and 0.05 a league's worth further down. A fixed SD put the same error on a fringe player as on a
    star, and since his dollar value is near zero, teams disagreed by more than his whole value (an SD across teams of
    $1.71 on a mean of $1.14 around replacement level, 2026-10-21); a lesser player's per-game stats are smaller, and so
    is the error in them. A player the neutral board does not rank is placed below every ranked one."""
    agent = session.agent
    neutral_order = agent.default_h_scores.sort_values(ascending=False).index
    ranks = pd.Series(np.arange(len(neutral_order), dtype=float), index=neutral_order)
    ranks = ranks.reindex(agent.x_scores.index).fillna(float(len(neutral_order)))
    players_drafted = agent.n_drafters * agent.n_picks
    return X_SCORE_NOISE_STANDARD_DEVIATION * 0.5 ** (ranks / players_drafted)


def count_remaining_picks(
    session: Session
    , player_assignments: dict[str, list[int]]
) -> int:
    """Active roster spots still to fill across the league: the players still to be bought before replacement level
    (perma-bench picks are not scored, as in _build_candidates)."""
    active_board, _ = split_off_bench(player_assignments, session.agent.n_picks)
    return session.agent.n_drafters * session.agent.n_picks - sum(len(roster) for roster in active_board.values())


def compute_team_dollars(
    session: Session
    , player_assignments: dict[str, list[int]]
    , remaining_cash: dict[str, float]
    , team: str
    , team_h_scores: TeamHScores
) -> tuple[pd.Series, pd.Series]:
    """(the team's "Your $" for every available player, the part of it that is the cash share): its H-scores
    rescaled against the replacement-level player (the one at the remaining-picks rank) to the league's spendable cash
    (sum_spendable_league_cash),
    then SAVOR, plus its best options' shares of the cash it could not otherwise use (compute_unusable_cash_shares) --
    the valuation _build_candidates gives the H-score table."""
    league_dollars = auction_value_adjuster(
        team_h_scores.h_scores.sort_values(ascending=False),
        count_remaining_picks(session, player_assignments),
        sum_spendable_league_cash(player_assignments, remaining_cash, session.agent.n_picks),
        session.current_settings['streaming_noise'],
        session.current_settings['replacement_threshold'],
    )
    fits_roster = team_h_scores.fits_roster.reindex(league_dollars.index)
    cash_shares = compute_unusable_cash_shares(
        your_dollar_series   = league_dollars,
        rosterable_players   = league_dollars.index[fits_roster.to_numpy(dtype=bool)],
        team_cash            = remaining_cash[team],
        open_active_slots    = session.agent.n_picks - min(len(player_assignments[team]), session.agent.n_picks),
        empty_roster_slots   = session.current_settings['n_picks'] - len(player_assignments[team]),
        rule                 = read_cash_share_rule(session, team),
        smooth_rule_softness = session.current_settings['cash_share_softness'],
    )
    return league_dollars + cash_shares, cash_shares


def read_cash_share_rule(
    session: Session
    , team: str
) -> str:
    """The cash-share rule this autodrafter prices with: the one the session sets for it (an experiment comparing
    rules), else the app's."""
    if team in session.auction_cash_share_rules:
        return session.auction_cash_share_rules[team]
    return APP_CASH_SHARE_RULE



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
