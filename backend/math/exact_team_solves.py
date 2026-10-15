"""Exact team solves: one concrete, legal team the H-scoring plan for a candidate could become.

H-scoring never builds a team -- it models the drafter's future picks as a smooth expectation that depends on the
category weights. These solves turn that plan into a named roster for display (the example team in the candidate
detail panel). They are display only: nothing here feeds back into H-scoring or the ranking, and nothing here writes
agent state.

naive_exact_team_solve: the best legal team on the LINEAR value x . w, w being the candidate's H-scoring category
weights. Linear value has no diminishing returns, so this team over-stacks whichever categories carry the most weight.

fw_exact_team_solve: Frank-Wolfe from the naive team on the app's real complete-team H-score, minus a penalty for the
team's category win rates falling short of the ones H-scoring predicted for the candidate. The real objective sees a
category saturate (a nearly-won category's gradient goes flat) and moves the effort elsewhere; the penalty keeps the
team from giving up categories H-scoring planned to compete in. Beating a prediction is not penalized: it already
shows in the H-score. Frank-Wolfe moves through fractional mixes of teams, so its whole-team answer is the best of the
corners it visited and its final mix rounded to the nearest legal team; then swaps -- one seat's player for another
outside the team, the best improving swap each time -- polish that team to a local optimum. The swaps matter most at
the start of a draft, where Frank-Wolfe's whole teams can sit several points below a team one or two swaps away.

assignment_by_specificity: the finished team re-seated so its earlier picks hold the more specific slots. Display only.

A legal team holds my current players and the candidate in the slots H-scoring seated them in, seats every other player
in an eligible slot, fills each flex slot type with each base position as often as H-scoring's flex allocation expects
(its share times the type's open slots, rounded down or up), and:
  - draft:   takes one player per remaining round, each one whose predicted round (his place in the predicted draft
             order / number of teams, rounded up) is that round or later. The predicted order is the caller's: base
             H-score order, or that blended with market draft positions (services/market_draft_values). Deliberately
             approximate -- it ignores where in the round my pick falls -- and needs no platform draft order.
  - auction: pays every player's predicted price (generic dollars, whole, at least $1) out of my remaining cash.
When NBA teams are known (the market data's, for the season being drafted), no two players predicted to go in a draft's
first DRAFT_DISTINCT_TEAM_ROUNDS rounds share an NBA team, and an auction roster holds at most AUCTION_PLAYERS_PER_TEAM
from one NBA team.

A seat is (player, slot type, base position): in a flex slot a multi-position player fills it AS one of his eligible
base positions, which the flex allocation rows count. Each step's best legal team is a linear program over seats. The
draft rows form two laminar families -- per player (each player once, nested round prefixes) and per slot (each slot
type's count, each flex cell inside it) -- so they are totally unimodular and the vertices are whole teams; the
auction's budget row breaks that, and so do the NBA-team rows (they cut across the round prefixes), so an auction, or a
draft with NBA-team rows, solves an integer program instead.
(The flex allocation was first a penalty in the objective. At every weight tried it either lost to the win-rate penalty
or stalled Frank-Wolfe on the naive team, so it is a hard rule.)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from backend.math.algorithm_agents import HAgent


FRANK_WOLFE_ITERATIONS = 30
LINE_SEARCH_STEPS      = 21          # step sizes 0, 0.05, ..., 1
# Lambda of the win-rate penalty, in H-score units per squared win-rate shortfall: falling 10 points short of
# H-scoring's predicted rate in one category costs one point of H-score. Fixed, not tuned against outcomes; the team is
# display only.
WIN_RATE_PENALTY       = 1.0
GRADIENT_STEP          = 1e-4        # finite-difference step, in x-score (or volume unit) of one team total
# Swaps bring in only players in the top SWAP_RANK_LIMIT of the predicted draft order: anyone later is all but never an
# improvement, and the limit keeps each round of swaps to one small batch.
SWAP_RANK_LIMIT        = 250
# Realism rules on NBA teams, where teams are known: no two players predicted for a draft's first rounds share a team;
# an auction roster holds at most a couple from one team.
DRAFT_DISTINCT_TEAM_ROUNDS = 7
AUCTION_PLAYERS_PER_TEAM   = 2
# The pool stops at SWAP_RANK_LIMIT in the predicted order, or this many rounds past the draft's last pick in a league
# big enough to draft further: later players are all but never an improvement (late specialists before it often are).
POOL_EXTRA_ROUNDS = 2
MIP_RELATIVE_GAP  = 1e-3


@dataclass
class TeamSolveProblem:
    """A team solve's pool and constraints, built once per request (build_team_solve_problem).

    The pool lists my current players first, then the candidate, then every available player. y[seat] = 1 seats player
    seat_players[seat] in slot type seat_slot_types[seat], filling it as base position seat_bases[seat]."""
    pool_player_ids: list[int]
    n_fixed_players: int                  # my current players + the candidate, the first entries of the pool
    x_scores: np.ndarray                  # (P, C)
    volume_units: Optional[np.ndarray]    # (P, R) when percentage categories are volume-aware, else None
    volume_category_indices: Optional[np.ndarray]
    slot_types: list[str]
    slot_base_counts: list[int]           # how many base positions each slot type takes (PG: 1, G: 2, Util: 5)
    base_positions: list[str]
    seat_players: np.ndarray              # (n_seats,) pool index
    seat_slot_types: np.ndarray           # (n_seats,) index into slot_types
    seat_bases: np.ndarray                # (n_seats,) index into base_positions
    flex_cell_bounds: dict[tuple[str, str], tuple[int, int]]   # (flex slot type, base) -> open players (low, high)
    inequality_rows: np.ndarray           # A_ub over seats
    inequality_bounds: np.ndarray         # b_ub
    equality_rows: np.ndarray             # A_eq over seats
    equality_bounds: np.ndarray           # b_eq
    is_auction: bool
    uses_integer_program: bool            # the best legal team needs an integer program (auction, or NBA-team rows)
    predicted_prices: Optional[np.ndarray]  # (P,) whole dollars, auction only
    base_rounds: np.ndarray               # (P,) each player's predicted round
    base_ranks: np.ndarray                # (P,) each player's place in the predicted draft order (0 = first); -1 if absent


@dataclass
class CompleteTeamField:
    """The other teams' side of a complete-team score, captured on a reference team (capture_complete_team_field).
    Any team's differences against the field are the reference's plus the change in its own totals."""
    diff_means: np.ndarray                # (1, C, D-1), the reference team's
    diff_vars: np.ndarray
    opponent_units: Optional[np.ndarray]
    reference_totals: np.ndarray          # (C,)


# ── Shared ────────────────────────────────────────────────────────────────────

def build_team_solve_problem(
    h_agent: HAgent
    , player_assignments: dict[str, list[int]]
    , my_team_id: str
    , candidate_player_id: int
    , exclusion_list: list[int]
    , availability_order: list[int]
    , fixed_slot_types: list[str]
    , expected_flex_position_counts: dict[str, dict[str, float]]
    , remaining_cash: Optional[float]
    , predicted_prices: Optional[pd.Series]
    , player_teams: Optional[pd.Series]
) -> TeamSolveProblem:
    """The pool and every constraint row of a team solve for one candidate. Draft when remaining_cash and
    predicted_prices are both None; auction when both are given. fixed_slot_types holds the slot type H-scoring seated
    each of my players and the candidate in (board order, then the candidate): they stay there, and only the open
    slots are solved for. availability_order lists the players in the order other drafters are predicted to take them
    (whom the round rule and the swaps' candidate limit go by). expected_flex_position_counts is H-scoring's flex
    allocation of the open slots, flex slot type -> base position -> expected count; the open players filling each pair
    are held to it, rounded down or up. player_teams maps player ids to NBA teams for the team rules (None: no team
    rules; a player it does not name is bound by none)."""
    if (remaining_cash is None) != (predicted_prices is None):
        raise ValueError('An auction team solve needs both the remaining cash and the predicted prices; '
                         'a draft solve takes neither.')
    is_auction = remaining_cash is not None

    my_players = [p for p in player_assignments[my_team_id] if p == p]
    if candidate_player_id in my_players:
        raise ValueError(f'Candidate {candidate_player_id} is already on the roster.')
    n_roster = h_agent.n_picks
    if len(my_players) + 1 > n_roster:
        raise ValueError(f'The roster has no slot left for a candidate ({len(my_players)} of {n_roster} filled).')

    taken = {p for roster in player_assignments.values() for p in roster if p == p} | set(exclusion_list)
    ranked_players = [p for p in availability_order if p in h_agent.x_scores.index]
    base_round_of = {player: rank // h_agent.n_drafters + 1 for rank, player in enumerate(ranked_players)}
    pool_depth = max(SWAP_RANK_LIMIT, h_agent.n_drafters * (n_roster + POOL_EXTRA_ROUNDS))
    available = [p for p in ranked_players[:pool_depth]
                 if p not in taken and p != candidate_player_id and p in h_agent.positions.index]
    pool_player_ids = my_players + [candidate_player_id] + available
    n_fixed_players = len(my_players) + 1
    if len(fixed_slot_types) != n_fixed_players:
        raise ValueError(f'{len(fixed_slot_types)} fixed slots given for {n_fixed_players} fixed players.')
    n_pool = len(pool_player_ids)

    position_config = h_agent.position_config
    base_positions = list(position_config.position_structure['base_list'])
    flex = position_config.position_structure['flex']
    slot_types = [slot for slot in base_positions + position_config.position_structure['flex_list']
                  if position_config.position_numbers[slot] > 0]
    slot_bases = {slot: list(flex[slot]['bases']) if slot in flex else [slot] for slot in slot_types}
    seats = [(index, slot_index, base_positions.index(base))
             for index, player in enumerate(pool_player_ids)
             for slot_index, slot in enumerate(slot_types)
             if index >= n_fixed_players or slot == fixed_slot_types[index]
             for base in slot_bases[slot]
             if base in h_agent.positions.loc[player]]
    seat_players = np.array([index for index, _, _ in seats])
    seat_slot_types = np.array([slot_index for _, slot_index, _ in seats])
    seat_bases = np.array([base_index for _, _, base_index in seats])
    unseatable = set(range(n_fixed_players)) - set(seat_players.tolist())
    if unseatable:
        raise ValueError(f'Players {[pool_player_ids[i] for i in sorted(unseatable)]} fit no roster slot.')

    # Only open picks are bound by rounds; the fixed players (who may include the unranked replacement player) get 0.
    base_rounds = np.array([0] * n_fixed_players + [base_round_of[player] for player in available])
    base_rank_of = {player: rank for rank, player in enumerate(ranked_players)}
    base_ranks = np.array([base_rank_of.get(player, -1) for player in pool_player_ids])
    is_open_pick = (seat_players >= n_fixed_players).astype(float)

    inequality_rows = ([(seat_players == index).astype(float) for index in range(n_pool)]
                       + [(seat_slot_types == slot_index).astype(float) for slot_index in range(len(slot_types))])
    inequality_bounds = ([1.0] * n_pool
                         + [float(position_config.position_numbers[slot]) for slot in slot_types])

    # H-scoring's flex allocation: the open players filling each (flex slot type, base position) pair.
    flex_cell_bounds = {}
    for slot_index, slot in enumerate(slot_types):
        if slot not in flex:
            continue
        for base in slot_bases[slot]:
            expected = expected_flex_position_counts[slot][base]
            low, high = int(np.floor(expected + 1e-9)), int(np.ceil(expected - 1e-9))
            flex_cell_bounds[(slot, base)] = (low, high)
            cell = is_open_pick * ((seat_slot_types == slot_index) & (seat_bases == base_positions.index(base)))
            inequality_rows += [cell, -cell]
            inequality_bounds += [float(high), float(-low)]

    predicted_price_array = None
    if is_auction:
        # The candidate is bought at its predicted price too; my current players are already paid for (price 0).
        candidate_index = n_fixed_players - 1
        buyable_prices = np.maximum(
            np.round(predicted_prices.reindex(pool_player_ids[candidate_index:]).to_numpy(dtype=float)), 1.0)
        if np.isnan(buyable_prices).any():
            missing = [pool_player_ids[candidate_index + i] for i in np.flatnonzero(np.isnan(buyable_prices))]
            raise ValueError(f'No predicted price for players {missing[:10]}.')
        predicted_price_array = np.concatenate([np.zeros(candidate_index), buyable_prices])
        candidate_and_open = (seat_players >= candidate_index).astype(float)
        inequality_rows.append(predicted_price_array[seat_players] * candidate_and_open)
        inequality_bounds.append(float(remaining_cash))
    else:
        # One open pick per round after the candidate's; the pick in round r takes a player of base round r or
        # later, i.e. at most (open picks before round r) of the open players come from base rounds before r.
        candidate_round = len(my_players) + 1
        for later_round in range(candidate_round + 1, n_roster + 1):
            inequality_rows.append(is_open_pick * (base_rounds[seat_players] < later_round))
            inequality_bounds.append(float(later_round - candidate_round - 1))

    # NBA-team realism rules, over the open picks; my players and the candidate already on the team use up the room.
    uses_team_rows = False
    if player_teams is not None:
        teams = np.array([player_teams.get(player) for player in pool_player_ids], dtype=object)
        if is_auction:
            fixed_counted = range(n_fixed_players)
            open_counted = is_open_pick.astype(bool)
            limit = AUCTION_PLAYERS_PER_TEAM
        else:
            # No two of the players predicted to go in the first DRAFT_DISTINCT_TEAM_ROUNDS rounds share a team: my
            # players picked in those rounds (pick k is round k), the candidate if his round is one of them, and the open
            # players predicted for them.
            fixed_counted = range(min(n_fixed_players, DRAFT_DISTINCT_TEAM_ROUNDS))
            open_counted = is_open_pick.astype(bool) & (base_rounds[seat_players] <= DRAFT_DISTINCT_TEAM_ROUNDS)
            limit = 1
        fixed_teams = [teams[index] for index in fixed_counted if teams[index] is not None]
        for team in sorted({team for team in teams[seat_players[open_counted]] if team is not None}):
            row = (open_counted & (teams[seat_players] == team)).astype(float)
            inequality_rows.append(row)
            inequality_bounds.append(float(max(0, limit - fixed_teams.count(team))))
            uses_team_rows = True

    equality_rows = [np.ones(len(seats))] + [(seat_players == index).astype(float)
                                              for index in range(n_fixed_players)]
    equality_bounds = [float(n_roster)] + [1.0] * n_fixed_players

    volume_units = None
    volume_category_indices = None
    if len(h_agent.volume_categories) > 0:
        volume_units = h_agent.player_volume_units.loc[pool_player_ids].to_numpy()
        volume_category_indices = h_agent.volume_category_indices

    return TeamSolveProblem(
        pool_player_ids         = pool_player_ids,
        n_fixed_players         = n_fixed_players,
        x_scores                = h_agent.x_scores.loc[pool_player_ids].to_numpy(),
        volume_units            = volume_units,
        volume_category_indices = volume_category_indices,
        slot_types              = slot_types,
        slot_base_counts        = [len(slot_bases[slot]) for slot in slot_types],
        base_positions          = base_positions,
        seat_players            = seat_players,
        seat_slot_types         = seat_slot_types,
        seat_bases              = seat_bases,
        flex_cell_bounds        = flex_cell_bounds,
        inequality_rows         = np.array(inequality_rows),
        inequality_bounds       = np.array(inequality_bounds),
        equality_rows           = np.array(equality_rows),
        equality_bounds         = np.array(equality_bounds),
        is_auction              = is_auction,
        uses_integer_program    = is_auction or uses_team_rows,
        predicted_prices        = predicted_price_array,
        base_rounds             = base_rounds,
        base_ranks              = base_ranks,
    )


def find_best_legal_team(
    problem: TeamSolveProblem
    , seat_values: np.ndarray
) -> np.ndarray:
    """The legal whole team maximizing the sum of seat_values (n_seats,), as a 0/1 seating over the problem's seats --
    an integer program where the problem needs one."""
    if problem.uses_integer_program:
        result = milp(
            -seat_values,
            constraints=[LinearConstraint(problem.inequality_rows, -np.inf, problem.inequality_bounds),
                         LinearConstraint(problem.equality_rows, problem.equality_bounds, problem.equality_bounds)],
            integrality=np.ones(len(seat_values)),
            bounds=Bounds(0, 1),
            # Within 0.1% of the best: the naive start and the rounding need a very good legal team, not a proof that
            # none is better.
            options={'mip_rel_gap': MIP_RELATIVE_GAP},
        )
        check_solved(problem, result)
        return np.round(result.x)
    seating = find_best_legal_mix(problem, seat_values)
    if not np.allclose(seating, np.round(seating), atol=1e-6):
        raise RuntimeError('The best legal team came back fractional; the constraints are not totally unimodular.')
    return np.round(seating)


def find_best_legal_mix(
    problem: TeamSolveProblem
    , seat_values: np.ndarray
) -> np.ndarray:
    """The best point of the legal polytope by seat_values: the linear program alone. A whole team where the constraints
    are totally unimodular (a draft without NBA-team rows); otherwise possibly a mix of teams, which is all a
    Frank-Wolfe step needs -- its path runs through mixes anyway."""
    result = linprog(
        -seat_values,
        A_ub=problem.inequality_rows, b_ub=problem.inequality_bounds,
        A_eq=problem.equality_rows, b_eq=problem.equality_bounds,
        bounds=(0, 1), method='highs',
    )
    check_solved(problem, result)
    return result.x


def check_solved(problem: TeamSolveProblem, result) -> None:
    """Raises when a solve found no legal team (or mix)."""
    if result.status != 0:
        constraint = 'fits the budget at predicted prices' if problem.is_auction else 'exists for this roster'
        raise ValueError(f'No legal team {constraint} ({result.message}).')


def convert_seatings_to_selections(
    problem: TeamSolveProblem
    , seatings: np.ndarray
) -> np.ndarray:
    """(N, P) how much of each pool player N seatings (or mixes of seatings, (N, n_seats)) take."""
    selections = np.zeros((seatings.shape[0], len(problem.pool_player_ids)))
    np.add.at(selections.T, problem.seat_players, seatings.T)
    return selections


def capture_complete_team_field(
    h_agent: HAgent
    , problem: TeamSolveProblem
    , player_assignments: dict[str, list[int]]
    , my_team_id: str
    , exclusion_list: list[int]
    , reference_seating: np.ndarray
    , remaining_cash_per_team: Optional[dict[str, float]]
) -> CompleteTeamField:
    """The field, captured with the reference team as my roster. It is held fixed while other teams are scored, though
    the other teams' remaining pool shifts slightly with who I take: accepted, as the team is display only."""
    reference_selection = convert_seatings_to_selections(problem, reference_seating[np.newaxis, :])[0]
    reference_team = [problem.pool_player_ids[i] for i in np.flatnonzero(reference_selection)]
    board = {team: list(roster) for team, roster in player_assignments.items()}
    board[my_team_id] = reference_team
    diff_means, diff_vars, volume_field = h_agent.get_complete_team_field(
        board, my_team_id, exclusion_list, remaining_cash_per_team)
    return CompleteTeamField(
        diff_means       = diff_means,
        diff_vars        = diff_vars,
        opponent_units   = None if volume_field is None else volume_field['opponent_units'],
        reference_totals = reference_selection @ problem.x_scores,
    )


def score_team_totals(
    h_agent: HAgent
    , problem: TeamSolveProblem
    , field: CompleteTeamField
    , team_totals: np.ndarray
    , team_units: Optional[np.ndarray]
    , h_scoring_win_rates: np.ndarray
) -> np.ndarray:
    """(N,) the penalized objective of N teams given by their category totals (N, C) and volume units (N, R):
    complete-team H-score minus WIN_RATE_PENALTY times the squared shortfall of each category's win rate below
    H-scoring's predicted one."""
    diff_means = field.diff_means + (team_totals - field.reference_totals)[:, :, np.newaxis]
    volume_context = None
    if problem.volume_units is not None:
        volume_context = {
            'my_x_total':     team_totals[:, problem.volume_category_indices],
            'my_units':       team_units,
            'opponent_units': field.opponent_units,
        }
    scores, win_rates = h_agent.compute_complete_team_scores_and_win_rates_batched(
        diff_means, field.diff_vars, volume_context)
    shortfalls = np.maximum(h_scoring_win_rates - win_rates, 0.0)
    return scores - WIN_RATE_PENALTY * (shortfalls ** 2).sum(axis=1)


def score_seatings(
    h_agent: HAgent
    , problem: TeamSolveProblem
    , field: CompleteTeamField
    , seatings: np.ndarray
    , h_scoring_win_rates: np.ndarray
) -> np.ndarray:
    """(N,) the penalized objective of N seatings (or mixes of seatings)."""
    selections = convert_seatings_to_selections(problem, seatings)
    totals = selections @ problem.x_scores
    units = None if problem.volume_units is None else selections @ problem.volume_units
    return score_team_totals(h_agent, problem, field, totals, units, h_scoring_win_rates)


def compute_seating_gradient(
    h_agent: HAgent
    , problem: TeamSolveProblem
    , field: CompleteTeamField
    , seating: np.ndarray
    , h_scoring_win_rates: np.ndarray
) -> np.ndarray:
    """(n_seats,) gradient of the penalized objective in each seat. It depends on the seating only through the team's
    C category totals and R volume units, so it is differenced in those (one batched call) and carried back to the
    players by the chain rule."""
    selection = convert_seatings_to_selections(problem, seating[np.newaxis, :])
    totals = selection @ problem.x_scores
    units = None if problem.volume_units is None else selection @ problem.volume_units
    n_categories = totals.shape[1]
    n_units = 0 if units is None else units.shape[1]
    nudged_totals = np.repeat(totals, 1 + n_categories + n_units, axis=0)
    nudged_totals[1:1 + n_categories] += GRADIENT_STEP * np.eye(n_categories)
    nudged_units = None
    if units is not None:
        nudged_units = np.repeat(units, 1 + n_categories + n_units, axis=0)
        nudged_units[1 + n_categories:] += GRADIENT_STEP * np.eye(n_units)
    objectives = score_team_totals(h_agent, problem, field, nudged_totals, nudged_units, h_scoring_win_rates)
    slopes = (objectives[1:] - objectives[0]) / GRADIENT_STEP
    player_gradient = problem.x_scores @ slopes[:n_categories]
    if units is not None:
        player_gradient = player_gradient + problem.volume_units @ slopes[n_categories:]
    return player_gradient[problem.seat_players]


def run_frank_wolfe(
    h_agent: HAgent
    , problem: TeamSolveProblem
    , field: CompleteTeamField
    , h_scoring_win_rates: np.ndarray
    , naive_seating: np.ndarray
) -> np.ndarray:
    """Frank-Wolfe on the penalized objective from the naive team. Each step heads for the linear program's best point
    -- a whole team where the constraints are totally unimodular, possibly a mix where they are not (an auction, NBA-team
    rows), which spares an integer program per step. Returns, as a seating, the best whole team among the corners it
    visited (the naive team included) and its final mix rounded to the legal team keeping the most of each player's
    share in it -- a mix itself is never returned."""
    seating = naive_seating
    visited = [naive_seating]
    for _ in range(FRANK_WOLFE_ITERATIONS):
        gradient = compute_seating_gradient(h_agent, problem, field, seating, h_scoring_win_rates)
        vertex = find_best_legal_mix(problem, gradient)
        if np.allclose(vertex, np.round(vertex), atol=1e-6):
            visited.append(np.round(vertex))
        direction = vertex - seating
        if gradient @ direction < 1e-7:
            break
        steps = np.linspace(0.0, 1.0, LINE_SEARCH_STEPS)
        trial_seatings = seating[np.newaxis, :] + steps[:, np.newaxis] * direction[np.newaxis, :]
        best_step = steps[int(np.argmax(score_seatings(h_agent, problem, field, trial_seatings, h_scoring_win_rates)))]
        if best_step == 0.0:
            break
        seating = seating + best_step * direction
    player_shares = convert_seatings_to_selections(problem, seating[np.newaxis, :])[0]
    visited.append(find_best_legal_team(problem, player_shares[problem.seat_players]))
    return visited[int(np.argmax(score_seatings(h_agent, problem, field, np.array(visited), h_scoring_win_rates)))]


def improve_team_by_swaps(
    h_agent: HAgent
    , problem: TeamSolveProblem
    , field: CompleteTeamField
    , h_scoring_win_rates: np.ndarray
    , seating: np.ndarray
) -> np.ndarray:
    """Best-improvement swaps to a local optimum: each round tries every swap of one open player's seat for a seat of a
    player outside the team (in the top SWAP_RANK_LIMIT of the predicted draft order), keeps those that leave every rule satisfied,
    and makes the one that most improves the penalized objective; it stops when none does. Legality is exact without a
    solve: replacing one seat by another keeps the equality rows (team size, fixed players), and the inequality rows
    are checked as the current row loads plus the two seats' columns. Returns the improved seating."""
    seating = seating.copy()
    rows = problem.inequality_rows
    is_outside_candidate_seat = ((problem.seat_players >= problem.n_fixed_players)
                                 & (problem.base_ranks[problem.seat_players] >= 0)
                                 & (problem.base_ranks[problem.seat_players] < SWAP_RANK_LIMIT))
    selection = convert_seatings_to_selections(problem, seating[np.newaxis, :])
    current_score = score_seatings(h_agent, problem, field, seating[np.newaxis, :], h_scoring_win_rates)[0]
    while True:
        on_team = selection[0] > 0.5
        outgoing_seats = [int(seat) for seat in np.flatnonzero(seating > 0.5)
                          if problem.seat_players[seat] >= problem.n_fixed_players]
        incoming_seats = np.flatnonzero(is_outside_candidate_seat & ~on_team[problem.seat_players])
        loads = rows @ seating
        swaps = []
        for outgoing in outgoing_seats:
            changed_loads = loads[:, np.newaxis] - rows[:, [outgoing]] + rows[:, incoming_seats]
            legal = (changed_loads <= problem.inequality_bounds[:, np.newaxis] + 1e-9).all(axis=0)
            swaps += [(outgoing, int(incoming)) for incoming in incoming_seats[legal]]
        if not swaps:
            return seating
        outgoing_players = problem.seat_players[[outgoing for outgoing, _ in swaps]]
        incoming_players = problem.seat_players[[incoming for _, incoming in swaps]]
        totals = (selection @ problem.x_scores - problem.x_scores[outgoing_players]
                  + problem.x_scores[incoming_players])
        units = None
        if problem.volume_units is not None:
            units = (selection @ problem.volume_units - problem.volume_units[outgoing_players]
                     + problem.volume_units[incoming_players])
        scores = score_team_totals(h_agent, problem, field, totals, units, h_scoring_win_rates)
        best = int(np.argmax(scores))
        if scores[best] <= current_score + 1e-12:
            return seating
        outgoing, incoming = swaps[best]
        seating[outgoing], seating[incoming] = 0.0, 1.0
        selection = convert_seatings_to_selections(problem, seating[np.newaxis, :])
        current_score = scores[best]


# ── naive_exact_team_solve ────────────────────────────────────────────────────

def solve_naive_exact_team_for_draft(
    problem: TeamSolveProblem
    , h_scoring_weights: np.ndarray
) -> np.ndarray:
    """The best legal draft team on linear value x . w; a seating."""
    return find_best_legal_team(problem, (problem.x_scores @ h_scoring_weights)[problem.seat_players])


def solve_naive_exact_team_for_auction(
    problem: TeamSolveProblem
    , h_scoring_weights: np.ndarray
) -> np.ndarray:
    """The best legal auction team on linear value x . w, within budget at predicted prices; a seating."""
    return find_best_legal_team(problem, (problem.x_scores @ h_scoring_weights)[problem.seat_players])


# ── fw_exact_team_solve ───────────────────────────────────────────────────────

def solve_fw_exact_team_for_draft(
    h_agent: HAgent
    , problem: TeamSolveProblem
    , field: CompleteTeamField
    , h_scoring_win_rates: np.ndarray
    , naive_seating: np.ndarray
) -> np.ndarray:
    """Frank-Wolfe refinement of the naive draft team, polished by swaps; a seating."""
    refined = run_frank_wolfe(h_agent, problem, field, h_scoring_win_rates, naive_seating)
    return improve_team_by_swaps(h_agent, problem, field, h_scoring_win_rates, refined)


def solve_fw_exact_team_for_auction(
    h_agent: HAgent
    , problem: TeamSolveProblem
    , field: CompleteTeamField
    , h_scoring_win_rates: np.ndarray
    , naive_seating: np.ndarray
) -> np.ndarray:
    """Frank-Wolfe refinement of the naive auction team, polished by swaps; a seating."""
    refined = run_frank_wolfe(h_agent, problem, field, h_scoring_win_rates, naive_seating)
    return improve_team_by_swaps(h_agent, problem, field, h_scoring_win_rates, refined)


# ── assignment_by_specificity ─────────────────────────────────────────────────

def seat_team_by_specificity(
    problem: TeamSolveProblem
    , seating: np.ndarray
) -> np.ndarray:
    """The same team, re-seated so its earlier picks hold the more specific slots: among every legal seating of exactly
    these players (eligibility, slot counts, the flex allocation, my players and the candidate where they are), the one
    maximizing the sum of pick priority x slot specificity over the open players. A slot's specificity is the inverse
    of how many base positions it takes (PG: 1, G: 1/2, Util: 1/5); a pick's priority is its rank among the open picks
    from the latest -- in a draft the earliest in the predicted draft order (the earliest planned round) ranks highest,
    in an auction the
    highest predicted price. Display only: the team, and so its score, is unchanged."""
    team = convert_seatings_to_selections(problem, seating[np.newaxis, :])[0] > 0.5
    open_players = [index for index in np.flatnonzero(team) if index >= problem.n_fixed_players]
    if problem.is_auction:
        open_players.sort(key=lambda index: (-problem.predicted_prices[index], index))
    priority = np.zeros(len(problem.pool_player_ids))
    for order, index in enumerate(open_players):
        priority[index] = len(open_players) - order

    slot_specificity = 1.0 / np.array(problem.slot_base_counts, dtype=float)
    seat_values = priority[problem.seat_players] * slot_specificity[problem.seat_slot_types]

    # Only the team's own players may be seated, each exactly once (the problem's rows already hold every other rule).
    team_rows = [(problem.seat_players == index).astype(float) for index in np.flatnonzero(team)]
    outsider_seats = ~team[problem.seat_players]
    result = linprog(
        -seat_values,
        A_ub=problem.inequality_rows, b_ub=problem.inequality_bounds,
        A_eq=np.vstack([problem.equality_rows] + team_rows),
        b_eq=np.concatenate([problem.equality_bounds, np.ones(len(team_rows))]),
        bounds=[(0, 0) if is_outsider else (0, 1) for is_outsider in outsider_seats], method='highs',
    )
    if result.status != 0:
        raise RuntimeError(f'Re-seating a legal team failed ({result.message}).')
    if not np.allclose(result.x, np.round(result.x), atol=1e-6):
        raise RuntimeError('The re-seated team came back fractional.')
    return np.round(result.x)
