"""Builds the hypothetical team for one candidate (math/exact_team_solves) and converts it to the
/hypothetical-team response. Display only: reads the strategy the last evaluate of this board computed, and writes
no agent state."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from backend.math import exact_team_solves
from backend.models import HypotheticalPick, HypotheticalTeamResponse, Roster, RosterAssignment
from backend.services.auction_pricing import sum_spendable_league_cash
from backend.services.build_agent import count_unscorable_players_as_replacement
from backend.services.market_draft_values import (
    compute_availability_order, compute_blended_prices, load_market_draft_values, uses_market_draft_values,
)
from backend.services.ranking import (
    compute_league_generic_dollars, make_candidate_strategy_key, split_off_bench, make_slot_names,
)
from backend.state.session import Session


class StaleBoardError(Exception):
    """The board has no evaluate on record for this candidate, so there is no strategy to build from."""


def build_hypothetical_team(
    session: Session
    , player_assignments: dict[str, list[int]]
    , my_team_id: str
    , exclusion_list: list[int]
    , remaining_cash: Optional[dict[str, float]]
    , forced_category_weights: Optional[dict[str, Optional[float]]]
    , candidate_player_id: int
) -> HypotheticalTeamResponse:
    """The naive exact team for the candidate's H-scoring weights, refined by Frank-Wolfe, as a full roster."""
    h_agent = session.agent
    board_key = make_candidate_strategy_key(
        player_assignments, my_team_id, exclusion_list, remaining_cash, forced_category_weights)
    entry = session.candidate_strategy_cache.get(board_key)
    if entry is None or entry[0] is not h_agent or candidate_player_id not in entry[1]:
        raise StaleBoardError('This board has changed since the candidates were ranked. Wait for the ranking '
                              'to refresh, then try again.')
    strategy = entry[1][candidate_player_id]
    if strategy.seating is None:
        raise ValueError('H-scoring found no roster slot assignment for this candidate.')

    # The same board transforms the evaluate applied before scoring.
    player_assignments = count_unscorable_players_as_replacement(session, player_assignments)
    player_assignments, benched_players = split_off_bench(player_assignments, h_agent.n_picks)
    exclusion_list = list(exclusion_list) + benched_players

    # When other drafters are predicted to take each player (and, in an auction, at what price): H-scoring's base order
    # and generic dollars, blended with the market's draft positions and costs for the season being drafted.
    market_values = load_market_draft_values() if uses_market_draft_values(session.current_settings) else None
    availability_order = compute_availability_order(
        [int(p) for p in h_agent.default_h_scores.index], market_values,
        h_agent.n_drafters * session.current_settings['n_picks'])

    is_auction = remaining_cash is not None
    my_remaining_cash = None
    predicted_prices = None
    if is_auction:
        current_settings = session.current_settings
        players_chosen = {p for roster in player_assignments.values() for p in roster}
        available_player_ids = h_agent.x_scores.index[
            ~h_agent.x_scores.index.isin(list(players_chosen) + exclusion_list)]
        generic_prices = compute_league_generic_dollars(
            h_agent.default_h_scores, available_player_ids, h_agent.n_drafters, h_agent.n_picks,
            current_settings['cash_per_team'],
            sum_spendable_league_cash(player_assignments, remaining_cash, h_agent.n_picks),
            current_settings['streaming_noise'], current_settings['replacement_threshold'],
        )
        predicted_prices = compute_blended_prices(generic_prices, market_values)
        my_remaining_cash = remaining_cash[my_team_id]

    # My players and the candidate keep the exact slots H-scoring's roster gave them.
    slot_names = make_slot_names(h_agent.position_config)
    fixed_slot_names = [slot_names[slot_index] for slot_index in strategy.seating]
    fixed_slot_types = [name.rstrip('0123456789') for name in fixed_slot_names]
    expected_flex_position_counts = compute_expected_flex_position_counts(
        strategy.flex_shares, fixed_slot_types, h_agent.position_config.position_numbers)
    problem = exact_team_solves.build_team_solve_problem(
        h_agent, player_assignments, my_team_id, candidate_player_id, exclusion_list, availability_order,
        anchors_to_h_scoring          = True,   # the example team illustrates H-scoring's plan
        fixed_slot_types              = fixed_slot_types,
        expected_flex_position_counts = expected_flex_position_counts,
        remaining_cash                = my_remaining_cash,
        predicted_prices              = predicted_prices,
        player_teams                  = None if market_values is None else market_values['team'].dropna(),
        my_remaining_pick_numbers     = None)
    if is_auction:
        naive_seating = exact_team_solves.solve_naive_exact_team_for_auction(problem, strategy.category_weights)
    else:
        naive_seating = exact_team_solves.solve_naive_exact_team_for_draft(problem, strategy.category_weights)
    field = exact_team_solves.capture_complete_team_field(
        h_agent, problem, player_assignments, my_team_id, exclusion_list, naive_seating, remaining_cash)
    if is_auction:
        seating = exact_team_solves.solve_fw_exact_team_for_auction(
            h_agent, problem, field, strategy.win_rates, naive_seating)
    else:
        seating = exact_team_solves.solve_fw_exact_team_for_draft(
            h_agent, problem, field, strategy.win_rates, naive_seating)

    seating = exact_team_solves.seat_team_by_specificity(problem, seating)
    return convert_seating_to_response(problem, seating, slot_names, fixed_slot_names)


def compute_expected_flex_position_counts(
    flex_shares: dict[str, pd.Series]
    , fixed_slot_types: list[str]
    , position_numbers: dict[str, int]
) -> dict[str, dict[str, float]]:
    """H-scoring's flex allocation as counts, flex slot type -> base position -> expected open players: each share times
    the slots of that type my players and the candidate leave open (what the candidate panel's flex table shows)."""
    return {
        slot: {base: float(share) * (position_numbers[slot] - fixed_slot_types.count(slot))
               for base, share in shares.items()}
        for slot, shares in flex_shares.items()
    }


def convert_seating_to_response(
    problem: exact_team_solves.TeamSolveProblem
    , seating: np.ndarray
    , slot_names: list[str]
    , fixed_slot_names: list[str]
) -> HypotheticalTeamResponse:
    """My players and the candidate in the exact slots H-scoring gave them; each hypothetical pick in the next free slot
    of the type the solve chose. Draft picks are labelled with their planned round, in predicted draft order
    (the earliest-predicted open player takes the earliest round)."""
    candidate_index = problem.n_fixed_players - 1
    assignments: dict[str, Optional[RosterAssignment]] = {name: None for name in slot_names}
    for pool_index, slot_name in enumerate(fixed_slot_names):
        assignments[slot_name] = RosterAssignment(
            player_id=int(problem.pool_player_ids[pool_index]),
            kind='candidate' if pool_index == candidate_index else 'rostered')
    free_slot_names = {slot_type: [name for name in slot_names
                                   if name.rstrip('0123456789') == slot_type and assignments[name] is None]
                       for slot_type in problem.slot_types}

    hypothetical_indices = []
    for seat in np.flatnonzero(seating):
        pool_index = int(problem.seat_players[seat])
        if pool_index <= candidate_index:
            continue
        slot_name = free_slot_names[problem.slot_types[problem.seat_slot_types[seat]]].pop(0)
        assignments[slot_name] = RosterAssignment(
            player_id=int(problem.pool_player_ids[pool_index]), kind='hypothetical')
        hypothetical_indices.append(pool_index)

    first_open_round = problem.n_fixed_players + 1
    hypothetical_picks = []
    # Pool order past the fixed players is the predicted draft order, so sorting the indices assigns rounds
    # earliest-first.
    for pick_offset, pool_index in enumerate(sorted(hypothetical_indices)):
        player_id = int(problem.pool_player_ids[pool_index])
        if problem.is_auction:
            hypothetical_picks.append(HypotheticalPick(
                player_id=player_id, predicted_price=int(problem.predicted_prices[pool_index])))
        else:
            hypothetical_picks.append(HypotheticalPick(player_id=player_id, pick_round=first_open_round + pick_offset))

    return HypotheticalTeamResponse(
        roster             = Roster(slots=slot_names, assignments=assignments),
        hypothetical_picks = hypothetical_picks,
    )
