# testing_files/test_exact_team_solves.py
# The hypothetical team (math/exact_team_solves, services/hypothetical_team, POST /hypothetical-team):
#   - the complete-team field reproduces the app's own full-roster H-score and win rates, in every objective
#   - draft teams are legal (slots, held players, the candidate, one pick per round from that round or later), and
#     my players and the candidate keep the slots H-scoring's roster gave them
#   - auction teams are legal and fit the budget at whole predicted prices
#   - Frank-Wolfe never ends below the naive team, the swaps never end below Frank-Wolfe and leave a legal team at a
#     local optimum, and every team fills each flex slot
#     type with each base position as often as H-scoring's flex allocation expects (rounded down or up), also after
#     re-seating the team so its earlier picks hold the more specific slots
#   - asking for a hypothetical team changes nothing the evaluate returns afterwards
#   - a board with no evaluate on record is refused (409), and so is a draft/auction mismatch (400)
# Uses 2024-25 historical data with default parameters.

import numpy as np
import pandas as pd
import pytest

from benchmark_helpers import client, _build_session_request
from backend.state.session import get_session
from backend.math import exact_team_solves
from backend.services.ranking import (
    make_candidate_strategy_key, make_slot_names, compute_original_and_generic_dollars,
)
from backend.services.hypothetical_team import compute_expected_flex_position_counts


def create_session(objective: str, cash_per_team: int = None) -> tuple[str, object]:
    response = client.post('/sessions', json=_build_session_request(objective=objective, cash_per_team=cash_per_team))
    assert response.status_code == 201, response.text
    session_id = response.json()['session_id']
    return session_id, get_session(session_id)


def make_two_round_board(session) -> dict[str, list[int]]:
    """Two snake rounds taken in base H-score order: Drafter 1 holds the 1st and 24th players."""
    agent = session.agent
    teams = [f'Drafter {seat + 1}' for seat in range(session.current_settings['n_drafters'])]
    ranked = [int(player) for player in agent.default_h_scores.index]
    board = {team: [] for team in teams}
    for pick, team in enumerate(teams + teams[::-1]):
        board[team].append(ranked[pick])
    return board


@pytest.fixture(scope='module')
def draft_session():
    return create_session('Each Category')


@pytest.mark.parametrize('objective', ['Each Category', 'Most Categories', 'Rotisserie'])
def test_complete_team_field_reproduces_the_app(objective):
    session_id, session = create_session(objective)
    agent = session.agent
    board = make_two_round_board(session)
    taken = {player for roster in board.values() for player in roster}
    full_team = board['Drafter 1'] + [int(p) for p in agent.default_h_scores.index if p not in taken][:agent.n_picks - 2]
    full_board = {**board, 'Drafter 1': full_team}
    app = client.post(f'/sessions/{session_id}/evaluate',
                      json={'player_assignments': full_board, 'my_team_id': 'Drafter 1'}).json()['candidates'][0]

    diff_means, diff_vars, volume_field = agent.get_complete_team_field(full_board, 'Drafter 1', [], None)
    volume_context = None if volume_field is None else {
        'my_x_total':     agent.x_scores.loc[full_team].sum().to_numpy()[agent.volume_category_indices].reshape(1, -1),
        'my_units':       agent.player_volume_units.loc[full_team].sum().to_numpy().reshape(1, -1),
        'opponent_units': volume_field['opponent_units'],
    }
    scores, win_rates = agent.compute_complete_team_scores_and_win_rates_batched(diff_means, diff_vars, volume_context)
    # The app rounds to 0.01 percentage points.
    assert abs(scores[0] * 100 - app['h_score']) <= 0.0051
    assert np.allclose(win_rates[0] * 100, app['win_rates'], atol=0.0051)


def test_draft_hypothetical_team_is_legal(draft_session):
    session_id, session = draft_session
    agent = session.agent
    board = make_two_round_board(session)
    request = {'player_assignments': board, 'my_team_id': 'Drafter 1'}
    candidates = client.post(f'/sessions/{session_id}/evaluate', json=request).json()['candidates']
    candidate = candidates[0]['player_id']

    response = client.post(f'/sessions/{session_id}/hypothetical-team', json={**request, 'candidate_player_id': candidate})
    assert response.status_code == 200, response.text
    data = response.json()
    assignments = {slot: a for slot, a in data['roster']['assignments'].items() if a is not None}
    assert len(assignments) == agent.n_picks, 'every active slot should be filled'

    kinds = {a['player_id']: a['kind'] for a in assignments.values()}
    assert {p for p, kind in kinds.items() if kind == 'rostered'} == set(board['Drafter 1'])
    assert [p for p, kind in kinds.items() if kind == 'candidate'] == [candidate]
    taken = {player for roster in board.values() for player in roster}
    assert not ({p for p, kind in kinds.items() if kind == 'hypothetical'} & taken), 'a hypothetical pick is rostered'

    flex = agent.position_config.position_structure['flex']
    for slot, assignment in assignments.items():
        slot_type = slot.rstrip('0123456789')
        eligible = set(flex[slot_type]['bases']) if slot_type in flex else {slot_type}
        assert set(agent.positions.loc[assignment['player_id']]) & eligible, f'{assignment} cannot play {slot}'

    # My players and the candidate sit exactly where the evaluate's roster put them.
    evaluated_roster = candidates[0]['roster']['assignments']
    for slot, assignment in evaluated_roster.items():
        if assignment is not None:
            assert data['roster']['assignments'][slot]['player_id'] == assignment['player_id'], slot

    # One pick per round from round 4 on, each a player of base round = that round or later.
    ranked = [p for p in agent.default_h_scores.index if p in agent.x_scores.index]
    base_round = {player: rank // agent.n_drafters + 1 for rank, player in enumerate(ranked)}
    picks = data['hypothetical_picks']
    assert [pick['pick_round'] for pick in picks] == list(range(4, agent.n_picks + 1))
    for pick in picks:
        assert base_round[pick['player_id']] >= pick['pick_round'], pick


def test_frank_wolfe_never_ends_below_the_naive_team(draft_session):
    session_id, session = draft_session
    agent = session.agent
    board = make_two_round_board(session)
    request = {'player_assignments': board, 'my_team_id': 'Drafter 1'}
    candidates = client.post(f'/sessions/{session_id}/evaluate', json=request).json()['candidates']
    strategies = session.candidate_strategy_cache[
        make_candidate_strategy_key(board, 'Drafter 1', [], None, None)][1]
    slot_names = make_slot_names(agent.position_config)
    for candidate in [c['player_id'] for c in candidates[:5]]:
        strategy = strategies[candidate]
        fixed_slot_types = [slot_names[slot_index].rstrip('0123456789') for slot_index in strategy.seating]
        expected = compute_expected_flex_position_counts(
            strategy.flex_shares, fixed_slot_types, agent.position_config.position_numbers)
        problem = exact_team_solves.build_team_solve_problem(
            agent, board, 'Drafter 1', candidate, [], [int(p) for p in agent.default_h_scores.index],
            True, fixed_slot_types, expected, None, None, None, None)
        naive = exact_team_solves.solve_naive_exact_team_for_draft(problem, strategy.category_weights)
        field = exact_team_solves.capture_complete_team_field(agent, problem, board, 'Drafter 1', [], naive, None)
        frank_wolfe = exact_team_solves.run_frank_wolfe(agent, problem, field, strategy.win_rates, naive)
        refined = exact_team_solves.solve_fw_exact_team_for_draft(agent, problem, field, strategy.win_rates, naive)
        naive_objective, frank_wolfe_objective, refined_objective = exact_team_solves.score_seatings(
            agent, problem, field, np.array([naive, frank_wolfe, refined]), strategy.win_rates)
        assert frank_wolfe_objective >= naive_objective - 1e-12
        assert refined_objective >= frank_wolfe_objective - 1e-12, 'the swaps never lower the objective'

        # The swapped team is a whole, legal team under every rule of the solve.
        assert np.array_equal(refined, np.round(refined))
        assert (problem.inequality_rows @ refined <= problem.inequality_bounds + 1e-9).all()
        assert np.allclose(problem.equality_rows @ refined, problem.equality_bounds)
        # ...and is a local optimum: running the swaps again changes nothing.
        assert np.array_equal(
            exact_team_solves.improve_team_by_swaps(agent, problem, field, strategy.win_rates, refined), refined)

        # Re-seating by specificity keeps exactly the same players.
        reseated = exact_team_solves.seat_team_by_specificity(problem, refined)
        selections = exact_team_solves.convert_seatings_to_selections(problem, np.array([refined, reseated]))
        assert np.array_equal(selections[0], selections[1])

        # Every flex slot type holds each base position as often as H-scoring's flex allocation expects, rounded.
        is_open = problem.seat_players >= problem.n_fixed_players
        for (slot, base), (low, high) in problem.flex_cell_bounds.items():
            assert low <= expected[slot][base] <= high
            in_cell = (is_open & (np.array(problem.slot_types)[problem.seat_slot_types] == slot)
                       & (np.array(problem.base_positions)[problem.seat_bases] == base))
            assert low <= refined[in_cell].sum() <= high, (slot, base)
            assert low <= reseated[in_cell].sum() <= high, (slot, base)


def test_an_unanchored_solve_drops_h_scorings_plan(draft_session):
    """Without anchoring, my players may take any slot they are eligible for, there are no flex-allocation rows, the
    objective is the plain complete-team H-score, and the solve still returns a legal team that never ends below the naive
    one. Mixing the switch with plan arguments it does not take is refused."""
    session_id, session = draft_session
    agent = session.agent
    board = make_two_round_board(session)
    request = {'player_assignments': board, 'my_team_id': 'Drafter 1'}
    candidate = client.post(f'/sessions/{session_id}/evaluate', json=request).json()['candidates'][0]['player_id']
    strategy = session.candidate_strategy_cache[make_candidate_strategy_key(board, 'Drafter 1', [], None, None)][1][candidate]
    order = [int(p) for p in agent.default_h_scores.index]
    slot_names = make_slot_names(agent.position_config)
    fixed_slot_types = [slot_names[slot_index].rstrip('0123456789') for slot_index in strategy.seating]
    expected = compute_expected_flex_position_counts(
        strategy.flex_shares, fixed_slot_types, agent.position_config.position_numbers)
    anchored = exact_team_solves.build_team_solve_problem(
        agent, board, 'Drafter 1', candidate, [], order, True, fixed_slot_types, expected, None, None, None, None)
    unanchored = exact_team_solves.build_team_solve_problem(
        agent, board, 'Drafter 1', candidate, [], order, False, None, None, None, None, None, None)

    assert unanchored.flex_cell_bounds == {} and anchored.flex_cell_bounds != {}
    flex_rows = 2 * len(anchored.flex_cell_bounds)
    assert len(unanchored.inequality_rows) == len(anchored.inequality_rows) - flex_rows
    for index in range(anchored.n_fixed_players):
        locked = set(anchored.seat_slot_types[anchored.seat_players == index])
        free = set(unanchored.seat_slot_types[unanchored.seat_players == index])
        assert len(locked) == 1 and locked < free, (index, locked, free)

    naive = exact_team_solves.solve_naive_exact_team_for_draft(unanchored, strategy.category_weights)
    field = exact_team_solves.capture_complete_team_field(agent, unanchored, board, 'Drafter 1', [], naive, None)
    seating = exact_team_solves.solve_fw_exact_team_for_draft(agent, unanchored, field, strategy.win_rates, naive)
    assert np.all(unanchored.inequality_rows @ seating <= unanchored.inequality_bounds + 1e-9), 'every constraint holds'
    assert np.allclose(unanchored.equality_rows @ seating, unanchored.equality_bounds)
    naive_objective, final_objective = exact_team_solves.score_seatings(
        agent, unanchored, field, np.array([naive, seating]), strategy.win_rates)
    assert final_objective >= naive_objective - 1e-12
    selection = exact_team_solves.convert_seatings_to_selections(unanchored, seating[np.newaxis, :])
    totals = selection @ unanchored.x_scores
    units = None if unanchored.volume_units is None else selection @ unanchored.volume_units
    plain = exact_team_solves.score_team_totals(agent, unanchored, field, totals, units, np.ones_like(strategy.win_rates))
    assert np.isclose(plain[0], final_objective), 'no penalty, even against impossible win-rate targets'

    with pytest.raises(ValueError):
        exact_team_solves.build_team_solve_problem(
            agent, board, 'Drafter 1', candidate, [], order, False, fixed_slot_types, expected, None, None, None, None)
    with pytest.raises(ValueError):
        exact_team_solves.build_team_solve_problem(
            agent, board, 'Drafter 1', candidate, [], order, True, None, None, None, None, None, None)


def test_a_solve_given_pick_numbers_follows_the_pick_order(draft_session):
    """With my remaining pick numbers, the planned team is draftable in the actual snake order: before each of my later
    picks the other drafters take the players ahead of it in the availability order, and at most i - 1 of my later
    players come from those taken before my i-th later pick. A wrong count of pick numbers is refused."""
    session_id, session = draft_session
    agent = session.agent
    board = make_two_round_board(session)
    request = {'player_assignments': board, 'my_team_id': 'Drafter 1'}
    candidate = client.post(f'/sessions/{session_id}/evaluate', json=request).json()['candidates'][0]['player_id']
    strategy = session.candidate_strategy_cache[make_candidate_strategy_key(board, 'Drafter 1', [], None, None)][1][candidate]
    order = [int(p) for p in agent.default_h_scores.index]
    n_drafters = agent.n_drafters
    # Drafter 1 picks first in odd rounds and last in even ones; two rounds are done, so the candidate is pick 25.
    my_pick_numbers = [n_drafters * (draft_round - 1) + 1 if draft_round % 2 == 1 else n_drafters * draft_round
                       for draft_round in range(3, agent.n_picks + 1)]
    problem = exact_team_solves.build_team_solve_problem(
        agent, board, 'Drafter 1', candidate, [], order, False, None, None, None, None, None, my_pick_numbers)

    naive = exact_team_solves.solve_naive_exact_team_for_draft(problem, strategy.category_weights)
    field = exact_team_solves.capture_complete_team_field(agent, problem, board, 'Drafter 1', [], naive, None)
    seating = exact_team_solves.solve_fw_exact_team_for_draft(agent, problem, field, strategy.win_rates, naive)
    assert np.all(problem.inequality_rows @ seating <= problem.inequality_bounds + 1e-9), 'every constraint holds'
    assert np.allclose(problem.equality_rows @ seating, problem.equality_bounds)
    chosen_places = sorted(problem.seat_players[(seating > 0.5) & (problem.seat_players >= problem.n_fixed_players)]
                           - problem.n_fixed_players)
    for later_index, later_pick in enumerate(my_pick_numbers[1:], start=1):
        taken_by_others = later_pick - my_pick_numbers[0] - later_index
        assert sum(place < taken_by_others for place in chosen_places) <= later_index - 1, later_pick

    with pytest.raises(ValueError):
        exact_team_solves.build_team_solve_problem(
            agent, board, 'Drafter 1', candidate, [], order, False, None, None, None, None, None, my_pick_numbers[1:])


def test_hypothetical_team_changes_nothing_the_evaluate_returns(draft_session):
    session_id, session = draft_session
    board = make_two_round_board(session)
    request = {'player_assignments': board, 'my_team_id': 'Drafter 1'}
    before = client.post(f'/sessions/{session_id}/evaluate', json=request).json()['candidates']
    for candidate in before[:3]:
        response = client.post(f'/sessions/{session_id}/hypothetical-team',
                               json={**request, 'candidate_player_id': candidate['player_id']})
        assert response.status_code == 200, response.text
    after = client.post(f'/sessions/{session_id}/evaluate', json=request).json()['candidates']
    assert [(c['player_id'], c['h_score'], c['win_rates'], c['category_weights']) for c in before] == \
           [(c['player_id'], c['h_score'], c['win_rates'], c['category_weights']) for c in after]


def test_auction_hypothetical_team_fits_the_budget():
    session_id, session = create_session('Each Category', cash_per_team=200)
    agent = session.agent
    teams = [f'Drafter {seat + 1}' for seat in range(session.current_settings['n_drafters'])]
    ranked = [int(player) for player in agent.default_h_scores.index]
    board = {team: [] for team in teams}
    board['Drafter 1'] = [ranked[0]]
    board['Drafter 2'] = [ranked[1]]
    remaining_cash = {team: 200.0 for team in teams}
    remaining_cash['Drafter 1'] = 140.0
    remaining_cash['Drafter 2'] = 150.0
    request = {'player_assignments': board, 'my_team_id': 'Drafter 1', 'remaining_cash': remaining_cash}
    candidates = client.post(f'/sessions/{session_id}/evaluate', json=request).json()['candidates']
    affordable = next(c for c in candidates if c['auction_values']['generic_dollar'] < 60)

    response = client.post(f'/sessions/{session_id}/hypothetical-team',
                           json={**request, 'candidate_player_id': affordable['player_id']})
    assert response.status_code == 200, response.text
    data = response.json()
    assert sum(a is not None for a in data['roster']['assignments'].values()) == agent.n_picks
    prices = [pick['predicted_price'] for pick in data['hypothetical_picks']]
    assert all(isinstance(price, int) and price >= 1 for price in prices)
    candidate_price = max(round(affordable['auction_values']['generic_dollar']), 1)
    assert sum(prices) + candidate_price <= 140


def solve_team_with_teams(agent, strategy, board, candidate, player_teams, remaining_cash=None, predicted_prices=None):
    """The app's pipeline (naive -> Frank-Wolfe and swaps) for one candidate with NBA-team rules; returns the problem and
    the final seating."""
    slot_names = make_slot_names(agent.position_config)
    fixed_slot_types = [slot_names[slot_index].rstrip('0123456789') for slot_index in strategy.seating]
    expected = compute_expected_flex_position_counts(
        strategy.flex_shares, fixed_slot_types, agent.position_config.position_numbers)
    problem = exact_team_solves.build_team_solve_problem(
        agent, board, 'Drafter 1', candidate, [], [int(p) for p in agent.default_h_scores.index],
        True, fixed_slot_types, expected, remaining_cash, predicted_prices, player_teams, None)
    if remaining_cash is None:
        naive = exact_team_solves.solve_naive_exact_team_for_draft(problem, strategy.category_weights)
    else:
        naive = exact_team_solves.solve_naive_exact_team_for_auction(problem, strategy.category_weights)
    field = exact_team_solves.capture_complete_team_field(agent, problem, board, 'Drafter 1', [], naive, None)
    if remaining_cash is None:
        seating = exact_team_solves.solve_fw_exact_team_for_draft(agent, problem, field, strategy.win_rates, naive)
    else:
        seating = exact_team_solves.solve_fw_exact_team_for_auction(agent, problem, field, strategy.win_rates, naive)
    return problem, seating


def test_draft_example_team_takes_its_first_picks_from_different_nba_teams(draft_session):
    session_id, session = draft_session
    agent = session.agent
    board = make_two_round_board(session)
    request = {'player_assignments': board, 'my_team_id': 'Drafter 1'}
    candidates = client.post(f'/sessions/{session_id}/evaluate', json=request).json()['candidates']
    strategies = session.candidate_strategy_cache[make_candidate_strategy_key(board, 'Drafter 1', [], None, None)][1]
    # Ten artificial NBA teams, so the rule binds hard: plenty of top players share one.
    player_teams = pd.Series({int(p): f'team {int(p) % 10}' for p in agent.default_h_scores.index})
    for candidate in [c['player_id'] for c in candidates[:3]]:
        problem, seating = solve_team_with_teams(agent, strategies[candidate], board, candidate, player_teams)
        assert problem.uses_integer_program
        team = [problem.pool_player_ids[i] for i in np.flatnonzero(
            exact_team_solves.convert_seatings_to_selections(problem, seating[np.newaxis, :])[0] > 0.5)]
        # My two players and the candidate are rounds 1-3; open players predicted for rounds 4-7 join them.
        top_picks = board['Drafter 1'] + [candidate] + [
            player for player in team if problem.pool_player_ids.index(player) >= problem.n_fixed_players
            and problem.base_rounds[problem.pool_player_ids.index(player)] <= exact_team_solves.DRAFT_DISTINCT_TEAM_ROUNDS]
        fixed_teams = [player_teams[p] for p in board['Drafter 1'] + [candidate]]
        open_teams = [player_teams[p] for p in top_picks[len(fixed_teams):]]
        for nba_team in set(open_teams):
            assert open_teams.count(nba_team) + fixed_teams.count(nba_team) <= max(1, fixed_teams.count(nba_team)), \
                f'{nba_team} repeats among the first picks'


def test_auction_example_team_holds_at_most_two_players_per_nba_team():
    session_id, session = create_session('Each Category', cash_per_team=200)
    agent = session.agent
    teams = [f'Drafter {seat + 1}' for seat in range(session.current_settings['n_drafters'])]
    board = {team: [] for team in teams}
    remaining_cash = {team: 200.0 for team in teams}
    request = {'player_assignments': board, 'my_team_id': 'Drafter 1', 'remaining_cash': remaining_cash}
    candidates = client.post(f'/sessions/{session_id}/evaluate', json=request).json()['candidates']
    strategies = session.candidate_strategy_cache[
        make_candidate_strategy_key(board, 'Drafter 1', [], remaining_cash, None)][1]
    player_teams = pd.Series({int(p): f'team {int(p) % 10}' for p in agent.default_h_scores.index})
    # Generic dollars for the whole available pool, as the service computes them.
    _, prices = compute_original_and_generic_dollars(
        agent.default_h_scores, agent.x_scores.index, agent.n_drafters, agent.n_picks,
        session.current_settings['n_picks'], 200, sum(remaining_cash.values()),
        session.current_settings['streaming_noise'], session.current_settings['replacement_threshold'],
        session.current_settings['cash_share_softness'])
    candidate = next(c for c in candidates if c['auction_values']['generic_dollar'] < 60)['player_id']
    problem, seating = solve_team_with_teams(agent, strategies[candidate], board, candidate, player_teams,
                                             remaining_cash=200.0, predicted_prices=prices)
    team = [problem.pool_player_ids[i] for i in np.flatnonzero(
        exact_team_solves.convert_seatings_to_selections(problem, seating[np.newaxis, :])[0] > 0.5)]
    nba_teams = [player_teams[p] for p in team]
    assert max(nba_teams.count(t) for t in set(nba_teams)) <= exact_team_solves.AUCTION_PLAYERS_PER_TEAM


def test_requests_without_an_evaluate_on_record_are_refused(draft_session):
    session_id, session = draft_session
    board = make_two_round_board(session)
    client.post(f'/sessions/{session_id}/evaluate', json={'player_assignments': board, 'my_team_id': 'Drafter 1'})
    some_player = next(int(p) for p in session.agent.default_h_scores.index
                       if p not in {q for roster in board.values() for q in roster})

    other_seat = client.post(f'/sessions/{session_id}/hypothetical-team', json={
        'player_assignments': board, 'my_team_id': 'Drafter 5', 'candidate_player_id': some_player})
    assert other_seat.status_code == 409

    auction_fields_on_a_draft = client.post(f'/sessions/{session_id}/hypothetical-team', json={
        'player_assignments': board, 'my_team_id': 'Drafter 1', 'candidate_player_id': some_player,
        'remaining_cash': {team: 200.0 for team in board}})
    assert auction_fields_on_a_draft.status_code == 400
