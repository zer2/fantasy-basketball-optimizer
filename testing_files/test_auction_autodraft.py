# testing_files/test_auction_autodraft.py
# The auction autodrafter's valuations, kept between nominations (services/auction_autodraft.read_team_h_scores):
# a full solve gives exactly the "Your $" the H-score table shows (without the autodrafters' noise); after a purchase
# only the buyer re-solves, and only its top candidates; a board that is not the kept one plus purchases solves in full.
# Each autodrafter values players through its own noisy view of the X-scores (read_team_x_scores).

import numpy as np
import pytest

from benchmark_helpers import client, _build_session_request
from backend.state.session import get_session
from backend.services import auction_autodraft
from backend.services.auction_autodraft import (
    INITIAL_NOISY_CANDIDATES, REFRESH_CANDIDATE_MARGIN, X_SCORE_NOISE_STANDARD_DEVIATION,
    compute_noise_standard_deviations, count_remaining_picks, extends_board, is_in_scored_pool,
    nominate_auction_player, read_team_x_scores, value_auction_player,
)
from backend.services.ranking import rank_candidates

CASH = 200


@pytest.fixture(scope='module')
def auction_session():
    request = _build_session_request(objective='Each Category', cash_per_team=CASH)
    response = client.post('/sessions', json=request)
    assert response.status_code == 201, response.text
    return get_session(response.json()['session_id'])


def _empty_board(session):
    teams = session.current_settings['team_names']
    return {team: [] for team in teams}, {team: float(CASH) for team in teams}


def _count_available_pool(session, assignments):
    """The players a full solve covers: the scored pool less everyone rostered."""
    rostered = {player for roster in assignments.values() for player in roster}
    return sum(1 for player in session.agent.x_scores.index
               if player not in rostered and is_in_scored_pool(session, player))


def _record_solves(monkeypatch, session):
    """Every H-score solve the autodrafter asks for, as (team, number of candidates). A full solve names the whole
    available pool (so the noise applies on an empty board too)."""
    solves = []
    original = session.agent.get_h_scores

    def recording(**arguments):
        solves.append((arguments['drafter'], len(arguments['candidate_subset'])))
        return original(**arguments)

    monkeypatch.setattr(session.agent, 'get_h_scores', recording)
    return solves


def test_a_board_extends_another_only_by_purchases():
    kept = {'A': (1, 2), 'B': (3,)}
    assert extends_board(kept, {'A': (1, 2, 4), 'B': (3,)})
    assert not extends_board(kept, {'A': (1,), 'B': (3,)}), 'an undo'
    assert not extends_board(kept, {'A': (2, 1), 'B': (3,)}), 'a reordered roster'
    assert not extends_board(kept, {'A': (1, 2), 'B': (3,), 'C': ()}), 'another team'


def test_a_full_solve_values_players_as_the_h_score_table_does(auction_session, monkeypatch):
    """On a board with purchases, a team solved from nothing values every player at the table's "Your $", once the
    autodrafters' noise is taken out."""
    session = auction_session
    monkeypatch.setattr(auction_autodraft, 'X_SCORE_NOISE_STANDARD_DEVIATION', 0.0)
    session.auction_team_h_scores.clear()
    session.auction_team_x_scores.clear()
    assignments, cash = _empty_board(session)
    teams = list(assignments)
    first_players = list(session.agent.default_h_scores.index[:3])
    assignments[teams[0]] = [first_players[0]]
    assignments[teams[1]] = [first_players[1]]
    cash[teams[0]] -= 60
    cash[teams[1]] -= 45
    table = rank_candidates(session, assignments, teams[2], [], cash)
    table_dollars = {candidate.player_id: candidate.auction_values.your_dollar for candidate in table.candidates}
    sample = list(table_dollars)[:25]
    valued = {player: value_auction_player(session, assignments, cash, player, [teams[2]])[teams[2]]
              for player in sample}
    assert valued == {player: table_dollars[player] for player in sample}


def test_after_a_purchase_only_the_buyer_re_solves_its_top_candidates(auction_session, monkeypatch):
    session = auction_session
    session.auction_team_h_scores.clear()
    assignments, cash = _empty_board(session)
    teams = list(assignments)
    first = nominate_auction_player(session, assignments, cash, teams[0], None, None, teams)

    solves = _record_solves(monkeypatch, session)
    assignments[teams[3]] = [first.nominated_player_id]
    cash[teams[3]] -= 40
    nominate_auction_player(session, assignments, cash, teams[1], None, None, teams)
    expected_refresh = count_remaining_picks(session, assignments) + REFRESH_CANDIDATE_MARGIN
    assert solves == [(teams[3], expected_refresh)], 'one solve: the buyer, over its top candidates'

    second = nominate_auction_player(session, assignments, cash, teams[1], None, None, teams)
    assignments[teams[4]] = [second.nominated_player_id]
    cash[teams[4]] -= 30
    nominate_auction_player(session, assignments, cash, teams[2], None, None, teams)
    solves.clear()
    assignments[teams[4]] = []   # an undo: the board is no longer the kept one plus purchases
    cash[teams[4]] += 30
    nominate_auction_player(session, assignments, cash, teams[2], None, None, teams)
    full_pool = _count_available_pool(session, assignments)
    assert sorted(solves) == sorted((team, full_pool) for team in teams), 'every team solves in full'

    solves.clear()
    assignments[teams[3]] = []   # undone back to an empty board
    cash[teams[3]] += 40
    nominate_auction_player(session, assignments, cash, teams[0], None, None, teams)
    assert sorted(solves) == sorted((team, INITIAL_NOISY_CANDIDATES) for team in teams),         'an empty board starts from the neutral board, re-solving the top players of each team'


def test_a_rebuilt_agent_starts_again(auction_session, monkeypatch):
    session = auction_session
    session.auction_team_h_scores.clear()
    assignments, cash = _empty_board(session)
    teams = list(assignments)
    nominate_auction_player(session, assignments, cash, teams[0], None, None, teams)
    for kept in session.auction_team_h_scores.values():
        kept.agent = object()   # as if the session had been rebuilt since
    solves = _record_solves(monkeypatch, session)
    nominate_auction_player(session, assignments, cash, teams[0], None, None, teams)
    assert sorted(solves) == sorted((team, INITIAL_NOISY_CANDIDATES) for team in teams), 'on an empty board'


def test_the_module_keeps_one_entry_per_team(auction_session):
    session = auction_session
    session.auction_team_h_scores.clear()
    assignments, cash = _empty_board(session)
    teams = list(assignments)
    nominate_auction_player(session, assignments, cash, teams[0], None, None, teams)
    assert set(session.auction_team_h_scores) == set(teams)
    assert all(isinstance(kept, auction_autodraft.TeamHScores) for kept in session.auction_team_h_scores.values())


def test_your_dollar_reports_the_unusable_cash_it_includes(auction_session):
    """A team with far more cash than its open slots can use, under the app's smooth rule: every player it can roster
    gets a share of the cash, in proportion to his value squared, reported apart (unusable_cash_share) so the table can
    mark it; its best options (one per open active slot) together spend the cash; and the table reports the ratio the
    rule acted on (cash_to_cost_ratio)."""
    session = auction_session
    assignments, cash = _empty_board(session)
    teams = list(assignments)
    open_slots = 2
    taken = list(session.agent.default_h_scores.index[: session.agent.n_picks - open_slots])
    assignments[teams[0]] = taken
    cash = {team: 30.0 for team in teams}
    cash[teams[0]] = 150.0
    table = rank_candidates(session, assignments, teams[0], [], cash)
    shares = {candidate.player_id: candidate.auction_values.unusable_cash_share for candidate in table.candidates}
    values_before = {candidate.player_id: candidate.auction_values.your_dollar - shares[candidate.player_id]
                     for candidate in table.candidates}
    best_options = sorted(values_before, key=values_before.get, reverse=True)[:open_slots]
    assert all(share > 0 for share in shares.values()), 'the smooth rule gives every rosterable player a share'
    first, second = best_options
    assert shares[first] / shares[second] == pytest.approx(
        (max(values_before[first], 1) / max(values_before[second], 1)) ** 2, rel=0.02), 'in proportion to value squared'
    best_options_value = sum(candidate.auction_values.your_dollar for candidate in table.candidates
                             if candidate.player_id in best_options)
    bench_reserve = session.current_settings['n_picks'] - session.agent.n_picks
    assert best_options_value == pytest.approx(150.0 - bench_reserve, abs=0.05), 'together they spend the cash'
    best_options_cost = sum(max(values_before[player], 1) for player in best_options)
    ratios = {candidate.auction_values.cash_to_cost_ratio for candidate in table.candidates}
    assert len(ratios) == 1, 'one ratio for the team'
    assert ratios.pop() == pytest.approx((150.0 - bench_reserve) / best_options_cost, rel=0.01)


def test_each_autodrafter_sees_the_players_through_its_own_noise(auction_session):
    """Independent noise per team, of each player's stated size, the replacement player left as he is; the same view
    on every call for the same agent; and the agent's own X-scores untouched by a solve."""
    session = auction_session
    session.auction_team_x_scores.clear()
    teams = session.current_settings['team_names']
    agent_x_scores = session.agent.x_scores.copy()
    first, second = read_team_x_scores(session, teams[0]), read_team_x_scores(session, teams[1])
    first_noise = (first - agent_x_scores).drop(index=-1, errors='ignore').to_numpy()
    second_noise = (second - agent_x_scores).drop(index=-1, errors='ignore').to_numpy()
    standard_deviations = compute_noise_standard_deviations(session).drop(index=-1, errors='ignore')
    standardized = first_noise / standard_deviations.to_numpy()[:, np.newaxis]
    assert np.std(standardized) == pytest.approx(1.0, rel=0.05), 'each player at his own standard deviation'
    assert abs(np.corrcoef(first_noise.ravel(), second_noise.ravel())[0, 1]) < 0.05, 'independent between teams'
    if -1 in agent_x_scores.index:
        assert (first.loc[-1] == agent_x_scores.loc[-1]).all(), 'the replacement player is not a player'
    assert read_team_x_scores(session, teams[0]) is first, 'drawn once per agent'

    assignments, cash = _empty_board(session)
    nominate_auction_player(session, assignments, cash, teams[0], None, None, teams)
    assert session.agent.x_scores.equals(agent_x_scores), "the agent's own X-scores are restored after the solves"


def test_autodrafters_value_the_same_player_differently(auction_session):
    session = auction_session
    session.auction_team_h_scores.clear()
    session.auction_team_x_scores.clear()
    assignments, cash = _empty_board(session)
    teams = list(assignments)
    nomination = nominate_auction_player(session, assignments, cash, teams[0], None, None, teams)
    assert len(set(nomination.valuations.values())) > 1, 'the noise separates the teams on an empty board'


def test_the_noise_halves_every_league_of_players_down_the_neutral_board(auction_session):
    """0.2 for the best player, 0.1 at replacement level (a league's worth of players down), 0.05 a league further."""
    session = auction_session
    standard_deviations = compute_noise_standard_deviations(session)
    neutral_order = list(session.agent.default_h_scores.sort_values(ascending=False).index)
    players_drafted = session.agent.n_drafters * session.agent.n_picks
    assert standard_deviations[neutral_order[0]] == pytest.approx(X_SCORE_NOISE_STANDARD_DEVIATION)
    assert standard_deviations[neutral_order[players_drafted]] == pytest.approx(X_SCORE_NOISE_STANDARD_DEVIATION / 2)
    assert standard_deviations[neutral_order[2 * players_drafted]] == pytest.approx(X_SCORE_NOISE_STANDARD_DEVIATION / 4)
