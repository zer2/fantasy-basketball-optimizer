"""Auctions with one H-scoring autodrafter against eleven G-score bidders: the auction counterpart of the H-scoring vs a
G-score field draft experiment.

The H seat is the app's autodrafter (backend/services/auction_autodraft): it nominates its best available player that
fits its roster and values players by its "Your $". A G-score bidder values a player at the auction table's G-score
"Gnrc. $": the league's G-score dollars for the players still available (compute_league_generic_dollars over the
G-score totals), plus the team's own share of the cash it could not otherwise use, with the app's cash-share rule --
$0 for a player its roster cannot place, $1 once only perma-bench slots remain, $0 with a full roster, as the
autodrafter's own valuation rules. It nominates its highest-G-score available player that its roster can place,
opening at $1. The bidding and the nomination order are the browser's (auction_simulation.run_bidding,
advance_nominator).

As in the draft experiment, every bidder sees the season's actual statistics: the autodrafter's own-view noise is off,
SAVOR is off (S = 0, S_R = 0), beth = 0 and psi = 0, and the H seat's opponent model is off (C = 0; Rotisserie pins it
on). The H seat's finished team is scored by its complete-team H-score against the field and, head to head, in real
weeks (real_matchups).
"""

from __future__ import annotations

import multiprocessing
from contextlib import contextmanager

import numpy as np
import pandas as pd

from .league_sessions import prepare_import_path
from .real_matchups import load_season_weeks, simulate_real_win_rates

OBJECTIVES = {'EC': 'Each Category', 'MC': 'Most Categories', 'Roto': 'Rotisserie'}
REAL_MEASURES = {'EC': 'category', 'MC': 'matchup'}
CASH_PER_TEAM = 200
AUCTION_SETTINGS = {'beth': 0, 'psi': 0, 'opponent_model_confidence': 0, 'streaming_noise': 0,
                    'replacement_threshold': 0}
REAL_MATCHUP_SIMULATIONS = 1000


@contextmanager
def without_own_view_noise():
    """The autodrafter's own view of the players without its noise: every team bids from the actual X-scores."""
    import backend.services.auction_autodraft as autodraft
    original = autodraft.compute_noise_standard_deviations

    def no_noise(session):
        return original(session) * 0.0
    autodraft.compute_noise_standard_deviations = no_noise
    try:
        yield
    finally:
        autodraft.compute_noise_standard_deviations = original


def build_h_versus_g_auction_session(format_key: str
                                     , season: str):
    """An auction session on the season's actual stats with the settings above."""
    prepare_import_path()
    import benchmark_helpers
    from backend.state.session import get_session
    request = benchmark_helpers._build_session_request(objective=OBJECTIVES[format_key], cash_per_team=CASH_PER_TEAM)
    request['model_settings'].update(AUCTION_SETTINGS)
    request['data_source'] = {'type': 'historical', 'season': season}
    response = benchmark_helpers.client.post('/sessions', json=request)
    if response.status_code != 201:
        raise RuntimeError(f'Session build failed ({response.status_code}): {response.text[:300]}')
    return get_session(response.json()['session_id'])


def list_fitting_players(players: list[int]
                         , roster: list[int]
                         , position_config
                         , has_positions: bool
                         , positions_by_player: dict) -> list[int]:
    """The players a roster can still place (the G-score drafter's eligibility check, simulate._pick_gscore_player)."""
    from backend.math.position_optimization import check_single_player_eligibility
    if not has_positions:
        return list(players)
    roster_positions = [positions_by_player[teammate] for teammate in roster]
    return [player for player in players
            if check_single_player_eligibility(positions_by_player[player], roster_positions, position_config)]


def value_for_g_bidder(session
                       , player_assignments: dict[str, list[int]]
                       , remaining_cash: dict[str, float]
                       , team: str
                       , player_id: int
                       , g_totals: pd.Series
                       , fitting_players: list[int]) -> float:
    """A G-score bidder's valuation of a player: the auction table's G-score "Gnrc. $" for its team, under the
    autodrafter's roster rules."""
    from backend.services.auction_pricing import sum_spendable_league_cash
    from backend.services.auction_autodraft import count_remaining_picks
    from backend.services.ranking import add_team_cash_shares_to_league_dollars, compute_league_generic_dollars
    agent, settings = session.agent, session.current_settings
    roster_size = len(player_assignments[team])
    if roster_size >= settings['n_picks']:
        return 0.0
    if roster_size >= agent.n_picks:
        return 1.0
    if player_id not in fitting_players:
        return 0.0
    rostered = {player for roster in player_assignments.values() for player in roster}
    available = pd.Index([player for player in g_totals.sort_values(ascending=False).index if player not in rostered])
    if count_remaining_picks(session, player_assignments) <= 0:
        raise ValueError('No active picks remain, so the G-score dollars have nothing to price.')
    league_dollars = compute_league_generic_dollars(
        g_totals, available, agent.n_drafters, agent.n_picks, CASH_PER_TEAM,
        sum_spendable_league_cash(player_assignments, remaining_cash, agent.n_picks), 0.0, 0.0)
    fits = pd.Series(False, index=league_dollars.index)
    fits[[player for player in fitting_players if player in fits.index]] = True
    dollars = add_team_cash_shares_to_league_dollars(
        league_dollars, fits, remaining_cash[team], agent.n_picks - min(roster_size, agent.n_picks),
        settings['n_picks'] - roster_size, settings['cash_share_softness'])
    return round(float(dollars[player_id]), 2)


def run_h_versus_g_auction(format_key: str
                           , season: str
                           , h_seat: int) -> dict:
    """One complete auction with the H-scoring autodrafter at board position h_seat. Returns its H-score and (head to
    head) real-week win rate, in percent, and what it spent and kept."""
    prepare_import_path()
    from simulate import _has_position_data
    from .auction_simulation import advance_nominator, name_auction_teams, run_bidding, score_final_teams
    from backend.services.auction_autodraft import select_autodrafter_nominee, value_player_for_team
    from backend.services.auction_pricing import compute_max_allowed_bids
    session = build_h_versus_g_auction_session(format_key, season)
    agent, settings = session.agent, session.current_settings
    teams = name_auction_teams(session)
    h_team = teams[h_seat]
    g_totals = agent.info['G-scores']['Total'].drop(index=[player for player in agent.info['G-scores'].index
                                                           if player not in session.player_registry])
    g_order = list(g_totals.sort_values(ascending=False).index)
    has_positions = _has_position_data(session)
    positions_by_player = {player: identity.positions for player, identity in session.player_registry.items()}
    player_assignments = {team: [] for team in teams}
    remaining_cash = {team: float(CASH_PER_TEAM) for team in teams}
    total_roster_picks = settings['n_picks']
    nominator_index = 0
    with without_own_view_noise():
        for _ in range(len(teams) * total_roster_picks):
            nominator = teams[nominator_index]
            rostered = {player for roster in player_assignments.values() for player in roster}
            available_g = [player for player in g_order if player not in rostered]
            fitting = {team: list_fitting_players(available_g, player_assignments[team], agent.position_config,
                                                  has_positions, positions_by_player)
                       for team in teams if team != h_team}
            if nominator == h_team:
                nominee = select_autodrafter_nominee(session, player_assignments, remaining_cash, h_team)
            else:
                # Its best fitting player; with none left to fit (a cornered roster), its best available one.
                nominee = (fitting[nominator] or available_g)[0]
            valuations = {team: (value_player_for_team(session, player_assignments, remaining_cash, team, nominee,
                                                       {})[0]
                                 if team == h_team else
                                 value_for_g_bidder(session, player_assignments, remaining_cash, team, nominee,
                                                    g_totals, fitting[team]))
                          for team in teams}
            max_allowed_bids = compute_max_allowed_bids(player_assignments, remaining_cash, total_roster_picks)
            winner_index, price = run_bidding(nominator_index, 1, valuations, max_allowed_bids, teams)
            player_assignments[teams[winner_index]].append(nominee)
            remaining_cash[teams[winner_index]] -= price
            nominator_index = advance_nominator(nominator_index, player_assignments, teams, total_roster_picks)
    if not all(len(roster) == total_roster_picks for roster in player_assignments.values()):
        raise RuntimeError('The auction ended with a roster unfilled.')
    h_score = score_final_teams(session, player_assignments, remaining_cash, [h_team])[h_team]   # in percent
    real_rate = None
    if format_key in REAL_MEASURES:
        real_rate = 100 * simulate_real_win_rates(load_season_weeks(season), player_assignments, h_team,
                                                  settings['categories'], REAL_MATCHUP_SIMULATIONS,
                                                  seed=h_seat)[REAL_MEASURES[format_key]]
    return {'format': format_key, 'season': season, 'h_seat': h_seat, 'h_score': h_score, 'real_rate': real_rate,
            'cash_left': remaining_cash[h_team]}


def _run_reporting_failure(*job) -> tuple:
    import traceback
    try:
        return 'measured', run_h_versus_g_auction(*job)
    except Exception:
        return 'failed', job, traceback.format_exc()


def run_h_versus_g_auctions_in_parallel(format_key: str
                                        , seasons: list[str]
                                        , n_seats: int
                                        , n_workers: int) -> tuple[list[dict], list[dict]]:
    """Every (season, H seat) auction of this format across n_workers processes. Returns (results, failures)."""
    jobs = [(format_key, season, seat) for season in seasons for seat in range(n_seats)]
    with multiprocessing.get_context('spawn').Pool(min(n_workers, len(jobs))) as pool:
        outcomes = pool.starmap(_run_reporting_failure, jobs, chunksize=1)
    results = [outcome[1] for outcome in outcomes if outcome[0] == 'measured']
    failures = [{'job': outcome[1], 'traceback': outcome[2]} for outcome in outcomes if outcome[0] == 'failed']
    return results, failures
