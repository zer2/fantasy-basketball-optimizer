"""Opponent awareness in auctions: the draft awareness experiment, with the auction autodrafter.

Two comparisons, as for drafts:
  - Against an unaware H-scoring field: twelve autodrafters, each with its own session (a shared agent would model one
    team from another team's view of it). One seat bids with the opponent model on (the app's confidence, 0.5), the
    other eleven with it off; the seat's H-score is compared with the same seat's in the all-off auction.
  - Against a G-score field: the H vs G auction (auction_h_versus_g) with the H seat's opponent model on, compared with
    the same seat's auction with it off.
Every team bids on the season's actual statistics, as in H vs G (auction_h_versus_g.AUCTION_SETTINGS: own-view noise
off, SAVOR off, beth = 0, psi = 0). Head to head only: Rotisserie pins the opponent model on, so it has no 'off'.
Every finished team is scored by one session with the opponent model off, so both arms are scored alike.
"""

from __future__ import annotations

import multiprocessing

from .auction_h_versus_g import (CASH_PER_TEAM, build_h_versus_g_auction_session, run_h_versus_g_auction,
                                 without_own_view_noise)
from .league_sessions import prepare_import_path

APP_CONFIDENCE = 0.5                    # parameters.yaml: opponent_model_confidence
H_FIELD = 'vs unaware H field'
G_FIELD = 'vs G field'


def run_autodrafter_auction(format_key: str
                            , season: str
                            , aware_seat: int | None) -> dict[str, float]:
    """One complete auction of twelve autodrafters, each with its own session: the one at aware_seat (if any) with the
    opponent model on, the rest with it off. Returns every team's H-score, in percent, scored with the model off."""
    prepare_import_path()
    from .auction_simulation import advance_nominator, name_auction_teams, run_bidding, score_final_teams
    from backend.services.auction_autodraft import select_autodrafter_nominee, value_player_for_team
    from backend.services.auction_pricing import compute_max_allowed_bids
    scorer = build_h_versus_g_auction_session(format_key, season, 0.0)
    teams = name_auction_teams(scorer)
    sessions = {team: build_h_versus_g_auction_session(format_key, season,
                                                       APP_CONFIDENCE if seat == aware_seat else 0.0)
                for seat, team in enumerate(teams)}
    total_roster_picks = scorer.current_settings['n_picks']
    player_assignments = {team: [] for team in teams}
    remaining_cash = {team: float(CASH_PER_TEAM) for team in teams}
    nominator_index = 0
    with without_own_view_noise():
        for _ in range(len(teams) * total_roster_picks):
            nominator = teams[nominator_index]
            nominee = select_autodrafter_nominee(sessions[nominator], player_assignments, remaining_cash, nominator)
            valuations = {team: value_player_for_team(sessions[team], player_assignments, remaining_cash, team, nominee,
                                                      {})[0]
                          for team in teams}
            max_allowed_bids = compute_max_allowed_bids(player_assignments, remaining_cash, total_roster_picks)
            winner_index, price = run_bidding(nominator_index, 1, valuations, max_allowed_bids, teams)
            player_assignments[teams[winner_index]].append(nominee)
            remaining_cash[teams[winner_index]] -= price
            nominator_index = advance_nominator(nominator_index, player_assignments, teams, total_roster_picks)
    if not all(len(roster) == total_roster_picks for roster in player_assignments.values()):
        raise RuntimeError('The auction ended with a roster unfilled.')
    return score_final_teams(scorer, player_assignments, remaining_cash, teams)


def measure_awareness_job(format_key: str
                          , season: str
                          , field: str
                          , seat: int | None
                          , opponent_model_confidence: float) -> dict:
    """One auction of the experiment. H field: seat None is the all-off auction (every team's score), otherwise the aware
    seat's. G field: the H seat's score at this confidence."""
    if field == H_FIELD:
        scores = run_autodrafter_auction(format_key, season, seat)
        return {'format': format_key, 'season': season, 'field': field, 'seat': seat, 'scores': scores}
    result = run_h_versus_g_auction(format_key, season, seat, opponent_model_confidence)
    return {'format': format_key, 'season': season, 'field': field, 'seat': seat,
            'opponent_model_confidence': opponent_model_confidence, 'h_score': result['h_score']}


def _run_reporting_failure(*job) -> tuple:
    import traceback
    try:
        return 'measured', measure_awareness_job(*job)
    except Exception:
        return 'failed', job, traceback.format_exc()


def measure_auction_awareness_in_parallel(format_key: str
                                          , seasons: list[str]
                                          , n_seats: int
                                          , n_workers: int) -> tuple[list[dict], list[dict]]:
    """Every auction of both comparisons for this format: per season, the all-off H-field auction, one with each seat
    aware, and each seat's G-field auction with the model on and off. Returns (results, failures)."""
    jobs = []
    for season in seasons:
        jobs.append((format_key, season, H_FIELD, None, 0.0))
        jobs += [(format_key, season, H_FIELD, seat, APP_CONFIDENCE) for seat in range(n_seats)]
        jobs += [(format_key, season, G_FIELD, seat, confidence) for seat in range(n_seats)
                 for confidence in (APP_CONFIDENCE, 0.0)]
    with multiprocessing.get_context('spawn').Pool(min(n_workers, len(jobs))) as pool:
        outcomes = pool.starmap(_run_reporting_failure, jobs, chunksize=1)
    results = [outcome[1] for outcome in outcomes if outcome[0] == 'measured']
    failures = [{'job': outcome[1], 'traceback': outcome[2]} for outcome in outcomes if outcome[0] == 'failed']
    return results, failures


def compute_awareness_gains(results: list[dict]
                            , season: str
                            , field: str
                            , n_seats: int) -> list[float]:
    """Per seat, its H-score with the opponent model on minus with it off, in percentage points."""
    from .auction_simulation import name_auction_teams   # noqa: F401  (team names follow 'Team i')
    season_results = [result for result in results if result['season'] == season and result['field'] == field]
    gains = []
    for seat in range(n_seats):
        team = f'Team {seat + 1}'
        if field == H_FIELD:
            baseline = next(result for result in season_results if result['seat'] is None)
            aware = next(result for result in season_results if result['seat'] == seat)
            gains.append(aware['scores'][team] - baseline['scores'][team])
        else:
            by_confidence = {result['opponent_model_confidence']: result['h_score']
                             for result in season_results if result['seat'] == seat}
            gains.append(by_confidence[APP_CONFIDENCE] - by_confidence[0.0])
    return gains
