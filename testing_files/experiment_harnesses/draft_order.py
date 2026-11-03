"""Early-seat bias under a snake draft and under third-round reversal, with every seat drafting by H-scoring.

One draft per (format, season, order): all twelve seats draft by H-scoring at the app's opponent-model confidence (0.5
head to head; Rotisserie pins it at 1), each with its own session -- a shared agent would model one seat's team from
another seat's view of it. Identical drafters make the draft deterministic, so one draft gives every seat's outcome.

Each team is scored by its complete-team H-score against the actual field, all by one agent, and head to head also in
real weeks (real_matchups: the same win rate the H-score expects, from random real weeks of the season). beth = 0 and
psi = 0, the H vs G conventions, so the H-score and the real weeks treat a missed game the same way.

The orders:
  - SNAKE: forward in the first round, then alternating.
  - THIRD_ROUND_REVERSAL: rounds one and two as a snake, round three in round two's direction again (the last seat of
    round one picks first in rounds two and three), then alternating from there.
The app models neither: H-scoring's future-pick model does not depend on the draft order, as it would not in a real
third-round-reversal league.
"""

from __future__ import annotations

import multiprocessing

from .league_sessions import build_league_session
from .real_matchups import load_season_weeks, simulate_real_win_rates

SNAKE = 'snake'
THIRD_ROUND_REVERSAL = 'third-round reversal'
ORDERS = (SNAKE, THIRD_ROUND_REVERSAL)
OBJECTIVES = {'EC': 'Each Category', 'MC': 'Most Categories', 'Roto': 'Rotisserie'}
# Head to head, the real counterpart of the H-score: Each Category's category win rate, Most Categories' matchup win
# rate. Rotisserie has none: one real season is one draw of the standings.
REAL_MEASURES = {'EC': 'category', 'MC': 'matchup'}
DRAFTER_SETTINGS = {'beth': 0, 'psi': 0, 'opponent_model_confidence': 0.5}
CANDIDATE_LIMIT = 40
REAL_MATCHUP_SIMULATIONS = 1000


def list_pick_order(order: str
                    , n_drafters: int
                    , n_rounds: int) -> list[int]:
    """The seat that picks at each pick of the draft."""
    from self_play import list_snake_order
    if order == SNAKE:
        return list_snake_order(n_drafters, n_rounds)
    if order == THIRD_ROUND_REVERSAL:
        forward, backward = list(range(n_drafters)), list(range(n_drafters))[::-1]
        # 0-based rounds: 0 forward, 1 backward, 2 backward again, then alternating (3 forward, 4 backward, ...).
        return [seat for pick_row in range(n_rounds)
                for seat in (forward if pick_row == 0 or (pick_row >= 3 and pick_row % 2 == 1) else backward)]
    raise ValueError(f'Unknown draft order {order!r}; expected one of {ORDERS}.')


def draft_and_record_picks(sessions: list
                           , n_drafters: int
                           , pick_order: list[int]) -> tuple[dict, list[dict]]:
    """draft_population's draft (every seat its own session, the top candidate of its first CANDIDATE_LIMIT), also
    recording every pick: its number, round, team, player and the category weights behind it (display units, 100 =
    neutral; None at a team's last pick, which has no future picks). Returns (assignments, picks)."""
    from backend.services.ranking import rank_candidates
    teams = [f'Drafter {seat + 1}' for seat in range(n_drafters)]
    assignments = {team: [] for team in teams}
    for session in sessions:
        session.agent.reset_draft_state()
    picks = []
    for pick_number, seat in enumerate(pick_order):
        team = teams[seat]
        result = rank_candidates(sessions[seat], assignments, team, [], None, 0, CANDIDATE_LIMIT)
        if not result.candidates:
            raise RuntimeError(f'No candidates for {team} at pick {pick_number + 1}.')
        chosen = result.candidates[0]
        is_last_pick = len(assignments[team]) == len(pick_order) // n_drafters - 1
        if chosen.category_weights is None and not is_last_pick:
            raise RuntimeError(f'No category weights for {team} at pick {pick_number + 1}, before its last pick.')
        # A team's last pick has no future picks, so no weights behind it.
        weights = None if chosen.category_weights is None else [float(weight) for weight in chosen.category_weights]
        picks.append({'pick': pick_number + 1, 'round': pick_number // n_drafters + 1, 'team': team,
                      'player_id': int(chosen.player_id), 'weights': weights})
        assignments[team].append(chosen.player_id)
    return assignments, picks


def measure_draft_order(format_key: str
                        , season: str
                        , order: str) -> dict:
    """One draft of twelve H-scoring seats in this order. Returns every seat's H-score and (head to head) real-week
    win rate, in percent, seat 1 first, and every pick (draft_and_record_picks)."""
    sessions = [build_league_session(OBJECTIVES[format_key], season, DRAFTER_SETTINGS)[2] for _ in range(12)]
    settings = sessions[0].current_settings
    n_drafters, n_picks, n_iterations = settings['n_drafters'], settings['n_picks'], settings['n_iterations']
    if n_drafters != len(sessions):
        raise ValueError(f'The league has {n_drafters} drafters; this experiment builds {len(sessions)} sessions.')
    assignments, picks = draft_and_record_picks(sessions, n_drafters, list_pick_order(order, n_drafters, n_picks))
    teams = [f'Drafter {seat + 1}' for seat in range(n_drafters)]
    scorer = sessions[0].agent
    h_scores = []
    for team in teams:
        scores = scorer.get_h_scores(assignments, team, n_iterations)['Scores']
        h_scores.append(100 * float(scores[scores.idxmax()]))
    real_rates = None
    if format_key in REAL_MEASURES:
        season_weeks = load_season_weeks(season)
        real_rates = [100 * simulate_real_win_rates(season_weeks, assignments, team, settings['categories'],
                                                    REAL_MATCHUP_SIMULATIONS, seed=seat)[REAL_MEASURES[format_key]]
                      for seat, team in enumerate(teams)]
    return {'format': format_key, 'season': season, 'order': order, 'h_scores': h_scores, 'real_rates': real_rates,
            'categories': list(settings['categories']),
            'value_direction': [float(value) for value in scorer.v.reshape(-1)], 'picks': picks}


def _measure_reporting_failure(*job) -> tuple:
    """measure_draft_order, returning ('measured', result) or ('failed', job, traceback) so one broken draft neither
    hides which it was nor discards the others."""
    import traceback
    try:
        return 'measured', measure_draft_order(*job)
    except Exception:
        return 'failed', job, traceback.format_exc()


def measure_draft_orders_in_parallel(format_key: str
                                     , seasons: list[str]
                                     , n_workers: int) -> tuple[list[dict], list[dict]]:
    """Both orders for every season of this format, one draft per job across n_workers processes (each job holds
    twelve sessions, so n_workers bounds the memory). Returns (results, failures)."""
    jobs = [(format_key, season, order) for season in seasons for order in ORDERS]
    with multiprocessing.get_context('spawn').Pool(min(n_workers, len(jobs))) as pool:
        outcomes = pool.starmap(_measure_reporting_failure, jobs, chunksize=1)
    results = [outcome[1] for outcome in outcomes if outcome[0] == 'measured']
    failures = [{'job': outcome[1], 'traceback': outcome[2]} for outcome in outcomes if outcome[0] == 'failed']
    return results, failures
