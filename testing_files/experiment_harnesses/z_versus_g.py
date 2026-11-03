"""Z-scores against G-scores: static drafts, half the seats ranking by each, scored by H-score and in real weeks.

Z-scores are the G-score pipeline without the weekly variance. process_player_data computes a player's scores as his
deviation over sqrt(alpha x variance of means + beta x mean of weekly variances): G-scores at alpha = beta = 1 (and
X-scores at alpha = 0). Z-scores are every G-score computation of that pipeline at beta = 0 -- the representative
player set, the percentage-volume correction, the games-played re-centring and the tiebreaker scaling included -- run on
exactly the inputs the session's own pipeline received (recorded while the session is built).

Every split of a 12-team league is drafted: 1 seat ranking by Z-score and 11 by G-score, 2 and 10, ... 11 and 1. Each
split is drafted DRAFTS_PER_SPLIT times as a snake, the Z seats chosen at random each time, so draft position averages
out (unequal groups cannot be paired by swapping them). Every seat drafts statically: the best player left on its
ranking whom its roster can still place (the H vs G experiment's G drafters). Every team is scored by its complete-team
H-score against the actual field, and head to head also in real weeks (real_matchups); a draft records each group's
mean. beth = 0 and psi = 0, as in H vs G.
"""

from __future__ import annotations

import multiprocessing
from contextlib import contextmanager

import numpy as np

from .league_sessions import build_league_session
from .real_matchups import load_season_weeks, simulate_real_win_rates

OBJECTIVES = {'EC': 'Each Category', 'MC': 'Most Categories', 'Roto': 'Rotisserie'}
REAL_MEASURES = {'EC': 'category', 'MC': 'matchup'}
SESSION_SETTINGS = {'beth': 0, 'psi': 0}
REAL_MATCHUP_SIMULATIONS = 1000
RANKINGS = ('Z', 'G')
# How many seats rank by Z-score in a draft: every split of the twelve, each group at least one seat.
Z_DRAFTER_COUNTS = tuple(range(1, 12))


@contextmanager
def record_player_data_inputs():
    """While a session is built: the keyword arguments its process_player_data call receives (yields the list they are
    appended to)."""
    import backend.services.build_agent as build_agent
    original = build_agent.process_player_data
    recorded = []

    def recording(**keyword_arguments):
        recorded.append(keyword_arguments)
        return original(**keyword_arguments)
    build_agent.process_player_data = recording
    try:
        yield recorded
    finally:
        build_agent.process_player_data = original


@contextmanager
def drop_weekly_variance():
    """Every G-score computation of process_player_data (alpha = beta = 1) at beta = 0: Z-scores. X-scores (alpha = 0)
    are untouched."""
    import backend.math.process_player_data as pipeline
    original = pipeline.calculate_scores_from_coefficients

    def without_weekly_variance(player_means, coefficients, sport_params, alpha_weight, beta_weight, *rest):
        if (alpha_weight, beta_weight) == (1, 1):
            beta_weight = 0
        return original(player_means, coefficients, sport_params, alpha_weight, beta_weight, *rest)
    pipeline.calculate_scores_from_coefficients = without_weekly_variance
    try:
        yield
    finally:
        pipeline.calculate_scores_from_coefficients = original


def build_rankings(format_key: str
                   , season: str) -> tuple:
    """(session, {'Z': ranking, 'G': ranking}) for this format and season: each ranking every player id, best first,
    by total score."""
    with record_player_data_inputs() as recorded:
        session = build_league_session(OBJECTIVES[format_key], season, SESSION_SETTINGS)[2]
    if len(recorded) != 1:
        raise RuntimeError(f'Expected one process_player_data call while building the session, saw {len(recorded)}.')
    from backend.math.process_player_data import process_player_data
    with drop_weekly_variance():
        z_info = process_player_data(**recorded[0])
    g_scores = session.agent.info['G-scores']
    z_scores = z_info['G-scores']
    if set(g_scores.index) != set(z_scores.index):
        raise RuntimeError('The Z-score pipeline scored a different set of players from the session\'s.')
    rankings = {'G': list(g_scores.sort_values('Total', ascending=False).index),
                'Z': list(z_scores.sort_values('Total', ascending=False).index)}
    return session, rankings


def draft_statically(session
                     , rankings: dict
                     , ranking_by_seat: list[str]) -> dict:
    """One snake draft, every seat taking the best player left on its own ranking whom its roster can still place."""
    from simulate import _has_position_data, _pick_gscore_player
    from self_play import list_snake_order
    settings = session.current_settings
    n_drafters, n_picks = settings['n_drafters'], settings['n_picks']
    teams = [f'Drafter {seat + 1}' for seat in range(n_drafters)]
    assignments = {team: [] for team in teams}
    drafted: set = set()
    position_config = session.agent.position_config
    has_positions = _has_position_data(session)
    positions_by_player = {player_id: identity.positions for player_id, identity in session.player_registry.items()}
    for seat in list_snake_order(n_drafters, n_picks):
        team = teams[seat]
        chosen = _pick_gscore_player(rankings[ranking_by_seat[seat]], drafted, assignments[team], position_config,
                                     has_positions, positions_by_player)
        assignments[team].append(chosen)
        drafted.add(chosen)
    return assignments


def measure_drafter_counts(format_key: str
                           , season: str
                           , drafts_per_split: int
                           , seed: int) -> list[dict]:
    """Every split in Z_DRAFTER_COUNTS, drafts_per_split drafts each on random seatings. Each draft record: how many seats
    ranked by Z-score and which, and each group's mean H-score and (head to head) real-week win rate, in percent."""
    session, rankings = build_rankings(format_key, season)
    settings = session.current_settings
    n_drafters, n_iterations = settings['n_drafters'], settings['n_iterations']
    teams = [f'Drafter {seat + 1}' for seat in range(n_drafters)]
    season_weeks = load_season_weeks(season) if format_key in REAL_MEASURES else None
    rng = np.random.default_rng(seed)
    records = []
    for n_z_drafters in Z_DRAFTER_COUNTS:
        for _ in range(drafts_per_split):
            z_seats = {int(seat) for seat in rng.choice(n_drafters, size=n_z_drafters, replace=False)}
            ranking_by_seat = ['Z' if seat in z_seats else 'G' for seat in range(n_drafters)]
            assignments = draft_statically(session, rankings, ranking_by_seat)
            h_scores, real_rates = [], []
            for seat, team in enumerate(teams):
                scores = session.agent.get_h_scores(assignments, team, n_iterations)['Scores']
                h_scores.append(100 * float(scores[scores.idxmax()]))
                if season_weeks is not None:
                    real_rates.append(100 * simulate_real_win_rates(
                        season_weeks, assignments, team, settings['categories'], REAL_MATCHUP_SIMULATIONS,
                        seed=seat)[REAL_MEASURES[format_key]])
            record = {'format': format_key, 'season': season, 'n_z_drafters': n_z_drafters,
                      'z_seats': sorted(z_seats)}
            for ranking in RANKINGS:
                seats = [seat for seat in range(n_drafters) if ranking_by_seat[seat] == ranking]
                record[f'{ranking}_h'] = float(np.mean([h_scores[seat] for seat in seats]))
                if real_rates:
                    record[f'{ranking}_real'] = float(np.mean([real_rates[seat] for seat in seats]))
            records.append(record)
    return records


def _measure_reporting_failure(*job) -> tuple:
    import traceback
    try:
        return 'measured', measure_drafter_counts(*job)
    except Exception:
        return 'failed', job, traceback.format_exc()


def measure_drafter_counts_in_parallel(format_key: str
                                       , seasons: list[str]
                                       , drafts_per_split: int
                                       , n_workers: int) -> tuple[list[dict], list[dict]]:
    """measure_drafter_counts for every season of this format, one season per job, its seatings seeded by the season's
    first year. Returns (draft records, failures)."""
    jobs = [(format_key, season, drafts_per_split, int(season[:4])) for season in seasons]
    with multiprocessing.get_context('spawn').Pool(min(n_workers, len(jobs))) as pool:
        outcomes = pool.starmap(_measure_reporting_failure, jobs, chunksize=1)
    records = [record for outcome in outcomes if outcome[0] == 'measured' for record in outcome[1]]
    failures = [{'job': outcome[1], 'traceback': outcome[2]} for outcome in outcomes if outcome[0] == 'failed']
    return records, failures
