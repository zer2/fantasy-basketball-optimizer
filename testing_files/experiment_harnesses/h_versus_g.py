"""H-scoring against a G-score field: one seat drafts with H-scoring, the other eleven take the best G-score player their
rosters can place. Shared by the H vs G experiment (test_h_scoring_beats_g_field) and the flex-regularization one.

The flex-regularization experiment drafts the H seat at several values of lambda_p (the pull of the flex-position shares
toward uniform), everything else at the H vs G conventions, so a season's drafts differ only in that dial. Every
finished team is scored by one session at the app's lambda_p: a completed roster has no future picks for lambda_p to
shape, and one scorer keeps the comparison purely about the drafting.
"""

from __future__ import annotations

import multiprocessing

from .league_sessions import build_league_session
from .real_matchups import load_season_weeks, simulate_real_win_rates

OBJECTIVES = {'EC': 'Each Category', 'MC': 'Most Categories', 'Roto': 'Rotisserie'}
# beth = 0 (historical stats are the truth), opponent model off (G drafters are not punt-pursuers), psi = 0 (a missed
# game counts as nothing, as the real-week simulation assumes).
H_VERSUS_G_SETTINGS = {'beth': 0, 'opponent_model_confidence': 0, 'psi': 0}
# Head to head, the real counterpart of the H-score: Each Category's category win rate, Most Categories' matchup win
# rate. Rotisserie has none: one real season is one draw of the standings.
REAL_MEASURES = {'EC': 'category', 'MC': 'matchup'}
REAL_MATCHUP_SIMULATIONS = 1000
APP_LAMBDA_P = 3.0                      # parameters.yaml
FLEX_LAMBDA_P_VALUES = (0.0, 1.0, 2.0, APP_LAMBDA_P)


def draft_h_seat_rosters(h_session
                         , seat: int
                         , candidate_limit: int = 40) -> dict:
    """One snake draft: `seat` drafts with the H agent, everyone else picks by G-score ranking.
    Returns every team's roster, {'Drafter i': [player id, ...]}."""
    from backend.services.ranking import rank_candidates
    from simulate import _gscore_ranking, _pick_gscore_player, _has_position_data   # season_simulation

    agent = h_session.agent
    agent.reset_draft_state()
    n_drafters = h_session.current_settings['n_drafters']
    n_picks = h_session.current_settings['n_picks']
    teams = [f'Drafter {i + 1}' for i in range(n_drafters)]
    assignments = {team: [] for team in teams}
    g_ranking = _gscore_ranking(h_session)
    has_positions = _has_position_data(h_session)
    position_config = agent.position_config
    positions_by_player = {player_id: identity.positions for player_id, identity in h_session.player_registry.items()}
    drafted: set = set()

    for pick_row in range(n_picks):
        for slot in range(n_drafters):
            index = slot if pick_row % 2 == 0 else (n_drafters - 1 - slot)
            team = teams[index]
            if index == seat:
                result = rank_candidates(h_session, assignments, team, [], None, 0, candidate_limit)
                chosen = result.candidates[0].player_id
            else:
                chosen = _pick_gscore_player(g_ranking, drafted, assignments[team], position_config, has_positions,
                                             positions_by_player)
            assignments[team].append(chosen)
            drafted.add(chosen)
    return assignments


def measure_flex_regularization(format_key: str
                                , season: str
                                , lambda_p_values: tuple) -> list[dict]:
    """Every seat's H vs G draft at each of these lambda_p: per (lambda_p, seat), the H seat's H-score and (head to
    head) real-week win rate, in percent."""
    objective = OBJECTIVES[format_key]
    scorer = build_league_session(objective, season, H_VERSUS_G_SETTINGS)[2]
    settings = scorer.current_settings
    if settings['lambda_p'] != APP_LAMBDA_P:
        raise ValueError(f'The scorer should run at the app\'s lambda_p {APP_LAMBDA_P}, not {settings["lambda_p"]}.')
    season_weeks = load_season_weeks(season) if format_key in REAL_MEASURES else None
    records = []
    for lambda_p in lambda_p_values:
        drafter = build_league_session(objective, season, {**H_VERSUS_G_SETTINGS, 'lambda_p': lambda_p})[2]
        for seat in range(settings['n_drafters']):
            assignments = draft_h_seat_rosters(drafter, seat)
            team = f'Drafter {seat + 1}'
            scores = scorer.agent.get_h_scores(assignments, team, settings['n_iterations'])['Scores']
            real_rate = None
            if season_weeks is not None:
                real_rate = 100 * simulate_real_win_rates(season_weeks, assignments, team, settings['categories'],
                                                          REAL_MATCHUP_SIMULATIONS, seed=seat)[REAL_MEASURES[format_key]]
            records.append({'format': format_key, 'season': season, 'lambda_p': lambda_p, 'seat': seat + 1,
                            'h_score': 100 * float(scores[scores.idxmax()]), 'real_rate': real_rate})
    return records


def _measure_reporting_failure(*job) -> tuple:
    """measure_flex_regularization, returning ('measured', records) or ('failed', job, traceback)."""
    import traceback
    try:
        return 'measured', measure_flex_regularization(*job)
    except Exception:
        return 'failed', job, traceback.format_exc()


def measure_flex_regularization_in_parallel(format_key: str
                                            , seasons: list[str]
                                            , lambda_p_values: tuple
                                            , n_workers: int) -> tuple[list[dict], list[dict]]:
    """Every season of this format at these lambda_p, one season per job across n_workers processes. Returns (records,
    failures)."""
    jobs = [(format_key, season, lambda_p_values) for season in seasons]
    with multiprocessing.get_context('spawn').Pool(min(n_workers, len(jobs))) as pool:
        outcomes = pool.starmap(_measure_reporting_failure, jobs, chunksize=1)
    records = [record for outcome in outcomes if outcome[0] == 'measured' for record in outcome[1]]
    failures = [{'job': outcome[1], 'traceback': outcome[2]} for outcome in outcomes if outcome[0] == 'failed']
    return records, failures
