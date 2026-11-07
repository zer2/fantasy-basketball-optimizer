"""Snake vs third-round reversal with a preseason projection as the ground truth.

The draft-order experiment (draft_order.py) drafts each season's actual statistics, which knows in hindsight who the
best players were: the first picks are the season's real stars, which may flatter the first seats. Here the pool is a
preseason projection instead -- mean outcomes, as a drafter would have seen them -- and the same projection scores every
team. Everything else is draft_order's: twelve H-scoring seats, each its own session, at the app's opponent-model
confidence, beth = 0 and psi = 0 (the projection stands in for the truth).

The projections are ESPN's stored loads (ESPN_PROJECTION_TABLE), exported to a CSV and uploaded as a custom projection
source, the way a user brings their own. Real-week win rates do not apply: the truth is the projection, not games.
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path

from .draft_order import CANDIDATE_LIMIT, DRAFTER_SETTINGS, ORDERS, OBJECTIVES, draft_and_record_picks, list_pick_order
from .league_sessions import build_league_session, prepare_import_path
from .real_matchups import load_season_weeks, simulate_real_win_rates

# The season each projection is for, where that season has been played: its drafts can be scored on what it was.
PLAYED_SEASONS = {'ESPN preseason 2025-26': '2025-26'}
# Head to head, the real counterpart of the H-score (as in draft_order.py).
REAL_MEASURES = {'EC': 'category', 'MC': 'matchup'}
REAL_MATCHUP_SIMULATIONS = 1000
# Each projection: its label and the ESPN_PROJECTION_TABLE load (UPDATE_TIMESTAMP, to the second) it is.
PROJECTIONS = {
    'ESPN preseason 2025-26': '2025-09-23 19:56:59',
    'ESPN preseason 2026-27': '2026-09-24 19:46:13',
}
PROJECTION_DIRECTORY = Path(__file__).resolve().parents[1] / 'experiment_results' / 'projections'
# Players a projection still lists who were known before the season to miss all of it, kept out of its sessions through
# the injured / excluded list, as a user would: {projection: {player id: why}}.
KNOWN_ABSENCES = {
    'ESPN preseason 2025-26': {202681: 'Kyrie Irving: out for the season before it began (ESPN still projected 43 games)'},
    'ESPN preseason 2026-27': {},
}


def export_projection_csv(label: str) -> Path:
    """The projection's ESPN load as a CSV in the app's column names (the espn-renamer's), written once."""
    path = PROJECTION_DIRECTORY / f'{label.replace(" ", "_")}.csv'
    if path.exists():
        return path
    prepare_import_path()
    from backend.infra.snowflake_connection import run_query
    from backend.services.build_agent import load_all_params
    renamer = load_all_params()['NBA']['espn-renamer']
    frame = run_query(f"SELECT * FROM ESPN_PROJECTION_TABLE "
                      f"WHERE TO_CHAR(UPDATE_TIMESTAMP, 'YYYY-MM-DD HH24:MI:SS') = '{PROJECTIONS[label]}'")
    if frame.empty:
        raise LookupError(f'No ESPN_PROJECTION_TABLE load at {PROJECTIONS[label]} for {label}.')
    PROJECTION_DIRECTORY.mkdir(parents=True, exist_ok=True)
    frame[list(renamer)].rename(columns=renamer).to_csv(path, index=False)
    return path


def build_projection_session(objective: str
                             , csv_path: Path
                             , absent_player_ids: list[int]):
    """A session (the backend tests' league, 12 x 13) whose whole pool is this projection CSV, uploaded as a custom
    source at weight 1, with these players on the injured / excluded list."""
    prepare_import_path()
    from benchmark_helpers import client, _build_session_request
    from backend.state.session import get_session
    with open(csv_path, 'rb') as upload:
        uploaded = client.post('/data/upload', files={'file': (csv_path.name, upload, 'text/csv')})
    if uploaded.status_code != 200:
        raise RuntimeError(f'Upload of {csv_path.name} failed: {uploaded.text[:300]}')
    data_id = uploaded.json()['data_id']
    request = _build_session_request(objective=objective)
    request['data_source'] = {'type': 'projections', 'season': None, 'blend_weights': {'ESPN': 0.0, 'DARKO': 0.0, data_id: 1.0},
                              'custom_data_ids': [data_id]}
    request['model_settings'].update(DRAFTER_SETTINGS)
    request['injured_players'] = absent_player_ids
    response = client.post('/sessions', json=request)
    if response.status_code != 201:
        raise RuntimeError(f'Session creation failed: {response.text[:300]}')
    return get_session(response.json()['session_id'])


def measure_projection_draft_order(format_key: str
                                   , label: str
                                   , order: str) -> dict:
    """One draft of twelve H-scoring seats on this projection, in this order: every seat's H-score (percent, scored on
    the same projection), seat 1 first, and every pick."""
    csv_path = export_projection_csv(label)
    sessions = [build_projection_session(OBJECTIVES[format_key], csv_path, list(KNOWN_ABSENCES[label]))
                for _ in range(12)]
    settings = sessions[0].current_settings
    n_drafters, n_picks, n_iterations = settings['n_drafters'], settings['n_picks'], settings['n_iterations']
    if n_drafters != len(sessions):
        raise ValueError(f'The league has {n_drafters} drafters; this experiment builds {len(sessions)} sessions.')
    assignments, picks = draft_and_record_picks(sessions, n_drafters, list_pick_order(order, n_drafters, n_picks))
    scorer = sessions[0].agent
    h_scores = []
    for seat in range(n_drafters):
        scores = scorer.get_h_scores(assignments, f'Drafter {seat + 1}', n_iterations)['Scores']
        h_scores.append(100 * float(scores[scores.idxmax()]))
    # 'season' is the projection's label, so save_draft_picks keys these drafts as it keys the historical ones.
    return {'format': format_key, 'projection': label, 'season': label, 'order': order, 'h_scores': h_scores,
            'categories': list(settings['categories']),
            'value_direction': [float(value) for value in scorer.v.reshape(-1)], 'picks': picks}


def _measure_reporting_failure(*job) -> tuple:
    """measure_projection_draft_order, returning ('measured', result) or ('failed', job, traceback)."""
    import traceback
    try:
        return 'measured', measure_projection_draft_order(*job)
    except Exception:
        return 'failed', job, traceback.format_exc()


def measure_projection_draft_orders_in_parallel(format_key: str
                                                , n_workers: int) -> tuple[list[dict], list[dict]]:
    """Both orders on every projection for this format, one draft per job. Returns (results, failures)."""
    for label in PROJECTIONS:
        export_projection_csv(label)            # once, before the workers race to write it
    jobs = [(format_key, label, order) for label in PROJECTIONS for order in ORDERS]
    with multiprocessing.get_context('spawn').Pool(min(n_workers, len(jobs))) as pool:
        outcomes = pool.starmap(_measure_reporting_failure, jobs, chunksize=1)
    results = [outcome[1] for outcome in outcomes if outcome[0] == 'measured']
    failures = [{'job': outcome[1], 'traceback': outcome[2]} for outcome in outcomes if outcome[0] == 'failed']
    return results, failures


def score_drafts_on_played_season(format_key: str
                                  , drafts: list[dict]) -> list[dict]:
    """Each recorded draft on a projection whose season has been played, its teams scored on that season's actual
    statistics (draft_order's conventions) and, head to head, in its real weeks. A drafted player the season has no
    statistics for raises: there is no right way to score him silently."""
    by_season: dict = {}
    for draft in drafts:
        if draft['format'] == format_key and draft['season'] in PLAYED_SEASONS:
            by_season.setdefault(PLAYED_SEASONS[draft['season']], []).append(draft)
    results = []
    for season, season_drafts in by_season.items():
        actual = build_league_session(OBJECTIVES[format_key], season, DRAFTER_SETTINGS)[2]
        settings = actual.current_settings
        season_weeks = load_season_weeks(season) if format_key in REAL_MEASURES else None
        for draft in season_drafts:
            teams = sorted({pick['team'] for pick in draft['picks']}, key=lambda team: int(team.split()[-1]))
            assignments = {team: [pick['player_id'] for pick in draft['picks'] if pick['team'] == team] for team in teams}
            absent = sorted({player for roster in assignments.values() for player in roster
                             if player not in actual.player_registry})
            if absent:
                raise LookupError(f'{draft["season"]} {draft["order"]} ({format_key}): drafted players with no '
                                  f'{season} statistics: {absent}')
            h_scores, real_rates = [], []
            for seat, team in enumerate(teams):
                scores = actual.agent.get_h_scores(assignments, team, settings['n_iterations'])['Scores']
                h_scores.append(100 * float(scores[scores.idxmax()]))
                if season_weeks is not None:
                    real_rates.append(100 * simulate_real_win_rates(
                        season_weeks, assignments, team, settings['categories'], REAL_MATCHUP_SIMULATIONS,
                        seed=seat)[REAL_MEASURES[format_key]])
            results.append({'format': format_key, 'season': draft['season'], 'order': draft['order'],
                            'h_scores': h_scores, 'real_rates': real_rates or None})
    return results
