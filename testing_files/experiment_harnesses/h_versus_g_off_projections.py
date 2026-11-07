"""H-scoring against a G-score field, drafted off a preseason projection and scored on what the season actually was.

Every drafter sees only the projection: the H seat's session is built on it (the app's settings, the opponent model
off as in H vs G), and the eleven G-score drafters rank by its G-scores. Each finished team is then scored on the
season's actual statistics, at the H vs G conventions (beth = 0, psi = 0), and head to head also in real weeks.

The projections are the 2025-26 preseason ones, the one season with a full preseason projection and an actual season to
score it on:
  - ESPN: ESPN's preseason projection (ESPN_PROJECTION_TABLE, loaded 2025-09-23).
  - ESPN + DARKO: that and DARKO's preseason projection (DARKO_PLAYER_TABLE, the projection dated 2025-07-04 as loaded
    the morning the season opened), blended 50/50 as the app blends sources. DARKO's per-100 rates become per-game with
    ESPN's preseason minutes, as the app does (backend.data_retrieval.convert_darko_to_per_game).
Each is uploaded as custom projection sources, the way a user brings their own.
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path

from .h_versus_g import H_VERSUS_G_SETTINGS, OBJECTIVES, REAL_MATCHUP_SIMULATIONS, REAL_MEASURES, draft_h_seat_rosters
from .league_sessions import build_league_session, prepare_import_path
from .projection_draft_order import KNOWN_ABSENCES, PROJECTION_DIRECTORY, PROJECTIONS, export_projection_csv
from .real_matchups import load_season_weeks, simulate_real_win_rates

SEASON = '2025-26'
ESPN_LABEL = 'ESPN preseason 2025-26'
# DARKO's preseason projection for 2025-26, as loaded the morning the season opened (2025-10-21).
DARKO_PRESEASON_LOAD = '2025-10-21 00:03:34.812'
# Each projection: its sources and their blend weights.
PROJECTION_BLENDS = {
    'ESPN': {'ESPN': 1.0},
    'ESPN + DARKO (50/50)': {'ESPN': 0.5, 'DARKO': 0.5},
}
# The drafting session: the app's settings, with the opponent model off (the field drafts by G-score, as in H vs G)
# and beth = 0: beth allows for the projection being wrong, and here the projection is the only view anyone drafts by.
DRAFTING_SETTINGS = {'opponent_model_confidence': 0, 'beth': 0}


def export_darko_preseason_csv() -> Path:
    """DARKO's 2025-26 preseason projection, per game, as an upload CSV (written once)."""
    path = PROJECTION_DIRECTORY / 'DARKO_preseason_2025-26.csv'
    if path.exists():
        return path
    prepare_import_path()
    from backend.data_retrieval import convert_darko_to_per_game
    from backend.infra.snowflake_connection import run_query
    from backend.services.build_agent import load_all_params
    sport_params = load_all_params()['NBA']
    # DARKO_VIEW's own definition, at that load instead of the latest.
    definition = run_query("select get_ddl('view', 'DARKO_VIEW')").iloc[0, 0]
    latest = '(SELECT MAX(UPDATE_TIMESTAMP) FROM DARKO_PLAYER_TABLE)'
    if definition.count(latest) != 1:
        raise RuntimeError('DARKO_VIEW no longer selects its load as MAX(UPDATE_TIMESTAMP); update this export.')
    body = definition[definition.upper().index(') AS') + len(') AS'):].strip().rstrip(';')
    darko_rows = run_query(body.replace(latest, f"'{DARKO_PRESEASON_LOAD}'"))
    espn_rows = run_query("SELECT * FROM ESPN_PROJECTION_TABLE "
                          f"WHERE TO_CHAR(UPDATE_TIMESTAMP, 'YYYY-MM-DD HH24:MI:SS') = '{PROJECTIONS[ESPN_LABEL]}'")
    per_game = convert_darko_to_per_game(darko_rows, espn_rows, sport_params).drop(columns=['player_id'])
    PROJECTION_DIRECTORY.mkdir(parents=True, exist_ok=True)
    per_game.to_csv(path, index=False)
    return path


def build_blend_session(objective: str
                        , blend: dict):
    """A session (12 x 13) on this blend of the preseason projections, each uploaded as a custom source."""
    prepare_import_path()
    from benchmark_helpers import client, _build_session_request
    from backend.state.session import get_session
    paths = {'ESPN': export_projection_csv(ESPN_LABEL), 'DARKO': export_darko_preseason_csv()}
    weights, data_ids = {'ESPN': 0.0, 'DARKO': 0.0}, []
    for source, weight in blend.items():
        with open(paths[source], 'rb') as upload:
            uploaded = client.post('/data/upload', files={'file': (paths[source].name, upload, 'text/csv')})
        if uploaded.status_code != 200:
            raise RuntimeError(f'Upload of {paths[source].name} failed: {uploaded.text[:300]}')
        weights[uploaded.json()['data_id']] = weight
        data_ids.append(uploaded.json()['data_id'])
    request = _build_session_request(objective=objective)
    request['data_source'] = {'type': 'projections', 'season': None, 'blend_weights': weights, 'custom_data_ids': data_ids}
    request['model_settings'].update(DRAFTING_SETTINGS)
    # Players known before the season to miss all of it (Kyrie Irving), excluded as a user would.
    request['injured_players'] = list(KNOWN_ABSENCES[ESPN_LABEL])
    response = client.post('/sessions', json=request)
    if response.status_code != 201:
        raise RuntimeError(f'Session creation failed: {response.text[:300]}')
    return get_session(response.json()['session_id'])


def measure_off_projection(format_key: str
                           , projection: str) -> list[dict]:
    """Every seat's draft off this projection, scored on the actual season: per seat, the H seat's actual H-score and
    (head to head) real-week win rate in percent, its projected H-score, and the drafted players the actual season
    has no statistics for."""
    drafter = build_blend_session(OBJECTIVES[format_key], PROJECTION_BLENDS[projection])
    actual = build_league_session(OBJECTIVES[format_key], SEASON, H_VERSUS_G_SETTINGS)[2]
    settings = actual.current_settings
    season_weeks = load_season_weeks(SEASON) if format_key in REAL_MEASURES else None
    records = []
    for seat in range(settings['n_drafters']):
        assignments = draft_h_seat_rosters(drafter, seat)
        team = f'Drafter {seat + 1}'
        absent = sorted({player for roster in assignments.values() for player in roster
                         if player not in actual.player_registry})
        projected = drafter.agent.get_h_scores(assignments, team, drafter.current_settings['n_iterations'])['Scores']
        record = {'format': format_key, 'projection': projection, 'seat': seat + 1, 'absent': absent,
                  'h_team': tuple(sorted(assignments[team])),
                  'projected_h_score': 100 * float(projected[projected.idxmax()]), 'assignments': assignments}
        if not absent:
            scores = actual.agent.get_h_scores(assignments, team, settings['n_iterations'])['Scores']
            record['h_score'] = 100 * float(scores[scores.idxmax()])
        if season_weeks is not None:
            record['real_rate'] = 100 * simulate_real_win_rates(season_weeks, assignments, team, settings['categories'],
                                                                REAL_MATCHUP_SIMULATIONS, seed=seat)[REAL_MEASURES[format_key]]
        records.append(record)
    return records


def _measure_reporting_failure(*job) -> tuple:
    """measure_off_projection, returning ('measured', records) or ('failed', job, traceback)."""
    import traceback
    try:
        return 'measured', measure_off_projection(*job)
    except Exception:
        return 'failed', job, traceback.format_exc()


def measure_off_projections_in_parallel(format_key: str
                                        , n_workers: int) -> tuple[list[dict], list[dict]]:
    """Every projection for this format, one per job. Returns (records, failures)."""
    export_projection_csv(ESPN_LABEL)
    export_darko_preseason_csv()
    jobs = [(format_key, projection) for projection in PROJECTION_BLENDS]
    with multiprocessing.get_context('spawn').Pool(min(n_workers, len(jobs))) as pool:
        outcomes = pool.starmap(_measure_reporting_failure, jobs, chunksize=1)
    records = [record for outcome in outcomes if outcome[0] == 'measured' for record in outcome[1]]
    failures = [{'job': outcome[1], 'traceback': outcome[2]} for outcome in outcomes if outcome[0] == 'failed']
    return records, failures
