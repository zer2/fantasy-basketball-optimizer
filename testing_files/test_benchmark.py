# testing_files/test_benchmark.py
# Performance benchmark and correctness test for the H-score algorithm.
# Uses 2024-25 historical season data with default parameters from parameters.yaml.
#
# Covers:
#   - Session creation time (pipeline steps 1–5 with historical data)
#   - Evaluate timing and correctness across all three scoring formats
#   - Evaluate timing with a mid-draft board state
#   - All three scoring formats with 8 categories (Turnovers removed)

import cProfile
import io
import pstats
import time
import yaml
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.session import get_session
from backend.evaluate import run_evaluate

client = TestClient(app)

_PARAMS_PATH = 'parameters.yaml'
_SEASON      = '2024-25'
_SCORE_TOL   = 0.05   # allowed deviation from expected H-score (percentage points)

with open(_PARAMS_PATH) as _f:
    _DEFAULT_CATEGORIES = yaml.safe_load(_f)['NBA']['default-categories']

_NO_TO_CATEGORIES = [category for category in _DEFAULT_CATEGORIES if category != 'Turnovers']

# Each entry: (scoring_format, [(player_name_prefix, expected_h_score), ...], categories)
# categories=None uses the full default category list from parameters.yaml.
_FORMAT_CONFIGS = [
    pytest.param(
        ('Head to Head: Most Categories', [
            ('Shai Gilgeous-Alexander',  62.0),
            ('Nikola Jokic',             60.4),
            ('Tyrese Haliburton',        53.5),
            ('Giannis Antetokounmpo',    53.1),
        ], None),
        id='MC',
    ),
    pytest.param(
        ('Head to Head: Each Category', [
            ('Shai Gilgeous-Alexander',  54.4),
            ('Nikola Jokic',             53.8),
            ('Tyrese Haliburton',        51.2),
            ('Karl-Anthony Towns',       51.0),
        ], None),
        id='EC',
    ),
    pytest.param(
        ('Rotisserie', [
            ('Shai Gilgeous-Alexander',  13.4),
            ('Nikola Jokic',             12.9),
            ('James Harden',              9.6),
            ('Tyrese Haliburton',         9.3),
        ], None),
        id='Roto',
    ),
    pytest.param(
        ('Head to Head: Most Categories', [
            ('Shai Gilgeous-Alexander',  60.5),
            ('Nikola Jokic',             60.5),
            ('James Harden',             54.6),
            ('Giannis Antetokounmpo',    54.0),
        ], _NO_TO_CATEGORIES),
        id='MC-noTO',
    ),
    pytest.param(
        ('Head to Head: Each Category', [
            ('Nikola Jokic',             54.5),
            ('Shai Gilgeous-Alexander',  54.4),
            ('James Harden',             51.8),
            ('Giannis Antetokounmpo',    51.3),
        ], _NO_TO_CATEGORIES),
        id='EC-noTO',
    ),
    pytest.param(
        ('Rotisserie', [
            ('Shai Gilgeous-Alexander',  13.4),
            ('Nikola Jokic',             12.1),
            ('James Harden',              9.8),
            ('Tyrese Haliburton',         9.0),
        ], _NO_TO_CATEGORIES),
        id='Roto-noTO',
    ),
]


def _build_session_request(
    scoring_format: str = 'Head to Head: Most Categories'
    , categories: list = None
) -> dict:
    """Construct a session request using all default parameters from parameters.yaml."""
    with open(_PARAMS_PATH) as f:
        all_params = yaml.safe_load(f)

    nba              = all_params['NBA']
    nba_options      = nba['options']
    n_picks          = nba_options['n_picks']['default']
    n_drafters       = nba_options['n_drafters']['default']
    positions_config = nba_options['positions'][n_picks]
    slot_counts      = {**positions_config['base'], **positions_config['flex']}

    return {
        'league': {
            'sport':          'NBA',
            'n_drafters':     n_drafters,
            'n_picks':        n_picks,
            'scoring_format': scoring_format,
            'categories':     categories if categories is not None else nba['default-categories'],
        },
        'slot_counts': slot_counts,
        'parameters': {
            'omega':           nba_options['omega']['default'],
            'gamma':           nba_options['gamma']['default'],
            'n_iterations':    nba_options['n_iterations']['default'],
            'beth':            nba_options['beth']['default'],
            'upsilon':         nba_options['upsilon']['default'],
            'psi':             nba_options['psi']['default'],
            'chi':             nba_options['chi']['default'],
            'aleph':           nba_options['aleph']['default'],
            'streaming_noise': nba_options['S']['default'],
        },
        'data_source': {
            'type':   'historical',
            'season': _SEASON,
        },
    }


@pytest.fixture(scope='module', params=_FORMAT_CONFIGS)
def session_for_format(request):
    """Create one session per scoring format / category set. Shared across all parametrized tests."""
    scoring_format, expected_top_scores, categories = request.param
    session_request = _build_session_request(scoring_format=scoring_format, categories=categories)
    n_drafters      = session_request['league']['n_drafters']

    start    = time.perf_counter()
    response = client.post('/sessions', json=session_request)
    session_creation_seconds = time.perf_counter() - start

    assert response.status_code == 201, f'Session creation failed ({scoring_format}): {response.text}'
    print(f'\n[benchmark] Session creation — {scoring_format} ({_SEASON}, {n_drafters} teams): {session_creation_seconds:.2f}s')

    return response.json()['session_id'], scoring_format, expected_top_scores


def _print_profile(profiler: cProfile.Profile, scoring_format: str, label: str, top_n: int = 20):
    stream = io.StringIO()
    stats  = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats(top_n)
    print(f'\n[profile] {label} — {scoring_format}')
    print(stream.getvalue())


def test_evaluate_empty_board(session_for_format):
    """Per scoring format: times evaluate on an empty board and checks H-score values and ordering."""
    session_id, scoring_format, expected_top_scores = session_for_format

    session      = get_session(session_id)
    categories   = session.current_params['categories']
    n_drafters   = session.current_params['n_drafters']
    n_iterations = session.current_params['n_iterations']

    player_assignments = {f'Team {i + 1}': [] for i in range(n_drafters)}

    profiler = cProfile.Profile()
    start    = time.perf_counter()
    result   = profiler.runcall(
        run_evaluate
        , session_id         = session_id
        , player_assignments = player_assignments
        , my_team_id         = 'Team 1'
        , exclusion_list     = []
        , remaining_cash     = None
        , n_iterations       = n_iterations
    )
    evaluate_seconds = time.perf_counter() - start

    print(f'\n[benchmark] Evaluate — {scoring_format} ({n_iterations} iterations, empty board): {evaluate_seconds:.2f}s')
    _print_profile(profiler, scoring_format, 'empty board')

    candidates      = result.candidates
    candidate_names = [c.name for c in candidates]
    assert len(candidates) >= 200, f'Expected 200+ candidates, got {len(candidates)}'

    # All expected top players must appear somewhere in the player pool.
    for expected_name, _ in expected_top_scores:
        assert any(name.startswith(expected_name) for name in candidate_names), \
            f'{expected_name} not found in candidate pool'

    # Candidates must be in descending H-score order.
    h_scores = [c.h_score for c in candidates]
    assert h_scores == sorted(h_scores, reverse=True), 'Candidates are not sorted by H-score'

    # H-scores must be in the valid display range (0–100 scale).
    assert all(0.0 <= score <= 100.0 for score in h_scores), \
        f'H-score out of [0, 100]: {[s for s in h_scores if not (0 <= s <= 100)]}'

    # Per-category win rates must be in the valid range.
    for candidate in candidates:
        assert len(candidate.win_rates) == len(categories), \
            f'{candidate.name}: expected {len(categories)} win rates, got {len(candidate.win_rates)}'
        assert all(0.0 <= rate <= 100.0 for rate in candidate.win_rates), \
            f'{candidate.name}: win rate out of [0, 100]'

    # Each expected player must have the correct H-score within tolerance.
    candidates_by_name = {c.name: c for c in candidates}
    for expected_name, expected_score in expected_top_scores:
        match = next((name for name in candidates_by_name if name.startswith(expected_name)), None)
        assert match is not None, f'{expected_name} not found in candidates'
        actual_score = candidates_by_name[match].h_score
        assert abs(actual_score - expected_score) <= _SCORE_TOL, (
            f'{match} ({scoring_format}): expected H-score {expected_score}, got {actual_score:.1f}'
        )


def test_evaluate_mid_draft(session_for_format):
    """Per scoring format: times evaluate with a mid-draft board and checks drafted players are excluded."""
    session_id, scoring_format, _ = session_for_format

    session      = get_session(session_id)
    n_drafters   = session.current_params['n_drafters']
    n_iterations = session.current_params['n_iterations']
    g_scores     = session.info['G-scores']

    # Take the top 8 G-score players and split them across two teams.
    top_eight      = list(g_scores.sort_values('Total', ascending=False).head(8).index)
    team_one_picks = top_eight[:4]
    team_two_picks = top_eight[4:]

    player_assignments           = {f'Team {i + 1}': [] for i in range(n_drafters)}
    player_assignments['Team 1'] = team_one_picks
    player_assignments['Team 2'] = team_two_picks

    start  = time.perf_counter()
    result = run_evaluate(
        session_id         = session_id
        , player_assignments = player_assignments
        , my_team_id         = 'Team 1'
        , exclusion_list     = team_one_picks
        , remaining_cash     = None
        , n_iterations       = n_iterations
    )
    evaluate_seconds = time.perf_counter() - start

    print(f'\n[benchmark] Evaluate — {scoring_format} ({n_iterations} iterations, mid-draft): {evaluate_seconds:.2f}s')

    candidates = result.candidates
    assert len(candidates) > 0

    candidate_names = {c.name for c in candidates}
    for drafted_player in team_one_picks:
        assert drafted_player not in candidate_names, \
            f'Drafted player {drafted_player} appeared in candidates'
