# testing_files/test_and_benchmark_draft.py
# Draft mode benchmarks and correctness tests for the H-score algorithm.
# Uses 2024-25 historical season data with default parameters from parameters.yaml.
#
# Covers:
#   - Session creation time (pipeline steps 1–5 with historical data)
#   - Evaluate timing and correctness across all three scoring formats
#   - Evaluate timing with a mid-draft board state
#   - All three scoring formats with 8 categories (Turnovers removed)
#   - All three scoring formats evaluated from Team 5 after a full first round of picks
#   - Smoke tests: 2-category Roto, 25 drafters, 3 drafters

import cProfile
import io
import pstats
import time
import pytest

from benchmark_helpers import (
    client
    , _SCORE_TOL
    , _SEASON
    , _NO_TO_CATEGORIES
    , _ALL_CATEGORIES
    , _build_session_request
    , check_top_scores
    , resolve_player_ids
    , record_benchmark
)
from backend.state.session import get_session
from backend.services.ranking import rank_candidates

# Snake-draft first round: pick i goes to Team (i+1), in this order.
_FIRST_ROUND_PICKS = [
    'Shai Gilgeous-Alexander (PG,SG)'
    , 'Nikola Jokic (C)'
    , 'James Harden (PG,SG)'
    , 'Tyrese Haliburton (PG,SG)'
    , 'Trae Young (PG)'
    , 'LeBron James (SF,PF)'
    , 'Giannis Antetokounmpo (C,PF)'
    , 'Jayson Tatum (SF,PF)'
    , 'Anthony Edwards (SG,SF)'
    , 'Stephen Curry (PG,SG)'
    , 'Victor Wembanyama (C)'
    , 'Kevin Durant (SF,PF)'
]

# Each entry: (objective, [(player_name_prefix, expected_h_score), ...], categories)
# Board state: _FIRST_ROUND_PICKS assigned one per team.
# Evaluate is run from Team 5's perspective (Trae Young already drafted).
_FIRST_ROUND_CONFIGS = [
    pytest.param(
        ('Each Category', [
            ('Karl-Anthony Towns',  50.4),
            ('Brook Lopez',         49.8),
            ('Jaren Jackson Jr.',   49.7),
            ('Derrick White',       49.7),
        ], None),
        id='EC-first-round',
    ),
    pytest.param(
        ('Most Categories', [
            ('Karl-Anthony Towns',  51.6),
            ('Brook Lopez',         49.9),
            ('Jaren Jackson Jr.',   49.9),
            ('Myles Turner',        49.1),
        ], None),
        id='MC-first-round',
    ),
    pytest.param(
        ('Rotisserie', [
            ('Karl-Anthony Towns',  9.0),
            ('Ivica Zubac',          8.2),
            ('Dyson Daniels',        7.7),
            ('Josh Hart',            8.1),
        ], None),
        id='Roto-first-round',
    ),
    pytest.param(
        ('Each Category', [
            ('Karl-Anthony Towns',  51.5),
            ('Cade Cunningham',     51.1),
            ('Devin Booker',        50.9),
            ('Jaren Jackson Jr.',   50.8),
        ], _NO_TO_CATEGORIES),
        id='EC-first-round-noTO',
    ),
    pytest.param(
        ('Most Categories', [
            ('Karl-Anthony Towns',  53.9),
            ('Cade Cunningham',     53.3),
            ('Devin Booker',        52.7),
            ('Jaren Jackson Jr.',   52.5),
        ], _NO_TO_CATEGORIES),
        id='MC-first-round-noTO',
    ),
    pytest.param(
        ('Rotisserie', [
            ('Karl-Anthony Towns',   9.9),
            ('Ivica Zubac',          9.0),
            ('Cade Cunningham',      9.5),
            ('Dyson Daniels',        8.8),
        ], _NO_TO_CATEGORIES),
        id='Roto-first-round-noTO',
    ),
]

# Each entry: (objective, [(player_name_prefix, expected_h_score), ...], categories)
# categories=None uses the full default category list from parameters.yaml.
_FORMAT_CONFIGS = [
    pytest.param(
        ('Most Categories', [
            ('Shai Gilgeous-Alexander',  60.7),
            ('Nikola Jokic',             60.4),
            ('Tyrese Haliburton',        53.0),
            ('Giannis Antetokounmpo',    49.4),
        ], None),
        id='MC',
    ),
    pytest.param(
        ('Each Category', [
            ('Shai Gilgeous-Alexander',  54.0),
            ('Nikola Jokic',             53.9),
            ('Tyrese Haliburton',        51.0),
            ('Karl-Anthony Towns',       50.5),
        ], None),
        id='EC',
    ),
    pytest.param(
        ('Rotisserie', [
            ('Shai Gilgeous-Alexander',  13.8),
            ('Nikola Jokic',             13.3),
            ('James Harden',             8.4),
            ('Tyrese Haliburton',         9.6),
        ], None),
        id='Roto',
    ),
    pytest.param(
        ('Most Categories', [
            ('Shai Gilgeous-Alexander',  59.1),
            ('Nikola Jokic',             61.0),
            ('James Harden',             54.0),
            ('Giannis Antetokounmpo',    50.6),
        ], _NO_TO_CATEGORIES),
        id='MC-noTO',
    ),
    pytest.param(
        ('Each Category', [
            ('Nikola Jokic',             54.5),
            ('Shai Gilgeous-Alexander',  54.0),
            ('James Harden',             51.6),
            ('Giannis Antetokounmpo',    49.9),
        ], _NO_TO_CATEGORIES),
        id='EC-noTO',
    ),
    pytest.param(
        ('Rotisserie', [
            ('Shai Gilgeous-Alexander',  12.8),
            ('Nikola Jokic',             12.9),
            ('James Harden',              9.7),
            ('Tyrese Haliburton',         8.6),
        ], _NO_TO_CATEGORIES),
        id='Roto-noTO',
    ),
    pytest.param(
        ('Most Categories', [
            ('Nikola Jokic',             64.3),
            ('Shai Gilgeous-Alexander',  57.8),
            ('Karl-Anthony Towns',       54.8),
            ('Giannis Antetokounmpo',    53.6),
        ], _ALL_CATEGORIES),
        id='MC-all-cats',
    ),
]


@pytest.fixture(scope='module', params=_FORMAT_CONFIGS)
def session_for_format(request):
    """Create one session per scoring format / category set. Shared across all parametrized tests."""
    objective, expected_top_scores, categories = request.param
    session_request = _build_session_request(objective=objective, categories=categories)
    n_drafters      = session_request['league']['n_drafters']

    start    = time.perf_counter()
    response = client.post('/sessions', json=session_request)
    session_creation_seconds = time.perf_counter() - start

    assert response.status_code == 201, f'Session creation failed ({objective}): {response.text}'
    record_benchmark(f'Session creation — {objective} ({_SEASON}, {n_drafters} teams)', session_creation_seconds)

    return response.json()['session_id'], objective, expected_top_scores


def _print_profile(profiler: cProfile.Profile, objective: str, label: str, top_n: int = 20):
    stream = io.StringIO()
    stats  = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats(top_n)
    print(f'\n[profile] {label} — {objective}')
    print(stream.getvalue())


def test_evaluate_empty_board(session_for_format):
    """Per scoring format: times evaluate on an empty board and checks H-score values and ordering."""
    session_id, objective, expected_top_scores = session_for_format

    session      = get_session(session_id)
    categories   = session.current_params['categories']
    n_drafters   = session.current_params['n_drafters']
    n_iterations = session.current_params['n_iterations']

    player_assignments = {f'Team {i + 1}': [] for i in range(n_drafters)}

    profiler = cProfile.Profile()
    start    = time.perf_counter()
    result   = profiler.runcall(
        rank_candidates
        , session            = session
        , player_assignments = player_assignments
        , my_team_id         = 'Team 1'
        , exclusion_list     = []
        , remaining_cash     = None
    )
    evaluate_seconds = time.perf_counter() - start

    record_benchmark(f'Evaluate — {objective} ({n_iterations} iterations, empty board)', evaluate_seconds)
    _print_profile(profiler, objective, 'empty board')

    candidates      = result.candidates
    candidate_names = [session.player_registry[c.player_id].name for c in candidates]
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
            f'{candidate.player_id}: expected {len(categories)} win rates, got {len(candidate.win_rates)}'
        assert all(0.0 <= rate <= 100.0 for rate in candidate.win_rates), \
            f'{candidate.player_id}: win rate out of [0, 100]'

    # Each expected player must have the correct H-score within tolerance.
    check_top_scores(session, objective, expected_top_scores, candidates)


def test_evaluate_mid_draft(session_for_format):
    """Per scoring format: times evaluate with a mid-draft board and checks drafted players are excluded."""
    session_id, objective, _ = session_for_format

    session      = get_session(session_id)
    n_drafters   = session.current_params['n_drafters']
    n_iterations = session.current_params['n_iterations']
    g_scores     = session.agent.info['G-scores']

    # Take the top 8 G-score players and split them across two teams.
    top_eight      = list(g_scores.sort_values('Total', ascending=False).head(8).index)
    team_one_picks = top_eight[:4]
    team_two_picks = top_eight[4:]

    player_assignments           = {f'Team {i + 1}': [] for i in range(n_drafters)}
    player_assignments['Team 1'] = team_one_picks
    player_assignments['Team 2'] = team_two_picks

    start  = time.perf_counter()
    result = rank_candidates(
        session            = session
        , player_assignments = player_assignments
        , my_team_id         = 'Team 1'
        , exclusion_list     = team_one_picks
        , remaining_cash     = None
    )
    evaluate_seconds = time.perf_counter() - start

    record_benchmark(f'Evaluate — {objective} ({n_iterations} iterations, mid-draft)', evaluate_seconds)

    candidates = result.candidates
    assert len(candidates) > 0

    candidate_player_ids = {c.player_id for c in candidates}
    for drafted_player_id in team_one_picks:
        assert drafted_player_id not in candidate_player_ids, \
            f'Drafted player {drafted_player_id} appeared in candidates'


@pytest.fixture(scope='module', params=_FIRST_ROUND_CONFIGS)
def session_for_first_round(request):
    """Create one session per scoring format / category set for the first-round board state test."""
    objective, expected_top_scores, categories = request.param
    session_request = _build_session_request(objective=objective, categories=categories)
    n_drafters      = session_request['league']['n_drafters']

    start    = time.perf_counter()
    response = client.post('/sessions', json=session_request)
    session_creation_seconds = time.perf_counter() - start

    assert response.status_code == 201, f'Session creation failed ({objective}): {response.text}'
    record_benchmark(f'Session creation — {objective} ({_SEASON}, {n_drafters} teams)', session_creation_seconds)

    return response.json()['session_id'], objective, expected_top_scores


def test_evaluate_first_round(session_for_first_round):
    """Per scoring format: evaluates from Team 5's perspective after a full first round of picks."""
    session_id, objective, expected_top_scores = session_for_first_round

    session      = get_session(session_id)
    n_drafters   = session.current_params['n_drafters']
    n_iterations = session.current_params['n_iterations']

    assert len(_FIRST_ROUND_PICKS) == n_drafters, (
        f'_FIRST_ROUND_PICKS has {len(_FIRST_ROUND_PICKS)} entries but n_drafters={n_drafters}'
    )

    first_round_player_ids = resolve_player_ids(session, _FIRST_ROUND_PICKS)
    player_assignments = {f'Team {i + 1}': [first_round_player_ids[i]] for i in range(n_drafters)}

    start  = time.perf_counter()
    result = rank_candidates(
        session            = session
        , player_assignments = player_assignments
        , my_team_id         = 'Team 5'
        , exclusion_list     = []
        , remaining_cash     = None
    )
    evaluate_seconds = time.perf_counter() - start

    record_benchmark(f'Evaluate — {objective} ({n_iterations} iterations, first round, Team 5)', evaluate_seconds)

    candidates = result.candidates
    assert len(candidates) > 0

    h_scores = [c.h_score for c in candidates]
    assert h_scores == sorted(h_scores, reverse=True), 'Candidates are not sorted by H-score'

    check_top_scores(session, objective, expected_top_scores, candidates)


def test_evaluate_two_category_roto():
    """Smoke + correctness test: Rotisserie with only Points and Threes, Team 12 after first round."""
    session_request = _build_session_request(
        objective = 'Rotisserie'
        , categories   = ['Points', 'Threes']
    )
    n_drafters = session_request['league']['n_drafters']

    response = client.post('/sessions', json=session_request)
    assert response.status_code == 201, f'Session creation failed: {response.text}'
    session_id = response.json()['session_id']

    session      = get_session(session_id)

    assert len(_FIRST_ROUND_PICKS) == n_drafters, (
        f'_FIRST_ROUND_PICKS has {len(_FIRST_ROUND_PICKS)} entries but n_drafters={n_drafters}'
    )

    first_round_player_ids = resolve_player_ids(session, _FIRST_ROUND_PICKS)
    player_assignments = {f'Team {i + 1}': [first_round_player_ids[i]] for i in range(n_drafters)}

    result = rank_candidates(
        session            = session
        , player_assignments = player_assignments
        , my_team_id         = 'Team 12'
        , exclusion_list     = []
        , remaining_cash     = None
    )

    candidates = result.candidates
    assert len(candidates) > 0

    h_scores = [c.h_score for c in candidates]
    assert h_scores == sorted(h_scores, reverse=True), 'Candidates are not sorted by H-score'
    assert all(0.0 <= score <= 100.0 for score in h_scores), \
        f'H-score out of [0, 100]: {[s for s in h_scores if not (0 <= s <= 100)]}'

    expected_top_scores = [
        ('Tyler Herro',    5.9),
        ('Jordan Poole',   7.5),
        ('Dillon Brooks',  8.8),
        ('Klay Thompson',  5.6),
    ]
    check_top_scores(session, 'Rotisserie, 2-cat', expected_top_scores, candidates)


def test_evaluate_twenty_five_drafters():
    """Smoke + correctness test: EC with 25 drafters, default categories, empty board."""
    session_request = _build_session_request(
        objective = 'Each Category'
        , n_drafters   = 25
    )
    response = client.post('/sessions', json=session_request)
    assert response.status_code == 201, f'Session creation failed: {response.text}'
    session_id = response.json()['session_id']

    session      = get_session(session_id)
    n_drafters   = session.current_params['n_drafters']

    result = rank_candidates(
        session            = session
        , player_assignments = {f'Team {i + 1}': [] for i in range(n_drafters)}
        , my_team_id         = 'Team 1'
        , exclusion_list     = []
        , remaining_cash     = None
    )

    candidates = result.candidates
    assert len(candidates) > 0

    h_scores = [c.h_score for c in candidates]
    assert h_scores == sorted(h_scores, reverse=True), 'Candidates are not sorted by H-score'
    assert all(0.0 <= score <= 100.0 for score in h_scores), \
        f'H-score out of [0, 100]: {[s for s in h_scores if not (0 <= s <= 100)]}'

    expected_top_scores = [
        ('Shai Gilgeous-Alexander',  54.6),
        ('Nikola Jokic',             54.4),
        ('James Harden',             51.0),
        ('Giannis Antetokounmpo',    50.5),
    ]
    check_top_scores(session, 'EC, 25 drafters', expected_top_scores, candidates)


def test_evaluate_three_drafters():
    """Smoke + correctness test: EC with 3 drafters, default categories, empty board."""
    session_request = _build_session_request(
        objective = 'Each Category'
        , n_drafters   = 3
    )
    response = client.post('/sessions', json=session_request)
    assert response.status_code == 201, f'Session creation failed: {response.text}'
    session_id = response.json()['session_id']

    session      = get_session(session_id)
    n_drafters   = session.current_params['n_drafters']

    result = rank_candidates(
        session            = session
        , player_assignments = {f'Team {i + 1}': [] for i in range(n_drafters)}
        , my_team_id         = 'Team 1'
        , exclusion_list     = []
        , remaining_cash     = None
    )

    candidates = result.candidates
    assert len(candidates) > 0

    h_scores = [c.h_score for c in candidates]
    assert h_scores == sorted(h_scores, reverse=True), 'Candidates are not sorted by H-score'
    assert all(0.0 <= score <= 100.0 for score in h_scores), \
        f'H-score out of [0, 100]: {[s for s in h_scores if not (0 <= s <= 100)]}'

    expected_top_scores = [
        ('Shai Gilgeous-Alexander',  52.3),
        ('Nikola Jokic',             52.2),
        ('Karl-Anthony Towns',       48.3),
        ('Stephen Curry',            48.1),
    ]
    check_top_scores(session, 'EC, 3 drafters', expected_top_scores, candidates)
