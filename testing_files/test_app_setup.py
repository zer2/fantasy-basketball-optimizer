# testing_files/test_app_setup.py
# Adapted from the original Streamlit-based test_app_setup.py.
# Tests the FastAPI backend via TestClient instead of AppTest.
#
# Covers:
#   - GET /config/NBA  returns correct structure and defaults
#   - POST /sessions   creates a session with expected defaults
#   - POST /sessions/{id}/evaluate  returns a valid candidate list

import yaml
from fastapi.testclient import TestClient

from backend.main import app
from backend.state.session import get_session

client = TestClient(app)

_PARAMS_PATH = 'parameters.yaml'

NUMERIC_OPTIONS = [
    'n_drafters', 'n_picks', 'psi', 'omega', 'gamma', 'beth', 'n_iterations'
]


def _load_params() -> dict:
    with open(_PARAMS_PATH) as f:
        return yaml.safe_load(f)


def _build_default_session_request() -> dict:
    all_params  = _load_params()
    nba         = all_params['NBA']
    opts        = nba['options']
    n_drafters  = opts['n_drafters']['default']
    n_picks     = opts['n_picks']['default']
    pos_config  = opts['positions'][n_picks]
    slot_counts = {**pos_config['base'], **pos_config['flex']}
    return {
        'league': {
            'sport':          'NBA'
            , 'n_drafters':   n_drafters
            , 'n_picks':      n_picks
            , 'scoring_format': 'Head to Head: Most Categories'
            , 'categories':   nba['default-categories']
        }
        , 'slot_counts': slot_counts
        , 'parameters': {
            'omega':           opts['omega']['default']
            , 'gamma':         opts['gamma']['default']
            , 'beth':          opts['beth']['default']
            , 'upsilon':       opts['upsilon']['default']
            , 'psi':           opts['psi']['default']
            , 'chi':           opts['chi']['default']
            , 'aleph':         opts['aleph']['default']
            , 'n_iterations':  opts['n_iterations']['default']
            , 'streaming_noise': opts['S']['default']
        }
        , 'data_source': {'type': 'historical', 'season': '2024-25'}
    }


def test_config_defaults():
    """GET /config/NBA returns all expected numeric options with correct defaults, min, and max."""
    response = client.get('/config/NBA')
    assert response.status_code == 200

    body    = response.json()
    options = body['options']

    all_params   = _load_params()
    yaml_options = all_params['NBA']['options']

    for option_name in NUMERIC_OPTIONS:
        assert option_name in options, f'Missing option: {option_name}'
        assert options[option_name]['default'] == yaml_options[option_name]['default']
        assert options[option_name]['min']     == yaml_options[option_name]['min']
        assert options[option_name]['max']     == yaml_options[option_name]['max']


def test_config_categories():
    """GET /config/NBA returns non-empty default and all_categories lists."""
    response = client.get('/config/NBA')
    assert response.status_code == 200

    body = response.json()
    assert len(body['default_categories']) > 0
    assert len(body['all_categories'])     > 0
    assert set(body['default_categories']).issubset(set(body['all_categories']))


def test_config_unknown_sport():
    """GET /config for an unknown sport returns 400."""
    response = client.get('/config/BASEBALL')
    assert response.status_code == 400


def test_session_creation_defaults():
    """POST /sessions with default NBA mock parameters creates a session and returns G-scores."""
    response = client.post('/sessions', json=_build_default_session_request())
    assert response.status_code == 201

    body = response.json()
    assert 'session_id' in body
    assert body['n_players_loaded'] > 0
    assert len(body['categories'])  > 0
    assert len(body['g_scores'])    > 0

    # Every G-score entry should have a name, total, and per-category values
    for entry in body['g_scores']:
        assert 'name'   in entry
        assert 'total'  in entry
        assert 'values' in entry
        assert len(entry['values']) == len(body['categories'])


def test_session_creation_stores_correct_params():
    """POST /sessions stores default parameter values matching parameters.yaml."""
    all_params   = _load_params()
    yaml_options = all_params['NBA']['options']

    response = client.post('/sessions', json=_build_default_session_request())
    assert response.status_code == 201

    session = get_session(response.json()['session_id'])
    cp      = session.current_params

    assert cp['sport']      == 'NBA'
    assert cp['n_drafters'] == yaml_options['n_drafters']['default']
    assert cp['n_picks']    == yaml_options['n_picks']['default']


def test_session_creation_insufficient_player_pool():
    """A league whose total roster capacity exceeds the available player pool is rejected with 400."""
    request = _build_default_session_request()
    request['league']['n_drafters'] = 200   # 200 teams x roster spots far exceeds the NBA player pool
    response = client.post('/sessions', json=request)
    assert response.status_code == 400
    assert 'fill every roster' in response.json()['detail']


def test_evaluate_empty_board():
    """POST /sessions + evaluate with an empty board returns a valid candidate list."""
    session_response = client.post('/sessions', json=_build_default_session_request())
    assert session_response.status_code == 201
    session_id = session_response.json()['session_id']

    n_drafters         = _load_params()['NBA']['options']['n_drafters']['default']
    player_assignments = {f'Team {i + 1}': [] for i in range(n_drafters)}
    evaluate_response  = client.post(
        f'/sessions/{session_id}/evaluate'
        , json={
            'player_assignments': player_assignments
            , 'my_team_id': 'Team 1'
        }
    )
    assert evaluate_response.status_code == 200, evaluate_response.text

    body       = evaluate_response.json()
    candidates = body['candidates']
    assert len(candidates) > 0

    # Candidates should be in descending H-score order
    h_scores = [c['h_score'] for c in candidates]
    assert h_scores == sorted(h_scores, reverse=True)

    # Each candidate should have a name, position, and G-score rows
    for candidate in candidates:
        assert candidate['name']   != ''
        assert candidate['position'] != ''
        assert len(candidate['g_score_rows']) > 0


def test_evaluate_cash_mode_mismatch():
    """Sending remaining_cash to a draft session (no cash_per_team) is rejected with 400."""
    session_response = client.post('/sessions', json=_build_default_session_request())
    assert session_response.status_code == 201
    session_id = session_response.json()['session_id']

    response = client.post(
        f'/sessions/{session_id}/evaluate'
        , json={
            'player_assignments': {'Team 1': []}
            , 'my_team_id': 'Team 1'
            , 'remaining_cash': {'Team 1': 200.0}
        }
    )
    assert response.status_code == 400
    assert 'remaining_cash' in response.json()['detail']


def test_patch_toggles_league_type_via_cash_per_team():
    """cash_per_team is the auction-vs-draft discriminator: a patch can set it (session becomes
    an auction league, remaining_cash required) and an explicit null clears it (back to a draft
    league, remaining_cash forbidden). Regression test for the mode-switch bug where leaving
    Auction Mode could never un-set cash_per_team and every draft/season evaluate 400'd."""
    session_response = client.post('/sessions', json=_build_default_session_request())
    assert session_response.status_code == 201
    session_id = session_response.json()['session_id']

    n_drafters         = _load_params()['NBA']['options']['n_drafters']['default']
    player_assignments = {f'Team {i + 1}': [] for i in range(n_drafters)}
    draft_request      = {'player_assignments': player_assignments, 'my_team_id': 'Team 1'}
    auction_request    = {**draft_request, 'remaining_cash': {team: 200.0 for team in player_assignments}}

    # Entering Auction Mode: patch sets cash_per_team -> auction evaluates work, draft evaluates 400
    patch_response = client.patch(
        f'/sessions/{session_id}'
        , json={'from_step': 4, 'league': {'cash_per_team': 200}}
    )
    assert patch_response.status_code == 200, patch_response.text
    assert client.post(f'/sessions/{session_id}/evaluate', json=auction_request).status_code == 200
    assert client.post(f'/sessions/{session_id}/evaluate', json=draft_request).status_code == 400

    # Leaving Auction Mode: an explicit null clears cash_per_team -> the reverse holds
    patch_response = client.patch(
        f'/sessions/{session_id}'
        , json={'from_step': 4, 'league': {'cash_per_team': None}}
    )
    assert patch_response.status_code == 200, patch_response.text
    assert client.post(f'/sessions/{session_id}/evaluate', json=draft_request).status_code == 200
    assert client.post(f'/sessions/{session_id}/evaluate', json=auction_request).status_code == 400


def test_evaluate_nonexistent_session():
    """Evaluate against a session ID that does not exist returns 404."""
    response = client.post(
        '/sessions/doesnotexist/evaluate'
        , json={
            'player_assignments': {'Team 1': []}
            , 'my_team_id': 'Team 1'
        }
    )
    assert response.status_code == 404
