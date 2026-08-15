# testing_files/test_app_setup.py
# Adapted from the original Streamlit-based test_app_setup.py.
# Tests the FastAPI backend via TestClient instead of AppTest.
#
# Covers:
#   - GET /config/NBA  returns correct structure and defaults
#   - POST /sessions   creates a session with expected defaults
#   - POST /sessions/{id}/evaluate  returns a valid candidate list

import pytest
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

    # Every G-score entry should carry a player id, total, and per-category values, and the
    # registry payload should name every one of those ids.
    registry_names = {entry['player_id']: entry['name'] for entry in body['players']}
    for entry in body['g_scores']:
        assert 'player_id' in entry
        assert 'total'     in entry
        assert 'values'    in entry
        assert len(entry['values']) == len(body['categories'])
        assert entry['player_id'] in registry_names


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

    # Each candidate should carry a player id and G-score rows
    for candidate in candidates:
        assert isinstance(candidate['player_id'], int)
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


def test_is_auction_patch_toggles_league_type():
    """is_auction is the session's league type: auction sessions require remaining_cash on every
    evaluate, non-auction sessions forbid it. Patching is_auction toggles the requirement; a
    cash_per_team value left over from an earlier auction lingers harmlessly. Regression test
    for the bug where league type was inferred from cash_per_team presence, which no patch could
    ever unset — poisoning every draft/season evaluate after visiting Auction Mode."""
    session_response = client.post('/sessions', json=_build_default_session_request())
    assert session_response.status_code == 201
    session_id = session_response.json()['session_id']

    n_drafters         = _load_params()['NBA']['options']['n_drafters']['default']
    player_assignments = {f'Team {i + 1}': [] for i in range(n_drafters)}
    draft_request      = {'player_assignments': player_assignments, 'my_team_id': 'Team 1'}
    auction_request    = {**draft_request, 'remaining_cash': {team: 200.0 for team in player_assignments}}

    # Entering Auction Mode: auction evaluates work, draft-style evaluates 400
    patch_response = client.patch(
        f'/sessions/{session_id}'
        , json={'from_step': 4, 'is_auction': True, 'league': {'cash_per_team': 200}}
    )
    assert patch_response.status_code == 200, patch_response.text
    assert client.post(f'/sessions/{session_id}/evaluate', json=auction_request).status_code == 200
    assert client.post(f'/sessions/{session_id}/evaluate', json=draft_request).status_code == 400

    # A patch that omits is_auction must preserve it: e.g. a scoring-format change in
    # Auction Mode must not flip the session's league type.
    patch_response = client.patch(
        f'/sessions/{session_id}'
        , json={'from_step': 4, 'league': {'scoring_format': 'Head to Head: Most Categories'}}
    )
    assert patch_response.status_code == 200, patch_response.text
    assert client.post(f'/sessions/{session_id}/evaluate', json=auction_request).status_code == 200

    # Leaving Auction Mode: is_auction False flips the league type back, even though
    # cash_per_team is still set on the session.
    patch_response = client.patch(
        f'/sessions/{session_id}'
        , json={'from_step': 4, 'is_auction': False}
    )
    assert patch_response.status_code == 200, patch_response.text
    assert client.post(f'/sessions/{session_id}/evaluate', json=draft_request).status_code == 200
    assert client.post(f'/sessions/{session_id}/evaluate', json=auction_request).status_code == 400


def test_v0_cache_key_reflects_blend_weights_and_uploads():
    """The v0 cache serves blended player pools across sessions, so its key must fully
    describe the blend: every source weight, plus the data_id of any uploaded table
    (uploads are immutable per data_id, so the id doubles as a content key). Regression
    test for the key that excluded uploaded-source weights and upload ids — changing an upload's
    weight served the stale cached blend, silently, and an uploaded blend could leak
    into sessions that never uploaded anything."""
    from backend.services.build_agent import _v0_cache_key

    base_blend = {
        'sport': 'NBA',
        'data_source_type': 'projections',
        'blend_weights': {'ESPN': 0.5, 'DARKO': 0.5, 'data_abc123': 0.5},
        'custom_data_ids': ['data_abc123'],
    }
    assert _v0_cache_key(base_blend) is not None, 'uploaded blends are cacheable when fully keyed'

    upload_reweighted = {**base_blend,
                         'blend_weights': {**base_blend['blend_weights'], 'data_abc123': 1.0}}
    assert _v0_cache_key(upload_reweighted) != _v0_cache_key(base_blend), \
        "changing an upload's weight must change the key"

    different_upload = {**base_blend, 'custom_data_ids': ['data_def456'],
                        'blend_weights': {'ESPN': 0.5, 'DARKO': 0.5, 'data_def456': 0.5}}
    assert _v0_cache_key(different_upload) != _v0_cache_key(base_blend), \
        'a different uploaded table must change the key'

    no_upload = {**base_blend, 'custom_data_ids': None,
                 'blend_weights': {'ESPN': 0.5, 'DARKO': 0.5}}
    assert _v0_cache_key(no_upload) != _v0_cache_key(base_blend), \
        'an upload-less blend must never share a key with an uploaded one'

    espn_reweighted = {**no_upload,
                       'blend_weights': {**no_upload['blend_weights'], 'ESPN': 1.0, 'DARKO': 0.0}}
    assert _v0_cache_key(espn_reweighted) != _v0_cache_key(no_upload), \
        'changing a Snowflake source weight must change the key'


def test_parse_projection_csv_reads_any_recognized_spelling():
    """Files are interpreted column by column through the alias table, not matched against
    named export formats: differently-spelled headers for the same stat all land on the
    canonical column, casing is irrelevant, and unrecognized columns are dropped."""
    from backend.services.build_agent import parse_projection_csv
    params = _load_params()['NBA']

    # Extra unmapped columns (Rank, Value, m/g) must be dropped: they would otherwise
    # join the blend's column union, where every player from the other sources is
    # "missing" them — and the blend drops players missing any column across all
    # sources, so one junk column can wipe out the entire pool.
    slash_per_game_csv = (
        'Rank,Name,Pos,Value,g,m/g,p/g,r/g,a/g,s/g,b/g,to/g,3/g,fg%,fga/g,ft%,fta/g\n'
        '1,Test Player,C,12.3,70,34.0,25.0,10.0,5.0,1.0,1.0,3.0,2.0,0.55,18.0,0.8,6.0\n'
    ).encode()
    parsed = parse_projection_csv(slash_per_game_csv, params)
    assert parsed.loc['Test Player', 'Points'] == 25.0
    for junk_column in ('Rank', 'Value', 'm/g'):
        assert junk_column not in parsed.columns, f'unmapped column {junk_column} should be dropped'

    # Entirely different spellings, mixed casing, and padded headers — same canonical result.
    abbreviated_csv = (
        'PLAYER , POS ,GP,PTS,TREB,AST,STL,BLK,TO,3PM,FG%,FGA,FT%,FTA\n'
        'Test Player,C,70,25.0,10.0,5.0,1.0,1.0,3.0,2.0,0.55,18.0,0.8,6.0\n'
    ).encode()
    abbreviated = parse_projection_csv(abbreviated_csv, params)
    assert abbreviated.loc['Test Player', 'Points'] == 25.0
    assert abbreviated.loc['Test Player', 'Rebounds'] == 10.0
    assert abbreviated.loc['Test Player', 'Free Throw %'] == 0.8

    # A spreadsheet of something else is still refused, naming what it could not find.
    unrelated_csv = (
        'Ticker,Sector,Close,Volume\n'
        'ABC,Tech,101.5,20000\n'
    ).encode()
    with pytest.raises(ValueError) as exc_info:
        parse_projection_csv(unrelated_csv, params)
    assert 'Player' in str(exc_info.value), 'the error should name what could not be found'
    assert 'Ticker' in str(exc_info.value), "the error should list the file's unrecognized headers"


def test_ratio_cells_supply_missing_attempt_columns():
    """Some sources print a ratio stat as its percentage followed by the makes and attempts
    behind it — '0.583 (10.2/17.5)' — and ship no attempts column at all. Attempt volume
    weights the percentage deviation in a ratio G-score, so it must be recovered from the
    cell rather than discarded with the rest of the text. A file's own attempts column
    always wins, and a percentage with no attempts anywhere is reported as missing."""
    from backend.services.build_agent import parse_projection_csv
    params = _load_params()['NBA']

    volumes_inside_the_cell = (
        'PLAYER,POS,GP,FG%,FT%,3PM,PTS,TREB,AST,STL,BLK,TO\n'
        'Test Player,C,70,"0.583 (10.2/17.5)","0.821 (6.0/7.3)",1.1,26.5,12.5,9.0,1.3,0.9,3.0\n'
    ).encode()
    recovered = parse_projection_csv(volumes_inside_the_cell, params)
    assert recovered.loc['Test Player', 'Field Goal %'] == 0.583, 'the percentage still parses'
    assert recovered.loc['Test Player', 'Field Goal Attempts'] == 17.5, 'attempts read from the cell'
    assert recovered.loc['Test Player', 'Free Throw Attempts'] == 7.3

    own_attempts_columns = (
        'PLAYER,POS,GP,FG%,FGA,FT%,FTA,3PM,PTS,TREB,AST,STL,BLK,TO\n'
        'Test Player,C,70,"0.583 (10.2/17.5)",21.0,"0.821 (6.0/7.3)",8.5,1.1,26.5,12.5,9.0,1.3,0.9,3.0\n'
    ).encode()
    explicit = parse_projection_csv(own_attempts_columns, params)
    assert explicit.loc['Test Player', 'Field Goal Attempts'] == 21.0, "the file's own column wins"
    assert explicit.loc['Test Player', 'Free Throw Attempts'] == 8.5

    percentages_only = (
        'PLAYER,POS,GP,FG%,FT%,3PM,PTS,TREB,AST,STL,BLK,TO\n'
        'Test Player,C,70,0.583,0.821,1.1,26.5,12.5,9.0,1.3,0.9,3.0\n'
    ).encode()
    response = client.post('/data/upload',
                           files={'file': ('percentages-only.csv', percentages_only, 'text/csv')})
    assert response.status_code == 200, response.text
    missing = response.json()['missing_stats']
    assert 'Field Goal Attempts' in missing and 'Free Throw Attempts' in missing, \
        'a percentage that cannot be weighted must be reported, not silently dropped later'


def test_percentage_without_attempts_narrows_out_instead_of_failing():
    """A percentage whose attempt volume is nowhere to be found cannot be scored, because the
    coefficient pass reads the volume column directly. The category must drop out of the
    session the same way an absent category does, rather than failing the whole build."""
    rows = ['PLAYER,POS,GP,FG%,FT%,3PM,PTS,TREB,AST,STL,BLK,TO']
    for player_index in range(60):
        # Eligible everywhere, so the position-means slice always covers each base position
        # regardless of how this small synthetic pool happens to rank.
        rows.append(f'Fake Player {player_index:03d},"PG,SG,SF,PF,C",70,0.500,0.800,'
                    f'1.1,{20 - player_index * 0.2:.1f},8.0,4.0,1.0,1.0,3.0')
    csv_bytes = '\n'.join(rows).encode()

    upload_response = client.post('/data/upload',
                                  files={'file': ('percentages-only.csv', csv_bytes, 'text/csv')})
    assert upload_response.status_code == 200, upload_response.text
    data_id = upload_response.json()['data_id']

    session_request = _build_default_session_request()
    session_request['league']['n_drafters'] = 2
    session_request['data_source'] = {
        'type': 'projections',
        'blend_weights': {'DARKO': 0.0, 'ESPN': 0.0, data_id: 1.0},
        'custom_data_ids': [data_id],
    }
    create_response = client.post('/sessions', json=session_request)
    assert create_response.status_code == 201, create_response.text

    categories = create_response.json()['categories']
    for unweightable in ('Field Goal %', 'Free Throw %'):
        assert unweightable not in categories, \
            f'{unweightable} has no attempt volume in this pool and cannot be scored'
    assert 'Points' in categories, 'the countable categories are unaffected'


def test_parse_projection_csv_tolerates_files_missing_categories():
    """A source that pairs each projection set with a league exports only that league's
    active categories, so a file may legitimately lack e.g. Turnovers or the free-throw
    columns. It must still parse; the absent stats simply stay absent (the blend covers
    them from other active sources, and step 4 narrows the category list when nothing
    carries them). The upload endpoint reports the absences so the caption can show them."""
    from backend.services.build_agent import parse_projection_csv
    params = _load_params()['NBA']

    partial_csv = (
        'Rank,Name,Pos,Value,g,p/g,r/g,a/g,s/g,b/g,3/g,fg%,fga/g\n'
        '1,Test Player,C,12.3,70,25.0,10.0,5.0,1.0,1.0,2.0,0.55,18.0\n'
    ).encode()
    parsed = parse_projection_csv(partial_csv, params)
    assert parsed.loc['Test Player', 'Points'] == 25.0
    assert parsed.loc['Test Player', 'Field Goal %'] == 0.55
    for absent_column in ('Turnovers', 'Free Throw %', 'Free Throw Attempts'):
        assert absent_column not in parsed.columns, \
            f'{absent_column} is not in the file and must not be invented'

    upload_response = client.post('/data/upload', files={
        'file': ('partial.csv', partial_csv, 'text/csv'),
    })
    assert upload_response.status_code == 200, upload_response.text
    assert upload_response.json()['missing_stats'] == ['Turnovers', 'Free Throw %'], \
        'the upload response should name the standard stats the file lacks'

    # A file with too few recognizable stat columns is still rejected — tolerance for
    # missing categories must not admit arbitrary CSVs that happen to have Name/Pos.
    too_sparse_csv = (
        'Name,Pos,g,p/g,junk\n'
        'Test Player,C,70,25.0,1\n'
    ).encode()
    with pytest.raises(ValueError):
        parse_projection_csv(too_sparse_csv, params)


def test_uploads_survive_restarts_and_stay_alive_while_in_use():
    """An upload must outlive every session that references it: a session whose upload has
    vanished cannot be rebuilt (creation 404s), which strands the app mid-draft. Two ways
    that used to happen — the store was memory-only (a dev auto-reload or a cold start wiped
    it) and the 2h TTL ran from upload time regardless of use (so it died under any draft
    longer than that, while the session it fed lived on a sliding 4h clock)."""
    import time as time_module
    from backend.state import upload_store

    csv_bytes = (
        'Rank,Name,Pos,Value,g,p/g,r/g,a/g,s/g,b/g,to/g,3/g,fg%,fga/g,ft%,fta/g\n'
        '1,Test Player,C,12.3,70,25.0,10.0,5.0,1.0,1.0,3.0,2.0,0.55,18.0,0.8,6.0\n'
    ).encode()
    response = client.post('/data/upload', files={'file': ('projections.csv', csv_bytes, 'text/csv')})
    assert response.status_code == 200, response.text
    data_id = response.json()['data_id']

    # A restart keeps only what reached disk: drop the in-memory entry and re-resolve.
    upload_store._upload_store.clear()
    rehydrated = upload_store.get_upload(data_id)
    assert rehydrated is not None, 'an upload must survive a process restart'
    assert rehydrated['bytes'] == csv_bytes, 'the rehydrated file must be byte-identical'
    assert rehydrated['n_players'] == response.json()['n_players'], 'its metadata must survive too'

    # The TTL is idle time, not age: an upload still in use never expires. Backdate the
    # entry to just inside the window, read it, and confirm the clock restarted.
    upload_store._upload_store[data_id]['last_accessed'] = time_module.time() - (upload_store.UPLOAD_TTL - 60)
    assert upload_store.get_upload(data_id) is not None, 'an in-window upload must resolve'
    assert (time_module.time() - upload_store._upload_store[data_id]['last_accessed']) < 5, \
        'reading an upload must refresh its clock (sliding window)'

    # Past the idle window it is dropped from memory AND disk, and stays gone.
    upload_store._upload_store[data_id]['last_accessed'] = time_module.time() - (upload_store.UPLOAD_TTL + 60)
    upload_store._touch_upload_on_disk(data_id, time_module.time() - (upload_store.UPLOAD_TTL + 60))
    assert upload_store.get_upload(data_id) is None, 'an idle upload must expire'
    assert upload_store.get_upload(data_id) is None, 'an expired upload must not rehydrate from disk'


def test_partial_upload_as_sole_source_narrows_categories_and_builds():
    """A file lacking Turnovers and free-throw data as the ONLY active projection source,
    end to end. The session must build, and the categories the pool cannot score must be
    narrowed out of the response's category list (the frontend syncs its checkboxes from
    it) instead of crashing the pipeline."""
    positions = ('PG', 'SG', 'SF', 'PF', 'C')
    rows = ['Rank,Name,Pos,Value,g,p/g,r/g,a/g,s/g,b/g,3/g,fg%,fga/g']
    for player_index in range(60):
        rows.append(
            f'{player_index + 1},Fake Player {player_index:03d},{positions[player_index % 5]},'
            f'9.9,70,{20 - player_index * 0.2:.1f},8.0,4.0,1.0,1.0,2.0,0.5,15.0'
        )
    csv_bytes = '\n'.join(rows).encode()

    upload_response = client.post('/data/upload', files={'file': ('partial.csv', csv_bytes, 'text/csv')})
    assert upload_response.status_code == 200, upload_response.text
    data_id = upload_response.json()['data_id']

    session_request = _build_default_session_request()
    session_request['league']['n_drafters'] = 2   # 26 players needed; the pool has 60
    session_request['data_source'] = {
        'type': 'projections',
        'blend_weights': {'DARKO': 0.0, 'ESPN': 0.0, data_id: 1.0},
        'custom_data_ids': [data_id],
    }
    create_response = client.post('/sessions', json=session_request)
    assert create_response.status_code == 201, create_response.text
    body = create_response.json()
    assert body['n_players_loaded'] == 60

    categories = body['categories']
    for absent_category in ('Turnovers', 'Free Throw %'):
        assert absent_category not in categories, \
            f'{absent_category} has no data in the pool and must be narrowed out'
    for present_category in ('Points', 'Rebounds', 'Field Goal %'):
        assert present_category in categories


def test_parse_projection_csv_filters_non_numeric_rows():
    """Some sources repeat the header row inside the table body (every stat cell a string)
    and format ratio stats as '0.583 (10.2/17.5)'. The parser must extract the leading
    numbers, drop the embedded header rows, and never crash dividing a string by 82."""
    from backend.services.build_agent import parse_projection_csv
    params = _load_params()['NBA']

    messy_csv = (
        'R#,ADP,PLAYER,POS,TEAM,GP,MPG,FG%,FT%,3PM,PTS,TREB,AST,STL,BLK,TO,TOTAL\n'
        '1,1.5,Nikola Jokic,C,DEN,70,34.0,"0.583 (10.2/17.5)","0.821 (6.0/7.3)",1.1,26.5,12.5,9.0,1.3,0.9,3.0,15.2\n'
        'R#,ADP,PLAYER,POS,TEAM,GP,MPG,FG%,FT%,3PM,PTS,TREB,AST,STL,BLK,TO,TOTAL\n'
        '2,2.1,Luka Doncic,PG,DAL,72,36.0,"0.490 (9.8/20.0)","0.786 (7.0/8.9)",3.0,32.0,9.0,9.5,1.4,0.5,4.0,14.8\n'
    ).encode()
    parsed = parse_projection_csv(messy_csv, params)
    assert len(parsed) == 2, 'embedded header rows must be dropped'
    assert parsed.loc['Nikola Jokic', 'Points'] == 26.5
    assert parsed.loc['Nikola Jokic', 'Field Goal %'] == 0.583, 'leading number extracted from the compound cell'
    assert abs(parsed.loc['Luka Doncic', 'Games Played %'] - 72 / 82.0) < 1e-9


def test_uploaded_positions_use_canonical_eligibility():
    """Positions must not depend on which blend sources are active. An upload carrying its
    own position formatting ('PF, C' with a space, or a different eligibility order) must
    not become the position source for a known player — eligibility comes from the
    canonical Yahoo table regardless of sources. (Identity itself is now an id, so a
    position string can no longer rename anyone; this guards the POSITIONS.)"""
    darko_only = _build_default_session_request()
    darko_only['data_source'] = {
        'type': 'projections',
        'blend_weights': {'ESPN': 0.0, 'DARKO': 1.0},
        'custom_data_ids': [],
    }
    darko_response = client.post('/sessions', json=darko_only)
    assert darko_response.status_code == 201, darko_response.text
    jokic_entry = next(entry for entry in darko_response.json()['players']
                       if entry['name'] == 'Nikola Jokic')

    conflicting_upload = (
        'Rank,Name,Pos,g,p/g,r/g,a/g,s/g,b/g,to/g,3/g,fg%,fga/g,ft%,fta/g\n'
        '1,Nikola Jokic,"PF, C",70,26.5,12.2,9.0,1.3,0.9,3.0,1.1,0.583,17.5,0.820,7.4\n'
    ).encode()
    upload_response = client.post('/data/upload',
                                  files={'file': ('conflicting.csv', conflicting_upload, 'text/csv')})
    assert upload_response.status_code == 200, upload_response.text
    data_id = upload_response.json()['data_id']

    blended = _build_default_session_request()
    blended['data_source'] = {
        'type': 'projections',
        'blend_weights': {'ESPN': 0.0, 'DARKO': 1.0, data_id: 1.0},
        'custom_data_ids': [data_id],
    }
    blended_response = client.post('/sessions', json=blended)
    assert blended_response.status_code == 201, blended_response.text
    blended_by_id = {entry['player_id']: entry for entry in blended_response.json()['players']}

    assert jokic_entry['player_id'] in blended_by_id, \
        'blending an upload must not remove a known player from the pool'
    assert blended_by_id[jokic_entry['player_id']]['positions'] == jokic_entry['positions'], \
        "blending an upload must not change a known player's positions — the upload's own " \
        'position string must be ignored'


def test_new_data_source_added_mid_draft_keeps_the_board_valid():
    """Adding a projection source mid-draft must not invalidate the existing board.

    The exact production sequence that used to 500: draft players under one blend, then
    upload a file (with its own position formatting) and patch it into the blend. In the
    string-identity era the upload's position strings renamed its players and every
    subsequent evaluate crashed on the board's now-unknown names; ids make that
    structurally impossible, and this guards the flow end-to-end. The reverse case — a
    drafted player genuinely leaving the pool — must be a 400 naming them, never a 500."""
    request = _build_default_session_request()
    request['data_source'] = {
        'type': 'projections',
        'blend_weights': {'ESPN': 0.0, 'DARKO': 1.0},
        'custom_data_ids': [],
    }
    create_response = client.post('/sessions', json=request)
    assert create_response.status_code == 201, create_response.text
    session_id = create_response.json()['session_id']

    n_drafters = _load_params()['NBA']['options']['n_drafters']['default']
    board = {f'Team {i + 1}': [] for i in range(n_drafters)}
    drafted_ids = [entry['player_id'] for entry in create_response.json()['g_scores'][:4]]
    registry_by_id = {entry['player_id']: entry for entry in create_response.json()['players']}
    board['Team 1'] = drafted_ids[:2]
    board['Team 2'] = drafted_ids[2:]

    def evaluate_board():
        return client.post(
            f'/sessions/{session_id}/evaluate'
            , json={'player_assignments': board, 'my_team_id': 'Team 1',
                    'candidate_offset': 0, 'candidate_limit': 10}
        )

    assert evaluate_board().status_code == 200, 'the board must evaluate under the original blend'

    # Upload covering the drafted players, with deliberately hostile position formatting:
    # reversed order plus a space ('PF,C' -> 'C, PF'). Positions come from the canonical
    # eligibility table, so the upload's own formatting must be ignored.
    def scramble_position(positions: list[str]) -> str:
        return ', '.join(reversed(positions))

    upload_rows = [
        f'{rank + 1},{registry_by_id[player_id]["name"]},"{scramble_position(registry_by_id[player_id]["positions"])}"'
        ',70,20.0,8.0,4.0,1.0,1.0,2.0,1.5,0.5,15.0,0.8,5.0'
        for rank, player_id in enumerate(drafted_ids)
    ]
    mid_draft_csv = (
        'Rank,Name,Pos,g,p/g,r/g,a/g,s/g,b/g,to/g,3/g,fg%,fga/g,ft%,fta/g\n'
        + '\n'.join(upload_rows)
    ).encode()
    upload_response = client.post('/data/upload',
                                  files={'file': ('mid_draft.csv', mid_draft_csv, 'text/csv')})
    assert upload_response.status_code == 200, upload_response.text
    data_id = upload_response.json()['data_id']

    patch_response = client.patch(f'/sessions/{session_id}', json={
        'from_step': 1,
        'data_source': {
            'type': 'projections',
            'custom_data_ids': [data_id],
            'blend_weights': {'ESPN': 0.0, 'DARKO': 1.0, data_id: 1.0},
        },
    })
    assert patch_response.status_code == 200, patch_response.text

    evaluate_response = evaluate_board()
    assert evaluate_response.status_code == 200, \
        f'the board drafted before the source change must still evaluate: {evaluate_response.text}'

    # A drafted player genuinely leaving the pool (here: marked injured) is a clear,
    # named 400 — not a KeyError 500 from deep inside the H-score math. The injured list
    # is the one name-typed input (free text), resolved server-side against the registry.
    dropped_name = registry_by_id[drafted_ids[0]]['name']
    patch_response = client.patch(f'/sessions/{session_id}',
                                  json={'from_step': 2, 'injured_players': [dropped_name]})
    assert patch_response.status_code == 200, patch_response.text
    evaluate_response = evaluate_board()
    assert evaluate_response.status_code == 400, evaluate_response.text
    detail = evaluate_response.json()['detail']
    assert dropped_name in detail, 'the error should name the vanished player'
    assert 'player pool' in detail, 'the error should explain the cause'


def test_evaluate_rejects_rostered_players_missing_from_pool():
    """A board whose players no longer exist in the pool (e.g. a stale board after a
    data-source switch) must get a clear 400, not a KeyError 500."""
    session_response = client.post('/sessions', json=_build_default_session_request())
    assert session_response.status_code == 201
    session_id = session_response.json()['session_id']

    n_drafters = _load_params()['NBA']['options']['n_drafters']['default']
    player_assignments = {f'Team {i + 1}': [] for i in range(n_drafters)}
    player_assignments['Team 1'] = [999999999]   # no such player id in any pool

    evaluate_response = client.post(
        f'/sessions/{session_id}/evaluate'
        , json={'player_assignments': player_assignments, 'my_team_id': 'Team 1'}
    )
    assert evaluate_response.status_code == 400, evaluate_response.text
    detail = evaluate_response.json()['detail']
    assert '999999999' in detail, 'the error should identify the missing player'
    assert 'player pool' in detail, 'the error should explain the cause'


def test_pipeline_cache_restores_prior_builds():
    """Returning to a configuration the session already built must restore that build's
    agent from the per-session pipeline cache instead of re-running the pipeline (and its
    expensive baseline H-scoring pass) — the toggling-weights-mid-draft scenario. Asserted
    by object identity: the exact agent instance comes back."""
    request_body = _build_default_session_request()
    session_response = client.post('/sessions', json=request_body)
    assert session_response.status_code == 201
    session_id = session_response.json()['session_id']

    original_agent = get_session(session_id).agent
    assert original_agent is not None

    base_parameters = request_body['parameters']
    changed_parameters = {**base_parameters, 'upsilon': base_parameters['upsilon'] + 0.1}

    patch_response = client.patch(f'/sessions/{session_id}',
                                  json={'from_step': 3, 'parameters': changed_parameters})
    assert patch_response.status_code == 200, patch_response.text
    changed_agent = get_session(session_id).agent
    assert changed_agent is not original_agent, 'a new configuration builds a new agent'

    patch_response = client.patch(f'/sessions/{session_id}',
                                  json={'from_step': 3, 'parameters': base_parameters})
    assert patch_response.status_code == 200, patch_response.text
    assert get_session(session_id).agent is original_agent, \
        'returning to the original configuration must restore its cached build'

    # The restored build must still serve evaluates
    n_drafters         = _load_params()['NBA']['options']['n_drafters']['default']
    player_assignments = {f'Team {i + 1}': [] for i in range(n_drafters)}
    evaluate_response  = client.post(
        f'/sessions/{session_id}/evaluate'
        , json={'player_assignments': player_assignments, 'my_team_id': 'Team 1'}
    )
    assert evaluate_response.status_code == 200, evaluate_response.text

    # And the other configuration was stashed too — toggling forward restores it as well
    patch_response = client.patch(f'/sessions/{session_id}',
                                  json={'from_step': 3, 'parameters': changed_parameters})
    assert patch_response.status_code == 200, patch_response.text
    assert get_session(session_id).agent is changed_agent, \
        'both sides of a toggle should be served from the cache'


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
