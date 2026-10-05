# testing_files/test_projection_blend.py
# The projection blend keeps every player some active source projects, and a league excludes only the players
# its own categories cannot score (live Snowflake data).

import math

from benchmark_helpers import client, _build_session_request
from backend.data_retrieval import combine_projections
from backend.parameters import load_all_params
from backend.player_identity import RP_PLAYER_ID
from backend.services.build_agent import count_unscorable_players_as_replacement
from backend.state.session import get_session

_SPORT_PARAMS = load_all_params()['NBA']
_BOOZER_ID = 1643409     # a 2026 rookie: ESPN projects him, DARKO does not
_JOKIC_ID = 203999


def _blend(espn_weight, darko_weight):
    return combine_projections(blend_weights={'ESPN': espn_weight, 'DARKO': darko_weight},
                               sport_params=_SPORT_PARAMS, uploaded_dfs={})


def test_a_player_one_source_lacks_is_blended_from_the_others():
    # Judging every column used to drop anyone not in every active source: at 50/50 the pool was DARKO's alone, and
    # every rookie DARKO did not yet carry was missing from the rankings and recorded as RP when drafted.
    espn_only, darko_only, blend = _blend(1.0, 0.0), _blend(0.0, 1.0), _blend(0.5, 0.5)
    assert _BOOZER_ID not in darko_only.index
    assert set(blend.index) == set(espn_only.index) | set(darko_only.index)
    for column in ['Points', 'Rebounds', 'Assists', 'Turnovers', 'Field Goal %', 'Field Goal Attempts']:
        assert abs(blend.loc[_BOOZER_ID, column] - espn_only.loc[_BOOZER_ID, column]) < 1e-9, column


def test_ratios_are_derived_where_the_source_implies_them_and_missing_where_nothing_does():
    blend, darko_only = _blend(0.5, 0.5), _blend(0.0, 1.0)
    rookie = blend.loc[_BOOZER_ID]
    # ESPN has no Field Goals Made or Assist to TO, but implies both
    assert abs(rookie['Field Goals Made'] - rookie['Field Goal %'] * rookie['Field Goal Attempts']) < 1e-9
    assert abs(rookie['Assist to TO'] - rookie['Assists'] / rookie['Turnovers']) < 1e-9
    # Nobody projects his three-point attempts: missing, not a made-up number
    assert math.isnan(rookie['Three Attempts']) and math.isnan(rookie['Three %'])
    # A player DARKO does cover keeps DARKO's figure rather than averaging in anything
    assert blend.loc[_JOKIC_ID, 'Three Attempts'] == darko_only.loc[_JOKIC_ID, 'Three Attempts']


def _build_blended_session(categories):
    request = _build_session_request(categories=categories)
    request['data_source'] = {'type': 'projections', 'season': None,
                              'blend_weights': {'ESPN': 0.5, 'DARKO': 0.5}, 'custom_data_ids': []}
    response = client.post('/sessions', json=request)
    assert response.status_code == 201, response.text
    return get_session(response.json()['session_id'])


def test_a_league_excludes_only_the_players_its_categories_cannot_score():
    nine_categories = _SPORT_PARAMS['default-categories']
    with_three_percent = nine_categories + ['Three %']

    without = _build_blended_session(nine_categories)
    assert _BOOZER_ID in without.agent.x_scores.index, 'no 3P% category: the missing 3PA does not matter'

    with_it = _build_blended_session(with_three_percent)
    assert _BOOZER_ID not in with_it.agent.x_scores.index, 'a 3P% league cannot score a player with no 3P%'
    # ...and drafted, he is replacement level rather than an error that stops the draft
    board = {'Team 1': [_BOOZER_ID, _JOKIC_ID], 'Team 2': []}
    assert count_unscorable_players_as_replacement(with_it, board) == {'Team 1': [RP_PLAYER_ID, _JOKIC_ID], 'Team 2': []}
