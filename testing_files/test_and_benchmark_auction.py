# testing_files/test_and_benchmark_auction.py
# Auction mode correctness tests for the H-score and dollar-value algorithms.
# Uses 2024-25 historical season data with default parameters from parameters.yaml.
#
# Covers:
#   - EC auction: Drafter 1 has Giannis ($50), Drafter 2 has Jokic ($50), cash_per_team=200

import os

from benchmark_helpers import (
    client
    , _SCORE_TOL
    , _build_session_request
    , resolve_player_ids
    , name_candidates
)
from backend.state.session import get_session
from backend.services.ranking import rank_candidates


def test_evaluate_auction():
    """EC auction, 12 drafters, Drafter 1 has Giannis ($50), Drafter 2 has Jokic ($50)."""
    session_request = _build_session_request(
        objective = 'Each Category'
        , cash_per_team = 200
    )
    response = client.post('/sessions', json=session_request)
    assert response.status_code == 201, f'Session creation failed: {response.text}'
    session_id = response.json()['session_id']

    session      = get_session(session_id)
    n_drafters   = session.current_settings['n_drafters']

    team_names = [f'Drafter {i + 1}' for i in range(n_drafters)]

    # The neutral (full-cash) baseline that anchors the dollar values is built once at session
    # creation (agent.populate_default_h_scores), so we go straight to the real assignments.
    player_assignments = {name: [] for name in team_names}
    player_assignments['Drafter 1'] = resolve_player_ids(session, ['Giannis Antetokounmpo'])
    player_assignments['Drafter 2'] = resolve_player_ids(session, ['Nikola Jokic'])

    remaining_cash = {name: 200.0 for name in team_names}
    remaining_cash['Drafter 1'] = 150.0
    remaining_cash['Drafter 2'] = 150.0

    result = rank_candidates(
        session            = session
        , player_assignments = player_assignments
        , my_team_id         = 'Drafter 1'
        , exclusion_list     = resolve_player_ids(session, ['Giannis Antetokounmpo'])
        , remaining_cash     = remaining_cash
    )

    candidates = result.candidates
    assert len(candidates) > 0

    h_scores = [c.h_score for c in candidates]
    assert h_scores == sorted(h_scores, reverse=True), 'Candidates are not sorted by H-score'

    # (expected_name, diff, your_dollar, generic_dollar, original_dollar)
    # Regenerated 2026-10-11 for the auction differential's roster-size term (72e9e869): the candidate
    # now fills one of the drafter's slots, where before it was charged as two replacement players too
    # few. The generic (neutral-board) values move most -- Haliburton and Daniels drop ~$14-15 -- since
    # the bias applied to every seat of the baseline.
    expected_auction_values = [
        ('Shai Gilgeous-Alexander',     -4.9,  87.2, 92.1, 90.8),
        ('Tyrese Haliburton',           9.2,  58.9, 49.7, 49.0),
        ('Dyson Daniels',               18.8,  55.3, 36.5, 36.0),
        ('Jayson Tatum',                2.2,  44.9, 42.7, 42.1),
    ]
    candidates_by_name = name_candidates(session, candidates)
    if os.environ.get('REGEN_GOLDENS'):
        rows = []
        for expected_name, *_ in expected_auction_values:
            match = next((n for n in candidates_by_name if n.startswith(expected_name)), None)
            a = candidates_by_name[match].auction_values
            rows.append(f"        ({repr(expected_name) + ',':30} {round(a.your_dollar - a.generic_dollar, 1)},"
                        f"  {round(a.your_dollar, 1)}, {round(a.generic_dollar, 1)}, {round(a.original_dollar, 1)})")
        print('\n# REGEN auction\n' + ',\n'.join(rows))
        return
    for expected_name, expected_diff, expected_your, expected_gnrc, expected_orig in expected_auction_values:
        match = next((name for name in candidates_by_name if name.startswith(expected_name)), None)
        assert match is not None, f'{expected_name} not found in candidates'
        auction = candidates_by_name[match].auction_values
        assert auction is not None, f'{match}: auction_values is None'
        actual_diff = round(auction.your_dollar - auction.generic_dollar, 1)
        assert abs(actual_diff - expected_diff) <= _SCORE_TOL, (
            f'{match}: expected diff {expected_diff}, got {actual_diff:.1f}'
        )
        assert abs(auction.your_dollar - expected_your) <= _SCORE_TOL, (
            f'{match}: expected your_dollar {expected_your}, got {auction.your_dollar:.1f}'
        )
        assert abs(auction.generic_dollar - expected_gnrc) <= _SCORE_TOL, (
            f'{match}: expected generic_dollar {expected_gnrc}, got {auction.generic_dollar:.1f}'
        )
        assert abs(auction.original_dollar - expected_orig) <= _SCORE_TOL, (
            f'{match}: expected original_dollar {expected_orig}, got {auction.original_dollar:.1f}'
        )
