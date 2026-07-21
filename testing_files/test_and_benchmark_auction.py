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
)
from backend.state.session import get_session
from backend.services.ranking import rank_candidates


def test_evaluate_auction():
    """EC auction, 12 drafters, Drafter 1 has Giannis ($50), Drafter 2 has Jokic ($50)."""
    session_request = _build_session_request(
        scoring_format = 'Head to Head: Each Category'
        , cash_per_team = 200
    )
    response = client.post('/sessions', json=session_request)
    assert response.status_code == 201, f'Session creation failed: {response.text}'
    session_id = response.json()['session_id']

    session      = get_session(session_id)
    n_drafters   = session.current_params['n_drafters']

    team_names = [f'Drafter {i + 1}' for i in range(n_drafters)]

    # The neutral (full-cash) baseline that anchors the dollar values is built once at session
    # creation (agent.populate_default_h_scores), so we go straight to the real assignments.
    player_assignments = {name: [] for name in team_names}
    player_assignments['Drafter 1'] = ['Giannis Antetokounmpo (C,PF)']
    player_assignments['Drafter 2'] = ['Nikola Jokic (C)']

    remaining_cash = {name: 200.0 for name in team_names}
    remaining_cash['Drafter 1'] = 150.0
    remaining_cash['Drafter 2'] = 150.0

    result = rank_candidates(
        session            = session
        , player_assignments = player_assignments
        , my_team_id         = 'Drafter 1'
        , exclusion_list     = ['Giannis Antetokounmpo (C,PF)']
        , remaining_cash     = remaining_cash
    )

    candidates = result.candidates
    assert len(candidates) > 0

    h_scores = [c.h_score for c in candidates]
    assert h_scores == sorted(h_scores, reverse=True), 'Candidates are not sorted by H-score'

    # (expected_name, diff, your_dollar, gnrc_dollar, orig_dollar)
    expected_auction_values = [
        ('Shai Gilgeous-Alexander', -11.9,  83.8, 95.7, 98.0),
        ('Tyrese Haliburton',         6.4,  59.0, 52.6, 54.1),
        ('Dyson Daniels',             7.0,  49.4, 42.4, 43.7),
        ('Jayson Tatum',             -0.3,  37.2, 37.5, 38.7),
    ]
    candidates_by_name = {c.name: c for c in candidates}
    if os.environ.get('REGEN_GOLDENS'):
        rows = []
        for expected_name, *_ in expected_auction_values:
            match = next((n for n in candidates_by_name if n.startswith(expected_name)), None)
            a = candidates_by_name[match].auction_values
            rows.append(f"        ({repr(expected_name) + ',':30} {round(a.your_dollar - a.gnrc_dollar, 1)},"
                        f"  {round(a.your_dollar, 1)}, {round(a.gnrc_dollar, 1)}, {round(a.orig_dollar, 1)})")
        print('\n# REGEN auction\n' + ',\n'.join(rows))
        return
    for expected_name, expected_diff, expected_your, expected_gnrc, expected_orig in expected_auction_values:
        match = next((name for name in candidates_by_name if name.startswith(expected_name)), None)
        assert match is not None, f'{expected_name} not found in candidates'
        auction = candidates_by_name[match].auction_values
        assert auction is not None, f'{match}: auction_values is None'
        actual_diff = round(auction.your_dollar - auction.gnrc_dollar, 1)
        assert abs(actual_diff - expected_diff) <= _SCORE_TOL, (
            f'{match}: expected diff {expected_diff}, got {actual_diff:.1f}'
        )
        assert abs(auction.your_dollar - expected_your) <= _SCORE_TOL, (
            f'{match}: expected your_dollar {expected_your}, got {auction.your_dollar:.1f}'
        )
        assert abs(auction.gnrc_dollar - expected_gnrc) <= _SCORE_TOL, (
            f'{match}: expected gnrc_dollar {expected_gnrc}, got {auction.gnrc_dollar:.1f}'
        )
        assert abs(auction.orig_dollar - expected_orig) <= _SCORE_TOL, (
            f'{match}: expected orig_dollar {expected_orig}, got {auction.orig_dollar:.1f}'
        )
