# testing_files/test_punt_popularity.py
# Golden test for the anti-crowded-punt ("kappa") popularity measurement.
#
# On the empty-board first run, the multi-start seed scan counts -- among the top _PUNT_POPULARITY_TOP_N
# players -- which single category each is weakest in (its best gentle punt). The resulting per-category
# popularity vector (counts / top-N) drives the kappa penalty that steers early picks away from the
# punts the field is crowding into. This test pins that vector for the 2024-25 season so any drift in
# the calculation -- or in the underlying player data / model -- is caught.
#
# The measurement is kappa-INDEPENDENT (it runs on the raw, penalty-off pass), so any kappa > 0 triggers
# it and yields the same vector. Categories are in the default NBA order:
#   [Field Goal %, Free Throw %, Threes, Points, Rebounds, Assists, Steals, Blocks, Turnovers]
#
# If the algorithm, data, or top-N changes intentionally, regenerate and paste into _GOLDEN below:
#   UPDATE_PUNT_POPULARITY=1 python -m pytest testing_files/test_punt_popularity.py -s

import os

import numpy as np
import pytest

from benchmark_helpers import client, _build_session_request
from backend.state.session import get_session

# popularity[c] = fraction of the top-40 players whose weakest category (best punt) is c. Values are
# multiples of 1/40 and sum to 1. Regenerate with UPDATE_PUNT_POPULARITY=1 (see module docstring).
_GOLDEN = {
    'Head to Head: Each Category':   [0.075, 0.5,   0.2,   0.0, 0.0, 0.0, 0.0, 0.225, 0.0],
    'Head to Head: Most Categories': [0.075, 0.525, 0.175, 0.0, 0.0, 0.0, 0.0, 0.225, 0.0],
}


@pytest.mark.parametrize('scoring_format', list(_GOLDEN))
def test_punt_popularity(scoring_format):
    """Pin the field punt-popularity vector the kappa penalty is built from."""
    request = _build_session_request(scoring_format=scoring_format)
    request['data_source']['season'] = '2024-25'
    request['parameters']['kappa']   = 0.5   # any > 0 triggers the (kappa-independent) measurement

    response = client.post('/sessions', json=request)
    assert response.status_code == 201, f'Session creation failed: {response.text}'
    agent = get_session(response.json()['session_id']).agent

    popularity = agent._punt_popularity
    assert popularity is not None, 'popularity should be measured on the empty board when kappa > 0'
    assert abs(float(np.sum(popularity)) - 1.0) < 1e-9, 'popularity should sum to 1'

    if os.environ.get('UPDATE_PUNT_POPULARITY'):
        print(f"\n    '{scoring_format}': {[round(float(x), 4) for x in popularity]},")
        return

    expected = np.array(_GOLDEN[scoring_format])
    assert np.allclose(popularity, expected, atol=1e-6), (
        f'{scoring_format}: punt popularity changed.\n'
        f'  expected {expected.tolist()}\n'
        f'  actual   {[round(float(x), 4) for x in popularity]}\n'
        'If intentional, regenerate: UPDATE_PUNT_POPULARITY=1 python -m pytest '
        'testing_files/test_punt_popularity.py -s'
    )
