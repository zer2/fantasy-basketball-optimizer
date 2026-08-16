import os

import pandas as pd
import numpy as np
import pytest

from src.data_retrieval.get_data_wnba import map_wnba_position, calculate_games_played_pct

ADJUSTER = {'C' : 'C', 'G' : 'PG,SG', 'F' : 'PF,SF'}

def test_map_wnba_position():
    """Make sure WNBA API positions are converted to the app's comma-separated convention"""

    assert map_wnba_position('C', ADJUSTER) == 'C'
    assert map_wnba_position('G', ADJUSTER) == 'PG,SG'
    assert map_wnba_position('F', ADJUSTER) == 'PF,SF'
    assert map_wnba_position('G-F', ADJUSTER) == 'PG,SG,PF,SF'
    assert map_wnba_position('F-C', ADJUSTER) == 'PF,SF,C'

    #unknown positions should fall back to NP
    assert map_wnba_position(None, ADJUSTER) == 'NP'
    assert map_wnba_position(np.nan, ADJUSTER) == 'NP'
    assert map_wnba_position('', ADJUSTER) == 'NP'

def test_calculate_games_played_pct():
    """Make sure Games Played % is relative to the most games played by any player"""

    game_counts = pd.Series({'Player A' : 40, 'Player B' : 20, 'Player C' : 10})
    pct = calculate_games_played_pct(game_counts)

    assert pct['Player A'] == 1.0
    assert pct['Player B'] == 0.5
    assert pct['Player C'] == 0.25
    assert (pct <= 1).all()
    assert (pct > 0).all()

@pytest.mark.skipif(os.environ.get('WNBA_LIVE') != '1', reason = 'set WNBA_LIVE=1 to run live API tests')
def test_wnba_player_index_live():
    """Make sure the WNBA player index endpoint is reachable and includes a full player pool"""

    from nba_api.stats.endpoints import playerindex

    index_df = playerindex.PlayerIndex(league_id = '10'
                                       , season = '2025'
                                       , historical_nullable = '1'
                                       , timeout = 60).get_data_frames()[0]

    assert len(index_df) > 100
    assert 'POSITION' in index_df.columns
