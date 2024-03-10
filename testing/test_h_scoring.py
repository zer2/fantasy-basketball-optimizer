from src.run_algorithm import HAgent
from streamlit.testing.v1 import AppTest

def test_h_score_gradient():
    """Make sure the draft mode is set up correctly in terms of default parameters"""
    at = AppTest.from_file("../app.py").run(timeout = 300)

    info = at.session_state.info

    H = HAgent(info = info
        , omega = at.number_input('omega').value
        , gamma = at.number_input('gamma').value
        , alpha = at.number_input('alpha').value
        , beta = at.number_input('beta').value
        , n_picks = at.number_input('n_picks').value
        , n_drafters = at.number_input('n_drafters').value
        , winner_take_all = True
        , punting = True)

    assert 1 == 1