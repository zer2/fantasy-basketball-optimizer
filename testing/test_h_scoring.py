from src.run_algorithm import HAgent
from streamlit.testing.v1 import AppTest
import numpy as np 

def test_h_score_calculation_and_gradient():
    """Make sure the H-score calculations are working"""
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

    c_list = [np.array([1/8] * 8 + [0]).reshape(1,9)
            ,np.array([1/4]*4 + [0] * 5).reshape(1,9)
            ,np.array([1/10] * 8 + [2/10]).reshape(1,9)]

    #check gradients

    for c in c_list:

        x_mu_long = H.get_x_mu_long_form(c)
        x_mu_simplified = H.get_x_mu_simplified_form(c)
        assert (abs(x_mu_long - x_mu_simplified) < 0.01).all()

        check_all_gradients(c, H.get_term_five_a, H.get_del_term_five_a)
        check_all_gradients(c, H.get_term_five_b, H.get_del_term_five_b)
        check_all_gradients(c, H.get_term_five, H.get_del_term_five)
        check_all_gradients(c, H.get_term_four, H.get_del_term_four)
        check_all_gradients(c, H.get_terms_four_five, H.get_del_terms_four_five)
        check_all_gradients(c, H.get_term_two, H.get_del_term_two)
        check_all_gradients(c, H.get_last_three_terms, H.get_del_last_three_terms)
        check_all_gradients(c, H.get_last_four_terms, H.get_del_last_four_terms)
        check_all_gradients(c, H.get_x_mu_simplified_form, H.get_del_full)
        
def check_all_gradients(c, func, del_func):
    for j in range(9):
        check_gradient(c, func, del_func, j)

def check_gradient(c, func, del_func, term):
    h = 0.0000001
    old = func(c)
    c2 = c.copy()
    c2[0,term] = c2[0,term] + h
    new = func(c2)
    del_real = (new - old)/h
    del_theoretical = del_func(c)

    if del_theoretical.shape == (1,1,9):
        res = del_theoretical[:,:,term]
    elif del_theoretical.shape == (1,9,9):
        res = del_theoretical[:,:,term].reshape(9,1)
    elif del_theoretical.shape == (1,9,9,9):
        res = del_theoretical[:,:,:,term].reshape(1,9,9)

    assert (abs(del_real - res) < 0.01).all()
