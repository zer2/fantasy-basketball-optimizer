from src.algorithm_agents import HAgent
from src.algorithm_helpers savor_calculation, combinatorial_calculation, calculate_tipping_points
from streamlit.testing.v1 import AppTest
import numpy as np 
import pandas as pd
from scipy.stats import norm


def test_x_mu_gradients():
    """Make sure the H-score calculations for x_mu are working"""
    at = AppTest.from_file("../app.py").run(timeout = 300)

    info = at.session_state.info

    H = HAgent(info = info
        , omega = at.number_input('omega').value
        , gamma = at.number_input('gamma').value
        , alpha = at.number_input('alpha').value
        , beta = at.number_input('beta').value
        , n_picks = at.number_input('n_picks').value
        , n_drafters = at.number_input('n_drafters').value
        , scoring_format = 'Head to Head: Most Categories'
        , punting = True
        , chi = None)

    c_list = [np.array([1/8] * 8 + [0]).reshape(1,9)
            ,np.array([1/4]*4 + [0] * 5).reshape(1,9)
            ,np.array([1/10] * 8 + [2/10]).reshape(1,9)]

    #check gradients

    for c in c_list:

        x_mu_long = H.get_x_mu_long_form(c)
        x_mu_simplified = H.get_x_mu_simplified_form(c)

        assert x_mu_long.shape == x_mu_simplified.shape
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

def test_objective_gradients():
    """Make sure the H-score calculations for x_mu are working"""
    at = AppTest.from_file("../app.py").run(timeout = 300)
    #at.selectbox('scoring_format').input('Rotisserie')

    info = at.session_state.info

    assert at.session_state.params is not None
    assert at.session_state

    H = HAgent(info = info
        , omega = at.number_input('omega').value
        , gamma = at.number_input('gamma').value
        , alpha = at.number_input('alpha').value
        , beta = at.number_input('beta').value
        , n_picks = at.number_input('n_picks').value
        , n_drafters = at.number_input('n_drafters').value
        , scoring_format = 'Rotisserie'
        , punting = True
        , chi = 0.6)

    #we're ok failing the c

    c_list = [  np.array([[[0.1] + [0.2] + [0.201] + [0.25] + [0.3] + [0.35] + [0.4] + [0.45] + [0.5] + [0.55] + [0.6]] * 9])
                , np.array([[[0.2] + [0.1] + [0.201] + [0.25] + [0.3] + [0.35] + [0.4] + [0.45] + [0.5] + [0.55] + [0.6]] * 9])
                , np.array([[[0.1] + [0.201] + [0.2] + [0.25] + [0.3] + [0.35] + [0.4] + [0.45] + [0.5] + [0.55] + [0.6]] * 9])
                , np.array([[[0.1] + [0.15] + [0.2] + [0.25] + [0.3] + [0.35] + [0.4] + [0.45] + [0.5] + [0.55] + [0.6]] * 9])
                , np.array([[[0.1] + [0.2] + [0.2] + [0.25] + [0.3] + [0.35] + [0.4] + [0.45] + [0.5] + [0.55] + [0.6]] * 9])
                 , np.array([[[0.1] + [0.15] + [0.2] + [0.25] + [0.3] + [0.35] + [0.4] + [0.45] + [0.5] + [0.55] + [0.6]] * 8
                            + [[0.1] + [0.17] + [0.2] + [0.25] + [0.3] + [0.35] + [0.4] + [0.45] + [0.5] + [0.55] + [0.6]]])
                 #, np.array([[[0.1] + [0.2] + [0.2] + [0.25] + [0.3] + [0.35] + [0.4] + [0.45] + [0.5] + [0.55] + [0.6]] * 8
                #            + [[0.1] + [0.2] + [0.2] + [0.25] + [0.3] + [0.35] + [0.4] + [0.45] + [0.5] + [0.55] + [0.6]]])
                #, np.array([ [[0.5]*11] * 9])
                #, np.array([[[0.4]*11] * 5 + [[0.5]*11] * 4])
                #, np.array([[[0.1] + [0.1] + [0.2] + [0.25] + [0.3] + [0.35] + [0.4] + [0.45] + [0.5] + [0.55] + [0.6]] * 8
                #            + [[0.1] + [0.1] + [0.2] + [0.25] + [0.3] + [0.35] + [0.4] + [0.45] + [0.5] + [0.55] + [0.6]]])
                #, np.array([[[0.3]*9 + [0.7]*2] * 7 + [[0.6]*11] * 1 + [[0.9]*11] * 1])
                #, np.array([[[0.7]*9 + [0.3]*2] * 7 + [[0.6]*11] * 1 + [[0.9]*11] * 1])
                #, np.array([[[0.7]*6 + [0.3]*5] * 7 + [[0.6]*11] * 1 + [[0.9]*11] * 1])
                #, np.array([[[0.4]*11] * 5 + [[0.5]*11] * 4])
                #, np.array([[[0.4]*11]* 7 + [[0.6]*11] * 1 + [[0.9]*11] * 1])
                #, np.array([[[0.3]*6 + [0.7]*5] * 5 + [[0.5]*11] * 4])
    ]

    #check gradients

    def rotisserie_objective(cdf_estimates):
        res = self.get_objective_and_pdf_weights_rotisserie(
                        cdf_estimates
                        , 1
                        , None
                        , False) 
        return res

    def rotisserie_gradient(cdf_estimates):
        _, res = self.get_objective_and_pdf_weights_rotisserie(
                        cdf_estimates
                        , 1
                        , None
                        , True) 
        return res

    for c in c_list:

        print('CHECKING NEXT. C = ')
        print(c)
        check_all_gradients_2(c, H.objective_function_rotisserie, H.get_gradient_weights_rotisserie)


def test_combinatorial_calculation():
    c = np.array([[[1/2,0]]*9] * 2)

    res = combinatorial_calculation(c, 1 -c)

    expected_result = np.array([[1/2,0],[1/2,0]])

    assert (abs(res - expected_result) < 0.01).all()

def test_tipping_point_calculation():
    x = np.array([[[1/2,0]]*9] * 2)

    res = calculate_tipping_points(x)

    expected_result = np.array([[[0.2734,0]] * 9] * 2)

    assert (abs(res - expected_result) < 0.01).all()

def check_all_gradients(c, func, del_func):
    for j in range(9):
        check_gradient(c, func, del_func, j)

def check_all_gradients_2(c, func, del_func):

    check_gradient_2(c, func, del_func)

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

def check_gradient_2(c, func, del_func):
    h = 0.0001
    old = func(c)

    #print('Objective value:')
    #print(old)

    all_del_real = []
    all_res = []

    for term in range(9):
        c2 = c.copy()

        c2[0,term,0] = c2[0,term,0] + h

        new = func(c2)
        del_real = (new - old)/h
        del_theoretical = del_func(c)

        res = del_theoretical[0,term,0]
        
        all_del_real = all_del_real + [del_real]
        all_res = all_res + [res]

    all_del_real_normalized = np.array(all_del_real).reshape(9,1)/sum(all_del_real)
    all_res_normalized = np.array(all_res).reshape(9,1)/sum(all_del_real)

    print(all_del_real_normalized)
    print(all_res_normalized)

    assert (abs(all_del_real_normalized - all_res_normalized) < 0.001).all()

def test_savor_calculation():
    raw_values_unselected = pd.Series([1,2,3,4,5]).sort_values(ascending = False)
    n_remaining_players = 3
    remaining_cash = 10
    noise = 1

    savor_result = savor_calculation(raw_values_unselected
                    , n_remaining_players
                    , remaining_cash
                    , noise = 1)

    replacement_level = raw_values_unselected.iloc[n_remaining_players]

    replacement_ev = np.mean(np.clip(np.random.normal(scale = noise
                                                        , size = 100000)
                                    ,0,None))

    def estimate_player_value(mean):
        return np.mean(np.clip(np.random.normal(loc = mean
                                                        ,scale = noise
                                                        , size = 100000)
                            ,0,None))

    player_net_evs = np.clip(np.array([estimate_player_value(x - replacement_level) - replacement_ev \
                            for x in raw_values_unselected])
                            ,0,None)

    regularized_player_net_evs = player_net_evs/player_net_evs.sum()
    regularized_savor = savor_result/savor_result.sum()

    assert all(abs(regularized_player_net_evs - regularized_savor) < .01)



