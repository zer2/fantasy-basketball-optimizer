# testing_files/test_algorithms.py
# Adapted from the original Streamlit-based test_algorithms.py.
# Tests backend math via the FastAPI TestClient instead of AppTest.
#
# Pure-math tests (combinatorial, tipping_point, savor) require no session setup.
# Gradient tests create a session through the API to obtain session.info, then
# build HAgent instances directly for targeted testing.

import numpy as np
import pandas as pd
import yaml
from fastapi.testclient import TestClient
from scipy.stats import norm

from backend.main import app
from backend.state.session import get_session
from backend.math.algorithm_agents import HAgent
from backend.math.algorithm_helpers import (
    combinatorial_calculation
    , calculate_tipping_points
    , savor_calculation
)

client = TestClient(app)

_PARAMS_PATH = 'parameters.yaml'


def _load_default_options() -> dict:
    with open(_PARAMS_PATH) as f:
        return yaml.safe_load(f)['NBA']['options']


def _build_default_session_request() -> dict:
    with open(_PARAMS_PATH) as f:
        all_params = yaml.safe_load(f)
    nba         = all_params['NBA']
    opts        = nba['options']
    n_drafters  = opts['n_drafters']['default']
    n_picks     = opts['n_picks']['default']
    pos_config  = opts['positions'][n_picks]
    slot_counts = {**pos_config['base'], **pos_config['flex']}
    return {
        'league': {
            'sport':            'NBA'
            , 'n_drafters':     n_drafters
            , 'n_picks':        n_picks
            , 'scoring_format': 'Head to Head: Most Categories'
            , 'categories':     nba['default-categories']
        }
        , 'slot_counts': slot_counts
        , 'parameters': {
            'omega':             opts['omega']['default']
            , 'gamma':           opts['gamma']['default']
            , 'beth':            opts['beth']['default']
            , 'upsilon':         opts['upsilon']['default']
            , 'psi':             opts['psi']['default']
            , 'chi':             opts['chi']['default']
            , 'aleph':           opts['aleph']['default']
            , 'n_iterations':    opts['n_iterations']['default']
            , 'streaming_noise': opts['S']['default']
        }
        , 'data_source': {'type': 'historical', 'season': '2024-25'}
    }


def _create_session() -> tuple[str, dict]:
    """POST /sessions with default NBA parameters using 2024-25 historical data.
    Returns (session_id, info) where info is the processed player data dict.
    """
    response = client.post('/sessions', json=_build_default_session_request())
    assert response.status_code == 201, response.text
    session_id = response.json()['session_id']
    session = get_session(session_id)
    return session_id, session.info


def _build_h_agent(info: dict, scoring_format: str) -> HAgent:
    opts = _load_default_options()
    with open(_PARAMS_PATH) as f:
        all_params = yaml.safe_load(f)
    sport_params = all_params['NBA']
    n_picks      = opts['n_picks']['default']
    pos_config   = opts['positions'][n_picks]
    slot_counts  = {**pos_config['base'], **pos_config['flex']}

    return HAgent(
        info           = info
        , omega        = opts['omega']['default']
        , gamma        = opts['gamma']['default']
        , n_picks      = opts['n_picks']['default']
        , n_drafters   = opts['n_drafters']['default']
        , dynamic      = True
        , scoring_format = scoring_format
        , sport        = 'NBA'
        , params       = sport_params
        , slot_counts  = slot_counts
        , beth         = opts['beth']['default']
    )


# ─── Gradient tests ───────────────────────────────────────────────────────────

def test_x_mu_gradients():
    """H-score gradient checks for Head to Head: Most Categories scoring format."""
    _, info = _create_session()
    H = _build_h_agent(info, 'Head to Head: Most Categories')

    c_list = [
        np.array([1/8] * 8 + [0]).reshape(1, 9)
        , np.array([1/4] * 4 + [0] * 5).reshape(1, 9)
        , np.array([1/10] * 8 + [2/10]).reshape(1, 9)
        , np.array([[[0.1] + [0.15] + [0.2] + [0.25] + [0.3] + [0.35] + [0.4] + [0.45] + [0.5] + [0.55] + [0.6]] * 8
                    + [[0.1] + [0.17] + [0.2] + [0.25] + [0.3] + [0.35] + [0.4] + [0.45] + [0.5] + [0.55] + [0.6]]]).reshape(11, 9)
    ]

    v = H.v
    for c in c_list:
        L = H.L.repeat(len(c), axis=0)
        _check_all_gradients(c,
            lambda c: H.get_term_two(c, v),
            lambda _: H.get_del_term_two(v),
        )
        _check_all_gradients(c,
            lambda c: H.get_term_five_a(c, L, v),
            lambda c: H.get_del_term_five_a(c, L, v),
        )
        _check_all_gradients(c,
            lambda c: H.get_term_five_b(c, L, v),
            lambda c: H.get_del_term_five_b(c, L, v),
        )
        _check_all_gradients(c,
            lambda c: H.get_term_four(c, v),
            lambda c: H.get_del_term_four(c, v),
        )
        _check_all_gradients(c,
            lambda c: H.get_terms_four_five(c, L, v),
            lambda c: H.get_del_terms_four_five(c, L, v),
        )
        _check_all_gradients(c,
            lambda c: H.get_last_three_terms(c, L, v),
            lambda c: H.get_del_last_three_terms(c, L, v),
        )
        _check_all_gradients(c,
            lambda c: H.get_last_four_terms(c, L, v),
            lambda c: H.get_del_last_four_terms(c, L, v),
        )
        _check_all_gradients(c,
            lambda c: H.get_x_mu_simplified_form(c, L, v),
            lambda c: H.get_del_full(c, L, v),
        )
        _check_all_gradients(c,
            lambda c: H.get_term_five(c, L, v),
            lambda c: H.get_del_term_five(c, L, v),
        )


def test_ec_gradients():
    """Each Category objective gradient checks (proportions across categories)."""
    _, info = _create_session()
    H = _build_h_agent(info, 'Head to Head: Each Category')

    x_diff_list = [
        np.array([[[0.1] + [0.15] + [0.2]  + [0.25] + [0.3] + [0.35] + [0.4] + [0.45] + [0.5] + [0.55] + [0.6]] * 9])
        , np.array([[[0.1] + [0.2]  + [0.201] + [0.25] + [0.3] + [0.35] + [0.4] + [0.45] + [0.5] + [0.55] + [0.6]] * 9])
        , np.array([[[0.1] + [0.15] + [0.2]  + [0.25] + [0.3] + [0.35] + [0.4] + [0.45] + [0.5] + [0.55] + [0.6]] * 8
                    + [[0.1] + [0.17] + [0.2]  + [0.25] + [0.3] + [0.35] + [0.4] + [0.45] + [0.5] + [0.55] + [0.6]]])
    ]

    diff_vars = 1.0

    def ec_objective(x_diff_array):
        cdf_estimates = norm.cdf(x_diff_array / np.sqrt(diff_vars))
        pdf_estimates = norm.pdf(x_diff_array / np.sqrt(diff_vars)) / np.sqrt(diff_vars)
        return H.get_objective_and_pdf_weights_ec(cdf_estimates, pdf_estimates)

    def ec_gradient(x_diff_array):
        cdf_estimates = norm.cdf(x_diff_array / np.sqrt(diff_vars))
        pdf_estimates = norm.pdf(x_diff_array / np.sqrt(diff_vars)) / np.sqrt(diff_vars)
        _, pdf_weights = H.get_objective_and_pdf_weights_ec(
            cdf_estimates, pdf_estimates, calculate_pdf_weights=True
        )
        return pdf_weights

    for x_diff in x_diff_list:
        _check_gradient_aggregate(x_diff, ec_objective, ec_gradient)


def test_mc_gradients():
    """Most Categories objective gradient checks (proportions across categories)."""
    _, info = _create_session()
    H = _build_h_agent(info, 'Head to Head: Most Categories')

    x_diff_list = [
        np.array([[[0.1] + [0.15] + [0.2]  + [0.25] + [0.3] + [0.35] + [0.4] + [0.45] + [0.5] + [0.55] + [0.6]] * 9])
        , np.array([[[0.1] + [0.2]  + [0.201] + [0.25] + [0.3] + [0.35] + [0.4] + [0.45] + [0.5] + [0.55] + [0.6]] * 9])
        , np.array([[[0.1] + [0.15] + [0.2]  + [0.25] + [0.3] + [0.35] + [0.4] + [0.45] + [0.5] + [0.55] + [0.6]] * 8
                    + [[0.1] + [0.17] + [0.2]  + [0.25] + [0.3] + [0.35] + [0.4] + [0.45] + [0.5] + [0.55] + [0.6]]])
    ]

    diff_vars = 1.0

    def mc_objective(x_diff_array):
        cdf_estimates = norm.cdf(x_diff_array / np.sqrt(diff_vars))
        pdf_estimates = norm.pdf(x_diff_array / np.sqrt(diff_vars)) / np.sqrt(diff_vars)
        return H.get_objective_and_pdf_weights_mc(cdf_estimates, pdf_estimates)

    def mc_gradient(x_diff_array):
        cdf_estimates = norm.cdf(x_diff_array / np.sqrt(diff_vars))
        pdf_estimates = norm.pdf(x_diff_array / np.sqrt(diff_vars)) / np.sqrt(diff_vars)
        _, pdf_weights = H.get_objective_and_pdf_weights_mc(
            cdf_estimates, pdf_estimates, calculate_pdf_weights=True
        )
        return pdf_weights

    for x_diff in x_diff_list:
        _check_gradient_aggregate(x_diff, mc_objective, mc_gradient)


def test_objective_gradients():
    """Rotisserie objective gradient checks."""
    _, info = _create_session()
    H = _build_h_agent(info, 'Rotisserie')

    # diff_vars is uniform across categories when both teams have equal rosters, which
    # is the simplest valid case. Non-uniform diff_vars would require a different gradient
    # check since the returned gradient is already scaled by sqrt(diff_vars) internally.
    diff_vars = 1.0

    x_diff_list = [
        np.array([[[-0.5, -0.3, -0.1,  0.0,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7]] * 9])
        , np.array([[[ 0.5,  0.3,  0.1,  0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7]] * 9])
        , np.array([[[-0.6, -0.4, -0.2,  0.0,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8]] * 9])
        , np.array([[[-0.3, -0.1,  0.0,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8]] * 9])
        , np.array([[[-0.5, -0.3, -0.1,  0.0,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7]] * 8
                    + [[-0.5, -0.3, -0.1,  0.0,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.75]]])
    ]

    for x_diff in x_diff_list:
        sigma_c   = x_diff[0, :, :].std(axis=1, ddof=1) * np.sqrt(2)
        h_m       = H.get_h_m(sigma_c, H.n_drafters)
        sigma_2_m = H.get_sigma_2_m(sigma_c, h_m, H.rho, H.n_drafters)

        def rotisserie_objective(x_diff_array, sigma_2_m=sigma_2_m):
            cdf_estimates = norm.cdf(x_diff_array / np.sqrt(diff_vars))
            return H.get_objective_and_pdf_weights_rotisserie(
                x_diff_array, diff_vars, cdf_estimates, None, sigma_2_m
            )

        def rotisserie_gradient(x_diff_array, sigma_2_m=sigma_2_m):
            cdf_estimates = norm.cdf(x_diff_array / np.sqrt(diff_vars))
            _, gradient = H.get_objective_and_pdf_weights_rotisserie(
                x_diff_array, diff_vars, cdf_estimates, None, sigma_2_m
                , calculate_pdf_weights=True
            )
            return gradient

        _check_gradient_aggregate(x_diff, rotisserie_objective, rotisserie_gradient)


# ─── Pure math tests ──────────────────────────────────────────────────────────

def test_combinatorial_calculation():
    c = np.array([[[1/2, 0]] * 9] * 2)
    result          = combinatorial_calculation(c, 1 - c)
    expected_result = np.array([[1/2, 0], [1/2, 0]])
    assert (abs(result - expected_result) < 0.01).all()


def test_tipping_point_calculation():
    x               = np.array([[[1/2, 0]] * 9] * 2)
    result          = calculate_tipping_points(x)
    expected_result = np.array([[[0.2734, 0]] * 9] * 2)
    assert (abs(result - expected_result) < 0.01).all()


def test_tipping_point_calculation_even_categories_uniform():
    # n=8 (e.g. removing turnovers), x=1/2 for all.
    # P(win exactly 4 of other 7 | x=1/2) = C(7,4)/2^7 = 35/128 ≈ 0.2734
    # P(tie = exactly 4 of 8 | x=1/2)    = C(8,4)/2^8 = 70/256 ≈ 0.2734
    # result[c] = (1/2 * 0.2734 + 1/2 * 0.2734)/2 + 0.2734/2 = 0.2734
    x               = np.array([[[1/2, 0]] * 8] * 2)
    result          = calculate_tipping_points(x)
    expected_result = np.array([[[0.2734, 0]] * 8] * 2)
    assert (abs(result - expected_result) < 0.01).all()


def test_tipping_point_calculation_even_categories_analytical():
    # For n=2 categories, every outcome has exactly one decisive category,
    # so each tipping point equals 0.5 for all x. Proof:
    #   result[0] = (x0*x1 + (1-x0)*(1-x1))/2 + (x0*(1-x1) + (1-x0)*x1)/2 = 1/2
    # This holds for all (x0, x1) and is a clean test of the even-n tie path.
    rng = np.random.default_rng(seed=42)
    x      = rng.uniform(0, 1, size=(5, 2, 4))
    result = calculate_tipping_points(x)
    assert result.shape == (5, 2, 4)
    assert (abs(result - 0.5) < 1e-10).all()


def test_savor_calculation():
    values = pd.Series([1, 2, 3, 4, 5]).sort_values(ascending=False)
    noise  = 2

    savor_result = savor_calculation(values, noise)

    replacement_ev = np.mean(np.clip(np.random.normal(scale=noise, size=100_000), 0, None))

    def estimate_player_value(mean: float) -> float:
        return np.mean(np.clip(np.random.normal(loc=mean, scale=noise, size=100_000), 0, None))

    player_net_evs = np.clip(
        np.array([estimate_player_value(x) - replacement_ev for x in values])
        , 0, None
    )
    regularized_simulated = player_net_evs / player_net_evs.sum()
    regularized_savor     = savor_result / savor_result.sum()

    assert all(abs(regularized_simulated - regularized_savor) < 0.01)


# ─── Gradient check helpers ───────────────────────────────────────────────────

def _check_all_gradients(c, func, del_func):
    for j in range(9):
        _check_gradient(c, func, del_func, j)


def _check_all_gradients_2(c, func, del_func):
    _check_gradient_2(c, func, del_func)


def _check_gradient(c, func, del_func, term: int):
    h         = 0.0000001
    old       = func(c)
    c2        = c.copy()
    c2[0, term] = c2[0, term] + h
    new       = func(c2)

    del_real        = (new - old) / h
    del_theoretical = del_func(c)

    if del_real.shape[0] > 1:
        del_real = del_real[0, :, :]

    if del_theoretical.shape[0] > 1:
        del_theoretical = np.expand_dims(del_theoretical[0, :, :], axis=0)

    if del_theoretical.shape == (1, 1, 9):
        result = del_theoretical[:, :, term]
    elif del_theoretical.shape == (1, 9, 9):
        result = del_theoretical[:, :, term].reshape(9, 1)
    elif del_theoretical.shape == (1, 9, 9, 9):
        result = del_theoretical[:, :, :, term].reshape(1, 9, 9)
    else:
        result = del_theoretical

    assert (abs(del_real - result) < 0.01).all()


def _check_gradient_2(c, func, del_func):
    h   = 0.0000001
    old = func(c)

    all_del_real = []
    all_results  = []

    for term in range(9):
        c2 = c.copy()
        c2[0, term, 0] = c2[0, term, 0] + h
        new            = func(c2)
        del_real       = (new - old) / h
        del_theoretical = del_func(c)
        all_del_real.append(del_real)
        all_results.append(del_theoretical[0, term, 0])

    all_del_real_normalized = np.array(all_del_real).reshape(9, 1) / sum(all_del_real)
    all_results_normalized  = np.array(all_results).reshape(9, 1)  / sum(all_del_real)

    assert (abs(all_del_real_normalized - all_results_normalized) < 1e-7).all()


def _check_gradient_aggregate(x_diff, func, del_func):
    """Check gradient proportions by perturbing all opponents of each category simultaneously.
    del_func must return shape (players, categories) — a gradient already averaged over opponents.
    Normalizes each array by its own sum so only relative proportions are checked, not absolute scale.
    """
    h            = 0.0001
    n_categories = x_diff.shape[1]

    all_del_real = []
    all_results  = []

    for term in range(n_categories):
        x_plus          = x_diff.copy()
        x_minus         = x_diff.copy()
        x_plus[0, term, :]  += h
        x_minus[0, term, :] -= h
        del_real        = float(np.asarray(func(x_plus) - func(x_minus)).ravel()[0]) / (2 * h)
        del_theoretical = del_func(x_diff)
        all_del_real.append(del_real)
        all_results.append(del_theoretical[0, term])

    all_del_real_array      = np.array(all_del_real).reshape(n_categories, 1)
    all_results_array       = np.array(all_results).reshape(n_categories, 1)
    all_del_real_normalized = all_del_real_array / all_del_real_array.sum()
    all_results_normalized  = all_results_array  / all_results_array.sum()

    assert (abs(all_del_real_normalized - all_results_normalized) < 1e-7).all()
