# testing_files/test_algorithms.py
# Adapted from the original Streamlit-based test_algorithms.py.
# Tests backend math via the FastAPI TestClient instead of AppTest.
#
# Pure-math tests (combinatorial, tipping_point, savor) require no session setup.
# Gradient tests create a session through the API to obtain session.agent.info, then
# build HAgent instances directly for targeted testing.

import itertools

import numpy as np
import pytest
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
    , calculate_win_probability_and_tipping_points
    , compute_win_probability
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
            , 'scoring_format': 'Head to Head'
            , 'most_categories_weight': 1.0
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
    return session_id, session.agent.info


def _build_h_agent(
    info: dict
    , scoring_format: str
    , most_categories_weight: float | None
    , tiebreaker_category: str | None = None
) -> HAgent:
    """Head to Head takes an objective weight (0 = Each Category, 1 = Most Categories, between =
    a blend of the two); Rotisserie takes None, since it scores neither way."""
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
        , most_categories_weight = most_categories_weight
        , tiebreaker_category = tiebreaker_category
        , sport        = 'NBA'
        , sport_params = sport_params
        , slot_counts  = slot_counts
        , beth         = opts['beth']['default']
    )


# ─── Gradient tests ───────────────────────────────────────────────────────────

def test_x_mu_gradients():
    """H-score gradient checks for Head to Head: Most Categories scoring format."""
    _, info = _create_session()
    H = _build_h_agent(info, 'Head to Head', most_categories_weight=1.0)

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
    H = _build_h_agent(info, 'Head to Head', most_categories_weight=0.0)

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
    H = _build_h_agent(info, 'Head to Head', most_categories_weight=1.0)

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
        return H.get_objective_and_pdf_weights_mc(x_diff_array, diff_vars,
                                                  cdf_estimates, pdf_estimates)

    def mc_gradient(x_diff_array):
        cdf_estimates = norm.cdf(x_diff_array / np.sqrt(diff_vars))
        pdf_estimates = norm.pdf(x_diff_array / np.sqrt(diff_vars)) / np.sqrt(diff_vars)
        _, pdf_weights = H.get_objective_and_pdf_weights_mc(
            x_diff_array, diff_vars, cdf_estimates, pdf_estimates, calculate_pdf_weights=True
        )
        return pdf_weights

    for x_diff in x_diff_list:
        finite_difference_tolerance = 2e-4 if H.mc_correlation_enabled else 1e-7
        _check_gradient_aggregate(x_diff, mc_objective, mc_gradient,
                                  tolerance=finite_difference_tolerance)


def test_objective_gradients():
    """Rotisserie objective gradient checks."""
    _, info = _create_session()
    H = _build_h_agent(info, 'Rotisserie', most_categories_weight=None)

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


# ─── The Head-to-Head objective dial ──────────────────────────────────────────
# Head to Head is one format with a weight on how much of the objective is winning the majority
# of categories rather than each category on its own. The endpoints must reproduce the two former
# formats exactly, the middle must be a true convex combination, and — the reason any of this is
# coherent — the two gradients must live on the same scale.

_OBJECTIVE_TEST_DIFFS = np.array(
    [[[0.1] + [0.15] + [0.2] + [0.25] + [0.3] + [0.35] + [0.4] + [0.45] + [0.5] + [0.55] + [0.6]] * 8
     + [[0.1] + [0.17] + [0.2] + [0.25] + [0.3] + [0.35] + [0.4] + [0.45] + [0.5] + [0.55] + [0.6]]])


def _objective_inputs(x_diff_array, diff_vars=1.0):
    """The (cdf, pdf) pair the objective functions take, for a given differential array."""
    z_scores = x_diff_array / np.sqrt(diff_vars)
    return norm.cdf(z_scores), norm.pdf(z_scores) / np.sqrt(diff_vars)


def test_each_category_gradient_matches_its_objective_in_absolute_scale():
    """Each Category's weights must be the gradient of the objective it returns, not of the sum
    over categories — which is what they were, n_categories too large.

    The other gradient tests normalise both sides by their own sum, so they only ever compared
    proportions and could not see this. It is invisible under Adam alone, but a blend adds the two
    objectives' gradients together, so a factor of nine on one of them would silently decide the
    result no matter what weight the user picked."""
    _, info = _create_session()
    agent = _build_h_agent(info, 'Head to Head', most_categories_weight=0.0)

    def each_category_objective(x_diff_array):
        cdf_estimates, pdf_estimates = _objective_inputs(x_diff_array)
        return agent.get_objective_and_pdf_weights_ec(cdf_estimates, pdf_estimates)

    cdf_estimates, pdf_estimates = _objective_inputs(_OBJECTIVE_TEST_DIFFS)
    _, analytic_weights = agent.get_objective_and_pdf_weights_ec(
        cdf_estimates, pdf_estimates, calculate_pdf_weights=True)

    step = 1e-4
    for category in range(_OBJECTIVE_TEST_DIFFS.shape[1]):
        raised  = _OBJECTIVE_TEST_DIFFS.copy()
        lowered = _OBJECTIVE_TEST_DIFFS.copy()
        raised[0, category, :]  += step
        lowered[0, category, :] -= step
        finite_difference = float(
            (each_category_objective(raised) - each_category_objective(lowered))[0]) / (2 * step)
        assert np.isclose(finite_difference, analytic_weights[0, category], rtol=1e-6), (
            f'category {category}: analytic weight {analytic_weights[0, category]} does not match '
            f'the objective it claims to differentiate ({finite_difference})'
        )


def test_the_objective_endpoints_reproduce_the_two_former_formats_exactly():
    """Weight 0 and weight 1 must be the old Each Category and Most Categories, bit for bit —
    that is what lets every existing golden stand as the regression test for this change."""
    _, info = _create_session()
    cdf_estimates, pdf_estimates = _objective_inputs(_OBJECTIVE_TEST_DIFFS)
    call = dict(x_diff_array=_OBJECTIVE_TEST_DIFFS, diff_vars=1.0,
                cdf_estimates=cdf_estimates, pdf_estimates=pdf_estimates,
                calculate_pdf_weights=True)

    each_category = _build_h_agent(info, 'Head to Head', most_categories_weight=0.0)
    objective, weights = each_category.get_objective_and_pdf_weights(**call)
    reference_objective, reference_weights = each_category.get_objective_and_pdf_weights_ec(
        cdf_estimates, pdf_estimates, calculate_pdf_weights=True)
    assert np.array_equal(objective, reference_objective)
    assert np.array_equal(weights, reference_weights)

    most_categories = _build_h_agent(info, 'Head to Head', most_categories_weight=1.0)
    objective, weights = most_categories.get_objective_and_pdf_weights(**call)
    reference_objective, reference_weights = most_categories.get_objective_and_pdf_weights_mc(
        _OBJECTIVE_TEST_DIFFS, 1.0, cdf_estimates, pdf_estimates, calculate_pdf_weights=True)
    assert np.array_equal(objective, reference_objective)
    assert np.array_equal(weights, reference_weights)


def test_a_blended_objective_is_the_convex_combination_of_the_two():
    """Half and half is exactly half of each — objective and gradient alike."""
    _, info = _create_session()
    cdf_estimates, pdf_estimates = _objective_inputs(_OBJECTIVE_TEST_DIFFS)

    blended = _build_h_agent(info, 'Head to Head', most_categories_weight=0.5)
    objective, weights = blended.get_objective_and_pdf_weights(
        _OBJECTIVE_TEST_DIFFS, 1.0, cdf_estimates, pdf_estimates, calculate_pdf_weights=True)

    each_objective, each_weights = blended.get_objective_and_pdf_weights_ec(
        cdf_estimates, pdf_estimates, calculate_pdf_weights=True)
    most_objective, most_weights = blended.get_objective_and_pdf_weights_mc(
        _OBJECTIVE_TEST_DIFFS, 1.0, cdf_estimates, pdf_estimates, calculate_pdf_weights=True)

    assert np.allclose(objective, 0.5 * each_objective + 0.5 * most_objective, rtol=0, atol=1e-15)
    assert np.allclose(weights, 0.5 * each_weights + 0.5 * most_weights, rtol=0, atol=1e-15)

    # Both ends are probabilities, which is what makes the combination meaningful rather than a
    # mix of incommensurable quantities.
    assert 0.0 <= float(each_objective[0]) <= 1.0
    assert 0.0 <= float(most_objective[0]) <= 1.0


def test_the_objective_weight_must_match_the_format():
    """Rotisserie scores neither way, so a weight there means the caller misunderstands the
    session; Head to Head cannot be scored without one. Both raise instead of being guessed at."""
    _, info = _create_session()

    with pytest.raises(ValueError, match='does not apply to Rotisserie'):
        _build_h_agent(info, 'Rotisserie', most_categories_weight=0.5)
    with pytest.raises(ValueError, match=r'most_categories_weight in \[0, 1\]'):
        _build_h_agent(info, 'Head to Head', most_categories_weight=None)
    with pytest.raises(ValueError, match=r'most_categories_weight in \[0, 1\]'):
        _build_h_agent(info, 'Head to Head', most_categories_weight=1.5)


# ─── Tiebreaker category ──────────────────────────────────────────────────────
# With an even number of categories a matchup can end level. A tiebreaker settles it by counting
# for two, which makes the total odd — so the matchup always has a winner, and only the level case
# changes hands. These check that claim against enumeration rather than against the DP's own logic.

def _enumerate_win_probability(probs, tiebreaker_index):
    """P(winning the matchup) by walking every win/loss combination and scoring it by hand."""
    n_players, n_categories, n_opponents = probs.shape
    weights = [2 if i == tiebreaker_index else 1 for i in range(n_categories)]
    needed  = sum(weights) / 2
    result  = np.zeros((n_players, n_opponents))
    for outcome in itertools.product([0, 1], repeat=n_categories):
        points = sum(weight for weight, won in zip(weights, outcome) if won)
        if points < needed:
            continue
        probability = np.ones((n_players, n_opponents))
        for category, won in enumerate(outcome):
            probability = probability * (probs[:, category, :] if won
                                         else 1 - probs[:, category, :])
        result += (0.5 if points == needed else 1.0) * probability
    return result


@pytest.mark.parametrize('n_categories', [4, 6, 8])
@pytest.mark.parametrize('tiebreaker_index', [None, 0, 2])
def test_tiebreaker_win_probability_matches_enumeration(n_categories, tiebreaker_index):
    probs = np.random.default_rng(11).uniform(0.15, 0.85, size=(3, n_categories, 2))
    expected = _enumerate_win_probability(probs, tiebreaker_index)

    assert np.allclose(compute_win_probability(probs, tiebreaker_index), expected, atol=1e-12)
    combined, _ = calculate_win_probability_and_tipping_points(probs, tiebreaker_index)
    assert np.allclose(combined, expected, atol=1e-12)


@pytest.mark.parametrize('tiebreaker_index', [None, 0, 3])
def test_tiebreaker_tipping_points_match_finite_differences(tiebreaker_index):
    """The tipping points are the objective's gradient, so they must track the enumerated
    objective — including for the doubled category, which can turn a matchup from either one or
    two points short and therefore carries two ways of being decisive."""
    probs = np.random.default_rng(3).uniform(0.2, 0.8, size=(2, 6, 2))
    _, tipping_points = calculate_win_probability_and_tipping_points(probs, tiebreaker_index)

    step = 1e-6
    for category in range(probs.shape[1]):
        raised, lowered = probs.copy(), probs.copy()
        raised[:, category, :]  += step
        lowered[:, category, :] -= step
        finite_difference = (_enumerate_win_probability(raised, tiebreaker_index)
                             - _enumerate_win_probability(lowered, tiebreaker_index)) / (2 * step)
        assert np.allclose(finite_difference, tipping_points[:, category, :], atol=1e-6)


def test_a_tiebreaker_decides_only_level_matchups():
    """The whole point of the feature, stated as the four cases that matter."""
    def certainty(outcomes, tiebreaker_index):
        probs = np.array([[[float(won)] for won in outcomes]])
        return float(compute_win_probability(probs, tiebreaker_index)[0, 0])

    assert certainty([1, 1, 0, 0], 0) == 1.0, 'level, tiebreaker won -> a win'
    assert certainty([0, 1, 1, 0], 0) == 0.0, 'level, tiebreaker lost -> a loss'
    assert certainty([0, 1, 1, 0], None) == 0.5, 'level with no tiebreaker -> half credit, as before'
    assert certainty([0, 1, 1, 1], 0) == 1.0, 'a majority still wins after losing the tiebreaker'
    assert certainty([1, 0, 0, 0], 0) == 0.0, 'the tiebreaker alone does not win a matchup'


def test_a_tiebreaker_is_refused_where_there_is_no_tie_to_break():
    """An odd number of categories already has a winner, and doubling one would reintroduce the
    ties a tiebreaker exists to remove — so it raises rather than being quietly ignored."""
    probs = np.full((1, 9, 1), 0.5)
    with pytest.raises(ValueError, match='even number of categories'):
        compute_win_probability(probs, 0)
    with pytest.raises(ValueError, match='outside the'):
        compute_win_probability(np.full((1, 8, 1), 0.5), 8)


def _create_even_category_session() -> dict:
    """A session with eight categories, so a tiebreaker has a tie to break. Returns its info."""
    request = _build_default_session_request()
    request['league']['categories'] = [category for category in request['league']['categories']
                                       if category != 'Turnovers']
    response = client.post('/sessions', json=request)
    assert response.status_code == 201, response.text
    return get_session(response.json()['session_id']).agent.info


def test_a_tiebreaker_is_priced_into_what_its_category_is_worth():
    """v is what a category is worth per unit of x-score (g = x * v exactly), so a tiebreaker —
    worth twice in the majority half of the objective and once in the per-category half — carries
    (1 + most_categories_weight) there.

    Putting it in v rather than bolting a tilt onto the opponents is what lets a tiebreaker league
    still punt: the neutral every team drafts toward already knows the rule, so punting is measured
    against a reference that accounts for it instead of fighting an assumption that all twelve
    seats chase the category in lockstep.
    """
    info = _create_even_category_session()
    categories = list(info['X-scores'].columns)
    blocks = categories.index('Blocks')
    other  = categories.index('Points')

    plain = _build_h_agent(info, 'Head to Head', most_categories_weight=1.0)
    plain_ratio = float(plain.v[blocks, 0] / plain.v[other, 0])

    for weight in (0.5, 1.0):
        agent = _build_h_agent(info, 'Head to Head', most_categories_weight=weight,
                               tiebreaker_category='Blocks')
        ratio = float(agent.v[blocks, 0] / agent.v[other, 0])
        assert np.isclose(ratio, (1 + weight) * plain_ratio),             f'at weight {weight} the tiebreaker should be worth {1 + weight}x an ordinary category'
        assert np.isclose(float(agent.v.sum()), 1.0), 'the neutral vector stays normalised'
        # The descent starts at that neutral, with nothing added on top -- adding the factor here
        # as well would apply it twice.
        assert np.allclose(agent.get_starting_category_weights(), agent.v.reshape(-1))


def test_a_tiebreaker_raises_its_category_in_the_g_scores():
    """G-scores are what a player is worth, and the pipeline orders everything by their total: the
    draftable pool, the position means drawn from it, and the anchors the opponent field is built
    from. So the field ends up holding more of the doubled category because those players are worth
    more, rather than because opponents were assumed to chase it."""
    request = _build_default_session_request()
    request['league']['most_categories_weight'] = 1.0
    request['league']['categories'] = [category for category in request['league']['categories']
                                       if category != 'Turnovers']

    without = client.post('/sessions', json=request)
    assert without.status_code == 201, without.text
    request['league']['tiebreaker_category'] = 'Blocks'
    with_tiebreaker = client.post('/sessions', json=request)
    assert with_tiebreaker.status_code == 201, with_tiebreaker.text

    plain_scores = get_session(without.json()['session_id']).agent.info['G-scores']
    tiebreaker_scores = get_session(with_tiebreaker.json()['session_id']).agent.info['G-scores']
    shared = plain_scores.index.intersection(tiebreaker_scores.index)

    ratio = (tiebreaker_scores.loc[shared, 'Blocks'] / plain_scores.loc[shared, 'Blocks']).dropna()
    assert np.allclose(ratio, 2.0), 'Blocks is worth double when it settles tied matchups'
    unchanged = (tiebreaker_scores.loc[shared, 'Points'] / plain_scores.loc[shared, 'Points']).dropna()
    assert np.allclose(unchanged, 1.0), 'and every other category is untouched'


def test_the_agent_rejects_a_tiebreaker_that_cannot_apply():
    _, info = _create_session()   # nine categories, so no tie is possible
    with pytest.raises(ValueError, match='even number of categories'):
        _build_h_agent(info, 'Head to Head', most_categories_weight=1.0,
                       tiebreaker_category='Blocks')
    with pytest.raises(ValueError, match='not one of'):
        _build_h_agent(info, 'Head to Head', most_categories_weight=1.0,
                       tiebreaker_category='Dunks')
    with pytest.raises(ValueError, match='only applies to the majority objective'):
        _build_h_agent(info, 'Rotisserie', most_categories_weight=None,
                       tiebreaker_category='Blocks')


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


def _check_gradient_aggregate(x_diff, func, del_func, tolerance=1e-7):
    """Check gradient proportions by perturbing all opponents of each category simultaneously.
    del_func must return shape (players, categories) — a gradient already averaged over opponents.
    Normalizes each array by its own sum so only relative proportions are checked, not absolute scale.

    tolerance: the finite-difference estimate inherits (objective noise / 2h). A float64-exact
    objective supports 1e-7; the correlation correction's single-precision evaluation adds ~1e-8
    objective noise, so callers with the correction enabled must pass a correspondingly looser
    bound (~2e-4) — the analytic gradient itself is unaffected, only the FD reference is noisy.
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

    assert (abs(all_del_real_normalized - all_results_normalized) < tolerance).all()
