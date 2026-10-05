# testing_files/test_volume_awareness.py
# Volume-aware percentage categories (percentage_volume_plan.md): a percentage is won on the RATE a team posts,
# so team volume dilutes excess makes and scales weekly noise. These pin the transform the agent applies
# (HAgent.apply_volume_awareness) and the gradient chain the descent relies on.

import numpy as np
import pytest

from backend.player_identity import RP_PLAYER_ID
from test_algorithms import _create_session, _build_h_agent

_N_OPPONENTS = 11


@pytest.fixture(scope='module')
def session_info():
    _, info = _create_session()
    return info


def _agent(info, scoring_format='Head to Head', most_categories_weight=0.0):
    return _build_h_agent(info, scoring_format,
                          most_categories_weight=None if scoring_format == 'Rotisserie' else most_categories_weight)


def _inputs(agent, my_units_fraction=1.0, opponent_units_fraction=1.0, seed=0):
    """A differential, today's variance, and a volume context for one candidate against 11 opponents."""
    rng = np.random.default_rng(seed)
    n, n_categories, n_volume = agent.n_picks, agent.n_categories, len(agent.volume_categories)
    x_diff = rng.normal(0, 2, size=(1, n_categories, _N_OPPONENTS))
    draft_term = rng.uniform(0, 3, size=(1, n_categories, _N_OPPONENTS))
    diff_vars = 2 * n + draft_term
    my_x_total = rng.normal(0, 3, size=(1, n_volume))
    my_units = np.full((1, n_volume), my_units_fraction * n)
    opponent_units = np.full((1, n_volume, _N_OPPONENTS), opponent_units_fraction * n)
    return x_diff, diff_vars, my_x_total, my_units, opponent_units


def test_typical_volume_on_both_sides_changes_nothing(session_info):
    agent = _agent(session_info)
    x_diff, diff_vars, my_x_total, my_units, opponent_units = _inputs(agent)
    differential, variance, my_k = agent.apply_volume_awareness(x_diff, diff_vars, my_x_total, my_units, opponent_units)
    assert np.allclose(differential, x_diff) and np.allclose(variance, diff_vars) and np.allclose(my_k, 1)


def test_only_percentage_categories_move(session_info):
    agent = _agent(session_info)
    x_diff, diff_vars, my_x_total, my_units, opponent_units = _inputs(agent, 0.8, 1.2)
    differential, variance, _ = agent.apply_volume_awareness(x_diff, diff_vars, my_x_total, my_units, opponent_units)
    counting = [c for c in range(agent.n_categories) if c not in agent.volume_category_indices]
    assert np.array_equal(differential[:, counting, :], x_diff[:, counting, :])
    assert np.array_equal(variance[:, counting, :], np.broadcast_to(diff_vars, variance.shape)[:, counting, :])


def test_equal_excess_makes_on_lower_volume_is_the_higher_percentage(session_info):
    # Both teams +5 excess makes (diff 0); mine on 20% less volume than typical, theirs on 20% more. Today's
    # model calls it a coin flip; the rate gap favours me.
    agent = _agent(session_info)
    x_diff, diff_vars, _, my_units, opponent_units = _inputs(agent, 0.8, 1.2)
    x_diff[:, agent.volume_category_indices, :] = 0.0
    my_x_total = np.full((1, len(agent.volume_categories)), 5.0)
    differential, _, _ = agent.apply_volume_awareness(x_diff, diff_vars, my_x_total, my_units, opponent_units)
    assert (differential[:, agent.volume_category_indices, :] > 0).all()


def test_the_variance_follows_the_model(session_info):
    agent = _agent(session_info)
    x_diff, diff_vars, my_x_total, my_units, opponent_units = _inputs(agent, 0.9, 1.15)
    _, variance, my_k = agent.apply_volume_awareness(x_diff, diff_vars, my_x_total, my_units, opponent_units)
    n, index = agent.n_picks, agent.volume_category_indices
    opponent_k = n / opponent_units
    expected = n * (my_k[:, :, None] + opponent_k) + opponent_k ** 2 * (diff_vars[:, index, :] - 2 * n)
    assert np.allclose(variance[:, index, :], expected)


@pytest.mark.parametrize('scoring_format, weight', [('Head to Head', 0.0), ('Head to Head', 1.0),
                                                    ('Head to Head', 0.5), ('Rotisserie', None)])
def test_the_gradient_chain_matches_finite_differences(session_info, scoring_format, weight):
    # The descent differentiates the objective with respect to MY future X tilt. In a percentage category it
    # moves the differential against every opponent and my absolute total together (the gap moves at rate
    # k_A), and in every category it moves my volume by B times the shift, which moves k_A in both the gap
    # and the variance. chain_volume_into_pdf_weights claims to carry all of that; check it against a finite
    # difference that moves all three together.
    agent = _agent(session_info, scoring_format, weight)
    x_diff, diff_vars, my_x_total, my_units, opponent_units = _inputs(agent, 0.85, 1.1, seed=3)
    opponent_units = opponent_units * np.linspace(0.85, 1.2, _N_OPPONENTS)

    def sigma_2_m_for(differential, variance):
        if scoring_format != 'Rotisserie':
            return None
        sigma_c = (differential / np.sqrt(variance))[0].std(axis=1, ddof=1) * np.sqrt(2)
        return agent.get_sigma_2_m(sigma_c, agent.get_h_m(sigma_c, agent.n_drafters), agent.rho, agent.n_drafters)

    # sigma_2_m is the field's spread, held fixed during a descent (computed once per evaluate)
    base_differential, base_variance, _ = agent.apply_volume_awareness(
        x_diff, diff_vars, my_x_total, my_units, opponent_units)
    sigma_2_m = sigma_2_m_for(base_differential, base_variance)

    def objective(shift_category=None, shift=0.0):
        shifted_diff, shifted_total, shifted_units = x_diff.copy(), my_x_total.copy(), my_units.copy()
        if shift_category is not None:
            shifted_diff[:, shift_category, :] += shift
            if shift_category in agent.volume_category_indices:
                shifted_total[:, list(agent.volume_category_indices).index(shift_category)] += shift
            shifted_units = shifted_units + shift * agent.volume_tilt_matrix[:, shift_category]
        differential, variance, my_k = agent.apply_volume_awareness(
            shifted_diff, diff_vars, shifted_total, shifted_units, opponent_units)
        cdf, pdf = agent.get_cdf(differential, variance), agent.get_pdf(differential, variance)
        score, cell_weights = agent.get_objective_and_pdf_weights(
            differential, variance, cdf, pdf, sigma_2_m, calculate_pdf_weights=True, correction_mode='skip')
        weights = agent.chain_volume_into_pdf_weights(
            cell_weights.sum(axis=2), cell_weights, differential, variance, shifted_total, shifted_units, my_k)
        return float(np.asarray(score).ravel()[0]), weights

    _, analytic = objective()
    h = 1e-4
    for category in range(agent.n_categories):
        finite_difference = (objective(category, h)[0] - objective(category, -h)[0]) / (2 * h)
        assert np.isclose(finite_difference, analytic[0, category], rtol=2e-4, atol=1e-9), (
            f'{scoring_format}/{weight}, category {category}: analytic {analytic[0, category]}, '
            f'finite difference {finite_difference}')


def test_the_auction_sees_the_same_volume_field_as_the_draft(session_info):
    # Volume is money-blind in the auction (D6 of percentage_volume_plan.md): rosters plus the generic level per
    # open slot, exactly as in the draft. A seat holding the three highest-volume players must show it.
    agent = _agent(session_info)
    teams = [f'Drafter {index + 1}' for index in range(agent.n_drafters)]
    high_volume = list(agent.player_volume_units.iloc[:, 0].drop(index=RP_PLAYER_ID).nlargest(3).index)
    assignments = {team: (high_volume if team == 'Drafter 2' else []) for team in teams}
    x_scores_available = agent.x_scores.drop(index=high_volume)
    cash = {team: 200 for team in teams}
    *_, draft_field = agent.get_diff_distributions(assignments, 'Drafter 1', x_scores_available)
    *_, auction_field = agent.get_diff_distributions(assignments, 'Drafter 1', x_scores_available,
                                                     cash_remaining_per_team=cash)
    assert auction_field is not None
    assert np.allclose(auction_field['opponent_units'], draft_field['opponent_units'])
    opponent_units = auction_field['opponent_units'][0, 0, :]
    assert opponent_units[0] > opponent_units[1:].max()
