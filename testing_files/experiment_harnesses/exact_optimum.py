"""Does H-scoring's gradient ascent find the global optimum? Measured against an exact solver, for Each Category.

The problem for one candidate: choose category weights w >= 0; the future picks then bring future_stats = n_remaining *
tilt(w), the truncated-max pick model's tilt (pick(w) - pick(neutral)), and the objective is the mean over categories
and opponent columns of Phi(gap / sqrt(variance)). The problem is taken from the app's own final objective call for the
candidate (capture_final_objective_call), with two things held at the app's solution so every category's term depends
on its own future stats alone -- the separable form branch and bound needs:
  - the percentage categories' volume scale (my team's k in apply_volume_awareness), which makes each percentage gap
    affine in its own future stats with a fixed variance;
  - the position term (the flex shares' contribution).
So every result here is for that slightly simplified ("frozen") objective; full_objective rescores a point with the
volume transform recomputed.

Four measurements, each a function below:
  - certify_optimum: sigmoidal-programming branch and bound (Udell & Boyd). Each node replaces every category's term by
    its concave envelope on the node's interval and solves the resulting convex problem through its dual -- minimize
    over dual weights >= 0 of sigma_K(dual weights) - sum of the envelopes' conjugates, where sigma_K, the support
    function of the reachable set, is n_remaining * (W(dual weights) - dual weights . pick(neutral)) with W the pick
    model's value function. Any dual weights give a valid upper bound (weak duality), so an inexact node solve never
    invalidates the search; the reachable point those weights select is a real team strategy, so its true objective is
    a lower bound. Branching splits the category whose envelope overstates its term most at the relaxation's solution.
  - count_starts_reaching_optimum: every single and double punt start, run to convergence -- does each reach the peak?
  - measure_loop_gain: at the peak, weights match marginal values (w proportional to g), and the peak is a strict local
    maximum exactly when every eigenvalue of the loop-gain matrix G = Dg J / kappa is below 1 (J = d future_stats / dw,
    Dg = d marginal value / d future_stats, kappa = g / w). Cross-checked against a finite-difference Hessian.
  - measure_quasiconcavity: along random segments of weight space, does the objective ever dip below the lower end?
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr

from backend.math.truncated_max_pick_model import (
    _evaluate_expected_max_score, _score_value_geometry, compute_expected_pick_tilt_jacobian,
)

ENVELOPE_GRID_POINTS = 401
# Annealed soft-min temperatures for the node's dual: the exact dual has kinks wherever an envelope has a straight piece,
# which is where L-BFGS stalls; each temperature warm-starts the next, and the bound reported is the exact dual.
SMOOTHING_TEMPERATURES = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
PUNT_DEPTHS = (0.05, 0.01, 0.0)
PEAK_TOLERANCE = 1e-5
LOCAL_WEIGHT_FLOOR = 1e-9


# ── The problem ───────────────────────────────────────────────────────────────

def capture_final_objective_call(client, session_id, agent, request) -> tuple:
    """Evaluate the board and return the (args, kwargs, result, candidate ids) of the app's last objective call -- the
    final descent iteration's, which the candidates' scores come from; candidate ids lists the player in each row."""
    calls, descents = [], []
    original_objective, original_descent = agent.get_objective_and_gradient, agent.perform_iterations

    def recording_descent(*args, **kwargs):
        descents.append(list(args[7]))           # result_index: the player in each candidate row
        return original_descent(*args, **kwargs)

    def recording_objective(*args, **kwargs):
        result = original_objective(*args, **kwargs)
        calls.append((args, kwargs, result, descents[-1]))
        return result
    agent.get_objective_and_gradient, agent.perform_iterations = recording_objective, recording_descent
    try:
        response = client.post(f'/sessions/{session_id}/evaluate', json=request)
        if response.status_code != 200:
            raise RuntimeError(f'Evaluate failed: {response.text[:300]}')
    finally:
        del agent.get_objective_and_gradient, agent.perform_iterations
    if not calls:
        raise RuntimeError('The evaluate made no objective call (an empty board serves the self-play rankings).')
    return calls[-1]


class PickModel:
    """The truncated-max pick model's value function W(w) = sigma_s e(rho) and its gradient, the pick."""

    def __init__(self, covariance, neutral_weights, pick_pool_size):
        self.covariance, self.neutral_weights, self.pick_pool_size = covariance, neutral_weights, pick_pool_size
        self.neutral_pick = self.compute_value_and_picks(neutral_weights.reshape(1, -1))[1][0]

    def compute_value_and_picks(self, weights):
        sigma_w, sigma_v, _, sigma_s, sigma_u, rho, _ = _score_value_geometry(
            weights, self.covariance, self.neutral_weights)
        expected_max, expected_max_slope, _ = _evaluate_expected_max_score(rho, self.pick_pool_size)
        picks = (sigma_w * ((expected_max - rho * expected_max_slope) / sigma_s)[:, None]
                 + sigma_v[None, :] * (expected_max_slope / sigma_u)[:, None])
        return sigma_s * expected_max, picks


@dataclass
class NodeResult:
    upper_bound: float
    lower_bound: float
    dual_weights: np.ndarray
    reachable_stats: np.ndarray       # the future stats the dual weights select (a real strategy)
    branch_point: np.ndarray          # that point clipped into the node's box: where the envelopes are compared
    envelope_gaps: np.ndarray         # per category: envelope minus term at the branch point


class FrozenCandidateProblem:
    """One candidate's optimization, frozen at the app's solution (see the module docstring)."""

    def __init__(self, agent, capture, row):
        args, kwargs, result, candidate_ids = capture
        weights, _, diff_means, diff_vars, x_scores_batch, _, _, n_players_selected = args[:8]
        self.agent, self.row = agent, row
        self.player_id = candidate_ids[row]
        self.volume_context = kwargs.get('volume_context')
        self.n_categories = agent.n_categories
        self.n_remaining = agent.n_picks - 1 - n_players_selected
        self.app_weights = np.asarray(weights[row], dtype=float)
        self.app_score = float(result['Score'][row])
        self.pick_model = PickModel(agent.L[0], agent.v.reshape(-1), agent.pick_pool_size)
        app_tilt = self.pick_model.compute_value_and_picks(self.app_weights[None])[1][0] - self.pick_model.neutral_pick
        self.position_part = result['Future-Diffs'][row, :, 0] - self.n_remaining * app_tilt
        row_diff_means = diff_means[row] if diff_means.shape[0] > 1 else diff_means[0]
        row_diff_vars = diff_vars[row] if diff_vars.shape[0] > 1 else diff_vars[0]
        self.base_differentials = row_diff_means + x_scores_batch[row] + self.position_part[:, None]
        self.base_variances = np.broadcast_to(row_diff_vars, self.base_differentials.shape).copy()
        self.app_future_stats = self.n_remaining * app_tilt
        self._freeze_volume()

    def _volume_inputs(self, future_stats):
        context, row, indices = self.volume_context, self.row, self.agent.volume_category_indices
        future = (future_stats + self.position_part)[None]
        opponent_units = context['opponent_units']
        opponent_units = opponent_units[row:row + 1] if opponent_units.shape[0] > 1 else opponent_units
        return (context['my_x_base'][row:row + 1] + future[:, indices],
                context['my_units'][row:row + 1] + future @ self.agent.volume_tilt_matrix.T, opponent_units)

    def compute_full_objective(self, future_stats):
        """The objective with the volume transform recomputed at these future stats (only the position part held)."""
        differentials = (self.base_differentials + future_stats[:, None])[None]
        variances = self.base_variances[None]
        if self.volume_context is not None:
            my_x_total, my_units, opponent_units = self._volume_inputs(future_stats)
            differentials, variances, _ = self.agent.apply_volume_awareness(
                differentials, variances, my_x_total, my_units, opponent_units)
        return float(ndtr(differentials / np.sqrt(variances)).mean())

    def _freeze_volume(self):
        """gap = intercept + slope * future_stats[category], variance fixed, with my volume k held at the app's point."""
        self.gap_intercepts = self.base_differentials.copy()
        self.gap_slopes = np.ones_like(self.base_differentials)
        self.gap_variances = self.base_variances.copy()
        if self.volume_context is None:
            return
        indices = self.agent.volume_category_indices
        _, my_units, opponent_units = self._volume_inputs(self.app_future_stats)

        def compute_gaps(shift):
            future_stats = self.app_future_stats.copy()
            future_stats[indices] += shift
            my_x_total = self._volume_inputs(future_stats)[0]
            differentials, variances, _ = self.agent.apply_volume_awareness(
                (self.base_differentials + future_stats[:, None])[None], self.base_variances[None],
                my_x_total, my_units, opponent_units)
            return differentials[0], variances[0]
        gaps_at_app, variances_at_app = compute_gaps(0.0)
        gaps_shifted, _ = compute_gaps(1.0)
        slopes = gaps_shifted - gaps_at_app
        self.gap_slopes[indices] = slopes[indices]
        self.gap_intercepts[indices] = gaps_at_app[indices] - slopes[indices] * self.app_future_stats[indices][:, None]
        self.gap_variances[indices] = variances_at_app[indices]

    def compute_category_term(self, category, future_stat_values):
        values = np.asarray(future_stat_values, dtype=float)
        z = ((self.gap_intercepts[category][None, :] + self.gap_slopes[category][None, :] * values[:, None])
             / np.sqrt(self.gap_variances[category])[None, :])
        return ndtr(z).mean(axis=1) / self.n_categories

    def compute_frozen_objective(self, future_stats):
        return float(ndtr((self.gap_intercepts + self.gap_slopes * future_stats[:, None])
                          / np.sqrt(self.gap_variances)).mean())

    def compute_frozen_objective_and_marginal_values(self, future_stats):
        """The objective and its gradient in future stats -- each category's marginal value."""
        z = (self.gap_intercepts + self.gap_slopes * future_stats[:, None]) / np.sqrt(self.gap_variances)
        density = np.exp(-z * z / 2) / np.sqrt(2 * np.pi)
        marginal_values = (density * self.gap_slopes / np.sqrt(self.gap_variances)).mean(axis=1) / self.n_categories
        return float(ndtr(z).mean()), marginal_values

    def compute_future_stats(self, weights):
        return self.n_remaining * (self.pick_model.compute_value_and_picks(np.atleast_2d(weights))[1]
                                   - self.pick_model.neutral_pick)

    def compute_support(self, dual_weights):
        """sigma_K at these dual weights, and the reachable future stats they select."""
        value, picks = self.pick_model.compute_value_and_picks(np.atleast_2d(dual_weights))
        return (self.n_remaining * (value - np.atleast_2d(dual_weights) @ self.pick_model.neutral_pick),
                self.n_remaining * (picks - self.pick_model.neutral_pick))

    def compute_weight_jacobian(self, weights):
        """d future_stats / d weights."""
        return self.n_remaining * compute_expected_pick_tilt_jacobian(
            weights[None], self.agent.L[0], self.agent.v, self.agent.pick_pool_size)[0]


# ── Local ascent (the reference the app's method is judged by) ───────────────

def find_local_maximum(problem, start_weights, weight_floors):
    """Gradient ascent on the weights to convergence (L-BFGS, each weight >= its floor). Picks depend only on the
    weights' direction, so the all-zero vector is no strategy at all; floors of 0 on every category would let the line
    search step onto it."""
    def negative_objective(weights):
        value, marginal_values = problem.compute_frozen_objective_and_marginal_values(
            problem.compute_future_stats(weights)[0])
        return -value, -(problem.compute_weight_jacobian(weights).T @ marginal_values)
    result = minimize(negative_objective, start_weights, jac=True, method='L-BFGS-B',
                      bounds=[(floor, None) for floor in weight_floors], options={'maxiter': 500})
    return -result.fun, result.x


def make_punt_starts(neutral_weights, depth):
    """Every single and double punt of the neutral weights at this depth (a punted weight = depth x neutral)."""
    n_categories = len(neutral_weights)
    starts = []
    for first in range(n_categories):
        weights = neutral_weights.copy(); weights[first] *= depth
        starts.append(((first,), weights))
        for second in range(first + 1, n_categories):
            weights = neutral_weights.copy(); weights[first] *= depth; weights[second] *= depth
            starts.append(((first, second), weights))
    return starts


def find_multistart_optimum(problem):
    """The best local maximum over the app's own weights, the neutral weights and every punt start (5% deep). Branch and
    bound certifies it (certify_optimum)."""
    neutral = problem.agent.v.reshape(-1)
    starts = [problem.app_weights, neutral] + [weights for _, weights in make_punt_starts(neutral, 0.05)]
    floors = np.full(problem.n_categories, LOCAL_WEIGHT_FLOOR)
    return max((find_local_maximum(problem, start, floors) for start in starts), key=lambda pair: pair[0])


def count_starts_reaching_optimum(problem, optimum):
    """For each punt depth, how many single and double punt starts end within PEAK_TOLERANCE of the optimum, and how far
    short the rest end. A start at exactly 0 may stay at 0: its punted categories' lower bound is 0 (every other
    weight's is LOCAL_WEIGHT_FLOOR, as for the deeper-than-zero starts)."""
    neutral = problem.agent.v.reshape(-1)
    counts = {}
    for depth in PUNT_DEPTHS:
        finishes = []
        for punted, weights in make_punt_starts(neutral, depth):
            floors = np.full(problem.n_categories, LOCAL_WEIGHT_FLOOR)
            if depth == 0:
                floors[list(punted)] = 0.0
            finishes.append(find_local_maximum(problem, weights, floors)[0])
        shortfalls = [optimum - value for value in finishes if optimum - value > PEAK_TOLERANCE]
        counts[depth] = {'starts': len(finishes), 'missed': len(shortfalls),
                         'largest_shortfall': max(shortfalls, default=0.0)}
    return counts


# ── Branch and bound ──────────────────────────────────────────────────────────

def compute_upper_concave_hull(xs, ys):
    hull = []
    for point in zip(xs, ys):
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1) >= 0:
                hull.pop()
            else:
                break
        hull.append(point)
    hull = np.array(hull)
    return hull[:, 0], hull[:, 1]


class Envelopes:
    """Each category term's concave envelope on the node's interval, as hull vertices built on a fine grid."""

    def __init__(self, problem, lower, upper):
        self.vertex_stats, self.vertex_values = [], []
        for category in range(problem.n_categories):
            grid = np.linspace(lower[category], upper[category], ENVELOPE_GRID_POINTS)
            stats, values = compute_upper_concave_hull(grid, problem.compute_category_term(category, grid))
            self.vertex_stats.append(stats)
            self.vertex_values.append(values)

    def compute_value(self, category, future_stat):
        return float(np.interp(future_stat, self.vertex_stats[category], self.vertex_values[category]))

    def compute_conjugate(self, dual_weights):
        """sum over categories of min over vertices of (dual weight x stat - envelope)."""
        return sum(float(np.min(dual_weights[category] * stats - values))
                   for category, (stats, values) in enumerate(zip(self.vertex_stats, self.vertex_values)))

    def compute_smoothed_conjugate(self, dual_weights, temperature):
        """A soft-min of the conjugate, never above it, so a dual built on it is still a valid upper bound; its gradient
        is the soft-min-weighted stat."""
        total, gradient = 0.0, np.empty(len(dual_weights))
        for category, (stats, values) in enumerate(zip(self.vertex_stats, self.vertex_values)):
            terms = dual_weights[category] * stats - values
            shift = terms.min()
            softmin_weights = np.exp(-(terms - shift) / temperature)
            total += shift - temperature * np.log(softmin_weights.sum())
            gradient[category] = (softmin_weights * stats).sum() / softmin_weights.sum()
        return total, gradient


def solve_node(problem, lower, upper, start_dual_weights) -> NodeResult:
    envelopes = Envelopes(problem, lower, upper)

    def compute_exact_dual(dual_weights):
        return float(problem.compute_support(dual_weights)[0][0] - envelopes.compute_conjugate(dual_weights))

    dual_weights = np.maximum(np.asarray(start_dual_weights, dtype=float), 1e-9)
    for temperature in SMOOTHING_TEMPERATURES:
        def compute_smoothed_dual(candidate, temperature=temperature):
            support_value, reachable = problem.compute_support(candidate)
            conjugate, soft_stats = envelopes.compute_smoothed_conjugate(candidate, temperature)
            return float(support_value[0] - conjugate), reachable[0] - soft_stats
        result = minimize(compute_smoothed_dual, dual_weights, jac=True, method='L-BFGS-B',
                          bounds=[(1e-9, None)] * problem.n_categories, options={'maxiter': 300, 'gtol': 1e-12})
        if compute_exact_dual(result.x) <= compute_exact_dual(dual_weights) or temperature == SMOOTHING_TEMPERATURES[0]:
            dual_weights = result.x
    reachable_stats = problem.compute_support(dual_weights)[1][0]
    # Branch at the relaxation's own solution: the reachable point clipped into the box. The exact conjugate's
    # minimizer is not that point when it sits on an envelope's straight piece -- it is a hull vertex, where the gap
    # reads zero, and branching there never tightens the bound.
    branch_point = np.clip(reachable_stats, lower, upper)
    gaps = np.array([envelopes.compute_value(category, branch_point[category])
                     - problem.compute_category_term(category, [branch_point[category]])[0]
                     for category in range(problem.n_categories)])
    return NodeResult(compute_exact_dual(dual_weights), problem.compute_frozen_objective(reachable_stats),
                      dual_weights, reachable_stats, branch_point, gaps)


def certify_optimum(problem, incumbent, tolerance, node_limit):
    """Branch and bound from the given incumbent (a real objective value): stops when no open box's bound beats the
    incumbent by more than tolerance (certified), or at node_limit. Returns the final gap, the incumbent (raised if the
    search found better), how much it was raised, nodes and seconds."""
    sample = np.vstack([np.eye(problem.n_categories) + 1e-3,
                        np.random.default_rng(0).dirichlet(np.full(problem.n_categories, 0.3), size=2000)])
    sampled_stats = problem.compute_future_stats(sample)
    lower, upper = sampled_stats.min(axis=0) - 1e-6, sampled_stats.max(axis=0) + 1e-6
    began = time.perf_counter()
    starting_incumbent = incumbent
    root = solve_node(problem, lower, upper, problem.app_weights)
    incumbent = max(incumbent, root.lower_bound)
    heap = [(-root.upper_bound, 0, lower, upper, root)]
    sequence, n_nodes = 1, 1
    while heap:
        if -heap[0][0] - incumbent <= tolerance or n_nodes >= node_limit:
            break
        negative_bound, _, lower, upper, node = heapq.heappop(heap)
        if -negative_bound - incumbent <= tolerance:
            continue
        category = int(np.argmax(node.envelope_gaps))
        width = upper[category] - lower[category]
        split = float(node.branch_point[category])
        if not (lower[category] + 1e-6 * width < split < upper[category] - 1e-6 * width):
            split = 0.5 * (lower[category] + upper[category])
        left_upper = upper.copy(); left_upper[category] = split
        right_lower = lower.copy(); right_lower[category] = split
        for child_lower, child_upper in ((lower, left_upper), (right_lower, upper)):
            child = solve_node(problem, child_lower, child_upper, node.dual_weights)
            n_nodes += 1
            incumbent = max(incumbent, child.lower_bound)
            if child.upper_bound - incumbent > tolerance:
                heapq.heappush(heap, (-child.upper_bound, sequence, child_lower, child_upper, child))
                sequence += 1
    best_open_bound = max([-entry[0] for entry in heap], default=incumbent)
    return {'gap': max(best_open_bound - incumbent, 0.0), 'incumbent': incumbent,
            'improvement': incumbent - starting_incumbent, 'nodes': n_nodes,
            'seconds': time.perf_counter() - began}


# ── Why it works ──────────────────────────────────────────────────────────────

def measure_loop_gain(problem, weights):
    """At a peak: how far weight shares are from marginal-value shares; the largest loop-gain eigenvalue (a strict peak
    when below 1); the largest Hessian eigenvalue across directions orthogonal to w (finite differences; negative at a
    peak); and, per category, the weight relative to neutral, the mean win probability and the diagonal gain."""
    weights = weights / weights.sum()
    future_stats = problem.compute_future_stats(weights)[0]
    z = (problem.gap_intercepts + problem.gap_slopes * future_stats[:, None]) / np.sqrt(problem.gap_variances)
    density = np.exp(-z * z / 2) / np.sqrt(2 * np.pi)
    scale = problem.gap_slopes / np.sqrt(problem.gap_variances)
    n_terms = z.size
    marginal_values = (density * scale).sum(axis=1) / n_terms
    marginal_value_slopes = (-z * density * scale ** 2).sum(axis=1) / n_terms
    jacobian = problem.compute_weight_jacobian(weights)
    kappa = (marginal_values @ weights) / (weights @ weights)
    eigenvalues, eigenvectors = np.linalg.eigh((jacobian + jacobian.T) / 2)
    keep = eigenvalues > 1e-10 * eigenvalues.max()
    root = eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])
    loop_gain_eigenvalues = np.linalg.eigvalsh(root.T @ np.diag(marginal_value_slopes) @ root / kappa)

    def compute_weight_gradient(at_weights):
        _, values = problem.compute_frozen_objective_and_marginal_values(problem.compute_future_stats(at_weights)[0])
        return problem.compute_weight_jacobian(at_weights).T @ values
    step = 1e-6
    hessian = np.stack([(compute_weight_gradient(weights + step * unit) - compute_weight_gradient(weights - step * unit))
                        / (2 * step) for unit in np.eye(len(weights))], axis=1)
    projector = np.eye(len(weights)) - np.outer(weights, weights) / (weights @ weights)
    hessian_eigenvalues = np.linalg.eigvalsh(projector @ ((hessian + hessian.T) / 2) @ projector)
    hessian_eigenvalues = hessian_eigenvalues[np.abs(hessian_eigenvalues) > 1e-12]
    neutral_share = problem.agent.v.reshape(-1) / problem.agent.v.sum()
    return {
        'share_gap': float(np.abs(weights - marginal_values / marginal_values.sum()).max()),
        'max_loop_gain': float(loop_gain_eigenvalues.max()),
        'max_hessian_eigenvalue': float(hessian_eigenvalues.max()) if len(hessian_eigenvalues) else 0.0,
        'weight_relative_to_neutral': weights / neutral_share,
        'win_probability': ndtr(z).mean(axis=1),
        'diagonal_gain': (marginal_value_slopes / marginal_values) * (weights * np.diag(jacobian)),
    }


def measure_quasiconcavity(problem, optimum_weights, rng, n_pairs=600, n_points=21):
    """Share of random weight segments along which the objective dips below the lower of its two ends (a quasiconcave
    function never does), and the dips' depth, for three ways of drawing the ends: anywhere on the simplex; single or
    double punts at 5-100% of neutral; within +-30% of the optimum."""
    neutral = problem.agent.v.reshape(-1)
    n_categories = problem.n_categories

    def draw_punts(count):
        weights = np.tile(neutral, (count, 1))
        for row in weights:
            for category in rng.choice(n_categories, size=rng.integers(1, 3), replace=False):
                row[category] *= rng.uniform(0.05, 1.0)
        return weights

    samplers = {
        'anywhere': lambda: (rng.dirichlet(np.ones(n_categories), n_pairs), rng.dirichlet(np.ones(n_categories), n_pairs)),
        'between punts': lambda: (draw_punts(n_pairs), draw_punts(n_pairs)),
        'near the optimum': lambda: (optimum_weights * rng.uniform(0.7, 1.3, (n_pairs, n_categories)),
                                     optimum_weights * rng.uniform(0.7, 1.3, (n_pairs, n_categories))),
    }
    fractions = np.linspace(0, 1, n_points)
    results = {}
    for name, draw in samplers.items():
        starts, ends = draw()
        starts = starts / starts.sum(axis=1, keepdims=True)
        ends = ends / ends.sum(axis=1, keepdims=True)
        points = fractions[None, :, None] * starts[:, None, :] + (1 - fractions)[None, :, None] * ends[:, None, :]
        stats = problem.compute_future_stats(points.reshape(-1, n_categories))
        z = ((problem.gap_intercepts[None] + problem.gap_slopes[None] * stats[:, :, None])
             / np.sqrt(problem.gap_variances)[None])
        values = ndtr(z).mean(axis=(1, 2)).reshape(n_pairs, n_points)
        dips = np.minimum(values[:, 0], values[:, -1]) - values[:, 1:-1].min(axis=1)
        violating = dips > 1e-9
        results[name] = {'share_dipping': float(violating.mean()),
                         'median_dip': float(np.median(dips[violating])) if violating.any() else 0.0,
                         'largest_dip': float(dips.max()) if violating.any() else 0.0}
    return results


# ── Driving it ────────────────────────────────────────────────────────────────

UNREGULARIZED_SETTINGS = {'lambda_c': 0, 'lambda_p': 0}


def build_board_session(season: str, n_rounds: int) -> tuple:
    """(test client, session id, session, evaluate request) for the board after n_rounds snake rounds taken in base
    H-score order, my seat Drafter 1, Each Category at the app's settings with no regularization anywhere
    (UNREGULARIZED_SETTINGS, the self-play field included): the exact solves are of the unregularized objective, so the
    app and they solve the same problem."""
    from .league_sessions import build_league_session
    client, session_id, session = build_league_session('Each Category', season, UNREGULARIZED_SETTINGS)
    teams = session.current_settings['team_names']
    ranked = [int(player) for player in session.agent.default_h_scores.index]
    board = {team: [] for team in teams}
    for pick, team in enumerate(((teams + teams[::-1]) * n_rounds)[:len(teams) * n_rounds]):
        board[team].append(ranked[pick])
    return client, session_id, session, {'player_assignments': board, 'my_team_id': teams[0]}


def convert_to_display_weights(weights, neutral_weights):
    """Category weights in the units the app displays: each category's share of the total over its neutral share,
    times 100 (100 = neutral emphasis). Picks depend only on the weights' direction, so their scale is dropped."""
    weights = np.asarray(weights, dtype=float)
    return 100 * (weights / weights.sum()) / (neutral_weights / neutral_weights.sum())


def measure_board(
    season: str
    , n_rounds: int
    , n_candidates: int
    , tolerance: float
    , node_limit: int
    , long_iterations: int
) -> list[dict]:
    """Every measurement for the top n_candidates of one board (build_board_session). The board is evaluated twice --
    at the app's iteration budget and at long_iterations -- and every board has players on it, so both evaluates
    cold-start from the seed scan (the frozen table warm-starts only the empty board): the long one is the same descent
    run longer. The frozen problem is held at the first evaluate's solution."""
    client, session_id, session, request = build_board_session(season, n_rounds)
    agent = session.agent
    capture = capture_final_objective_call(client, session_id, agent, request)
    app_iterations = session.current_settings['n_iterations']
    session.current_settings['n_iterations'] = long_iterations
    try:
        long_capture = capture_final_objective_call(client, session_id, agent, request)
    finally:
        session.current_settings['n_iterations'] = app_iterations
    long_scores = dict(zip(long_capture[3], long_capture[2]['Score']))

    measurements = []
    for row in np.argsort(-capture[2]['Score'])[:n_candidates]:
        problem = FrozenCandidateProblem(agent, capture, int(row))
        optimum, optimum_weights = find_multistart_optimum(problem)
        optimum_shares = optimum_weights / optimum_weights.sum()
        certification = certify_optimum(problem, optimum, tolerance, node_limit)
        gain = measure_loop_gain(problem, optimum_shares)
        rng = np.random.default_rng([int(season[:4]), n_rounds, int(row)])
        measurements.append({
            'season': season, 'rounds': n_rounds, 'player': session.player_registry[problem.player_id].name,
            'app_iterations': app_iterations, 'app_score': problem.app_score,
            'long_app_score': float(long_scores[problem.player_id]), 'optimum': optimum,
            'certification': certification,
            'punt_starts': count_starts_reaching_optimum(problem, optimum),
            'loop_gain': {key: value for key, value in gain.items() if np.isscalar(value)},
            'quasiconcavity': measure_quasiconcavity(problem, optimum_shares, rng),
        })
    return measurements


def _measure_board_reporting_failure(*job) -> tuple:
    """measure_board, returning ('measured', measurements) or ('failed', job, traceback) so one broken board neither
    hides which board it was nor discards the others' hours of work."""
    import traceback
    try:
        return 'measured', measure_board(*job)
    except Exception:
        return 'failed', job, traceback.format_exc()


def measure_boards_in_parallel(
    seasons: list[str]
    , rounds: list[int]
    , n_candidates: int
    , tolerance: float
    , node_limit: int
    , long_iterations: int
    , n_workers: int
) -> tuple[list[dict], list[dict]]:
    """measure_board for every (season, rounds) pair, one board per job across n_workers processes. Returns the
    measurements (in job order) and the failed boards, each {season, rounds, traceback}."""
    import multiprocessing
    jobs = [(season, n_rounds, n_candidates, tolerance, node_limit, long_iterations)
            for season in seasons for n_rounds in rounds]
    with multiprocessing.get_context('spawn').Pool(min(n_workers, len(jobs))) as pool:
        outcomes = pool.starmap(_measure_board_reporting_failure, jobs, chunksize=1)
    measurements = [measurement for outcome in outcomes if outcome[0] == 'measured' for measurement in outcome[1]]
    failures = [{'season': outcome[1][0], 'rounds': outcome[1][1], 'traceback': outcome[2]}
                for outcome in outcomes if outcome[0] == 'failed']
    return measurements, failures
