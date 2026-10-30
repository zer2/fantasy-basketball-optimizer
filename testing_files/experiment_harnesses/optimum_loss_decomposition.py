"""Where the app loses to the optimum: its H-score shortfall from the certified optimum, split into its causes.

Six answers to the same problem, each a step from the one before:
  0. No seeding: the app's descent from the neutral weights, perturbed by NEUTRAL_PERTURBATION (neutral itself is a
     stationary point), for the app's iteration budget. The gap 0 -> 1 is what the app's seeding already buys; it is
     not a loss, since the app does seed.
  1. The app as it runs: the seed scan picks one seed per candidate by its objective before any descent, then descends
     from it for the app's iteration budget.
  2. The app's descent from every seed in the app's menu, keeping the best result: the gap 1 -> 2 is the seed scan
     choosing the wrong seed.
  3. The app's descent from more seeds (the app's menu plus neutral and every single and double punt at 5% of neutral),
     keeping the best: the gap 2 -> 3 is what the app's menu does not cover.
  4. The descent that won 3, continued to a long budget (the same start and optimiser state, so its first iterations
     are 3's), keeping the best point it scored on the way: the gap 3 -> 4 is the iteration budget. Fixed-step Adam
     does not settle exactly, so its last point can fall below an earlier one; the best point cannot fall below 3.
  5. The certified optimum (exact_optimum: the best multi-start ascent, which branch and bound certified to 0.001pp):
     the gap 4 -> 5 is what the app's optimiser cannot reach however long it runs.
The four gaps from 1 add up to the whole shortfall 5 - 1.

ASSUMPTIONS
  - The problem. Every step solves the same problem: exact_optimum's frozen problem for the candidate, built from the
    app's own evaluate of the board. It is the app's final objective with two things held at the app's solution: the
    percentage categories' volume scale (my team's k), and the position term (the flex shares' contribution). The app
    itself re-optimises both as it descends; here they stay fixed, so only the category weights move. Step 5 is that
    problem's certified optimum, so steps 1 and 5 are on exactly the same problem.
  - No regularization. lambda_c = lambda_p = 0 everywhere, the self-play field included (exact_optimum's sessions), so
    the app and the optimum solve the same objective.
  - The app's optimiser, as it is in HAgent.perform_iterations: Adam (learning rate _CATEGORY_LEARNING_RATE, beta1
    0.9, beta2 0.999), a fresh optimiser per descent, the gradient centred over the categories (the simplex
    projection), negative weights set to 0 and the weights renormalised to sum 1 after every step. The result reported
    is the app's: the weights its last objective call scored, which is one update short of the last update.
  - The app's seed scan, as it is in HAgent._select_starting_weights: the menu is recorded from the app's own evaluate
    (its seed-scoring calls), so it is exactly the app's -- one gentle punt per category, the balanced build, and the
    candidate's builds from the self-play history -- and each candidate takes the seed with the best objective before
    any descent. Here that objective is the frozen one.
  - Budgets: the app's own (the session's n_iterations) for steps 1-3; LONG_ITERATIONS in all for step 4.
  - Step 1 is a copy of the app's descent on the frozen problem, not the app itself.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.special import ndtr

from . import exact_optimum as exact
from backend.math.algorithm_agents import _CATEGORY_LEARNING_RATE

LONG_ITERATIONS = 2000
EXTRA_SEED_DEPTH = 0.05
# The no-seeding start: each neutral weight times (1 + NEUTRAL_PERTURBATION x a standard normal draw).
NEUTRAL_PERTURBATION = 0.01


# ── The app's optimiser, on the frozen problem ────────────────────────────────

def compute_frozen_objectives_and_gradients(
    problem: exact.FrozenCandidateProblem
    , weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(S,) frozen objective and (S, C) its gradient in the weights, for S weight vectors at once."""
    future_stats = problem.compute_future_stats(weights)                                    # (S, C)
    z = ((problem.gap_intercepts[None] + problem.gap_slopes[None] * future_stats[:, :, None])
         / np.sqrt(problem.gap_variances)[None])                                            # (S, C, O)
    density = np.exp(-z * z / 2) / np.sqrt(2 * np.pi)
    marginal_values = (density * (problem.gap_slopes / np.sqrt(problem.gap_variances))[None]).mean(axis=2) \
        / problem.n_categories                                                              # (S, C)
    jacobians = problem.n_remaining * exact.compute_expected_pick_tilt_jacobian(
        weights, problem.agent.L[0], problem.agent.v, problem.agent.pick_pool_size)        # (S, C, C)
    gradients = np.einsum('sij,si->sj', jacobians, marginal_values)
    return ndtr(z).mean(axis=(1, 2)), gradients


ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON = 0.9, 0.999, 1e-8


def descend_like_the_app(
    problem: exact.FrozenCandidateProblem
    , start_weights: np.ndarray
    , n_iterations: int
    , keeps_best_point: bool
) -> tuple[np.ndarray, np.ndarray]:
    """The app's descent from each of S starts at once, for n_iterations: (S,) objectives and (S, C) weights. With
    keeps_best_point, the best point any objective call scored along the way; otherwise the app's report, the weights
    its last objective call scored."""
    objectives, weights, _, _, _ = descend_like_the_app_keeping_adam_state(problem, start_weights, n_iterations,
                                                                            keeps_best_point)
    return objectives, weights


def descend_like_the_app_keeping_adam_state(
    problem: exact.FrozenCandidateProblem
    , start_weights: np.ndarray
    , n_iterations: int
    , keeps_best_point: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    """descend_like_the_app, also returning Adam's state: (S, C) second moment when it stops and the step count (for
    the bias correction), which measure_adam_contraction linearises the update with, and (S, T, C) the second moment
    after every one of the T iterations, which the conditioning measurement scales the curvature with."""
    weights = start_weights / start_weights.sum(axis=1, keepdims=True)
    first_moment = second_moment = 0.0
    beta1, beta2, epsilon = ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON
    step = 0
    second_moment_history = []
    for step in range(1, max(1, n_iterations) + 1):
        objectives, gradients = compute_frozen_objectives_and_gradients(problem, weights)
        if keeps_best_point and step > 1:
            improved = objectives > reported_objectives
            reported_objectives = np.where(improved, objectives, reported_objectives)
            reported_weights = np.where(improved[:, None], weights, reported_weights)
        else:
            reported_objectives, reported_weights = objectives, weights
        centred = gradients - gradients.mean(axis=1, keepdims=True)
        first_moment = beta1 * first_moment + (1 - beta1) * centred
        second_moment = beta2 * second_moment + (1 - beta2) * centred ** 2
        second_moment_history.append(second_moment)
        update = (_CATEGORY_LEARNING_RATE * (first_moment / (1 - beta1 ** step))
                  / (np.sqrt(second_moment / (1 - beta2 ** step)) + epsilon))
        weights = np.clip(weights + update, 0, None)
        weights = weights / weights.sum(axis=1, keepdims=True)
    return reported_objectives, reported_weights, second_moment, step, np.stack(second_moment_history, axis=1)


# ── Curvature and Adam's contraction at the optimum ───────────────────────────

CURVATURE_STEP = 1e-6
# With Adam's momentum (m = beta1 m + (1 - beta1) g, then a step of size a along m), a direction of curvature c is a
# two-step linear system whose roots multiply to beta1. While a x (1 - beta1) x c lies between (1 - sqrt(beta1))^2 and
# (1 + sqrt(beta1))^2 the roots are complex: the direction spirals in, and both have magnitude exactly sqrt(beta1),
# whatever c is. The upper end is also the stability limit (2 + 2 beta1) / (1 - beta1) = 38 on a x c.
UNDERDAMPED_WINDOW = ((1 - np.sqrt(ADAM_BETA1)) ** 2 / (1 - ADAM_BETA1),
                      (1 + np.sqrt(ADAM_BETA1)) ** 2 / (1 - ADAM_BETA1))


def make_simplex_tangent_basis(n_categories: int) -> np.ndarray:
    """(C, C - 1) orthonormal directions that keep the weights summing to 1."""
    centring = np.eye(n_categories) - 1 / n_categories
    basis, _ = np.linalg.qr(centring[:, :-1])
    return basis


def compute_centred_gradient(
    problem: exact.FrozenCandidateProblem
    , weights: np.ndarray
) -> np.ndarray:
    """The gradient the app steps along at these weights: centred over the categories (the simplex projection)."""
    _, gradients = compute_frozen_objectives_and_gradients(problem, weights[None])
    return gradients[0] - gradients[0].mean()


def compute_tangent_hessian(
    problem: exact.FrozenCandidateProblem
    , optimum: np.ndarray
    , basis: np.ndarray
) -> np.ndarray:
    """(C - 1, C - 1) the Hessian along the simplex at the optimum: finite differences of the centred gradient in the
    tangent directions. Its eigenvalues are negative at a peak."""
    hessian_columns = []
    for direction in basis.T:
        upper = compute_centred_gradient(problem, optimum + CURVATURE_STEP * direction)
        lower = compute_centred_gradient(problem, optimum - CURVATURE_STEP * direction)
        hessian_columns.append(basis.T @ (upper - lower) / (2 * CURVATURE_STEP))
    return np.array(hessian_columns).T


def compute_step_scaled_curvatures(
    second_moments: np.ndarray
    , steps: np.ndarray
    , tangent_hessian: np.ndarray
    , basis: np.ndarray
) -> np.ndarray:
    """(..., C - 1) each direction's curvature times its Adam step size, for second moments (..., C) recorded at the
    step counts steps (...,): the learning rate over the root second moment, with the bias corrections, scaling the
    negated tangent Hessian. When every one lies in UNDERDAMPED_WINDOW, every direction spirals in at sqrt(beta1)."""
    step_sizes = (_CATEGORY_LEARNING_RATE / (1 - ADAM_BETA1 ** steps)[..., None]
                  / (np.sqrt(second_moments / (1 - ADAM_BETA2 ** steps)[..., None]) + ADAM_EPSILON))
    scaled_matrices = np.einsum('ca,...c,cb,bd->...ad', basis, step_sizes, basis, -tangent_hessian)
    return np.sort(np.abs(np.linalg.eigvals(scaled_matrices)), axis=-1)


def measure_adam_contraction(
    problem: exact.FrozenCandidateProblem
    , optimum_weights: np.ndarray
    , adam_states: dict
) -> dict:
    """The curvature at the optimum, and how fast the app's update closes in on it.

    Curvature: the Hessian along the simplex (finite differences of the centred gradient in the tangent directions);
    its eigenvalues are negative at a peak, and their magnitudes are reported, steepest and flattest.

    Contraction: the app's update -- momentum, the step divided by Adam's second moment, clipping at 0, renormalising
    -- as a map on (weights, momentum), with the second moment held at a recorded state (it moves on a 1/(1 - beta2) =
    1,000-iteration timescale). Its Jacobian at (optimum, no momentum), by finite differences in the tangent and
    momentum directions, has largest eigenvalue magnitude rho: the factor a small deviation shrinks by per iteration.
    One rho per entry of adam_states, {label: ((C,) second moment, step count)}."""
    n_categories = problem.n_categories
    basis = make_simplex_tangent_basis(n_categories)
    optimum = optimum_weights / optimum_weights.sum()

    tangent_hessian = compute_tangent_hessian(problem, optimum, basis)
    curvatures = np.linalg.eigvalsh((tangent_hessian + tangent_hessian.T) / 2)

    def apply_update(weights, momentum, second_moment, step):
        momentum = ADAM_BETA1 * momentum + (1 - ADAM_BETA1) * compute_centred_gradient(problem, weights)
        update = (_CATEGORY_LEARNING_RATE * (momentum / (1 - ADAM_BETA1 ** step))
                  / (np.sqrt(second_moment / (1 - ADAM_BETA2 ** step)) + ADAM_EPSILON))
        stepped = np.clip(weights + update, 0, None)
        return stepped / stepped.sum(), momentum

    contraction, step_scaled_curvatures = {}, {}
    for label, (second_moment, step) in adam_states.items():
        scaled = compute_step_scaled_curvatures(second_moment, np.array(step), tangent_hessian, basis)
        step_scaled_curvatures[label] = (float(scaled.min()), float(scaled.max()))
        fixed_weights, fixed_momentum = apply_update(optimum, np.zeros(n_categories), second_moment, step)
        columns = []
        for coordinate in range(2 * n_categories - 1):
            nudge_weights = basis[:, coordinate] if coordinate < n_categories - 1 else np.zeros(n_categories)
            nudge_momentum = (np.eye(n_categories)[coordinate - (n_categories - 1)] if coordinate >= n_categories - 1
                              else np.zeros(n_categories))
            upper = apply_update(optimum + CURVATURE_STEP * nudge_weights, CURVATURE_STEP * nudge_momentum,
                                 second_moment, step)
            lower = apply_update(optimum - CURVATURE_STEP * nudge_weights, -CURVATURE_STEP * nudge_momentum,
                                 second_moment, step)
            columns.append(np.concatenate([basis.T @ (upper[0] - lower[0]), upper[1] - lower[1]])
                           / (2 * CURVATURE_STEP))
        contraction[label] = float(np.abs(np.linalg.eigvals(np.array(columns).T)).max())
        # The optimum must be (nearly) a fixed point of the update for the linearisation to describe it.
        contraction[f'{label} fixed-point drift'] = float(np.abs(fixed_weights - optimum).max())
    return {'steepest_curvature': float(np.abs(curvatures).max()),
            'flattest_curvature': float(np.abs(curvatures).min()),
            'largest_curvature_eigenvalue': float(curvatures.max()),
            'step_scaled_curvatures': step_scaled_curvatures,
            'contraction': contraction}


def make_extra_seeds(neutral_weights: np.ndarray) -> np.ndarray:
    """Neutral and every single and double punt at EXTRA_SEED_DEPTH of neutral."""
    return np.vstack([neutral_weights] + [weights for _, weights in exact.make_punt_starts(neutral_weights,
                                                                                            EXTRA_SEED_DEPTH)])


def describe_seeds(
    categories: list[str]
    , menu_size: int
) -> list[dict]:
    """What each start in np.vstack([menu, make_extra_seeds(...)]) is, in that order: the app's menu (one gentle punt
    per category at the app's punt factor, the balanced build, then the candidate's builds from the self-play history;
    HAgent._select_starting_weights), then neutral and the single and double punts at EXTRA_SEED_DEPTH. 'kind' is
    gentle, balanced, history, neutral or punt; 'categories' the punted categories."""
    n_categories = len(categories)
    return ([{'kind': 'gentle', 'categories': (categories[index],), 'app_menu': True}
             for index in range(n_categories)]
            + [{'kind': 'balanced', 'categories': (), 'app_menu': True}]
            + [{'kind': 'history', 'categories': (), 'app_menu': True}
               for _ in range(menu_size - n_categories - 1)]
            + [{'kind': 'neutral', 'categories': (), 'app_menu': False}]
            + [{'kind': 'punt', 'categories': tuple(categories[index] for index in punted), 'app_menu': False}
               for punted, _ in exact.make_punt_starts(np.ones(n_categories), EXTRA_SEED_DEPTH)])


def measure_seeds(
    problem: exact.FrozenCandidateProblem
    , starts: np.ndarray
    , app_budget_results: np.ndarray
    , seeds: list[dict]
) -> list[dict]:
    """Each start's objective before any descent (what the seed scan compares), after the app's iteration budget (what
    multi-starting compares), and at the long budget (the app's descent continued to LONG_ITERATIONS, keeping the best
    point it scored: where the app's optimiser gets from that seed given unlimited iterations). The self-play history
    builds are left out: the seed-start section studies the menu's fixed seeds."""
    kept = [index for index, seed in enumerate(seeds) if seed['kind'] != 'history']
    normalized = starts[kept] / starts[kept].sum(axis=1, keepdims=True)
    pre_descent, _ = compute_frozen_objectives_and_gradients(problem, normalized)
    long_budget, _ = descend_like_the_app(problem, normalized, LONG_ITERATIONS, keeps_best_point=True)
    return [{**seeds[index], 'pre_descent': float(pre_descent[position]),
             'app_budget': float(app_budget_results[index]), 'long_budget': float(long_budget[position])}
            for position, index in enumerate(kept)]


# ── Capturing the app's evaluate, seed menu included ─────────────────────────

def capture_final_descent(client, session_id, agent, request) -> dict:
    """Evaluate the board and return the app's final descent: its last objective call (as
    exact_optimum.capture_final_objective_call returns it) and the seed menu it scored before descending, one (N, C)
    matrix per seed."""
    descents = []
    original_objective, original_descent = agent.get_objective_and_gradient, agent.perform_iterations

    def recording_descent(*args, **kwargs):
        descents.append({'candidate_ids': list(args[7]), 'seeds': [], 'final': None})
        return original_descent(*args, **kwargs)

    def recording_objective(*args, **kwargs):
        result = original_objective(*args, **kwargs)
        descent = descents[-1]
        if kwargs.get('correction_mode') == 'skip':
            descent['seeds'].append(np.array(args[0], dtype=float))
        else:
            descent['final'] = (args, kwargs, result, descent['candidate_ids'])
        return result
    agent.get_objective_and_gradient, agent.perform_iterations = recording_objective, recording_descent
    try:
        response = client.post(f'/sessions/{session_id}/evaluate', json=request)
        if response.status_code != 200:
            raise RuntimeError(f'Evaluate failed: {response.text[:300]}')
    finally:
        del agent.get_objective_and_gradient, agent.perform_iterations
    final_descent = descents[-1]
    if not final_descent['seeds'] or final_descent['final'] is None:
        raise RuntimeError('The final descent scored no seed menu (the evaluate did not run its seed scan).')
    return final_descent


# ── One board ─────────────────────────────────────────────────────────────────

def decompose_board(
    season: str
    , n_rounds: int
    , n_candidates: int
) -> list[dict]:
    """The five answers for the top n_candidates of one board (exact_optimum.build_board_session), and what the
    improvements cost."""
    client, session_id, session, request = exact.build_board_session(season, n_rounds)
    agent = session.agent
    n_iterations = session.current_settings['n_iterations']
    descent = capture_final_descent(client, session_id, agent, request)
    capture = descent['final']
    # The app's own time per candidate: one more evaluate of the board (the field is built by now), timed.
    began = time.perf_counter()
    timed = agent.get_h_scores(player_assignments=request['player_assignments'], drafter=request['my_team_id'],
                               n_iterations=n_iterations)
    app_seconds_per_candidate = (time.perf_counter() - began) / len(timed['Scores'])
    seeds = np.stack(descent['seeds'])                                   # (seeds, N, C)
    neutral_weights = agent.v.reshape(-1)
    extra_seeds = make_extra_seeds(neutral_weights)

    results = []
    for row in np.argsort(-capture[2]['Score'])[:n_candidates]:
        row = int(row)
        problem = exact.FrozenCandidateProblem(agent, capture, row)
        menu = seeds[:, row, :]
        menu_scores, _ = compute_frozen_objectives_and_gradients(problem, menu / menu.sum(axis=1, keepdims=True))
        chosen = int(np.argmax(menu_scores))
        step_1, app_weights, app_second_moment, app_steps, _ = descend_like_the_app_keeping_adam_state(
            problem, menu[[chosen]], n_iterations, keeps_best_point=False)
        rng = np.random.default_rng([int(season[:4]), n_rounds, row])
        perturbed_neutral = neutral_weights * (1 + NEUTRAL_PERTURBATION * rng.standard_normal(neutral_weights.shape))
        neutral_start, _ = descend_like_the_app(problem, perturbed_neutral[None], n_iterations, keeps_best_point=False)
        menu_results, _ = descend_like_the_app(problem, menu, n_iterations, keeps_best_point=False)
        all_starts = np.vstack([menu, extra_seeds])
        all_results, _ = descend_like_the_app(problem, all_starts, n_iterations, keeps_best_point=False)
        best_start = int(np.argmax(all_results))
        step_4, _, long_second_moment, long_steps, _ = descend_like_the_app_keeping_adam_state(
            problem, all_starts[[best_start]], LONG_ITERATIONS, keeps_best_point=True)
        optimum, optimum_weights = exact.find_multistart_optimum(problem)
        results.append({
            'season': season, 'rounds': n_rounds, 'player': session.player_registry[problem.player_id].name,
            'app': problem.app_score,
            # Where the app ends and where the optimum is (weights summing to 1, in the agent's category order), so later
            # measurements can start from them instead of solving again.
            'categories': list(agent.x_scores.columns),
            'app_weights': [float(weight) for weight in app_weights[0]],
            'optimum_weights': [float(weight) for weight in optimum_weights / optimum_weights.sum()],
            # Step 0, before the app's seeding; 'steps' runs from the app (step 1) to the optimum.
            'neutral_start': float(neutral_start[0]),
            'steps': [float(step_1[0]), float(menu_results.max()), float(all_results.max()), float(step_4[0]),
                      float(optimum)],
            # What each improvement costs: the seed and iteration counts behind the first three, and the app's own
            # time per candidate, which branch and bound's measured time is set against.
            'menu_size': int(menu.shape[0]),
            'extra_seed_count': int(extra_seeds.shape[0]),
            'n_iterations': int(n_iterations),
            'app_seconds_per_candidate': app_seconds_per_candidate,
            # The curvature at the optimum and the app's contraction rate there, with Adam's state as the app leaves it
            # and as the long descent leaves it.
            'curvature': measure_adam_contraction(problem, optimum_weights, {
                'app': (app_second_moment[0], app_steps), 'long': (long_second_moment[0], long_steps)}),
            # Every start of both menus, for the seed-start section.
            'seeds': measure_seeds(problem, all_starts, all_results,
                                   describe_seeds(list(agent.x_scores.columns), int(menu.shape[0]))),
        })
    return results


def decompose_boards_in_parallel(
    seasons: list[str]
    , rounds: list[int]
    , n_candidates: int
    , n_workers: int
) -> list[dict]:
    """decompose_board for every (season, rounds) pair, one board per job across n_workers processes."""
    import multiprocessing
    jobs = [(season, n_rounds, n_candidates) for season in seasons for n_rounds in rounds]
    with multiprocessing.get_context('spawn').Pool(min(n_workers, len(jobs))) as pool:
        per_board = pool.starmap(decompose_board, jobs, chunksize=1)
    return [case for board in per_board for case in board]


# ── Conditioning across the iterations ────────────────────────────────────────

def measure_conditioning_board(
    season: str
    , n_rounds: int
    , n_candidates: int
    , optimum_weights_by_player: dict
) -> list[dict]:
    """How the step x curvature at the optimum changes over the iterations, for the top n_candidates of one board: the
    curvature is the Hessian along the simplex at the optimum (fixed), and what changes is Adam's step size in each
    direction, recorded after every iteration of three kinds of descent:
      - every fixed seed of both menus (the seed-start section's seeds, no history builds), for the app's iterations;
      - the app's own descent from the seed it chooses, for the app's iterations;
      - the long descent from the best start (decompose_board's step 4), for LONG_ITERATIONS.
    All C - 1 step x curvature values are kept for every descent and iteration. The optimum is decompose_board's,
    passed in as {player name: weights}, so it is not solved again; that it is still a peak and a fixed point of the
    update on this board is recorded, as a check that the board is the same problem."""
    client, session_id, session, request = exact.build_board_session(season, n_rounds)
    agent = session.agent
    n_iterations = session.current_settings['n_iterations']
    descent = capture_final_descent(client, session_id, agent, request)
    capture = descent['final']
    seeds = np.stack(descent['seeds'])                                   # (seeds, N, C)
    extra_seeds = make_extra_seeds(agent.v.reshape(-1))
    categories = list(agent.x_scores.columns)
    basis = make_simplex_tangent_basis(agent.n_categories)

    results = []
    for row in np.argsort(-capture[2]['Score'])[:n_candidates]:
        row = int(row)
        problem = exact.FrozenCandidateProblem(agent, capture, row)
        player = session.player_registry[problem.player_id].name
        saved = optimum_weights_by_player[player]
        if saved['categories'] != categories:
            raise ValueError(f'{season} after {n_rounds} rounds, {player}: categories {saved["categories"]} were saved, '
                             f'the board has {categories}.')
        optimum = np.array(saved['optimum_weights'])
        tangent_hessian = compute_tangent_hessian(problem, optimum, basis)

        menu = seeds[:, row, :]
        menu_scores, _ = compute_frozen_objectives_and_gradients(problem, menu / menu.sum(axis=1, keepdims=True))
        chosen = int(np.argmax(menu_scores))
        all_starts = np.vstack([menu, extra_seeds])
        seed_descriptions = describe_seeds(categories, int(menu.shape[0]))
        kept = [index for index, seed in enumerate(seed_descriptions) if seed['kind'] != 'history']
        all_results, _ = descend_like_the_app(problem, all_starts, n_iterations, keeps_best_point=False)
        best_start = int(np.argmax(all_results))

        def record_step_scaled(starts, budget):
            *_, last_second_moment, last_step, history = descend_like_the_app_keeping_adam_state(
                problem, starts, budget, keeps_best_point=False)
            steps = np.broadcast_to(np.arange(1, history.shape[1] + 1), history.shape[:2])
            return compute_step_scaled_curvatures(history, steps, tangent_hessian, basis), last_second_moment, last_step

        seed_scaled, _, _ = record_step_scaled(all_starts[kept], n_iterations)
        app_scaled, app_second_moment, app_steps = record_step_scaled(menu[[chosen]], n_iterations)
        long_scaled, _, _ = record_step_scaled(all_starts[[best_start]], LONG_ITERATIONS)
        check = measure_adam_contraction(problem, optimum, {'app': (app_second_moment[0], app_steps)})
        results.append({
            'season': season, 'rounds': n_rounds, 'player': player, 'categories': categories,
            'seeds': [seed_descriptions[index] for index in kept],
            'seed_step_scaled': seed_scaled.astype(np.float32),       # (kept seeds, app iterations, C - 1)
            'app_step_scaled': app_scaled[0].astype(np.float32),      # (app iterations, C - 1)
            'long_step_scaled': long_scaled[0].astype(np.float32),    # (LONG_ITERATIONS, C - 1)
            'long_start_index': best_start,
            'largest_curvature_eigenvalue': check['largest_curvature_eigenvalue'],
            'optimum_fixed_point_drift': check['contraction']['app fixed-point drift'],
        })
    return results


def measure_conditioning_boards_in_parallel(
    seasons: list[str]
    , rounds: list[int]
    , n_candidates: int
    , n_workers: int
    , optimum_weights_by_board: dict
) -> list[dict]:
    """measure_conditioning_board for every (season, rounds) pair across n_workers processes; optimum_weights_by_board
    is {(season, rounds): {player name: {'categories': ..., 'optimum_weights': ...}}}."""
    import multiprocessing
    jobs = [(season, n_rounds, n_candidates, optimum_weights_by_board[(season, n_rounds)])
            for season in seasons for n_rounds in rounds]
    with multiprocessing.get_context('spawn').Pool(min(n_workers, len(jobs))) as pool:
        per_board = pool.starmap(measure_conditioning_board, jobs, chunksize=1)
    return [case for board in per_board for case in board]
