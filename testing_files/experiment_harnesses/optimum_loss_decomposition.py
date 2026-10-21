"""Where the app loses to the optimum: its H-score shortfall from the certified optimum, split into its causes.

Five answers to the same problem, each a step from the one before:
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
The four gaps add up to the whole shortfall 5 - 1.

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


def descend_like_the_app(
    problem: exact.FrozenCandidateProblem
    , start_weights: np.ndarray
    , n_iterations: int
    , keeps_best_point: bool
) -> tuple[np.ndarray, np.ndarray]:
    """The app's descent from each of S starts at once, for n_iterations: (S,) objectives and (S, C) weights. With
    keeps_best_point, the best point any objective call scored along the way; otherwise the app's report, the weights
    its last objective call scored."""
    weights = start_weights / start_weights.sum(axis=1, keepdims=True)
    first_moment = second_moment = 0.0
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
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
        update = (_CATEGORY_LEARNING_RATE * (first_moment / (1 - beta1 ** step))
                  / (np.sqrt(second_moment / (1 - beta2 ** step)) + epsilon))
        weights = np.clip(weights + update, 0, None)
        weights = weights / weights.sum(axis=1, keepdims=True)
    return reported_objectives, reported_weights


def make_extra_seeds(neutral_weights: np.ndarray) -> np.ndarray:
    """Neutral and every single and double punt at EXTRA_SEED_DEPTH of neutral."""
    return np.vstack([neutral_weights] + [weights for _, weights in exact.make_punt_starts(neutral_weights,
                                                                                            EXTRA_SEED_DEPTH)])


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
        raise RuntimeError('The final descent scored no seed menu (a warm-started evaluate is not a cold start).')
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
        step_1, _ = descend_like_the_app(problem, menu[[chosen]], n_iterations, keeps_best_point=False)
        menu_results, _ = descend_like_the_app(problem, menu, n_iterations, keeps_best_point=False)
        all_starts = np.vstack([menu, extra_seeds])
        all_results, _ = descend_like_the_app(problem, all_starts, n_iterations, keeps_best_point=False)
        best_start = int(np.argmax(all_results))
        step_4, _ = descend_like_the_app(problem, all_starts[[best_start]], LONG_ITERATIONS, keeps_best_point=True)
        optimum, _ = exact.find_multistart_optimum(problem)
        results.append({
            'season': season, 'rounds': n_rounds, 'player': session.player_registry[problem.player_id].name,
            'app': problem.app_score,
            'steps': [float(step_1[0]), float(menu_results.max()), float(all_results.max()), float(step_4[0]),
                      float(optimum)],
            # What each improvement costs: the seed and iteration counts behind the first three, and the app's own
            # time per candidate, which branch and bound's measured time is set against.
            'menu_size': int(menu.shape[0]),
            'extra_seed_count': int(extra_seeds.shape[0]),
            'n_iterations': int(n_iterations),
            'app_seconds_per_candidate': app_seconds_per_candidate,
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
