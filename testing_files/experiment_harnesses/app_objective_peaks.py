"""Where the app loses to the best build found, and how many peaks there are, for formats with no exact solver (Most
Categories, Rotisserie).

Each Category has a certified optimum (exact_optimum), and its sections measure the app against it on a frozen copy of
the objective. Most Categories and Rotisserie have nothing like an exact algorithm, so here the reference is the BEST
BUILD FOUND: the app's own descent, on the app's own objective, run from every start of a wide menu for a long budget.
That is a lower bound on the optimum, not a certificate.

Four answers to the same problem for each candidate, each a step from the one before, and one below them:
  0. No seeding (Each Category and Most Categories, which seed from a menu; Rotisserie has no seeding): the app's
     descent from the neutral build, perturbed by NEUTRAL_PERTURBATION (neutral itself is a stationary point), for the
     app's budget. The gap 0 -> 1 is what the app's seeding already gains; not a loss.
  1. The app as it runs: its start policy (Most Categories: the seed scan picks one seed of the shared menu by its
     objective before any descent; Rotisserie: its single low-variance seed), then the app's iteration budget.
  2. The app's descent from every start in the app's menu, keeping the best result after the app's budget: the gap
     1 -> 2 is the seed scan choosing the wrong seed. Rotisserie's menu is its one seed, so this gap is zero there.
  3. The same from the full menu (the app's menu plus every single and double punt at EXTRA_SEED_DEPTH of neutral, and
     for Rotisserie the shared menu too): the gap 2 -> 3 is what the app's menu does not cover.
  4. The best build found: every start of the full menu descended for LONG_ITERATIONS, keeping the best point each
     descent scored: the gap 3 -> 4 is the iteration budget.

ASSUMPTIONS
  - The problem is the app's own: positions, volume and the regulariser all live, the session at the app's settings,
    the opponent field as the app models it on that board. Nothing is frozen.
  - Every descent is the app's own (HAgent.perform_iterations), started through the agent's start-injection hook
    (initial_category_weights, uniform flex shares), for the candidates as one batch (candidate_subset), so their
    position solves are exact, as the app's are for the top of its ranking.
  - A long descent's first n_iterations are the app-budget descent from the same start (the descent is deterministic
    and nothing in it depends on the budget), so one long run per start gives both: the score of the objective call
    at the app's last iteration, and the best score over the whole run.
  - The board's opponent store is refreshed by step 1's normal solve before any injected descent, so the injected start
    never reaches the opponent model's nested solves.
"""

from __future__ import annotations

import numpy as np

from . import exact_optimum as exact
from .league_sessions import build_league_session
from .optimum_loss_decomposition import NEUTRAL_PERTURBATION

LONG_ITERATIONS = 2000
EXTRA_SEED_DEPTH = 0.05


def build_format_board(objective: str
                       , season: str
                       , n_rounds: int) -> tuple:
    """(session, board, my team) for the board after n_rounds snake rounds taken in base H-score order, my seat Drafter
    1, this objective at the app's own settings (no overrides). n_rounds = 0 is the empty board: the first pick, against
    the predicted field."""
    _, _, session = build_league_session(objective, season, {})
    teams = session.current_settings['team_names']
    ranked = [int(player) for player in session.agent.default_h_scores.index]
    board = {team: [] for team in teams}
    for pick, team in enumerate(((teams + teams[::-1]) * n_rounds)[:len(teams) * n_rounds]):
        board[team].append(ranked[pick])
    return session, board, teams[0]


def list_app_menu(agent) -> list[dict]:
    """The app's own starts, with what they are: the shared menu (one gentle punt per category, the balanced build)
    for head to head, the single low-variance seed for Rotisserie."""
    categories = list(agent.x_scores.columns)
    if agent.seed_mode == 'lowvar':
        return [{'kind': 'lowvar', 'categories': (), 'app_menu': True, 'weights': agent.compute_lowvar_seed()}]
    if agent.seed_mode != 'multistart':
        raise ValueError(f'No app menu for seed mode {agent.seed_mode!r}.')
    seeds = agent.list_shared_seeds()
    return ([{'kind': 'gentle', 'categories': (categories[index],), 'app_menu': True, 'weights': seeds[index]}
             for index in range(len(categories))]
            + [{'kind': 'balanced', 'categories': (), 'app_menu': True, 'weights': seeds[len(categories)]}])


def list_full_menu(agent) -> list[dict]:
    """The app's menu, then (Rotisserie) the shared menu it does not use, then every single and double punt at
    EXTRA_SEED_DEPTH of neutral."""
    categories = list(agent.x_scores.columns)
    starts = list_app_menu(agent)
    if agent.seed_mode == 'lowvar':
        shared = agent.list_shared_seeds()
        starts += ([{'kind': 'gentle', 'categories': (categories[index],), 'app_menu': False, 'weights': shared[index]}
                    for index in range(len(categories))]
                   + [{'kind': 'balanced', 'categories': (), 'app_menu': False, 'weights': shared[len(categories)]}])
    neutral = agent.get_starting_category_weights()
    starts += [{'kind': 'punt', 'categories': tuple(categories[index] for index in punted), 'app_menu': False,
                'weights': weights}
               for punted, weights in exact.make_punt_starts(neutral, EXTRA_SEED_DEPTH)]
    return starts


class DescentRecorder:
    """Records the scores of the outer descent's objective calls (one per iteration) and the seed scan's pre-descent
    scores, for the batch of exactly the candidates measured -- the opponent model's nested solves (one candidate each)
    are not recorded. Scores are put in the order of `candidates`: the solver's batch has its own row order (its
    descent receives the player in each row), which is not the order the candidates were passed in."""

    def __init__(self
                 , agent
                 , candidates):
        self.agent = agent
        self.candidates = list(candidates)
        self.original_objective = agent.get_objective_and_gradient
        self.original_descent = agent.perform_iterations
        self.batch_order = None
        self.iteration_scores = []
        self.scan_scores = []

    def reorder(self, scores):
        """Batch-ordered scores in the order of `candidates`."""
        position = {player: row for row, player in enumerate(self.batch_order)}
        return np.array([scores[position[player]] for player in self.candidates])

    def __enter__(self):
        def recording_descent(*arguments, **keyword_arguments):
            batch = [int(player) for player in arguments[7]]         # result_index: the player in each row
            if sorted(batch) == sorted(self.candidates):
                self.batch_order = batch
            return self.original_descent(*arguments, **keyword_arguments)

        def recording_objective(*arguments, **keyword_arguments):
            result = self.original_objective(*arguments, **keyword_arguments)
            scores = np.asarray(result['Score'], dtype=float)
            if self.batch_order is not None and len(scores) == len(self.candidates):
                if keyword_arguments.get('correction_mode') == 'skip':
                    self.scan_scores.append(self.reorder(scores))
                else:
                    self.iteration_scores.append(self.reorder(scores))
            return result
        self.agent.perform_iterations = recording_descent
        self.agent.get_objective_and_gradient = recording_objective
        return self

    def __exit__(self, *exception):
        # Drop the instance shadows, restoring the class methods.
        del self.agent.get_objective_and_gradient, self.agent.perform_iterations


def solve_app_step(agent
                   , board
                   , my_team
                   , candidates
                   , n_iterations) -> tuple:
    """Step 1: the app's own solve of these candidates (its start policy and budget). Returns (scores, scan scores
    (starts x candidates) or None for Rotisserie, which scans nothing). Also refreshes the board's opponent store."""
    with DescentRecorder(agent, candidates) as recorder:
        result = agent.get_h_scores(player_assignments=board, drafter=my_team, n_iterations=n_iterations,
                                    candidate_subset=candidates)
    scores = result['Scores'].loc[candidates].to_numpy(dtype=float)
    scan = np.vstack(recorder.scan_scores) if recorder.scan_scores else None
    return scores, scan


def descend_from_start(agent
                       , board
                       , my_team
                       , candidates
                       , weights
                       , n_iterations
                       , long_iterations) -> tuple:
    """Every candidate descended from one start for long_iterations. Returns (score at the app's budget, best score over
    the whole run), per candidate."""
    agent.initial_category_weights = np.asarray(weights, dtype=float) / np.sum(weights)
    agent.initial_shares = {code: {base: 1 / len(info['bases']) for base in info['bases']}
                            for code, info in agent.position_structure['flex'].items()}
    try:
        with DescentRecorder(agent, candidates) as recorder:
            agent.get_h_scores(player_assignments=board, drafter=my_team, n_iterations=long_iterations,
                               candidate_subset=candidates)
    finally:
        agent.clear_initial_weights()
    path = np.vstack(recorder.iteration_scores)
    if len(path) != long_iterations:
        raise RuntimeError(f'Expected {long_iterations} recorded iterations, got {len(path)}.')
    return path[n_iterations - 1], path.max(axis=0)


def descend_from_perturbed_neutral(agent
                                   , board
                                   , my_team
                                   , candidates
                                   , season
                                   , n_rounds
                                   , n_iterations) -> np.ndarray:
    """Step 0: every candidate descended for the app's budget from the neutral build, each weight times (1 +
    NEUTRAL_PERTURBATION x a standard normal draw), the draw seeded by the board. Returns the score at the last
    iteration, per candidate."""
    neutral = agent.get_starting_category_weights()
    rng = np.random.default_rng([int(season[:4]), n_rounds])
    perturbed = neutral * (1 + NEUTRAL_PERTURBATION * rng.standard_normal(neutral.shape))
    at_budget, _ = descend_from_start(agent, board, my_team, candidates, perturbed, n_iterations, n_iterations)
    return at_budget


def measure_board(objective: str
                  , season: str
                  , n_rounds: int
                  , n_candidates: int
                  , long_iterations: int) -> list[dict]:
    """Every start's descent for the top n_candidates of one board. Each case: the candidate, the app's score (step
    1), and per start of the full menu its pre-descent score (app menu, head to head only), its score at the app's
    budget and its best over the long run. Steps 2-4 are derived from those."""
    session, board, my_team = build_format_board(objective, season, n_rounds)
    agent = session.agent
    n_iterations = session.current_settings['n_iterations']
    ranking = agent.get_h_scores(player_assignments=board, drafter=my_team, n_iterations=n_iterations)['Scores']
    taken = {player for roster in board.values() for player in roster}
    candidates = [int(player) for player in ranking.sort_values(ascending=False).index if player not in taken]
    candidates = candidates[:n_candidates]
    app_scores, scan = solve_app_step(agent, board, my_team, candidates, n_iterations)
    neutral_scores = (descend_from_perturbed_neutral(agent, board, my_team, candidates, season, n_rounds, n_iterations)
                      if agent.seed_mode == 'multistart' else None)
    starts = list_full_menu(agent)
    if scan is not None and len(scan) != sum(start['app_menu'] for start in starts):
        raise RuntimeError(f'The seed scan scored {len(scan)} seeds; the app menu has '
                           f'{sum(start["app_menu"] for start in starts)}.')
    budget_scores, long_scores = [], []
    for start in starts:
        budget, best = descend_from_start(agent, board, my_team, candidates, start['weights'], n_iterations,
                                          long_iterations)
        budget_scores.append(budget)
        long_scores.append(best)
    agent.reset_draft_state()
    cases = []
    for column, player in enumerate(candidates):
        seeds = []
        for index, start in enumerate(starts):
            pre_descent = float(scan[index, column]) if scan is not None and start['app_menu'] else None
            seeds.append({'kind': start['kind'], 'categories': start['categories'], 'app_menu': start['app_menu'],
                          'pre_descent': pre_descent, 'app_budget': float(budget_scores[index][column]),
                          'long_budget': float(long_scores[index][column])})
        app_menu = [seed for seed in seeds if seed['app_menu']]
        steps = [float(app_scores[column]),
                 max(seed['app_budget'] for seed in app_menu),
                 max(seed['app_budget'] for seed in seeds),
                 max(seed['long_budget'] for seed in seeds)]
        cases.append({'objective': objective, 'season': season, 'rounds': n_rounds,
                      'player': session.player_registry[player].name, 'n_iterations': n_iterations,
                      'neutral_start': None if neutral_scores is None else float(neutral_scores[column]),
                      'long_iterations': long_iterations, 'menu_size': len(app_menu),
                      'full_menu_size': len(seeds), 'steps': steps, 'seeds': seeds})
    return cases


def _measure_board_reporting_failure(*job) -> tuple:
    """measure_board, returning ('measured', cases) or ('failed', job, traceback) so one broken board neither hides
    which board it was nor discards the others' work."""
    import traceback
    try:
        return 'measured', measure_board(*job)
    except Exception:
        return 'failed', job, traceback.format_exc()


def measure_boards_in_parallel(objective: str
                               , seasons: list[str]
                               , rounds: list[int]
                               , n_candidates: int
                               , long_iterations: int
                               , n_workers: int) -> tuple[list[dict], list[dict]]:
    """measure_board for every (season, rounds) pair, one board per job across n_workers processes. Returns the cases
    (in job order) and the failed boards, each {season, rounds, traceback}."""
    import multiprocessing
    jobs = [(objective, season, n_rounds, n_candidates, long_iterations) for season in seasons for n_rounds in rounds]
    with multiprocessing.get_context('spawn').Pool(min(n_workers, len(jobs))) as pool:
        outcomes = pool.starmap(_measure_board_reporting_failure, jobs, chunksize=1)
    cases = [case for outcome in outcomes if outcome[0] == 'measured' for case in outcome[1]]
    failures = [{'season': outcome[1][1], 'rounds': outcome[1][2], 'traceback': outcome[2]}
                for outcome in outcomes if outcome[0] == 'failed']
    return cases, failures
