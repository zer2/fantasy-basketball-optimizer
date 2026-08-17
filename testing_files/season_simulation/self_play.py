# testing_files/season_simulation/self_play.py
# Self-play SPSA learning of the punting parameters (gamma, omega, kappa) toward the symmetric draft
# equilibrium -- the parameter setting that is a best response to itself (no seat can gain by deviating).
#
# Each SPSA step (see the kappa memory for the full design):
#   1. sample ONE season and ONE deviator seat (the SGD mini-batch),
#   2. draw a random +/-1 perturbation direction Delta over (gamma, omega, kappa),
#   3. run two full-league drafts on that same season/seat -- field at theta, deviator at theta +/- delta*Delta
#      (common random numbers, so the season's idiosyncrasy cancels in the difference),
#   4. score each by the deviator's final-roster H-score vs the actual field (parameter-clean),
#   5. central-difference gradient  ghat = (f_plus - f_minus)/(2*delta) * Delta,
#   6. step the field  theta <- clip(theta + alpha*ghat), and Polyak-average.
# alpha and delta decay on the standard SPSA schedules. The final estimate is the Polyak average; validate
# it once on the full season set before trusting it.
#
# The whole field are H-agents (so it punts and crowds -- the prerequisite the G-drafter sim can't meet).
#
# Smoke test (one step, one season):
#   python testing_files/season_simulation/self_play.py --format EC --steps 1 --seasons 2024-25

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

os.environ.setdefault('SESSION_SECRET_KEY', 'self-play-only')
logging.disable(logging.CRITICAL)

import sys
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))   # repo root -> `backend`
sys.path.insert(0, str(_HERE.parent))           # testing_files -> `benchmark_helpers`

import numpy as np

from benchmark_helpers import client, _build_session_request
from backend.state.session import get_session
from backend.services.ranking import rank_candidates
from backend.data_retrieval import get_available_seasons

SCORING_FORMATS = {'EC': 'Each Category',
                   'MC': 'Most Categories',
                   'Roto': 'Rotisserie'}

# tuned parameter vector is (gamma, omega, kappa); clipped to these [min, max] (from parameters.yaml).
PARAMS  = ['gamma', 'omega', 'kappa']
BOUNDS  = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 5.0]])
DEFAULT = np.array([0.25, 0.7, 0.3])


def _clip(theta: np.ndarray) -> np.ndarray:
    return np.clip(theta, BOUNDS[:, 0], BOUNDS[:, 1])


def build_session(season: str, objective: str):
    """One backend session (=> one reusable H-agent) for a season/format. Built once and cached; theta
    is applied later via `configure`. The expensive processed data (G/X-scores, coefficients, position
    means) is theta-independent and shared via the backend's v0 cache. beth=0: real historical stats,
    so there is nothing for the strength adjustment to doubt."""
    request = _build_session_request(objective=objective)
    request['data_source']['season'] = season
    request['parameters']['beth']    = 0
    response = client.post('/sessions', json=request)
    assert response.status_code == 201, f'session build failed: {response.text}'
    return get_session(response.json()['session_id'])


def configure(session, theta: np.ndarray) -> None:
    """Point an existing agent at parameter vector theta. gamma/omega/kappa are read at scoring time, so
    we just reassign them; then refresh the theta-dependent empty-board solve (default ranking + kappa
    popularity). No rebuild, no data reprocessing."""
    agent = session.agent
    agent.gamma, agent.omega, agent.kappa = float(theta[0]), float(theta[1]), float(theta[2])
    agent.populate_default_h_scores(session.current_params['n_iterations'])


def draft_and_score(field_session, deviator_session, seat: int, candidate_limit: int) -> float:
    """Run one snake draft where `seat` drafts with the deviator agent and every other seat with the
    field agent, then return the deviator's final-roster H-score against the actual field. Full roster =>
    the score is parameter-independent (no future-pick weights), so it is a clean fitness signal.
    candidate_limit prunes each pick to the top-N by the cached generic ranking (same gate the
    autodrafters use) -- the chosen player is essentially always in that slice."""
    # Fresh draft: clear both agents' in-draft state so nothing leaks from a previous draft.
    field_session.agent.reset_draft_state()
    deviator_session.agent.reset_draft_state()
    n_drafters   = field_session.current_params['n_drafters']
    n_picks      = field_session.current_params['n_picks']
    n_iterations = field_session.current_params['n_iterations']
    team_names   = [f'Drafter {i + 1}' for i in range(n_drafters)]
    assignments  = {name: [] for name in team_names}
    deviator     = team_names[seat]

    for pick_row in range(n_picks):
        for slot in range(n_drafters):
            drafter_index = slot if pick_row % 2 == 0 else (n_drafters - 1 - slot)   # serpentine
            drafter_name  = team_names[drafter_index]
            session       = deviator_session if drafter_index == seat else field_session
            result        = rank_candidates(session, assignments, drafter_name, [], None, 0, candidate_limit)
            assert result.candidates, f'no candidates for {drafter_name}, round {pick_row + 1}'
            assignments[drafter_name].append(result.candidates[0].name)

    scores = deviator_session.agent.get_h_scores(assignments, deviator, n_iterations)['Scores']
    return float(scores[scores.idxmax()])


def draft_population(session_by_seat: dict
                     , n_drafters: int
                     , n_picks: int
                     , candidate_limit: int) -> dict:
    """Run one snake draft where every seat drafts with its OWN configured session (its own theta),
    rather than a single field agent plus one deviator. session_by_seat maps a seat index to the session
    that seat drafts with. Returns the completed assignments."""
    team_names  = [f'Drafter {i + 1}' for i in range(n_drafters)]
    assignments = {name: [] for name in team_names}
    for session in {id(s): s for s in session_by_seat.values()}.values():   # Session is unhashable
        session.agent.reset_draft_state()   # fresh draft: no state leaks from a previous draft

    for pick_row in range(n_picks):
        for slot in range(n_drafters):
            drafter_index = slot if pick_row % 2 == 0 else (n_drafters - 1 - slot)   # serpentine
            drafter_name  = team_names[drafter_index]
            session       = session_by_seat[drafter_index]
            result        = rank_candidates(session, assignments, drafter_name, [], None, 0, candidate_limit)
            assert result.candidates, f'no candidates for {drafter_name}, round {pick_row + 1}'
            assignments[drafter_name].append(result.candidates[0].name)
    return assignments


def score_all_teams(scorer_session
                    , assignments: dict
                    , n_iterations: int) -> dict:
    """Final-roster H-score for every team, all scored by one agent so they are directly comparable. A
    completed roster's H-score is parameter-independent (no future-pick weights), so which agent scores it
    does not matter -- the same call draft_and_score uses, looped over the twelve teams."""
    agent = scorer_session.agent
    return {team: float(scores[scores.idxmax()])
            for team in assignments
            for scores in [agent.get_h_scores(assignments, team, n_iterations)['Scores']]}


def estimate_gradient_population(pool: list
                                 , theta: np.ndarray
                                 , delta: float
                                 , candidate_limit: int
                                 , rng: np.random.Generator) -> np.ndarray:
    """Antithetic multi-deviator simultaneous perturbation. The seats of a single draft are split into
    len(PARAMS) equal groups; within each group half draft at theta+delta*e_i and half at theta-delta*e_i.
    TWO drafts are run with the +/- roles flipped between them, so every seat serves once on each side and
    seat-position bias cancels in the +delta-vs--delta contrast. The gradient is read off the collective
    final-roster H-scores of the +delta seats vs the -delta seats. Cheaper (2 drafts vs coord's 6) and
    higher-signal (4-vs-4 samples per parameter vs 1-vs-1), at the cost of a field that is itself perturbed
    rather than exactly theta -- a population / evolution-strategies estimate of the same gradient.

    pool holds 2*len(PARAMS) reusable sessions: pool[2*i] is configured to theta+delta*e_i and
    pool[2*i+1] to theta-delta*e_i."""
    n_params     = len(PARAMS)
    n_drafters   = pool[0].current_params['n_drafters']
    n_picks      = pool[0].current_params['n_picks']
    n_iterations = pool[0].current_params['n_iterations']
    assert n_drafters % (2 * n_params) == 0, (
        f'population gradient needs n_drafters ({n_drafters}) divisible by 2*len(PARAMS) ({2 * n_params})')
    seats_per_param = n_drafters // n_params
    half            = seats_per_param // 2
    E               = np.eye(n_params)

    for i in range(n_params):
        configure(pool[2 * i],     _clip(theta + delta * E[i]))
        configure(pool[2 * i + 1], _clip(theta - delta * E[i]))

    team_names   = [f'Drafter {j + 1}' for j in range(n_drafters)]
    groups       = rng.permutation(n_drafters).reshape(n_params, seats_per_param)
    plus_scores  = [[] for _ in range(n_params)]
    minus_scores = [[] for _ in range(n_params)]

    for test in range(2):   # test 1, then its exact +/- flip -> seat-position bias cancels
        session_by_seat = {}
        seat_side       = {}
        for i in range(n_params):
            group       = groups[i]
            plus_seats  = group[:half] if test == 0 else group[half:]
            minus_seats = group[half:] if test == 0 else group[:half]
            for seat in plus_seats:
                session_by_seat[int(seat)] = pool[2 * i];     seat_side[int(seat)] = (i, +1)
            for seat in minus_seats:
                session_by_seat[int(seat)] = pool[2 * i + 1]; seat_side[int(seat)] = (i, -1)

        assignments = draft_population(session_by_seat, n_drafters, n_picks, candidate_limit)
        team_scores = score_all_teams(pool[0], assignments, n_iterations)
        for seat, (i, sign) in seat_side.items():
            (plus_scores if sign > 0 else minus_scores)[i].append(team_scores[team_names[seat]])

    return np.array([(np.mean(plus_scores[i]) - np.mean(minus_scores[i])) / (2.0 * delta)
                     for i in range(n_params)])


def save_checkpoint(path, objective, theta, theta_avg, velocity, step, A, hyper) -> None:
    """Persist enough to resume the same SPSA trajectory: the current point, its Polyak average, the
    momentum velocity, the global step counter (so the alpha/delta decay continues rather than
    restarting), the stability offset A, and the schedule hyper-parameters."""
    json.dump({'format': objective, 'theta': [float(x) for x in theta],
               'theta_avg': [float(x) for x in theta_avg], 'velocity': [float(x) for x in velocity],
               'step': int(step), 'A': int(A), **hyper}, open(path, 'w'), indent=1)


def spsa(objective: str
         , seasons: list[str]
         , theta0: np.ndarray
         , n_steps: int
         , a: float
         , c: float
         , alpha_decay: float
         , gamma_decay: float
         , polyak_beta: float
         , candidate_limit: int
         , rng: np.random.Generator
         , start_step: int = 0
         , A: int | None = None
         , theta_avg0: np.ndarray | None = None
         , momentum: float = 0.0
         , velocity0: np.ndarray | None = None
         , gradient_mode: str = 'spsa'
         , checkpoint_path: str | None = None) -> np.ndarray:
    """SPSA on the deviator's advantage. Returns the Polyak-averaged parameter vector. Resumes by
    continuing the global step counter from `start_step` (so the decay schedule is unbroken).
    momentum>0 accumulates a heavy-ball velocity, so a consistent gradient direction builds up while
    noisy ones cancel -- note the effective step scales ~1/(1-momentum), so pair a high momentum with a
    smaller `a`."""
    theta     = _clip(theta0.copy())
    theta_avg = theta.copy() if theta_avg0 is None else theta_avg0.copy()
    velocity  = np.zeros(len(PARAMS)) if velocity0 is None else velocity0.copy()
    E         = np.eye(len(PARAMS))
    if A is None:
        A = max(1, n_steps // 10)   # SPSA stability offset
    hyper = dict(a=a, c=c, alpha_decay=alpha_decay, gamma_decay=gamma_decay,
                 polyak_beta=polyak_beta, momentum=momentum, gradient_mode=gradient_mode)

    # Two reusable sessions per season (field + deviator, which hold different theta at once during a
    # draft). Built on first use of a season and reconfigured each step -- no per-step rebuilds.
    sessions: dict[str, tuple] = {}
    pools: dict[str, list]     = {}

    def sessions_for(season):
        if season not in sessions:
            sessions[season] = (build_session(season, objective), build_session(season, objective))
        return sessions[season]

    def pool_for(season):   # 2*len(PARAMS) sessions for the multi-deviator population gradient
        if season not in pools:
            pools[season] = [build_session(season, objective) for _ in range(2 * len(PARAMS))]
        return pools[season]

    for step in range(start_step + 1, start_step + n_steps + 1):
        season = seasons[rng.integers(len(seasons))]
        alpha  = a / (step + A) ** alpha_decay
        delta  = c / step ** gamma_decay

        if gradient_mode == 'population':
            # Pack all six perturbations into one draft's seats; read the gradient off the +/- cross-section.
            ghat        = estimate_gradient_population(pool_for(season), theta, delta, candidate_limit, rng)
            batch_label = 'pop 2x'
        else:
            field_session, deviator_session = sessions_for(season)
            configure(field_session, theta)
            seat = int(rng.integers(field_session.current_params['n_drafters']))

            def probe(offset):   # deviator's final H-score at theta+offset, vs the field at theta
                configure(deviator_session, _clip(theta + offset))
                return draft_and_score(field_session, deviator_session, seat, candidate_limit)

            if gradient_mode == 'coord':
                # Coordinate finite-differences: perturb each parameter alone (6 evals), so every
                # component's step reflects ITS OWN gradient magnitude, not one shared scalar.
                ghat = np.array([(probe(delta * E[i]) - probe(-delta * E[i])) / (2.0 * delta)
                                 for i in range(len(PARAMS))])
            else:
                # SPSA: one shared central difference along a random +/-1 direction (2 evals, any dimension).
                Delta = rng.choice([-1.0, 1.0], size=len(PARAMS))
                ghat  = (probe(delta * Delta) - probe(-delta * Delta)) / (2.0 * delta) * Delta
            batch_label = f'seat {seat:2d}'

        velocity = momentum * velocity + ghat                   # heavy-ball (no-op when momentum=0)
        theta    = _clip(theta + alpha * velocity)
        theta_avg = (1.0 - polyak_beta) * theta_avg + polyak_beta * theta

        # log the gradient estimate directly (5 dp: the true per-step move is ~1e-4).
        print(f'step {step:4d} | {season} {batch_label} | ghat=({ghat[0]:+.4f},{ghat[1]:+.4f},{ghat[2]:+.4f}) '
              f'| theta=({theta[0]:.5f},{theta[1]:.5f},{theta[2]:.5f}) '
              f'avg=({theta_avg[0]:.5f},{theta_avg[1]:.5f},{theta_avg[2]:.5f})', flush=True)

        if checkpoint_path and step % 25 == 0:
            save_checkpoint(checkpoint_path, objective, theta, theta_avg, velocity, step, A, hyper)

    if checkpoint_path:
        save_checkpoint(checkpoint_path, objective, theta, theta_avg, velocity,
                        start_step + n_steps, A, hyper)
    return theta_avg


def main() -> None:
    parser = argparse.ArgumentParser(description='Self-play SPSA learning of (gamma, omega, kappa).')
    parser.add_argument('--format', choices=list(SCORING_FORMATS), default='EC')
    parser.add_argument('--seasons', nargs='*', default=None, help='Seasons to sample (default: all).')
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--gamma0', type=float, default=None)
    parser.add_argument('--omega0', type=float, default=None)
    parser.add_argument('--kappa0', type=float, default=None)
    parser.add_argument('--a', type=float, default=0.05, help='SPSA step-size scale.')
    parser.add_argument('--c', type=float, default=0.10, help='SPSA perturbation scale.')
    parser.add_argument('--alpha-decay', type=float, default=0.602)
    parser.add_argument('--gamma-decay', type=float, default=0.101)
    parser.add_argument('--polyak-beta', type=float, default=0.1)
    parser.add_argument('--candidate-limit', type=int, default=40,
                        help='Prune each pick to the top-N by the cached ranking (autodrafter gate).')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--momentum', type=float, default=None,
                        help='Heavy-ball momentum (default off; resume inherits the checkpoint). '
                             'Effective step ~1/(1-momentum), so pair a high value with a smaller --a.')
    parser.add_argument('--gradient', choices=['spsa', 'coord', 'population'], default='spsa',
                        help='spsa: 1 random +/-1 direction, 2 evals/step (uniform per-param magnitude). '
                             'coord: per-parameter finite differences, 6 evals/step (true per-param gradient). '
                             'population: antithetic multi-deviator, 2 drafts/step, all seats perturbed at '
                             'once (4-vs-4 samples/param; cheaper and higher-signal, slightly noisier field).')
    parser.add_argument('--checkpoint', default=None, help='Path to write/refresh a resume checkpoint.')
    parser.add_argument('--resume', default=None, help='Resume from a checkpoint; --steps adds more steps.')
    args = parser.parse_args()

    seasons = args.seasons if args.seasons else get_available_seasons()
    rng = np.random.default_rng(args.seed)

    if args.resume:
        # Continue an existing trajectory: point, Polyak average, velocity, step counter, offset and
        # schedule all come from the checkpoint, so the decay is unbroken. --steps is how many MORE steps.
        ck = json.load(open(args.resume))
        objective = ck['format']
        theta0, theta_avg0 = np.array(ck['theta']), np.array(ck['theta_avg'])
        velocity0 = np.array(ck['velocity']) if 'velocity' in ck else None
        start_step, A = ck['step'], ck['A']
        a, c, alpha_decay, gamma_decay, polyak_beta = (
            ck['a'], ck['c'], ck['alpha_decay'], ck['gamma_decay'], ck['polyak_beta'])
        momentum = args.momentum if args.momentum is not None else ck.get('momentum', 0.0)
        gradient_mode = ck.get('gradient_mode', 'spsa')
        print(f'resume {objective} from step {start_step} | +{args.steps} steps | {gradient_mode} '
              f'| theta=({theta0[0]:.4f},{theta0[1]:.4f},{theta0[2]:.4f}) | momentum={momentum}', flush=True)
    else:
        objective = SCORING_FORMATS[args.format]
        theta0 = DEFAULT.copy()
        for i, override in enumerate([args.gamma0, args.omega0, args.kappa0]):
            if override is not None:
                theta0[i] = override
        theta_avg0, velocity0, start_step, A = None, None, 0, None
        a, c, alpha_decay, gamma_decay, polyak_beta = (
            args.a, args.c, args.alpha_decay, args.gamma_decay, args.polyak_beta)
        momentum = args.momentum if args.momentum is not None else 0.0
        gradient_mode = args.gradient
        print(f'self-play | format={args.format} | {len(seasons)} seasons | {args.steps} steps | '
              f'{gradient_mode} | a={a} momentum={momentum} '
              f'| theta0=({theta0[0]:.4f},{theta0[1]:.4f},{theta0[2]:.4f})', flush=True)

    theta_star = spsa(objective, seasons, theta0, args.steps, a, c, alpha_decay, gamma_decay,
                      polyak_beta, args.candidate_limit, rng, start_step, A, theta_avg0,
                      momentum, velocity0, gradient_mode, args.checkpoint)
    print(f'\nPolyak-averaged theta* = gamma={theta_star[0]:.4f} omega={theta_star[1]:.4f} '
          f'kappa={theta_star[2]:.4f}', flush=True)
    print('Validate on the full season set before trusting (run a full-batch fitness at theta*).')


if __name__ == '__main__':
    main()
