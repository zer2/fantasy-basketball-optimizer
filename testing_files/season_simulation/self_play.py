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

SCORING_FORMATS = {'EC': 'Head to Head: Each Category',
                   'MC': 'Head to Head: Most Categories',
                   'Roto': 'Rotisserie'}

# tuned parameter vector is (gamma, omega, kappa); clipped to these [min, max] (from parameters.yaml).
PARAMS  = ['gamma', 'omega', 'kappa']
BOUNDS  = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 5.0]])
DEFAULT = np.array([0.25, 0.7, 0.3])


def _clip(theta: np.ndarray) -> np.ndarray:
    return np.clip(theta, BOUNDS[:, 0], BOUNDS[:, 1])


def build_session(season: str, scoring_format: str):
    """One backend session (=> one reusable H-agent) for a season/format. Built once and cached; theta
    is applied later via `configure`. The expensive processed data (G/X-scores, coefficients, position
    means) is theta-independent and shared via the backend's v0 cache. beth=0: real historical stats,
    so there is nothing for the strength adjustment to doubt."""
    request = _build_session_request(scoring_format=scoring_format)
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
    agent.clear_initial_weights()
    agent.populate_default_h_scores(session.current_params['n_iterations'])


def draft_and_score(field_session, deviator_session, seat: int, candidate_limit: int) -> float:
    """Run one snake draft where `seat` drafts with the deviator agent and every other seat with the
    field agent, then return the deviator's final-roster H-score against the actual field. Full roster =>
    the score is parameter-independent (no future-pick weights), so it is a clean fitness signal.
    candidate_limit prunes each pick to the top-N by the cached generic ranking (same gate the
    autodrafters use) -- the chosen player is essentially always in that slice."""
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


def spsa(scoring_format: str
         , seasons: list[str]
         , theta0: np.ndarray
         , n_steps: int
         , a: float
         , c: float
         , alpha_decay: float
         , gamma_decay: float
         , polyak_beta: float
         , candidate_limit: int
         , rng: np.random.Generator) -> np.ndarray:
    """SPSA on the deviator's advantage. Returns the Polyak-averaged parameter vector."""
    theta     = _clip(theta0.copy())
    theta_avg = theta.copy()
    A         = max(1, n_steps // 10)   # SPSA stability offset

    # Two reusable sessions per season (field + deviator, which hold different theta at once during a
    # draft). Built on first use of a season and reconfigured each step -- no per-step rebuilds.
    sessions: dict[str, tuple] = {}

    def sessions_for(season):
        if season not in sessions:
            sessions[season] = (build_session(season, scoring_format), build_session(season, scoring_format))
        return sessions[season]

    for step in range(1, n_steps + 1):
        season = seasons[rng.integers(len(seasons))]
        alpha  = a / (step + A) ** alpha_decay
        delta  = c / step ** gamma_decay
        Delta  = rng.choice([-1.0, 1.0], size=len(PARAMS))

        field_session, deviator_session = sessions_for(season)
        configure(field_session, theta)
        seat = int(rng.integers(field_session.current_params['n_drafters']))

        configure(deviator_session, _clip(theta + delta * Delta))
        f_plus = draft_and_score(field_session, deviator_session, seat, candidate_limit)
        configure(deviator_session, _clip(theta - delta * Delta))
        f_minus = draft_and_score(field_session, deviator_session, seat, candidate_limit)

        ghat  = (f_plus - f_minus) / (2.0 * delta) * Delta   # central-difference SPSA gradient
        theta = _clip(theta + alpha * ghat)
        theta_avg = (1.0 - polyak_beta) * theta_avg + polyak_beta * theta

        print(f'step {step:3d} | season {season} seat {seat:2d} | f+={f_plus:.4f} f-={f_minus:.4f} '
              f'| theta=({theta[0]:.3f},{theta[1]:.3f},{theta[2]:.3f}) '
              f'avg=({theta_avg[0]:.3f},{theta_avg[1]:.3f},{theta_avg[2]:.3f})', flush=True)

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
    args = parser.parse_args()

    theta0 = DEFAULT.copy()
    for i, override in enumerate([args.gamma0, args.omega0, args.kappa0]):
        if override is not None:
            theta0[i] = override

    seasons = args.seasons if args.seasons else get_available_seasons()
    rng = np.random.default_rng(args.seed)

    print(f'self-play SPSA | format={args.format} | {len(seasons)} seasons | {args.steps} steps '
          f'| theta0=({theta0[0]:.3f},{theta0[1]:.3f},{theta0[2]:.3f})', flush=True)
    theta_star = spsa(SCORING_FORMATS[args.format], seasons, theta0, args.steps,
                      args.a, args.c, args.alpha_decay, args.gamma_decay, args.polyak_beta,
                      args.candidate_limit, rng)
    print(f'\nPolyak-averaged theta* = gamma={theta_star[0]:.4f} omega={theta_star[1]:.4f} '
          f'kappa={theta_star[2]:.4f}', flush=True)
    print('Validate on the full season set before trusting (run a full-batch fitness at theta*).')


if __name__ == '__main__':
    main()
