"""Paired drafts: H-scoring against the example team's planner choosing picks.

The example team (backend/math/exact_team_solves) plans a whole roster: a naive exact team, then Frank-Wolfe over the seat
polytope, then swaps to a local optimum. Here it is used to CHOOSE picks, to measure whether planning concrete teams beats
H-scoring's smooth model of future picks.

One pair is two 12-team snake drafts on the same seats: six seats (chosen at random) draft with the planner and six by
H-scoring, then the methods swap seats. Every completed roster is scored with the complete-team H-score against the
actual field, and the pair's result is the planner seats' mean minus the H-scoring seats' mean, in percentage points,
averaged over its two drafts. The league is the backend tests' (2024-25, 12 x 13, Each Category), beth = 0 as in every
simulation.

A planner seat takes H-scoring's top TOP_CANDIDATES candidates, plans the rest of the team around each, and takes the
candidate whose planned team scores highest. A candidate the planner cannot plan for is skipped; with none plannable (the
last pick) it drafts like H-scoring.

Two variants:
  - APP_EXAMPLE_TEAM: the planner exactly as the app ships it for the example team -- a one-sided penalty on category win
    rates below H-scoring's, H-scoring's flex allocation enforced, my players held to H-scoring's slots, and the round
    rule for availability (one open pick per round, each a player predicted for that round or later).
  - UNCONSTRAINED_PLANNER: the strongest version of the planner. None of the three anchors (build_team_solve_problem with
    anchors_to_h_scoring off): the objective is the complete-team H-score alone and every player may take any slot he is
    eligible for. Availability follows the actual snake order, from the seat's real remaining pick numbers.
Both predict the other drafters to take players in base H-score order, which is how the H-scoring seats draft.
"""

from __future__ import annotations

import json
import multiprocessing
import sys
import time

import numpy as np

from .league_sessions import build_league_session

APP_EXAMPLE_TEAM = "app's example team"
UNCONSTRAINED_PLANNER = 'unconstrained planner'
VARIANTS = (APP_EXAMPLE_TEAM, UNCONSTRAINED_PLANNER)
TOP_CANDIDATES = 40
SEASON = '2024-25'


def run_paired_drafts_in_parallel(
    variant: str
    , n_pairs: int
    , n_workers: int
    , seed: int
) -> list[dict]:
    """n_pairs pairs of the variant, spread over n_workers processes (each builds its own session). The pairs' seat
    splits are drawn up front from one generator (draw_distinct_seat_splits), so no two pairs share a split: the drafts
    are deterministic given the split, and a repeated split would repeat its pair's result exactly."""
    if variant not in VARIANTS:
        raise ValueError(f'Unknown variant {variant!r}; expected one of {VARIANTS}.')
    n_drafters = _build_league_session().current_settings['n_drafters']
    splits = draw_distinct_seat_splits(n_drafters, n_pairs, seed)
    jobs = [(variant, splits[worker::n_workers]) for worker in range(n_workers) if splits[worker::n_workers]]
    with multiprocessing.get_context('spawn').Pool(len(jobs)) as pool:
        per_worker = pool.starmap(run_paired_drafts, jobs)
    return sorted((pair for pairs in per_worker for pair in pairs), key=lambda pair: splits.index(pair['split']))


def draw_distinct_seat_splits(
    n_drafters: int
    , n_pairs: int
    , seed: int
) -> list[tuple[int, ...]]:
    """n_pairs distinct splits of the seats into two halves, each given by its planner half in the first draft. A split
    and its complement are the same pair (the halves swap between a pair's two drafts), so they count as one."""
    rng = np.random.default_rng(seed)
    splits, seen = [], set()
    while len(splits) < n_pairs:
        half = tuple(sorted(int(seat) for seat in rng.permutation(n_drafters)[: n_drafters // 2]))
        complement = tuple(seat for seat in range(n_drafters) if seat not in half)
        if half in seen or complement in seen:
            continue
        seen.add(half)
        splits.append(half)
    return splits


def run_paired_drafts(
    variant: str
    , splits: list[tuple[int, ...]]
) -> list[dict]:
    """One pair of the variant per seat split, in this process. Each pair record: the split, and for each of the two
    drafts the planner seats, every seat's complete-team H-score and the planner and H-scoring seat means (percentage
    points); and the pair difference."""
    session = _build_league_session()
    n_drafters = session.current_settings['n_drafters']
    records = []
    for split in splits:
        halves = (set(split), set(range(n_drafters)) - set(split))
        drafts = []
        for planner_seats in halves:
            began = time.perf_counter()
            scores = _run_draft(session, variant, planner_seats)
            team_names = list(scores)
            planner_mean = 100 * float(np.mean([scores[team_names[seat]] for seat in planner_seats]))
            h_scoring_mean = 100 * float(np.mean([scores[team_names[seat]] for seat in range(n_drafters)
                                                  if seat not in planner_seats]))
            drafts.append({'planner_seats': sorted(int(seat) for seat in planner_seats),
                           'seat_scores': [100 * float(scores[team]) for team in team_names],
                           'planner_mean': planner_mean, 'h_scoring_mean': h_scoring_mean,
                           'seconds': time.perf_counter() - began})
        records.append({'variant': variant, 'split': split, 'drafts': drafts,
                        'difference': float(np.mean([draft['planner_mean'] - draft['h_scoring_mean']
                                                     for draft in drafts]))})
    return records


def _build_league_session():
    """The backend tests' league, Each Category, beth = 0."""
    return build_league_session('Each Category', SEASON, {'beth': 0})[2]


def _run_draft(session, variant: str, planner_seats: set[int]) -> dict[str, float]:
    """One snake draft; planner seats pick with the planner, the rest by H-scoring. Returns each team's complete-team
    H-score."""
    from backend.services.ranking import rank_candidates
    agent = session.agent
    n_drafters, n_picks = session.current_settings['n_drafters'], session.current_settings['n_picks']
    team_names = [f'Drafter {index + 1}' for index in range(n_drafters)]
    seat_order = [slot if row % 2 == 0 else n_drafters - 1 - slot for row in range(n_picks) for slot in range(n_drafters)]
    agent.reset_draft_state()
    assignments = {team: [] for team in team_names}
    for pick_index, seat in enumerate(seat_order):
        drafter = team_names[seat]
        ranked = [candidate for candidate in rank_candidates(session, assignments, drafter, [], None).candidates
                  if candidate.player_id in agent.x_scores.index]
        my_remaining_pick_numbers = [later_index + 1 for later_index in range(pick_index, len(seat_order))
                                     if seat_order[later_index] == seat]
        chosen = (_choose_planner_pick(session, variant, assignments, drafter, ranked, my_remaining_pick_numbers)
                  if seat in planner_seats else ranked[0].player_id)
        assignments[drafter].append(chosen)
    scores = {}
    for team in team_names:
        team_scores = agent.get_h_scores(assignments, team, session.current_settings['n_iterations'])['Scores']
        scores[team] = float(team_scores[team_scores.idxmax()])
    return scores


def _choose_planner_pick(session, variant, assignments, drafter, ranked, my_remaining_pick_numbers) -> int:
    from backend.services.ranking import make_candidate_strategy_key
    cached = session.candidate_strategy_cache.get(make_candidate_strategy_key(assignments, drafter, [], None, None))
    if cached is None:                       # the last pick: no future picks to plan
        return ranked[0].player_id
    strategies = cached[1]
    scored = []
    for candidate in ranked[:TOP_CANDIDATES]:
        strategy = strategies.get(candidate.player_id)
        if strategy is None or strategy.seating is None:
            continue
        score = plan_team_h_score(session, variant, assignments, drafter, candidate.player_id, strategy,
                                  my_remaining_pick_numbers)
        if score is not None:
            scored.append((score, candidate.player_id))
    return max(scored)[1] if scored else ranked[0].player_id


def plan_team_h_score(
    session
    , variant: str
    , assignments: dict[str, list[int]]
    , drafter: str
    , candidate_id: int
    , strategy
    , my_remaining_pick_numbers: list[int]
) -> float | None:
    """The complete-team H-score of the team the planner builds around the candidate, or None when it cannot build one.
    my_remaining_pick_numbers (the candidate's first) are used by the variant that follows the actual pick order."""
    from backend.math import exact_team_solves as solves
    from backend.services.hypothetical_team import compute_expected_flex_position_counts
    from backend.services.ranking import make_slot_names
    agent = session.agent
    availability_order = [int(player) for player in agent.default_h_scores.index]
    anchored = variant == APP_EXAMPLE_TEAM
    if anchored:
        slot_names = make_slot_names(agent.position_config)
        fixed_slot_types = [slot_names[index].rstrip('0123456789') for index in strategy.seating]
        expected_flex = compute_expected_flex_position_counts(
            strategy.flex_shares, fixed_slot_types, agent.position_config.position_numbers)
    else:
        fixed_slot_types, expected_flex = None, None
    try:
        problem = solves.build_team_solve_problem(
            agent, assignments, drafter, candidate_id, [], availability_order,
            anchors_to_h_scoring          = anchored,
            fixed_slot_types              = fixed_slot_types,
            expected_flex_position_counts = expected_flex,
            remaining_cash                = None,
            predicted_prices              = None,
            player_teams                  = None,
            my_remaining_pick_numbers     = None if anchored else my_remaining_pick_numbers)
        naive = solves.solve_naive_exact_team_for_draft(problem, strategy.category_weights)
        field = solves.capture_complete_team_field(agent, problem, assignments, drafter, [], naive, None)
        team = solves.solve_fw_exact_team_for_draft(agent, problem, field, strategy.win_rates, naive)
    except ValueError:
        return None
    selection = solves.convert_seatings_to_selections(problem, team[np.newaxis, :])
    totals = selection @ problem.x_scores
    units = None if problem.volume_units is None else selection @ problem.volume_units
    differential = field.diff_means + (totals - field.reference_totals)[:, :, np.newaxis]
    volume_context = None if units is None else {
        'my_x_total': totals[:, problem.volume_category_indices], 'my_units': units,
        'opponent_units': field.opponent_units}
    h_scores, _ = agent.compute_complete_team_scores_and_win_rates_batched(differential, field.diff_vars, volume_context)
    return float(h_scores[0])


def summarize_pairs(records: list[dict]) -> dict:
    """Mean pair difference with its standard error, the planner's wins, and the extremes."""
    differences = np.array([record['difference'] for record in records])
    return {
        'pairs': len(differences),
        'mean': float(differences.mean()),
        'standard_error': float(differences.std(ddof=1) / np.sqrt(len(differences))) if len(differences) > 1 else float('nan'),
        'planner_wins': int((differences > 0).sum()),
        'worst': float(differences.min()),
        'best': float(differences.max()),
    }


if __name__ == '__main__':
    # Standalone, from testing_files: python -m experiment_harnesses.frank_wolfe_drafts <variant> <pairs> <workers> <seed>
    chosen_variant, pairs, workers, chosen_seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    results = run_paired_drafts_in_parallel(chosen_variant, pairs, workers, chosen_seed)
    print(json.dumps(summarize_pairs(results), indent=1))
