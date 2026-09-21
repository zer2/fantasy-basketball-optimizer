"""Record a real self-play bootstrap: what the field looks like after every pass.

The docs describe this loop in `hscores.md` under "No model of other managers":

    Before running H-scoring on all candidates, the algorithm is run once to establish a set of
    top players. Then the process repeats using the identified top players and their
    corresponding builds as a context in which H-scoring runs. In order to dampen oscillations
    from one extreme to another, only half of the players are adjusted in each step, and they run
    against averaged versions of their opponents across previous iterations. This process slowly
    drifts strategy profiles towards a stationary point.

So the thing that converges is a set of PLAYER BUILDS, and the scene animates them settling. What
it draws per player is not the build's weights but what they buy: the shipped pass result's
'Rates', each build's per-category win probability against the modelled field, which is the same
frame the served-field punt gate reads. A category at fifty percent is a coin flip; the strategy
is visible in which ones a build lets go of.

This script runs the shipped bootstrap with `agent.self_play_trace` attached for the pass
boundaries, and with a recorder around `_run_bootstrap_pass` for the two things the trace does not
carry: which half of the universe each pass re-solves, and the win rates that solve produced. A
player the pass rests keeps the rates from the last pass that solved him, which is exactly what
the loop itself does with his incumbent build.

Nothing here models anything. An earlier version of this script simulated a twelve-seat field and
was thrown away: it converged in a single pass, which said more about the toy than about the
algorithm.

    python visualizations/prepare_self_play_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))   # the repository
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))          # visualizations/

from backend.api.routers.sessions import _build_current_settings            # noqa: E402
from backend.services.session_management import build_session               # noqa: E402
from backend.math.algorithm_agents import HAgent                            # noqa: E402
from shared.prepare_season_data import build_default_session_request, SEASON, SPORT    # noqa: E402

_VISUALIZATIONS_DIR = Path(__file__).resolve().parent.parent

# How many of the field's players the scene draws. The bootstrap tracks the whole top-player
# universe; a few more rows than fit on screen would be recorded and never shown.
TRACKED_PLAYERS = 14


def record_bootstrap(session) -> tuple[list[dict], list[dict]]:
    """Re-run the shipped self-play bootstrap with both recorders attached.

    Returns the loop's own per-pass trace and, separately, one entry per call to
    `_run_bootstrap_pass` -- the subset of players that call solved and the win rates it gave
    them. The calls are what the passes DO; the trace is where the loop's pass boundaries are.
    """
    agent = session.agent
    agent.self_play_trace = []
    solves: list[dict] = []
    original_pass = HAgent._run_bootstrap_pass

    def watched_pass(self, empty, n_iterations, cash_remaining_per_team, candidate_subset=None,
                     preserve_frozen_weights=False):
        result = original_pass(self, empty, n_iterations, cash_remaining_per_team,
                               candidate_subset, preserve_frozen_weights)
        solves.append({'solved': None if candidate_subset is None else list(candidate_subset),
                       'rates':  result['Rates'],
                       'scores': result['Scores']})
        return result

    HAgent._run_bootstrap_pass = watched_pass
    try:
        # The build already ran this once; running it again with the recorders on is the only way
        # to see inside it, and it resets its own state at the top, so the second run is not a
        # continuation of the first.
        agent.populate_default_h_scores(session.current_settings['n_iterations'])
    finally:
        HAgent._run_bootstrap_pass = original_pass

    trace = agent.self_play_trace
    agent.self_play_trace = None
    if not trace:
        raise RuntimeError(
            'The bootstrap recorded no passes. Opponent modelling is probably off for this '
            'session (a single neutral pass), so there is no self-play to animate.')
    return trace, solves


def build_passes(trace: list[dict], solves: list[dict], ranking) -> tuple[list[dict], list]:
    """Per-pass win rates for the players the scene follows, and who each pass re-solved.

    The rows are the universe's best players by settled H-score, in that order. They cannot be
    taken off the field frame directly: the loop builds it with `groupby(level=0)`, which returns
    rows sorted by PLAYER ID, so the first fourteen of it are the fourteen lowest ids and mean
    nothing. `ranking` is the agent's own sorted H-scores, which is what "top player" means.

    The tracked set is fixed once and held, so a row means the same player all the way through --
    rows that changed identity mid-animation would make the settling unreadable.

    Each trace entry is appended just before that pass runs its solve, so the solve that belongs
    to trace entry i is the call after the one Level 0 made: `solves[i + 1]`.
    """
    universe = set(trace[0]['field'].index)
    tracked = [player for player in ranking if player in universe][:TRACKED_PLAYERS]
    if len(tracked) < TRACKED_PLAYERS:
        raise RuntimeError(
            f'Only {len(tracked)} of the ranked players are in the self-play universe, but the '
            f'scene draws {TRACKED_PLAYERS} rows.')
    categories = list(trace[0]['field'].columns)
    if len(solves) < len(trace) + 1:
        raise RuntimeError(
            f'{len(trace)} passes were traced but only {len(solves)} solves were recorded, so a '
            f'pass cannot be paired with the solve it ran.')

    # Level 0 solved everybody against a neutral field, so the board is already full before the
    # first pass runs. That is the state the scene should open on -- an empty grid would be
    # showing a moment that never existed.
    opening = solves[0]
    opening_rates = [opening['rates'].loc[player].to_numpy(dtype=float).tolist()
                     for player in tracked]
    carried = np.array(opening_rates, dtype=float)
    merged_scores = opening['scores'].copy()
    # Where the rows stand before any self-play has happened, by Level-0 H alone.
    opening_order = sorted(range(len(tracked)),
                           key=lambda row: -float(merged_scores.get(tracked[row], -np.inf)))

    passes = []
    for index, entry in enumerate(trace):
        solve = solves[index + 1]
        if solve['solved'] is None:
            raise RuntimeError(
                f'Pass {entry["pass_index"]} was paired with a whole-pool solve, which is the '
                f'serve rather than a pass. The pairing rule is wrong.')
        rates = solve['rates']
        solving = [row for row, player in enumerate(tracked) if player in solve['solved']]
        carried = carried.copy()
        for row in solving:
            carried[row] = rates.loc[tracked[row]].to_numpy(dtype=float)

        # The loop keeps a MERGED score frame -- every player at his most recently solved
        # state -- and re-ranks off it every pass. That re-ranking is the thing the scene was
        # not showing: the field's idea of who the top players are moves while it settles.
        merged_scores = merged_scores.copy()
        merged_scores.loc[solve['scores'].index] = solve['scores']
        order = sorted(range(len(tracked)),
                       key=lambda row: -float(merged_scores.get(tracked[row], -np.inf)))

        passes.append({
            'pass_index': entry['pass_index'],
            'drift':      entry['drift'],
            'solving':    solving,
            'order':      order,
            'rates':      np.round(carried, 4).tolist(),
        })
    return passes, tracked, opening_rates, opening_order


def main() -> None:
    print(f'Building a {SEASON} session at the app defaults...')
    session = build_session(
        current_settings = _build_current_settings(
            build_default_session_request(SEASON, SPORT)),
        platform_config  = None,
        csv_bytes        = None,
        uploaded_dfs     = None,
    )

    print('Re-running the self-play bootstrap with the recorders attached...')
    trace, solves = record_bootstrap(session)
    # The bootstrap leaves its own ranking behind; the scene's rows are the top of it.
    ranking = list(session.agent.default_h_scores.index)
    passes, tracked, opening_rates, opening_order = build_passes(
        trace, solves, ranking)
    categories = list(trace[0]['field'].columns)

    drifts = [entry['drift'] for entry in passes if entry['drift'] is not None]
    solved_counts = [len(entry['solving']) for entry in passes]
    print(f'\n{len(passes)} passes recorded, {len(tracked)} players tracked')
    print(f'each pass re-solved {min(solved_counts)}-{max(solved_counts)} of them')
    print(f'drift: first {drifts[0]:.4f}, last {drifts[-1]:.4f}, '
          f'{drifts[0] / max(drifts[-1], 1e-9):.0f}x smaller by the end')

    settled = np.array(passes[-1]['rates'], dtype=float)
    print(f'final win rates {np.nanmin(settled):.2f} .. {np.nanmax(settled):.2f}')

    # How much a reordering animation would actually move. Measured rather than guessed: rows
    # that swap every pass would be unreadable, and this says whether that is the case.
    moves = []
    previous = opening_order
    for entry in passes:
        positions = {row: place for place, row in enumerate(entry['order'])}
        previous_positions = {row: place for place, row in enumerate(previous)}
        moves.append(sum(1 for row in range(len(tracked))
                         if positions[row] != previous_positions[row]))
        previous = entry['order']
    print(f'\nreordering churn: {sum(moves)} row moves over {len(passes)} passes, '
          f'{np.mean(moves):.1f} per pass, worst pass {max(moves)} of {len(tracked)}')
    print(f'passes with no movement at all: {sum(1 for m in moves if m == 0)}')
    first_order = passes[0]['order']
    last_order = passes[-1]['order']
    print(f'row 1 at the start: {session.player_registry[tracked[first_order[0]]].name}')
    print(f'row 1 at the end:   {session.player_registry[tracked[last_order[0]]].name}')

    data_path = _VISUALIZATIONS_DIR / 'data' / 'self_play.json'
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps({
        'season':         SEASON,
        # How many seats the drafting context is: the field the passes optimise against is the
        # league's own size, so the scene can mark where that cut falls rather than assume it.
        'context_size':   session.current_settings['n_drafters'],
        'categories':     categories,
        'player_names':   [session.player_registry[player].name
                           if player in session.player_registry else str(player)
                           for player in tracked],
        'opening_rates':  np.round(np.array(opening_rates, dtype=float), 4).tolist(),
        'opening_order':  opening_order,
        'passes':         passes,
    }), encoding='utf-8')
    print(f'Wrote {data_path.relative_to(_VISUALIZATIONS_DIR.parent)}')


if __name__ == '__main__':
    main()
