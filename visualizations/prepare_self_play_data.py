"""Record a real self-play bootstrap: what the field looks like after every pass.

The docs describe this loop in `hscores.md` under "No model of other managers":

    Before running H-scoring on all candidates, the algorithm is run once to establish a set of
    top players. Then the process repeats using the identified top players and their
    corresponding builds as a context in which H-scoring runs. In order to dampen oscillations
    from one extreme to another, only half of the players are adjusted in each step, and they run
    against averaged versions of their opponents across previous iterations. This process slowly
    drifts strategy profiles towards a stationary point.

So the thing that converges is a set of PLAYER BUILDS -- one category-weight profile per top
player -- and the scene animates them settling. This script runs the shipped bootstrap with
`agent.self_play_trace` attached and writes what it saw: for every pass, each tracked player's
category weights, the win rate those weights buy against the field, and the drift of the running
average that the loop already logs.

Nothing here models anything. An earlier version of this script simulated a twelve-seat field
and was thrown away: it converged in a single pass, which said more about the toy than about the
algorithm.

    python visualizations/prepare_self_play_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.api.routers.sessions import _build_current_settings            # noqa: E402
from backend.services.session_management import build_session               # noqa: E402
from prepare_season_data import build_default_session_request, SEASON, SPORT    # noqa: E402

_VISUALIZATIONS_DIR = Path(__file__).resolve().parent

# How many of the field's players the scene draws. The bootstrap tracks the whole top-player
# universe; a few more rows than fit on screen would be recorded and never shown.
TRACKED_PLAYERS = 14


def win_rates_for(weights: np.ndarray, field_mean: np.ndarray) -> np.ndarray:
    """What each player's build is worth per category, against the average build in the field.

    A weight is an input; this is the outcome it buys, which is what the scene colours. Half is a
    coin flip and the distance from it is the edge -- the same Phi the objective is built from,
    left per-category instead of summed into a total.
    """
    return norm.cdf(weights - field_mean)


def record_bootstrap(session) -> list[dict]:
    """Re-run the shipped self-play bootstrap with the per-pass recorder attached."""
    agent = session.agent
    agent.self_play_trace = []
    # The build already ran this once; running it again with the recorder on is the only way to
    # see inside it, and it resets its own state at the top, so the second run is not a
    # continuation of the first.
    agent.populate_default_h_scores(session.current_settings['n_iterations'])
    trace = agent.self_play_trace
    agent.self_play_trace = None
    if not trace:
        raise RuntimeError(
            'The bootstrap recorded no passes. Opponent modelling is probably off for this '
            'session (a single neutral pass), so there is no self-play to animate.')
    return trace


def build_passes(trace: list[dict], player_registry: dict) -> tuple[list[dict], list[int]]:
    """Per-pass weights and win rates for the players the scene follows.

    The tracked set is fixed by the FIRST pass and held, so a row means the same player all the
    way through -- rows that changed identity mid-animation would make the settling unreadable.
    """
    first_field = trace[0]['field']
    tracked = list(first_field.index[:TRACKED_PLAYERS])
    categories = list(first_field.columns)

    passes = []
    for entry in trace:
        field = entry['field']
        # A player can be absent from a pass's field (the loop re-solves half the universe at a
        # time); carrying the previous row forward is what the algorithm itself does with a
        # resting player's incumbent build.
        present = [player_id for player_id in tracked if player_id in field.index]
        weights = field.loc[present].to_numpy(dtype=float)
        field_mean = field.to_numpy(dtype=float).mean(axis=0)

        rows = {player_id: weights[index] for index, player_id in enumerate(present)}
        ordered = np.array([rows.get(player_id, np.full(len(categories), np.nan))
                            for player_id in tracked])
        passes.append({
            'pass_index': entry['pass_index'],
            'drift':      entry['drift'],
            'weights':    np.round(ordered, 4).tolist(),
            'win_rates':  np.round(win_rates_for(ordered, field_mean), 4).tolist(),
        })
    return passes, tracked


def main() -> None:
    print(f'Building a {SEASON} session at the app defaults...')
    session = build_session(
        current_settings = _build_current_settings(
            build_default_session_request(SEASON, SPORT)),
        platform_config  = None,
        csv_bytes        = None,
        uploaded_dfs     = None,
    )

    print('Re-running the self-play bootstrap with the recorder attached...')
    trace = record_bootstrap(session)
    passes, tracked = build_passes(trace, session.player_registry)
    categories = list(trace[0]['field'].columns)

    drifts = [entry['drift'] for entry in passes if entry['drift'] is not None]
    print(f'\n{len(passes)} passes recorded, {len(tracked)} players tracked')
    print(f'drift: first {drifts[0]:.4f}, last {drifts[-1]:.4f}, '
          f'{drifts[0] / max(drifts[-1], 1e-9):.0f}x smaller by the end')

    data_path = _VISUALIZATIONS_DIR / 'data' / 'self_play.json'
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps({
        'season':         SEASON,
        'categories':     categories,
        'player_names':   [session.player_registry[player_id].name
                           if player_id in session.player_registry else str(player_id)
                           for player_id in tracked],
        'passes':         passes,
    }), encoding='utf-8')
    print(f'Wrote {data_path.relative_to(_VISUALIZATIONS_DIR.parent)}')


if __name__ == '__main__':
    main()
