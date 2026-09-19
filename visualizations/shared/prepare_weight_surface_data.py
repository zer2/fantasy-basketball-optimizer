"""Measure the real objective over a plane of two category weights, and trace the real descent.

Two scenes read this: the gradient-descent explainer (one start, walking uphill) and the
multi-start one (the seed menu the algorithm actually uses).

Everything is recorded from inside the shipped self-play bootstrap -- the build that runs at
session creation, on an empty board, which is the FIRST PICK. That matters twice over. The field
the objective scores against is the equilibrium the passes settle into rather than a neutral
stand-in, and, unlike an ordinary evaluate, a bootstrap pass COLD-starts: it consults the seed
menu. Measured here, one pass hands out up to forty-two distinct starting points across a batch,
which is the multi-start behaviour the docs describe.

That cold start is why this is done the hard way rather than by calling evaluate. An ordinary
empty-board evaluate warm-starts from the bootstrap's own results, so forcing a seed into it
changes nothing at all -- every seed returns a byte-identical descent, and a scene built on that
would be showing a choice that was never made.

The plane is two category weights, each as a percentage of what that category is worth in a
balanced build: zero is the category given up entirely, a hundred is the balanced weight, and the
axis runs past a hundred so that the balanced build is an interior point rather than a corner --
otherwise a summit there could not be told from the edge of the picture. Z is the H-score the
shipped objective gives that build. The pair of categories is not chosen by hand: every pairing
is searched for a slice with three summits, one at balanced weights and one for each category
given up on its own, since that is the shape that shows what a starting point decides.

    python visualizations/prepare_weight_surface_data.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))   # the repository
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))          # visualizations/

from backend.api.routers.sessions import _build_current_settings          # noqa: E402
from backend.services.session_management import build_session             # noqa: E402
from backend.math.algorithm_agents import HAgent                          # noqa: E402
from backend.math.algorithm_agents import _PUNT_SEED_FACTOR                 # noqa: E402
from shared.prepare_season_data import build_default_session_request, SEASON, SPORT   # noqa: E402

_VISUALIZATIONS_DIR = Path(__file__).resolve().parent.parent

TRACKED_RANKS = range(10)      # whose solves are followed, by G-score rank; one is chosen below
SEARCH_STEPS = 15              # grid resolution while searching; the kept slice is measured finer
GRID_STEPS = 31
# The axis, as a multiple of each category's balanced weight: 0 gives the category up, 1 is the
# balanced weight, and the plane runs half again past it.
WEIGHT_RANGE = (0.0, 1.5)
PUNT_SEED_FACTOR = _PUNT_SEED_FACTOR   # the depth of the menu's gentle punt, as the app ships it

# What counts as the three summits: one with both categories at about their balanced weights, and
# one for each category given up while the OTHER is still near its balanced weight. Without that
# second condition a ridge running diagonally would qualify, and the point is that the two punts
# are alternatives rather than things to combine.
BALANCED_WEIGHTS  = (0.85, 1.35)
SUMMIT_MERGE_WEIGHT = 0.16     # summits closer than this on both axes are one broad hilltop
PUNTED_MAX_WEIGHT = 0.72
KEPT_MIN_WEIGHT   = 0.78


# ── Watching one player's solve inside the bootstrap ─────────────────────────────────

def narrow_to_one_candidate(args, row: int):
    """The objective's arguments, cut down to the single candidate at `row`.

    The bootstrap solves hundreds of players at once; a surface is about one of them.
    """
    (weights, position_shares, diff_means, diff_vars, x_scores_batch_array,
     candidate_player_array, team_so_far_array, n_players_selected,
     sigma_2_m, iteration, pitching_preference) = args
    batch_size = np.asarray(weights).shape[0]

    def narrow(value):
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == batch_size:
            return value[row:row + 1].copy()
        if isinstance(value, dict):
            return {key: (None if frame is None else frame.iloc[[row]].set_axis([0]))
                    for key, frame in value.items()}
        return value

    return (narrow(weights), narrow(position_shares), narrow(diff_means), narrow(diff_vars),
            narrow(x_scores_batch_array), narrow(candidate_player_array),
            narrow(team_so_far_array), n_players_selected, sigma_2_m, iteration,
            pitching_preference)


def run_bootstrap_watching(agent, n_iterations, tracked_players, forced_seeds=None):
    """Run the shipped bootstrap, following several players' solves through it.

    Returns {player_id: (context, trajectory)} -- the objective's arguments narrowed to that
    player, and his category weights at every iteration of the last pass that solved him. Several
    at once because a bootstrap run costs about fifteen seconds and the choice of WHICH player to
    draw is made afterwards, from what the surfaces turn out to look like.

    `forced_seeds` maps a player to a seed that replaces the menu's choice FOR HIM ONLY. Every
    other candidate keeps the seed the menu picked, so the field the passes settle into is the
    same in every run and the trajectories are comparable.
    """
    tracked_players = list(tracked_players)
    forced_seeds = forced_seeds or {}
    captured = {player: {'context': None, 'trajectory': []} for player in tracked_players}
    original_select = HAgent._select_starting_weights
    original_objective = HAgent.get_objective_and_gradient

    def tracked_rows(agent_self):
        batch_index = getattr(agent_self, '_current_batch_index', None)
        if batch_index is None:
            return {}
        batch = np.asarray(batch_index)
        rows = {}
        for player in tracked_players:
            positions = np.where(batch == player)[0]
            if len(positions):
                rows[player] = int(positions[0])
        return rows

    def select(self, *args, **kwargs):
        chosen = np.array(original_select(self, *args, **kwargs), dtype=float)
        for player, row in tracked_rows(self).items():
            if player in forced_seeds:
                chosen[row] = forced_seeds[player]
        return chosen

    def objective(self, *args, **kwargs):
        weights = np.asarray(args[0], dtype=float)
        for player, row in tracked_rows(self).items():
            if weights.shape[0] <= row:
                continue
            record = captured[player]
            # Each pass re-solves him from scratch, so the trajectory restarts rather than
            # accumulating: a scene shows one descent, not every pass concatenated.
            if args[9] == 0:
                record['trajectory'] = []
            record['trajectory'].append(weights[row].copy())
            record['context'] = narrow_to_one_candidate(args, row)
        return original_objective(self, *args, **kwargs)

    HAgent._select_starting_weights = select
    HAgent.get_objective_and_gradient = objective
    try:
        agent.populate_default_h_scores(n_iterations)
    finally:
        HAgent._select_starting_weights = original_select
        HAgent.get_objective_and_gradient = original_objective

    solved = {player: (record['context'], record['trajectory'])
              for player, record in captured.items() if record['context'] is not None}
    if not solved:
        raise RuntimeError(
            'None of the tracked players was solved during the bootstrap, so there is nothing to '
            'follow. Track candidates inside the bootstrap universe.')
    return solved


# ── The plane ────────────────────────────────────────────────────────────────────────

class WeightPlane:
    """The shipped objective over two category weights, for one candidate in one captured context."""

    def __init__(self, agent, context, neutral, index_a, index_b):
        self.agent = agent
        self.context = context
        self.neutral = neutral
        self.index_a = index_a
        self.index_b = index_b

    def weights_at(self, coordinates: np.ndarray) -> np.ndarray:
        """Plane coordinates to full weight vectors, renormalised as the solver keeps them."""
        weights = np.tile(self.neutral, (len(coordinates), 1))
        weights[:, self.index_a] *= np.clip(coordinates[:, 0], 0.0, None)
        weights[:, self.index_b] *= np.clip(coordinates[:, 1], 0.0, None)
        weights = np.clip(weights, 1e-6, None)
        return weights / weights.sum(axis=1, keepdims=True)

    def score(self, coordinates: np.ndarray) -> np.ndarray:
        """What the objective says about builds that lie IN the plane."""
        return self.score_weights(self.weights_at(coordinates))

    def score_weights(self, weight_vectors: np.ndarray) -> np.ndarray:
        """What the objective says about any builds at all, in this candidate's context.

        A descent moves all nine weights, so where it ENDS is not a point of this plane and its
        height here -- the other seven weights pinned at balanced -- is not its score. Measured:
        a descent started exactly on this slice's summit walks off it and reads LOWER, while the
        objective it is actually climbing goes up the whole way. So a climb's score is taken from
        the objective at its real build, and only its position is projected.
        """
        (_, position_shares, diff_means, diff_vars, x_scores_batch_array,
         candidate_player_array, team_so_far_array, n_players_selected,
         sigma_2_m, _, pitching_preference) = self.context
        repeats = len(weight_vectors)

        def tile(value):
            if isinstance(value, np.ndarray):
                return np.repeat(value, repeats, axis=0)
            if isinstance(value, dict):
                return {key: (None if frame is None
                              else frame.loc[frame.index.repeat(repeats)].set_axis(range(repeats)))
                        for key, frame in value.items()}
            return value

        # The position machinery caches per batch and the captured cache is sized to the solve it
        # came from; the shipped seed-scoring path clears exactly these before re-batching.
        self.agent._candidate_priority = None
        self.agent._position_rosters_cache = None
        self.agent._current_batch_index = None
        # diff_means and diff_vars are deliberately NOT tiled: the objective reshapes them as one
        # field shared by every candidate, which is exactly what a surface over weights wants, and
        # a tiled copy fails the reshape outright.
        scores = self.agent.get_objective_and_gradient(
            np.asarray(weight_vectors, dtype=float), tile(position_shares), diff_means,
            diff_vars, tile(x_scores_batch_array), tile(candidate_player_array),
            tile(team_so_far_array), n_players_selected, sigma_2_m, 0, pitching_preference,
        )['Score']
        return np.asarray(scores, dtype=float)


def project_onto_plane(weights, neutral, index_a: int, index_b: int) -> list[float]:
    """Where a real nine-dimensional build sits on the two-weight plane.

    A category's weight is read RELATIVE to the categories the plane does not show -- relative,
    because the solver renormalises every build to sum to one, so an absolute ratio would read a
    punt of one category as a mild lift of the other eight. Those categories supply the reference
    by their median ratio, which also discards whatever the descent did in the seven dimensions
    the plane does not show.
    """
    ratios = np.asarray(weights, dtype=float) / neutral
    untouched = [index for index in range(len(ratios)) if index not in (index_a, index_b)]
    scale = float(np.median(ratios[untouched]))
    return [float(np.clip(ratios[index_a] / scale, *WEIGHT_RANGE)),
            float(np.clip(ratios[index_b] / scale, *WEIGHT_RANGE))]


def interpolate_grid(surface: np.ndarray, axis: np.ndarray, point) -> float:
    """The measured surface at an arbitrary point, bilinear between grid samples."""
    span, steps = axis[-1] - axis[0], len(axis)
    position = [float(np.clip((value - axis[0]) / span, 0.0, 1.0)) * (steps - 1) for value in point]
    low = [int(np.floor(value)) for value in position]
    high = [min(index + 1, steps - 1) for index in low]
    fraction = [position[index] - low[index] for index in (0, 1)]
    return float(
        surface[low[0], low[1]]     * (1 - fraction[0]) * (1 - fraction[1])
        + surface[high[0], low[1]]  * fraction[0] * (1 - fraction[1])
        + surface[low[0], high[1]]  * (1 - fraction[0]) * fraction[1]
        + surface[high[0], high[1]] * fraction[0] * fraction[1]
    )


def local_maxima(surface: np.ndarray, axis: np.ndarray):
    """Every grid point at least as high as its eight neighbours."""
    steps = len(axis)
    return [(float(axis[i]), float(axis[j]), float(surface[i, j]))
            for i in range(steps) for j in range(steps)
            if surface[i, j] >= surface[max(0, i - 1):i + 2, max(0, j - 1):j + 2].max()]


def three_summit_prominence(maxima):
    """How far the LOWER punt summit stands above the balanced one, or None if the shape is absent."""
    low, high = BALANCED_WEIGHTS
    balanced = [peak for peak in maxima if low <= peak[0] <= high and low <= peak[1] <= high]
    punt_a = [peak for peak in maxima
              if peak[0] <= PUNTED_MAX_WEIGHT and KEPT_MIN_WEIGHT <= peak[1] <= high]
    punt_b = [peak for peak in maxima
              if peak[1] <= PUNTED_MAX_WEIGHT and KEPT_MIN_WEIGHT <= peak[0] <= high]
    if not (balanced and punt_a and punt_b):
        return None
    return min(max(punt_a, key=lambda peak: peak[2])[2],
               max(punt_b, key=lambda peak: peak[2])[2]) - max(balanced,
                                                               key=lambda peak: peak[2])[2]


def merge_summits(maxima, tolerance: float):
    """Grid points on one broad hilltop, reported as the single summit they are.

    A summit that is broad rather than sharp registers at several adjacent grid points, and
    counting each of them would make one hill look like a range.
    """
    kept: list = []
    for peak in sorted(maxima, key=lambda entry: -entry[2]):
        if any(abs(peak[0] - other[0]) <= tolerance and abs(peak[1] - other[1]) <= tolerance
               for other in kept):
            continue
        kept.append(peak)
    return kept


def single_summit_relief(maxima, surface: np.ndarray):
    """How much hill a one-summit slice has, or None if it is not one clean hill.

    The gradient-descent scene is about what a descent IS, so it wants a surface where walking
    uphill has exactly one answer. A slice that peaks on the edge of the plane would serve as
    badly as a three-summit one -- the climb would end at the crop rather than at a maximum -- so
    the summit has to stand inside the picture.
    """
    summits = merge_summits(maxima, SUMMIT_MERGE_WEIGHT)
    if len(summits) != 1:
        return None
    weight_a, weight_b, score = summits[0]
    low, high = WEIGHT_RANGE
    margin = (high - low) / (SEARCH_STEPS - 1)
    if not (low + margin < weight_a < high - margin and low + margin < weight_b < high - margin):
        return None
    return score - float(surface.min())


def search_slices(agent, solved, neutral, categories, player_names):
    """Every candidate and pairing measured once, and the two slices the scenes need picked out.

    One sweep rather than two: a surface costs a batched objective call, and both questions --
    where are there three summits, and where is there just one clean hill -- are answered off the
    same grid. Which player's weights make which shape is a fact about the field rather than
    something to assert, so both are searched for and neither is chosen by hand.
    """
    three_summit, single_summit = None, None
    for player, (context, _) in solved.items():
        best_three = None
        for index_a, index_b in itertools.combinations(range(len(categories)), 2):
            plane = WeightPlane(agent, context, neutral, index_a, index_b)
            surface = plane.score(search_grid()).reshape(SEARCH_STEPS, SEARCH_STEPS)
            maxima = local_maxima(surface, search_axis())

            prominence = three_summit_prominence(maxima)
            if prominence is not None and (best_three is None or prominence > best_three[0]):
                best_three = (prominence, index_a, index_b)

            relief = single_summit_relief(maxima, surface)
            if relief is not None and (single_summit is None or relief > single_summit[0]):
                single_summit = (relief, player, index_a, index_b)

        if best_three is None:
            print(f'   {player_names[player]:24} no three-summit slice')
            continue
        prominence, index_a, index_b = best_three
        print(f'   {player_names[player]:24} {categories[index_a]} against '
              f'{categories[index_b]}, the lower punt {prominence:+.5f} above balanced')
        if three_summit is None or prominence > three_summit[0]:
            three_summit = (prominence, player, index_a, index_b)

    if three_summit is None:
        raise RuntimeError(
            'No three-summit slice for any of the tracked candidates. The surfaces may genuinely '
            'have fewer basins, in which case the scene should say so rather than be handed a '
            'picture that flatters it.')
    if single_summit is None:
        raise RuntimeError(
            'No slice with a single interior summit. Every surface measured here is either '
            'multi-basin or still climbing at the edge of the plane, and the gradient-descent '
            'scene needs one hill with a top on it.')
    return three_summit[1:], single_summit[1:]


def as_percent(weight: float) -> float:
    """A weight, as the percentage of the balanced weight that the scene's axes are labelled in."""
    return round(float(weight) * 100.0, 2)


def search_axis() -> np.ndarray:
    return np.linspace(*WEIGHT_RANGE, SEARCH_STEPS)


def search_grid() -> np.ndarray:
    axis = search_axis()
    return np.array([[a, b] for a in axis for b in axis])


# ── Measuring and writing one slice ──────────────────────────────────────────────────

def measure_slice(agent, context, neutral, index_a, index_b):
    """The chosen pairing, re-measured on the fine grid the scene actually draws."""
    plane = WeightPlane(agent, context, neutral, index_a, index_b)
    axis = np.linspace(*WEIGHT_RANGE, GRID_STEPS)
    surface = plane.score(
        np.array([[a, b] for a in axis for b in axis])).reshape(GRID_STEPS, GRID_STEPS)
    peaks = [{'a': as_percent(weight_a), 'b': as_percent(weight_b), 'score': round(score, 6)}
             for weight_a, weight_b, score
             in merge_summits(local_maxima(surface, axis), SUMMIT_MERGE_WEIGHT)]
    print(f'   surface {surface.min():.5f} .. {surface.max():.5f}; {len(peaks)} summits')
    for peak in peaks:
        print(f"      ({peak['a']:5.1f}%, {peak['b']:5.1f}%)  {peak['score']:.5f}")
    return plane, axis, surface


def trace_from_seed(agent
                    , n_iterations
                    , tracked
                    , plane
                    , start
                    , label
                    , neutral
                    , index_a
                    , index_b
                    , axis
                    , surface):
    """Re-run the bootstrap with one seed forced on the tracked player, and keep his descent.

    The seed replaces the menu's choice for that player only, so the field the passes settle into
    is the same in every run and the climbs can be compared with each other.
    """
    seed = plane.weights_at(np.array([start]))[0]
    rerun = run_bootstrap_watching(agent, n_iterations, [tracked], forced_seeds={tracked: seed})
    trajectory = rerun[tracked][1]

    path = [project_onto_plane(weights, neutral, index_a, index_b) for weights in trajectory]
    trimmed = [path[0]]
    for point in path[1:]:
        if abs(point[0] - trimmed[-1][0]) > 1e-4 or abs(point[1] - trimmed[-1][1]) > 1e-4:
            trimmed.append(point)
    start_score, end_score = plane.score_weights(np.array([trajectory[0], trajectory[-1]]))
    entry = {'label': label
             , 'path': [[as_percent(value) for value in point] for point in trimmed]
             , 'start_score': round(float(start_score), 6)
             , 'score': round(float(end_score), 6)}
    print(f'   {label:28} {entry["start_score"]:.5f} -> ({entry["path"][-1][0]:5.1f}%, '
          f'{entry["path"][-1][1]:5.1f}%) {entry["score"]:.5f}  '
          f'(slice reads {interpolate_grid(surface, axis, trimmed[-1]):.5f})  '
          f'over {len(trimmed) - 1} real iterations')
    return entry


def write_measurements(filename: str, payload: dict) -> None:
    data_path = _VISUALIZATIONS_DIR / 'data' / filename
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps(payload), encoding='utf-8')
    print(f'Wrote {data_path.relative_to(_VISUALIZATIONS_DIR.parent)}')


def main() -> None:
    print(f'Building a {SEASON} session at the app defaults...')
    session = build_session(
        current_settings = _build_current_settings(
            build_default_session_request(SEASON, SPORT)),
        platform_config  = None,
        csv_bytes        = None,
        uploaded_dfs     = None,
    )
    agent = session.agent
    n_iterations = session.current_settings['n_iterations']
    categories = list(agent.x_scores.columns)
    neutral = agent.get_starting_category_weights()
    ranked = list(agent.info['G-scores'].index)
    tracked_players = [ranked[rank] for rank in TRACKED_RANKS]
    player_names = {player: session.player_registry[player].name for player in tracked_players}

    print(f'Re-running the self-play bootstrap, following the top {len(tracked_players)} picks...')
    solved = run_bootstrap_watching(agent, n_iterations, tracked_players)
    print(f'   {len(solved)} of them were solved during it')

    print('Searching every candidate and category pairing...')
    menu_slice, simple_slice = search_slices(agent, solved, neutral, categories, player_names)

    # ── The seed menu: three starts on a surface with three summits ───────────────────
    tracked, index_a, index_b = menu_slice
    category_a, category_b = categories[index_a], categories[index_b]
    print(f'\nSeed menu -- {player_names[tracked]}: {category_a} against {category_b}')
    plane, axis, surface = measure_slice(
        agent, solved[tracked][0], neutral, index_a, index_b)
    climbs = [
        trace_from_seed(agent, n_iterations, tracked, plane, start, label,
                        neutral, index_a, index_b, axis, surface)
        for label, start in ((f'punt {category_a}', [PUNT_SEED_FACTOR, 1.0])
                             , (f'punt {category_b}', [1.0, PUNT_SEED_FACTOR])
                             , ('balanced', [1.0, 1.0]))
    ]
    write_measurements('weight_surface.json', {
        'season':     SEASON
        , 'candidate':  player_names[tracked]
        , 'category_a': category_a
        , 'category_b': category_b
        , 'axis':       [as_percent(value) for value in axis]
        , 'surface':    np.round(surface, 6).tolist()
        , 'climbs':     climbs
    })

    # ── Gradient descent: one start on a surface with one hill ────────────────────────
    tracked, index_a, index_b = simple_slice
    category_a, category_b = categories[index_a], categories[index_b]
    print(f'\nGradient descent -- {player_names[tracked]}: {category_a} against {category_b}')
    plane, axis, surface = measure_slice(
        agent, solved[tracked][0], neutral, index_a, index_b)
    middle = [sum(WEIGHT_RANGE) / 2] * 2
    climbs = [trace_from_seed(agent, n_iterations, tracked, plane, middle,
                              'the middle of the surface', neutral, index_a, index_b,
                              axis, surface)]
    write_measurements('weight_surface_simple.json', {
        'season':     SEASON
        , 'candidate':  player_names[tracked]
        , 'category_a': category_a
        , 'category_b': category_b
        , 'axis':       [as_percent(value) for value in axis]
        , 'surface':    np.round(surface, 6).tolist()
        , 'climbs':     climbs
    })


if __name__ == '__main__':
    main()
