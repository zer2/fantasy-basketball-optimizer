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
shipped objective gives that build. Nothing here is chosen by hand: every candidate, every pair
of categories and every setting of the OTHER seven weights is searched for a surface where a
punt summit beats the balanced one, since that is the shape that shows what a starting point
decides. The ones that have it are then traced, and the slice whose climb walks furthest without
doubling back is kept.

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
from backend.math.algorithm_agents import HAgent, AdamOptimizer           # noqa: E402
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

# What counts as the shape the scene argues for: a summit with both categories at about their
# balanced weights, and at least one more with a category given up while the OTHER is still near
# its balanced weight. That second condition is what stops a ridge running diagonally from
# qualifying -- a punt gives ONE category up, not both shaded down together.
#
# A punt in EACH category was demanded here at one point, which rules out most of the field: a
# candidate usually has one category worth giving up rather than two. Measured on the first pick,
# 33 of Jokic's 36 pairings have a single punt peak and no second one -- including pairs where
# the punt plainly beats the balanced build.
BALANCED_WEIGHTS  = (0.85, 1.35)
SUMMIT_MERGE_WEIGHT = 0.16     # summits closer than this on both axes are one broad hilltop

# How many three-summit slices are traced before one is kept. The shape of a surface is cheap to
# measure and says nothing about where a descent ends on it, so several are run and the one whose
# climb finishes nearest its own summit is kept.
#
# Both scenes' climbs are now held to the two weights they draw (see trace_from_seed's
# `hold_to_plane`), so a path no longer drifts off the slice it is drawn on -- it cannot, the
# other seven weights do not move. What still varies between slices, and is still only knowable
# by running them, is whether the climb goes anywhere worth watching: which basin it settles in,
# how far it walks, and whether it doubles back getting there.
MENU_SLICES_TO_TRACE = 16
# At most this many cuts from any one candidate-and-anchor. Sorted by advantage alone the
# shortlist filled with the same surface over and over -- six of eight were one player at one
# anchor, differing only in which category sat on the second axis, and behaving identically
# because the anchor was doing the work. A shortlist of near-duplicates tests one surface eight
# times and calls it a search.
CUTS_PER_CANDIDATE = 2

# What makes a climb worth watching, rather than merely correct. A descent that starts next to
# the summit has nowhere to go: it shuffles about on the flat top, doubles back on itself, and
# shows none of the walking uphill the scene is there to show. So a slice is kept for the LENGTH
# of the journey its chosen seed makes, once the journey ends somewhere near the right peak.
# The complaint these are here to rule out is the doubling back, so the wander bound is the tight
# one: a free descent measured on the descent scene's own slice wandered 3.22 times the straight
# line, and anything under about two and a half is a different picture from that. The landing
# bound is looser, because a ball that stops fifteen points short of a broad summit still plainly
# climbed it.
LANDING_TOLERANCE = 15.0       # percentage points from the summit still counted as arriving
WANDER_TOLERANCE = 2.5         # path length over straight-line distance; above this it meanders

# Draw the winning slice with the roster assignment held still, which smooths away the notches
# the re-solving leaves down the hillside. This changes only what is DRAWN -- the pairing and its
# climbs are chosen on the honest surface either way, and cannot be chosen on a held one, because
# holding removes the balanced summit that punt_advantage weighs the punt against. See
# measure_slice for what holding costs in height.
HOLD_DRAWN_ASSIGNMENT = True


def anchor_menu(neutral, index_a, index_b):
    """The cuts worth trying: the balanced build, and one gentle punt in each other category.

    The seven weights that are not on the plane have to be held at something, and the balanced
    build is one choice among many rather than a privileged one. Every seed the app's own menu
    offers is an equally real build to cut through, and each gives a different surface with
    different summits and a different climb over it -- which turns a few dozen candidate
    pictures into a few thousand, and is the difference between taking whatever the field
    happens to offer and choosing.

    The punt is labelled by the category it gives up, for the log: a record that says which cut
    produced a picture is the difference between a result and a coincidence.
    """
    cuts = [('balanced', np.asarray(neutral, dtype=float))]
    for index in range(len(neutral)):
        if index in (index_a, index_b):
            continue
        anchor = np.asarray(neutral, dtype=float).copy()
        anchor[index] *= PUNT_SEED_FACTOR
        cuts.append((index, anchor))
    return cuts

# How many times the seed-menu slice is re-cut through its own answer. The seven category weights
# that are not on the plane are now pinned to the cut for the whole descent, so THEY no longer
# argue for re-cutting: the endpoint lies in the plane by construction.
#
# The FLEX SHARES still do. The solver returns gradients for them as well and the descent moves
# them, and they are a dimension the plane neither shows nor can show -- so a surface drawn at the
# shares the cut started from is cut through a slightly different place than the one the climb
# finished in. The second pass re-cuts through the finishing shares. Both are judged and the
# better kept, since the first sometimes gives the longer climb even when the second lands closer.
ANCHOR_PASSES = 2
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


def run_bootstrap_watching(agent, n_iterations, tracked_players, forced_seeds=None,
                           held_to_plane=None):
    """Run the shipped bootstrap, following several players' solves through it.

    Returns {player_id: (context, trajectory)} -- the objective's arguments narrowed to that
    player, and his category weights at every iteration of the last pass that solved him. Several
    at once because a bootstrap run costs about fifteen seconds and the choice of WHICH player to
    draw is made afterwards, from what the surfaces turn out to look like.

    `forced_seeds` maps a player to a seed that replaces the menu's choice FOR HIM ONLY. Every
    other candidate keeps the seed the menu picked, so the field the passes settle into is the
    same in every run and the trajectories are comparable.

    `held_to_plane` maps a player to the two category indices a scene draws, and confines his
    descent to them: the other seven weights take no step. See trace_from_seed's `hold_to_plane`,
    and the descent section of main(), for why a scene would want that.
    """
    tracked_players = list(tracked_players)
    forced_seeds = forced_seeds or {}
    held_to_plane = held_to_plane or {}
    captured = {player: {'context': None, 'trajectory': []} for player in tracked_players}
    original_select = HAgent._select_starting_weights
    original_objective = HAgent.get_objective_and_gradient
    original_minimize = AdamOptimizer.minimize
    # Which row of the current batch each tracked player occupies, refreshed on every objective
    # call and read by the Adam hook below -- the optimiser is handed a gradient and no idea whose
    # it is, and the update it returns has to be masked for one row only.
    live_rows: dict = {}

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
        rows = tracked_rows(self)
        live_rows.clear()
        live_rows.update(rows)
        for player, row in rows.items():
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

    def minimize(self, gradient):
        """Adam's step, with the off-plane categories zeroed for the players being held.

        The mask has to go on the UPDATE rather than on the gradient. The solver centres the
        category gradient row by row before handing it over, so a gradient zeroed beforehand comes
        back non-zero the moment the row mean is subtracted from it.

        Only the category optimiser is masked. It is told apart from the shares optimiser by the
        width of what it returns, which is asserted to be unambiguous before any of this runs.
        """
        update = original_minimize(self, gradient)
        if not held_to_plane:
            return update
        update = np.asarray(update, dtype=float)
        if update.ndim != 2 or update.shape[1] != agent.n_categories:
            return update
        update = update.copy()
        for player, row in live_rows.items():
            drawn = held_to_plane.get(player)
            if drawn is None or row >= update.shape[0]:
                continue
            off_plane = np.ones(agent.n_categories, dtype=bool)
            off_plane[list(drawn)] = False
            update[row, off_plane] = 0.0
        return update

    if held_to_plane and agent.position_means is not None:
        n_positions = agent.position_means.shape[1]
        if n_positions == agent.n_categories:
            raise RuntimeError(
                f'There are as many positions as categories ({n_positions}), so the Adam hook '
                f'cannot tell the category optimiser from the shares one by the width of its '
                f'update. Hold the descent some other way rather than masking the wrong thing.')

    HAgent._select_starting_weights = select
    HAgent.get_objective_and_gradient = objective
    AdamOptimizer.minimize = minimize
    try:
        agent.populate_default_h_scores(n_iterations)
    finally:
        HAgent._select_starting_weights = original_select
        HAgent.get_objective_and_gradient = original_objective
        AdamOptimizer.minimize = original_minimize

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

    def __init__(self, agent, context, neutral, index_a, index_b, anchor=None):
        self.agent = agent
        self.context = context
        self.neutral = neutral
        self.index_a = index_a
        self.index_b = index_b
        # Where the seven weights that are not on the plane are held. The balanced build unless a
        # caller says otherwise, which is the cut that contains no part of any descent.
        self.anchor = neutral if anchor is None else np.asarray(anchor, dtype=float)

    def weights_at(self, coordinates: np.ndarray) -> np.ndarray:
        """Plane coordinates to full weight vectors, renormalised as the solver keeps them.

        The two plane categories are set as multiples of their BALANCED weights however the cut
        is anchored, so the axes keep meaning what they are labelled with: nothing at zero, the
        balanced weight at a hundred. Only the other seven follow the anchor.
        """
        weights = np.tile(self.anchor, (len(coordinates), 1))
        weights[:, self.index_a] = (self.neutral[self.index_a]
                                    * np.clip(coordinates[:, 0], 0.0, None))
        weights[:, self.index_b] = (self.neutral[self.index_b]
                                    * np.clip(coordinates[:, 1], 0.0, None))
        weights = np.clip(weights, 1e-6, None)
        return weights / weights.sum(axis=1, keepdims=True)

    def score(self, coordinates: np.ndarray, keep_positions: bool = False) -> np.ndarray:
        """What the objective says about builds that lie IN the plane."""
        return self.score_weights(self.weights_at(coordinates), keep_positions)

    def score_weights(self, weight_vectors: np.ndarray,
                      keep_positions: bool = False) -> np.ndarray:
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

        # The position machinery caches per batch and the captured cache is sized to the solve
        # it came from; the shipped seed-scoring path clears exactly these before re-batching,
        # which makes every point re-solve its own roster assignment from its own weights.
        #
        # `keep_positions` leaves the cache alone instead, so a caller that has already filled it
        # gets every point scored against ONE assignment. The objective is a max over matchings,
        # so re-solving is what creases it; holding the matching still is how to see the surface
        # underneath those creases.
        if not keep_positions:
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


def punt_advantage(maxima):
    """How far the best punt summit stands above the balanced one, or None if the shape is absent.

    Positive is what the scene needs: the surface has a balanced build and at least one punt,
    and the punt is the better of the two. That makes the balanced start the one that would get
    stuck, which is what the narration says the seed menu is for.

    A punt summit is one category given up while the OTHER stays near its balanced weight. Both
    shaded down together is a diagonal ridge rather than a punt, and does not count.
    """
    low, high = BALANCED_WEIGHTS
    balanced = [peak for peak in maxima if low <= peak[0] <= high and low <= peak[1] <= high]
    punts = [peak for peak in maxima
             if (peak[0] <= PUNTED_MAX_WEIGHT and KEPT_MIN_WEIGHT <= peak[1] <= high)
             or (peak[1] <= PUNTED_MAX_WEIGHT and KEPT_MIN_WEIGHT <= peak[0] <= high)]
    if not (balanced and punts):
        return None
    return (max(punts, key=lambda peak: peak[2])[2]
            - max(balanced, key=lambda peak: peak[2])[2])


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
    three_summits, single_summit = [], None
    for player, (context, _) in solved.items():
        best_three = None
        for index_a, index_b in itertools.combinations(range(len(categories)), 2):
            for cut, anchor in anchor_menu(neutral, index_a, index_b):
                plane = WeightPlane(agent, context, neutral, index_a, index_b, anchor)
                surface = plane.score(search_grid()).reshape(SEARCH_STEPS, SEARCH_STEPS)
                maxima = local_maxima(surface, search_axis())

                # Every pairing AND every cut is shortlisted, not just each player's best
                # pairing at the balanced cut: which surface a descent stays on has nothing to
                # do with how prominent its summits are, and the old sweep threw away almost
                # every candidate before the question that matters was asked of it.
                #
                # A positive advantage is the shape the scene argues for: the best build on the
                # surface is a punt, and the balanced start is the one that would get stuck. On
                # a surface whose highest point is the balanced build, the scene's sentence is
                # simply false.
                prominence = punt_advantage(maxima)
                if prominence is not None and prominence > 0:
                    three_summits.append((prominence, player, index_a, index_b, cut, anchor))
                if prominence is not None and (best_three is None or prominence > best_three[0]):
                    best_three = (prominence, index_a, index_b)

                # The one-hill slice is measured at the balanced cut only. Its scene is signed
                # off, and its surface has to come back identical from every run.
                if cut != 'balanced':
                    continue
                relief = single_summit_relief(maxima, surface)
                if relief is not None and (single_summit is None or relief > single_summit[0]):
                    single_summit = (relief, player, index_a, index_b)

        if best_three is None:
            print(f'   {player_names[player]:24} no three-summit slice')
            continue
        prominence, index_a, index_b = best_three
        print(f'   {player_names[player]:24} best pairing {categories[index_a]} against '
              f'{categories[index_b]}, the lower punt {prominence:+.5f} above balanced')

    if not three_summits:
        raise RuntimeError(
            'No three-summit slice for any of the tracked candidates. The surfaces may genuinely '
            'have fewer basins, in which case the scene should say so rather than be handed a '
            'picture that flatters it.')
    if single_summit is None:
        raise RuntimeError(
            'No slice with a single interior summit. Every surface measured here is either '
            'multi-basin or still climbing at the edge of the plane, and the gradient-descent '
            'scene needs one hill with a top on it.')
    three_summits.sort(key=lambda entry: -entry[0])

    # Best first, but never more than a couple from the same candidate at the same anchor.
    spread, seen = [], {}
    for entry in three_summits:
        _, player, _, _, cut, _ = entry
        taken = seen.get((player, cut), 0)
        if taken >= CUTS_PER_CANDIDATE:
            continue
        seen[(player, cut)] = taken + 1
        spread.append(entry)
        if len(spread) == MENU_SLICES_TO_TRACE:
            break

    print(f'   {len(three_summits)} cuts where the best peak is a punt; tracing {len(spread)} '
          f'of them, spread across {len(seen)} candidate-and-anchor pairs')
    return [entry[1:] for entry in spread], single_summit[1:]


def climb_shape(climbs, axis, surface) -> tuple[dict, dict]:
    """The climb the scene will trace, and the three numbers that say whether it is worth watching.

    The scene climbs whichever seed scores best where it stands, so that is the one measured.

    `landing` is how far it stops from the surface's own highest point, `journey` how far it had
    to come, and `wander` how much further it travelled than the straight line between the two --
    one means it walked directly there, and a large number means it doubled back. All in the
    percentage units the axes are labelled in, so they are things a viewer could read off the
    picture rather than internal quantities.
    """
    chosen = max(climbs, key=lambda climb: climb['start_score'])
    path = np.array(chosen['path'])

    peak = np.unravel_index(np.argmax(surface), surface.shape)
    summit = np.array([as_percent(axis[peak[0]]), as_percent(axis[peak[1]])])

    straight = float(np.hypot(*(path[-1] - path[0])))
    travelled = float(np.hypot(*np.diff(path, axis=0).T).sum())
    return chosen, {
        'landing': float(np.hypot(*(path[-1] - summit)))
        , 'journey': float(np.hypot(*(path[0] - summit)))
        , 'wander':  travelled / straight if straight > 1e-6 else float('inf')
    }


def as_percent(weight: float) -> float:
    """A weight, as the percentage of the balanced weight that the scene's axes are labelled in."""
    return round(float(weight) * 100.0, 2)


def search_axis() -> np.ndarray:
    return np.linspace(*WEIGHT_RANGE, SEARCH_STEPS)


def search_grid() -> np.ndarray:
    axis = search_axis()
    return np.array([[a, b] for a in axis for b in axis])


# ── Measuring and writing one slice ──────────────────────────────────────────────────

def rescore_held(agent, plane, axis):
    """The same plane, every point scored against ONE roster assignment instead of its own.

    The first pass at the balanced build fills the position cache with a single assignment. After
    that nothing counts as active, so there is nothing left to re-solve and every later point
    reuses what is cached. Both hooks live on the agent instance; the algorithm is not touched.
    """
    grid = np.array([[a, b] for a in axis for b in axis])
    plane.score(np.tile([1.0, 1.0], (len(grid), 1)))
    count_active = agent._active_candidate_count
    agent._active_candidate_count = lambda iteration, n_candidates: 0
    agent._candidate_priority = np.array([0])
    try:
        return plane.score(grid, keep_positions=True).reshape(GRID_STEPS, GRID_STEPS)
    finally:
        agent._active_candidate_count = count_active


def measure_slice(agent, context, neutral, index_a, index_b, anchor=None,
                  hold_assignment=False):
    """The chosen pairing, re-measured on the fine grid the scene actually draws.

    The objective re-solves the roster assignment from the weights on every call, so its value is
    a max over matchings: smooth while one matching wins, notched where the winner changes. Those
    notches are the creases down the hillside.

    `hold_assignment` scores the whole grid against ONE matching, taken at the balanced build, so
    the surface underneath the notches shows through. On the cut the scene ships -- Scottie
    Barnes, Threes against Steals -- holding moves the surface by at most 0.00302 against a relief
    of 0.02222. That is 14% of the relief, so this is not a free cosmetic pass: it is a visible
    change of heights, and the reason it is still honest is that both summits stay in the same
    cells and in the same order. main() prints both figures on every run, so a slice that starts
    costing more than this says so rather than being taken on trust.

    Holding cannot be used for the SEARCH, only for the final drawing. Held still, some cuts lose
    their balanced summit altogether and read as two punts, and punt_advantage needs a balanced
    summit to weigh the punt against. So the pairing and its climbs are chosen on the honest
    surface, and only the winner is redrawn -- with main() checking first that the plane still
    scores as it did when it was chosen.
    """
    plane = WeightPlane(agent, context, neutral, index_a, index_b, anchor)
    axis = np.linspace(*WEIGHT_RANGE, GRID_STEPS)
    if hold_assignment:
        surface = rescore_held(agent, plane, axis)
    else:
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
                    , surface
                    , hold_to_plane=False):
    """Re-run the bootstrap with one seed forced on the tracked player, and keep his descent.

    The seed replaces the menu's choice for that player only, so the field the passes settle into
    is the same in every run and the climbs can be compared with each other.

    `hold_to_plane` confines the descent to the two weights the scene draws. See the descent
    section of main() for what that buys and how it narrows the claim being made.
    """
    seed = plane.weights_at(np.array([start]))[0]
    rerun = run_bootstrap_watching(
        agent, n_iterations, [tracked], forced_seeds={tracked: seed},
        held_to_plane={tracked: (index_a, index_b)} if hold_to_plane else None)
    rerun_context, trajectory = rerun[tracked]

    path = [project_onto_plane(weights, neutral, index_a, index_b) for weights in trajectory]
    trimmed = [path[0]]
    for point in path[1:]:
        if abs(point[0] - trimmed[-1][0]) > 1e-4 or abs(point[1] - trimmed[-1][1]) > 1e-4:
            trimmed.append(point)
    start_score, end_score = plane.score_weights(np.array([trajectory[0], trajectory[-1]]))
    # The context this run finished in, kept for the same reason the final weights are. The
    # descent moves the FLEX SHARES as well as the category weights -- the solver returns
    # gradients for both -- so a surface drawn at some other run's shares is cut through a
    # different place again, in a dimension the plane does not show and cannot show.
    entry = {'_final_weights': np.asarray(trajectory[-1], dtype=float)
             , '_context': rerun_context
             , 'label': label
             , 'path': [[as_percent(value) for value in point] for point in trimmed]
             , 'start_score': round(float(start_score), 6)
             , 'score': round(float(end_score), 6)}
    print(f'   {label:28} {entry["start_score"]:.5f} -> ({entry["path"][-1][0]:5.1f}%, '
          f'{entry["path"][-1][1]:5.1f}%) {entry["score"]:.5f}  '
          f'(slice reads {interpolate_grid(surface, axis, trimmed[-1]):.5f})  '
          f'over {len(trimmed) - 1} real iterations')
    return entry


def without_working(climbs):
    """The climbs as the scenes read them, with the nine-weight endpoints left behind.

    The endpoints are what the cut is re-anchored on; they are working, not something a scene
    has any use for, and writing them would put a nine-number vector in a file whose whole
    contents are otherwise plane coordinates.
    """
    return [{key: value for key, value in climb.items() if not key.startswith('_')}
            for climb in climbs]


def write_measurements(filename: str, payload: dict) -> None:
    data_path = _VISUALIZATIONS_DIR / 'prepared_data' / filename
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps(payload), encoding='utf-8')
    print(f'Wrote {data_path.relative_to(_VISUALIZATIONS_DIR.parent)}')


def main() -> None:
    build = sys.argv[1] if len(sys.argv) > 1 else 'both'
    if build not in ('both', 'menu', 'descent'):
        raise SystemExit(f'Build what? Expected both, menu or descent; got {build!r}.')

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
    menu_slices, simple_slice = search_slices(agent, solved, neutral, categories, player_names)

    # The seed-menu slice is the expensive half -- sixteen shortlisted cuts, each traced in
    # full -- and its scene is signed off. Rebuilding only the descent slice leaves that file
    # exactly as it is rather than re-deriving a blessed picture as a side effect.
    if build != 'descent':
        # ── The seed menu: three starts on a surface with three summits ───────────────────
        # Each shortlisted slice is traced in full and kept only if its climb lands nearer its own
        # summit than the best so far. Prominence alone picked a surface whose climb stopped fifteen
        # percentage points from the visible peak, which reads as the algorithm giving up early.
        print(f'\nTracing {len(menu_slices)} shortlisted slices to see where each climb lands...')
        best_menu = None
        for tracked, index_a, index_b, cut, found_anchor in menu_slices:
            category_a, category_b = categories[index_a], categories[index_b]
            held = 'balanced' if cut == 'balanced' else f'punting {categories[cut]}'
            print(f'\n   {player_names[tracked]}: {category_a} against {category_b}, '
                  f'others held {held}')

            anchor, context, best_pass = found_anchor, solved[tracked][0], None
            for attempt in range(ANCHOR_PASSES):
                plane, axis, surface = measure_slice(
                    agent, context, neutral, index_a, index_b, anchor)

                # The shape is re-checked on the fine grid the scene actually draws rather than
                # trusted from the coarse search: a punt that beats the balanced build by a hair at
                # fifteen steps can lose to it at thirty-one, and then the picture argues the
                # opposite of the narration.
                drawn = punt_advantage(merge_summits(local_maxima(surface, axis),
                                                     SUMMIT_MERGE_WEIGHT))
                if drawn is None or drawn <= 0:
                    # Two quite different failures, worth telling apart: the surface may have lost
                    # one of the two summits entirely, or it may still have both with the balanced
                    # one on top. Only the second is "the punt is not worth it".
                    reason = ('no balanced summit and punt summit to compare'
                              if drawn is None else
                              f'the balanced summit wins by {-drawn:.5f}')
                    print(f'      cut {attempt + 1}: {reason} -- dropped')
                    break

                climbs = [
                    trace_from_seed(agent, n_iterations, tracked, plane, start, label,
                                    neutral, index_a, index_b, axis, surface,
                                    hold_to_plane=True)
                    for label, start in ((f'punt {category_a}', [PUNT_SEED_FACTOR, 1.0])
                                         , (f'punt {category_b}', [1.0, PUNT_SEED_FACTOR])
                                         , ('balanced', [1.0, 1.0]))
                ]
                chosen, shape = climb_shape(climbs, axis, surface)
                arrives = shape['landing'] <= LANDING_TOLERANCE
                direct = shape['wander'] <= WANDER_TOLERANCE
                print(f'      cut {attempt + 1}: climbs {chosen["label"]!r:22} '
                      f'starts {shape["journey"]:5.1f} from the summit, lands {shape["landing"]:5.1f} '
                      f'away, wanders x{shape["wander"]:.2f}'
                      f'{"" if arrives and direct else "   -- rejected"}')

                # Kept for the JOURNEY, once it arrives and goes more or less straight there. A
                # climb that starts beside the summit is correct and dull; the scene needs one that
                # visibly walks uphill.
                if arrives and direct and (best_pass is None or shape['journey'] > best_pass[0]):
                    best_pass = (shape['journey'], shape, plane, axis, surface, climbs)
                # The next cut goes through this climb's own endpoint, in every dimension the
                # plane does not draw: the seven other category weights via the anchor, and the flex
                # shares via the context the climb itself finished in.
                anchor, context = chosen['_final_weights'], chosen['_context']

            if best_pass is None:
                print('      nothing usable from this pairing')
                continue
            journey, shape, plane, axis, surface, climbs = best_pass
            if best_menu is None or journey > best_menu[0]:
                best_menu = (journey, shape, tracked, index_a, index_b, plane, axis, surface, climbs)

        if best_menu is None:
            raise RuntimeError(
                'No shortlisted slice produced a climb that both arrives near its summit and goes '
                'more or less straight there. Widen MENU_SLICES_TO_TRACE or loosen LANDING_TOLERANCE '
                'and WANDER_TOLERANCE -- but loosening them means shipping a scene whose ball either '
                'stops short of the peak or doubles back on the way, which is what they are for.')

        journey, shape, tracked, index_a, index_b, plane, axis, surface, climbs = best_menu
        category_a, category_b = categories[index_a], categories[index_b]
        print(f'\nSeed menu -- {player_names[tracked]}: {category_a} against {category_b}; '
              f'the climb walks {journey:.1f} points to within {shape["landing"]:.1f} of the summit, '
              f'wandering x{shape["wander"]:.2f}')

        # Redrawn with the roster assignment held, which takes the notches out of the hillside. The
        # pairing and its climbs were chosen above on the honest surface and are not re-chosen here;
        # only the heights being drawn change. See measure_slice for what that costs.
        if HOLD_DRAWN_ASSIGNMENT:
            honest = surface
            # This plane was measured many bootstrap runs ago -- every later pairing in the menu ran
            # its own passes through the same agent. Scoring it again unheld has to reproduce what
            # was measured back then, exactly. If it does not, the agent has moved underneath the
            # plane, and the held surface would be cut through a different place than the climbs were
            # traced in: the drawn ball would descend a hill that is not the one drawn under it.
            # That mismatch is what every earlier version of this scene got wrong, so it is checked
            # rather than assumed.
            redrawn = plane.score(
                np.array([[a, b] for a in axis for b in axis])).reshape(GRID_STEPS, GRID_STEPS)
            drift = float(np.abs(redrawn - honest).max())
            if drift > 0.0:
                raise RuntimeError(
                    f'The chosen plane no longer scores as it did when it was measured: rescoring it '
                    f'unheld moves the surface by {drift:.6f}. The agent has changed underneath it, '
                    f'so holding the assignment now would draw a hill the climbs were never traced '
                    f'on. Measure and hold the surface inside the search loop instead of here.')

            surface = rescore_held(agent, plane, axis)
            moved = float(np.abs(surface - honest).max())
            relief = float(honest.max() - honest.min())
            print(f'   held still: the plane still scores as measured (drift {drift:.6f}); holding '
                  f'moved the surface by at most {moved:.5f} against a span of {relief:.5f} '
                  f'({moved / relief:.0%} of the relief)')
            for weight_a, weight_b, score in merge_summits(local_maxima(surface, axis)
                                                           , SUMMIT_MERGE_WEIGHT):
                print(f'      ({as_percent(weight_a):5.1f}%, {as_percent(weight_b):5.1f}%)  '
                      f'{score:.5f}')
        write_measurements('weight_surface.json', {
            'season':     SEASON
            , 'candidate':  player_names[tracked]
            , 'category_a': category_a
            , 'category_b': category_b
            , 'axis':       [as_percent(value) for value in axis]
            , 'surface':    np.round(surface, 6).tolist()
            , 'climbs':     without_working(climbs)
        })

    # ── Gradient descent: one start on a surface with one hill ────────────────────────
    if build == 'menu':
        return
    tracked, index_a, index_b = simple_slice
    category_a, category_b = categories[index_a], categories[index_b]
    print(f'\nGradient descent -- {player_names[tracked]}: {category_a} against {category_b}')
    plane, axis, surface = measure_slice(
        agent, solved[tracked][0], neutral, index_a, index_b)
    # This climb is held to the two weights being drawn: the other seven take no Adam step, so the
    # path is a path ON this surface rather than the shadow of one through nine dimensions.
    #
    # It is worth being plain about what that changes, because it is a real change and not a
    # cosmetic one. The free descent moves all nine weights, and the drawn curve divides its two
    # coordinates by the median of the other seven to undo the solver's renormalisation -- so when
    # those seven drift, and measured here they drift 65% out of proportion, the drawn path
    # wobbles in a way the descent itself never did. It also ends somewhere this plane does not
    # contain, twelve points from the summit drawn beneath it.
    #
    # Held, the same seed walks to the summit of the surface it is drawn on, and its readout is
    # the height of the hill under it. The scene's claim narrows honestly with it: this is what
    # climbing looks like in the two weights on screen, which is what the narration says and all
    # the picture was ever able to show. The menu scene is NOT held -- its subject is where a
    # descent ends up, and holding it would decide that.
    middle = [sum(WEIGHT_RANGE) / 2] * 2
    climbs = without_working([
        trace_from_seed(agent, n_iterations, tracked, plane, middle,
                        'the middle of the surface', neutral, index_a, index_b, axis, surface,
                        hold_to_plane=True)])
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
