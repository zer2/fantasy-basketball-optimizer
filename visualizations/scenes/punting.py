"""Why giving up on categories wins more of them.

Nine categories, nine units of effort, and an opponent who spreads theirs evenly. Put one unit
into each and every category is a coin flip. Then take the effort out of one category and spread
it over the rest, and the expected haul goes UP -- the ground lost in a category you have
abandoned is cheaper than the ground gained in one you are contesting.

The optimum is to abandon three outright and split the budget six ways, worth 4.625 categories
against the 4.500 that perfect balance gets. That is not a figure chosen to make a point: an
optimiser handed all nine weights and told only to keep them non-negative and summing to nine
finds exactly [0, 0, 0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5].

Reading a panel: the bell never moves -- it is the distribution of how the category comes out,
and it is the same distribution whatever you do. What moves is the threshold. You win the
category when the result lands to the LEFT of the bar, so pushing the bar right is buying win
probability. Yellow is the half you get for showing up, green is what the effort bought on top
of it, red is what abandoning the category gave back.

The scene carries no words; see `visualizations/narration_notes.md` for what it is saying.

    manim -ql visualizations/scenes/punting.py PuntingSearch
    manim -qh visualizations/scenes/punting.py PuntingSearch
"""

from __future__ import annotations

from math import erf

import numpy as np
from manim import (
    Scene, VGroup, Line, Polygon, Text, DecimalNumber, ValueTracker,
    FadeIn, always_redraw,
    DOWN, RIGHT,
    YELLOW, WHITE, GREY_B, GREEN_B, RED_C,
)


CATEGORY_NAMES = [
    'Field Goal %', 'Free Throw %', 'Threes',
    'Points', 'Rebounds', 'Assists',
    'Steals', 'Blocks', 'Turnovers',
]
CATEGORY_COUNT = 9
EFFORT_BUDGET = 9.0
OPPONENT_EFFORT = 1.0     # what the opponent puts into every category, so parity is a coin flip

# The order categories are abandoned in. Free Throw % first is not arbitrary -- it is the punt
# the algorithm reaches for most often, and the one fantasy players will recognise.
ABANDON_ORDER = [1, 7, 2, 0]

# One panel of the three-by-three grid.
PANEL_WIDTH, PANEL_HEIGHT = 4.15, 1.95
PANEL_ORIGIN_Y = 1.75
CURVE_HALF_WIDTH = 1.72   # horizontal half-extent of a bell inside its panel
CURVE_HEIGHT = 0.95       # height of a bell at its peak
CURVE_SPAN = 2.8          # how many standard deviations either side of centre are drawn
_SHADE_RESOLUTION = 120   # points along a shaded region's curved edge

_SCORE_SAMPLES = 2000     # resolution of the precomputed best-so-far ratchet


def weights_at(progress: float) -> np.ndarray:
    """The nine efforts once `progress` categories have been drained.

    Draining is continuous and one at a time: the category currently being emptied gives its
    effort up smoothly, and whatever it releases is shared equally among the categories still
    being contested. Equal shares are not a simplification -- for identical categories the
    optimum genuinely splits the contested budget evenly, because at a maximum every contested
    category must have the same marginal value, and a Normal density hits a given value at only
    one point on a given side of the threshold.
    """
    weights = np.full(CATEGORY_COUNT, EFFORT_BUDGET / CATEGORY_COUNT)
    abandoned = int(progress)
    fraction = progress - abandoned

    for position in range(abandoned):
        weights[ABANDON_ORDER[position]] = 0.0
    if abandoned < len(ABANDON_ORDER):
        draining = ABANDON_ORDER[abandoned]
        weights[draining] = (EFFORT_BUDGET / (CATEGORY_COUNT - abandoned)) * (1.0 - fraction)

    emptied = ABANDON_ORDER[:abandoned + 1]
    contested = [i for i in range(CATEGORY_COUNT) if i not in emptied]
    if contested:
        weights[contested] = (EFFORT_BUDGET - weights[emptied].sum()) / len(contested)
    return weights


def win_probability(weight: float) -> float:
    """The chance of taking a category, given the effort put into it against an even opponent."""
    return 0.5 * (1.0 + erf((weight - OPPONENT_EFFORT) / np.sqrt(2.0)))


def categories_won(weights: np.ndarray) -> float:
    return float(sum(win_probability(weight) for weight in weights))


class PuntingSearch(Scene):
    """Walk the punt one category at a time and watch the expected haul rise, then fall."""

    def setup(self) -> None:
        self.progress = ValueTracker(0.0)

        # The best-so-far readout ratchets, so it needs the whole path's history. Sampling it
        # once here keeps the per-frame cost to a lookup: recomputing a running maximum from
        # scratch every frame would redo the same work thousands of times.
        self._sampled_progress = np.linspace(0.0, len(ABANDON_ORDER), _SCORE_SAMPLES)
        sampled_scores = np.array([categories_won(weights_at(p)) for p in self._sampled_progress])
        self._best_so_far = np.maximum.accumulate(sampled_scores)

    # ── Geometry ──────────────────────────────────────────────────────────────────────

    def _panel_centre(self, category_index: int) -> np.ndarray:
        row, column = divmod(category_index, 3)
        return np.array([(column - 1) * PANEL_WIDTH,
                         PANEL_ORIGIN_Y - row * PANEL_HEIGHT,
                         0.0])

    def _point_on_curve(self, category_index: int, offset: float) -> np.ndarray:
        """A point on the bell, `offset` standard deviations from its centre."""
        return self._panel_centre(category_index) + np.array([
            offset * (CURVE_HALF_WIDTH / CURVE_SPAN),
            CURVE_HEIGHT * np.exp(-0.5 * offset ** 2),
            0.0,
        ])

    def _threshold_offset(self, weight: float) -> float:
        """Where the bar sits, in standard deviations: the effort's edge over the opponent.

        This is the one number the whole scene moves. At parity it is zero -- the bar stands on
        the mean and the category is a coin flip -- and every unit of effort pushes it a standard
        deviation to the right, which is exactly the win probability the score is counting.
        """
        return float(np.clip(weight - OPPONENT_EFFORT, -CURVE_SPAN, CURVE_SPAN))

    def _shaded_region(self, category_index: int, from_offset: float, to_offset: float, colour):
        """The area under the bell between two offsets, as a filled polygon."""
        if to_offset - from_offset < 1e-4:
            return None
        centre = self._panel_centre(category_index)
        top_edge = [self._point_on_curve(category_index, offset)
                    for offset in np.linspace(from_offset, to_offset, _SHADE_RESOLUTION)]
        base = [np.array([top_edge[-1][0], centre[1], 0.0]),
                np.array([top_edge[0][0], centre[1], 0.0])]
        return Polygon(*top_edge, *base, stroke_width=0, fill_color=colour, fill_opacity=0.6)

    # ── One category ──────────────────────────────────────────────────────────────────

    def current_weight(self, category_index: int) -> float:
        """The effort in one category right now.

        Read through a method rather than off `weights_at` directly so a scene that moves a
        single category on its own -- CategoryGradient does -- can override where one weight
        comes from without touching the drawing code or the drain path.
        """
        return float(weights_at(self.progress.get_value())[category_index])

    def _build_shading(self, category_index: int) -> VGroup:
        """The three regions that say what this category is worth, and the bar dividing them.

        Yellow is the half of the distribution you win at parity -- what showing up is worth.
        Green is territory beyond the mean that extra effort has bought. Red is territory short
        of the mean that abandoning the category has given back. Only ever two of the three are
        present at once, because the bar is either right of the mean or left of it.
        """
        threshold = self._threshold_offset(self.current_weight(category_index))

        regions = VGroup()
        # The part of the win region that was there before any effort was spent.
        baseline_edge = min(threshold, 0.0)
        for region in (
            self._shaded_region(category_index, -CURVE_SPAN, baseline_edge, YELLOW),
            self._shaded_region(category_index, 0.0, threshold, GREEN_B) if threshold > 0 else None,
            self._shaded_region(category_index, threshold, 0.0, RED_C) if threshold < 0 else None,
        ):
            if region is not None:
                regions.add(region)

        centre = self._panel_centre(category_index)
        bar_x = centre[0] + threshold * (CURVE_HALF_WIDTH / CURVE_SPAN)
        regions.add(Line([bar_x, centre[1] - 0.08, 0.0],
                         [bar_x, centre[1] + CURVE_HEIGHT * 1.14, 0.0],
                         color=WHITE, stroke_width=3))
        return regions

    def _build_bell(self, category_index: int) -> VGroup:
        """The curve itself, which never moves -- drawn once, not redrawn per frame."""
        points = [self._point_on_curve(category_index, offset)
                  for offset in np.linspace(-CURVE_SPAN, CURVE_SPAN, 140)]
        return VGroup(*[
            Line(start, end, color=WHITE, stroke_width=2.5)
            for start, end in zip(points, points[1:])
        ])

    # ── Readout ───────────────────────────────────────────────────────────────────────

    def _build_score_readout(self) -> VGroup:
        progress = self.progress.get_value()
        current = categories_won(weights_at(progress))
        best = float(np.interp(progress, self._sampled_progress, self._best_so_far))

        # Laid out across rather than stacked: the grid of bells already reaches well down the
        # frame, and a stacked readout leaves no room beneath it.
        current_value = DecimalNumber(current, num_decimal_places=3, font_size=50,
                                      color=YELLOW if current >= best - 1e-9 else GREY_B)
        readout = VGroup(
            Text('categories won', font_size=19, color=GREY_B),
            current_value,
            Text(f'best so far  {best:.3f}', font_size=19, color=GREY_B),
            Text(f'punted  {int(progress)}', font_size=19, color=GREY_B),
        )
        readout.arrange(RIGHT, buff=0.45)
        readout.move_to([0.0, -2.95, 0.0])
        return readout

    # ── The scene ─────────────────────────────────────────────────────────────────────

    def construct(self) -> None:
        titles = VGroup(*[
            Text(name, font_size=16, color=GREY_B).move_to(
                self._panel_centre(index) + np.array([0.0, CURVE_HEIGHT * 1.36, 0.0]))
            for index, name in enumerate(CATEGORY_NAMES)
        ])
        bells = VGroup(*[self._build_bell(index) for index in range(CATEGORY_COUNT)])
        shading = VGroup(*[
            always_redraw(lambda index=index: self._build_shading(index))
            for index in range(CATEGORY_COUNT)
        ])

        self.play(FadeIn(titles), FadeIn(bells), run_time=1.0)
        self.add(shading)
        self.wait(1.4)

        self.add(always_redraw(self._build_score_readout))
        self.wait(1.6)

        # Abandon them one at a time. The first three each pay for themselves; the fourth gives
        # ground back, and watching it fail is what makes three the answer rather than a claim.
        for abandoned in range(len(ABANDON_ORDER)):
            self.play(self.progress.animate.set_value(abandoned + 1.0), run_time=2.4)
            self.wait(1.0)

        # Walk back to the best the search found and rest there.
        self.play(self.progress.animate.set_value(3.0), run_time=1.2)
        self.wait(3.0)
