"""What a category is worth at the margin, which is a number already drawn on screen.

The punting scene shades the area under each bell and moves a bar across it. The quantity the
optimiser actually reasons about is not that area but its DERIVATIVE: how much win probability
one more unit of effort buys. For a Normal that derivative is the density at the threshold --
which is the HEIGHT OF THE BELL WHERE THE BAR CROSSES IT, already being drawn, and never
labelled.

That is the whole scene. Put the height on screen as a number, sweep one bar from far left to
far right and watch its number rise to a peak at parity and fall away on both sides, then read
all nine numbers off at the punt optimum: the six contested categories share one value and the
three abandoned ones sit lower. Six equal marginals is the first-order condition of the search
made visible -- at a maximum, every category you are still contesting has to be worth the same
at the margin, or effort would move from one to another.

It matters past punting. At convergence the optimiser's weights are proportional to these
gradients, so a weight IS the marginal value of its category -- the `Jw = 0` property from the
weight-model note, which this is the picture of.

Extends `punting.py` rather than standing alone: same nine panels, same bells, same geometry.

    manim -ql visualizations/scenes/category_gradient.py CategoryGradient
"""

from __future__ import annotations

import numpy as np
from manim import (
    VGroup, Line, Rectangle, Text, DecimalNumber, ValueTracker,
    FadeIn, FadeOut, Create, always_redraw,
    YELLOW, WHITE, GREY_B, GREY_D, BLUE_B,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / '5_punting'))
from punting import (
    PuntingSearch, CATEGORY_NAMES, CATEGORY_COUNT, OPPONENT_EFFORT,
    weights_at, CURVE_HEIGHT,
)


SWEPT_CATEGORY = 3            # Points, which is the one a viewer will not expect to be punted
SWEEP_LOW, SWEEP_HIGH = -1.6, 3.6     # effort, so parity (1.0) sits inside the sweep
PUNT_OPTIMUM = 3.0            # categories abandoned at the search's answer

MARGINAL_LABEL_OFFSET_Y = -0.33        # under each panel's baseline, clear of the threshold bar
SUMMARY_BASELINE_Y = -1.75
SUMMARY_SCALE = 9.0           # frame units per unit of marginal value


def marginal_value(weight: float) -> float:
    """How much win probability one more unit of effort buys, at this effort.

    The win probability is Phi(weight - opponent), so its derivative is phi(weight - opponent):
    the Normal density at the threshold, which is the height of the bell where the bar crosses.
    """
    return float(np.exp(-0.5 * (weight - OPPONENT_EFFORT) ** 2) / np.sqrt(2.0 * np.pi))


class CategoryGradient(PuntingSearch):
    """Label the height of the bell, sweep it, then read all nine off at the optimum."""

    def setup(self) -> None:
        super().setup()
        # The swept category moves on its own, off the drain path the parent scene walks. While
        # this holds a value that category ignores `progress` entirely.
        self.swept_effort = ValueTracker(float(weights_at(0.0)[SWEPT_CATEGORY]))
        self.sweeping = False

    def current_weight(self, category_index: int) -> float:
        if self.sweeping and category_index == SWEPT_CATEGORY:
            return float(self.swept_effort.get_value())
        return super().current_weight(category_index)

    # ── The number that was always on screen without being written down ───────────────

    def _build_marginal_readout(self, category_index: int) -> DecimalNumber:
        value = marginal_value(self.current_weight(category_index))
        highlighted = self.sweeping and category_index == SWEPT_CATEGORY
        readout = DecimalNumber(value, num_decimal_places=3, font_size=17,
                                color=YELLOW if highlighted else GREY_B)
        readout.move_to(self._panel_centre(category_index)
                        + np.array([0.0, MARGINAL_LABEL_OFFSET_Y, 0.0]))
        return readout

    def _build_height_marker(self) -> Line:
        """A tick at the top of the swept bar, so the number is visibly the bell's height.

        Without it the readout is a number that happens to be near a picture. With it the eye
        can see the two are the same quantity, which is the entire point of the scene.
        """
        threshold = self._threshold_offset(self.current_weight(SWEPT_CATEGORY))
        top = self._point_on_curve(SWEPT_CATEGORY, threshold)
        centre = self._panel_centre(SWEPT_CATEGORY)
        return Line([centre[0] - 1.9, top[1], 0.0], [centre[0] + 1.9, top[1], 0.0],
                    color=YELLOW, stroke_width=2)

    # ── Nine marginals, side by side ──────────────────────────────────────────────────

    def _build_summary_bars(self) -> VGroup:
        weights = weights_at(PUNT_OPTIMUM)
        contested = marginal_value(float(weights[weights > 0][0]))

        bars, labels = VGroup(), VGroup()
        for index in range(CATEGORY_COUNT):
            value = marginal_value(float(weights[index]))
            position_x = (index - (CATEGORY_COUNT - 1) / 2) * 1.16
            bars.add(Rectangle(
                width=0.72, height=value * SUMMARY_SCALE, stroke_width=0,
                fill_color=BLUE_B if weights[index] > 0 else GREY_D, fill_opacity=0.85,
            ).move_to([position_x, SUMMARY_BASELINE_Y + value * SUMMARY_SCALE / 2, 0.0]))
            labels.add(Text(CATEGORY_NAMES[index].replace(' %', '%'), font_size=12,
                            color=GREY_B if weights[index] > 0 else GREY_D)
                       .rotate(np.pi / 2.6)
                       .move_to([position_x, SUMMARY_BASELINE_Y - 0.78, 0.0]))

        level = Line([-5.6, SUMMARY_BASELINE_Y + contested * SUMMARY_SCALE, 0.0],
                     [5.6, SUMMARY_BASELINE_Y + contested * SUMMARY_SCALE, 0.0],
                     color=YELLOW, stroke_width=3)
        axis = Line([-5.6, SUMMARY_BASELINE_Y, 0.0], [5.6, SUMMARY_BASELINE_Y, 0.0],
                    color=GREY_B, stroke_width=2)
        return VGroup(axis, bars, labels, level)

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
        self.wait(1.0)

        # Every category at parity: nine identical bars, and so nine identical marginals.
        marginals = VGroup(*[
            always_redraw(lambda index=index: self._build_marginal_readout(index))
            for index in range(CATEGORY_COUNT)
        ])
        self.add(marginals)
        self.wait(2.0)

        # One bar, swept the whole way across. Its number rises to a peak exactly at parity --
        # where the bar stands on the mean, which is the tallest the bell ever is -- and falls
        # away on BOTH sides, which is why a category is worth most when it is closest to even.
        self.sweeping = True
        height_marker = always_redraw(self._build_height_marker)
        self.add(height_marker)
        self.play(self.swept_effort.animate.set_value(SWEEP_LOW), run_time=2.0)
        self.wait(0.8)
        self.play(self.swept_effort.animate.set_value(SWEEP_HIGH), run_time=5.0)
        self.wait(0.8)
        self.play(self.swept_effort.animate.set_value(OPPONENT_EFFORT), run_time=1.6)
        self.wait(1.4)
        self.remove(height_marker)
        self.sweeping = False

        # Now walk to the answer the punt search found and read the nine numbers off it.
        self.play(self.progress.animate.set_value(PUNT_OPTIMUM), run_time=3.0)
        self.wait(2.0)

        self.play(FadeOut(titles), FadeOut(bells), FadeOut(shading), FadeOut(marginals),
                  run_time=0.9)
        summary = self._build_summary_bars()
        self.play(Create(summary[0]), run_time=0.5)
        self.play(FadeIn(summary[1], lag_ratio=0.08), FadeIn(summary[2]), run_time=1.4)
        self.wait(1.0)
        self.play(Create(summary[3]), run_time=1.0)
        self.wait(3.0)
