"""Scene C: the scalar the whole model is standing on.

Once the pool has collapsed to (sigma_s, rho), the only thing left to compute is the expected
best of M draws from one univariate density. This scene is that one axis: the survivor's
skew-normal score, the best-of-M density stacked on top of it, the ledge where the tail is
1/M, and the Gumbel step from that ledge up to the mean. The shipped model's e(rho) is
exactly those last two pieces added together, and its value function is sigma_s times it.

    manim -ql visualizations/scenes/truncated_max_scalar_core.py TruncatedMaxScalarCore
"""

from __future__ import annotations

import numpy as np
from manim import (
    Scene, VGroup, VMobject, Line, Polygon, Text, MathTex, DecimalNumber, ValueTracker,
    FadeIn, FadeOut, Create, Write, always_redraw,
    DOWN, LEFT, RIGHT,
    GREY_B, ORANGE, WHITE,
)

from truncated_max_base import (
    CAPTION_COLOUR, PICK_COLOUR, SURVIVOR_COLOUR, TITLE_FONT_SIZE, VALUE_COLOUR,
    WEIGHT_COLOUR,
    build_caption, load_truncated_max_data,
)


# The plotted window of standardised score. Narrower than the prepared grid on both sides:
# every pool size the scene animates through puts its mass inside this, and the wider the
# window the smaller the Gumbel step becomes on screen -- and that step is the whole act.
WINDOW_LEFT, WINDOW_RIGHT = -2.4, 3.6
AXIS_LEFT_X, AXIS_RIGHT_X = -6.2, 6.2
BASELINE_Y = -2.05
DENSITY_SCALE = 2.4           # frame units per unit of probability density

TAIL_QUANTILE_LINE_TOP = 1.25
EXPECTED_MAX_LINE_TOP = 1.80

CAPTION_Y = -3.70
TITLE_Y = 3.55
LEFT_COLUMN_X = -4.85
RIGHT_COLUMN_X = 3.75
MAX_DENSITY_LABEL_X = 4.20

STARTING_POOL_SIZE = 5.0
DEFAULT_POOL_SIZE = 25.0      # the app default
LARGE_POOL_SIZE = 100.0

# The ledge and the step are different kinds of thing -- one is a quantile of the parent
# density, the other is the mean of the best-of-M -- so they never share a colour.
TAIL_QUANTILE_COLOUR = ORANGE
EXPECTED_MAX_COLOUR = VALUE_COLOUR


class TruncatedMaxScalarCore(Scene):
    """One axis, three densities, and the two-term approximation the model actually ships."""

    def setup(self) -> None:
        self.prepared = load_truncated_max_data()
        core = self.prepared['scalar_core']

        self.reference_correlation = core['reference_correlation']
        self.shape = core['shape']
        self.scalar_grid = np.array(core['scalar_grid'])
        self.parent_density = np.array(core['skew_normal_density'])
        self.parent_cumulative = np.array(core['skew_normal_cumulative'])
        self.pool_sizes = np.array(core['pool_sizes'])
        self.tail_quantiles = np.array(core['tail_quantile_by_pool_size'])
        self.expected_maxima = np.array(core['expected_max_by_pool_size'])
        self.exact_means = np.array(core['exact_mean_by_pool_size'])

        self.inside_window = ((self.scalar_grid >= WINDOW_LEFT)
                              & (self.scalar_grid <= WINDOW_RIGHT))
        self.pool_size = ValueTracker(STARTING_POOL_SIZE)

    # ── The axis ──────────────────────────────────────────────────────────────────────

    def x_of(
        self
        , standardised_score
    ) -> float:
        fraction = (standardised_score - WINDOW_LEFT) / (WINDOW_RIGHT - WINDOW_LEFT)
        return AXIS_LEFT_X + fraction * (AXIS_RIGHT_X - AXIS_LEFT_X)

    def point_at(
        self
        , standardised_score
        , density
    ) -> np.ndarray:
        return np.array([self.x_of(standardised_score),
                         BASELINE_Y + density * DENSITY_SCALE,
                         0.0])

    def build_axis(self) -> VGroup:
        axis = VGroup(Line([AXIS_LEFT_X, BASELINE_Y, 0.0], [AXIS_RIGHT_X, BASELINE_Y, 0.0],
                           color=GREY_B, stroke_width=2))
        for tick_value in range(int(np.ceil(WINDOW_LEFT)), int(WINDOW_RIGHT) + 1):
            axis.add(Line([self.x_of(tick_value), BASELINE_Y, 0.0],
                          [self.x_of(tick_value), BASELINE_Y - 0.10, 0.0],
                          color=GREY_B, stroke_width=2))
        label = Text('standardised score  t', font_size=19, color=GREY_B)
        label.move_to([self.x_of(WINDOW_RIGHT) - 1.0, BASELINE_Y - 0.42, 0.0])
        axis.add(label)
        return axis

    def build_density_curve(
        self
        , densities
        , colour
        , stroke_width=4
    ) -> VMobject:
        """One density as a single polyline.

        A single VMobject rather than a group of segments: the best-of-M curve is redrawn on
        every frame while M animates, and two hundred separate Lines per frame is what would
        make this scene take minutes to render instead of seconds.
        """
        curve = VMobject()
        curve.set_points_as_corners([
            self.point_at(score, density)
            for score, density in zip(self.scalar_grid[self.inside_window],
                                      np.asarray(densities)[self.inside_window])
        ])
        curve.set_stroke(colour, width=stroke_width)
        return curve

    # ── The quantities, at the pool size the sweep has reached ────────────────────────

    def current_pool_size(self) -> float:
        return self.pool_size.get_value()

    def current_max_density(self) -> np.ndarray:
        """M g(t) G(t)^(M-1): the density of the largest of M independent draws."""
        pool_size = self.current_pool_size()
        return pool_size * self.parent_density * self.parent_cumulative ** (pool_size - 1.0)

    def current_tail_quantile(self) -> float:
        return float(np.interp(self.current_pool_size(), self.pool_sizes, self.tail_quantiles))

    def current_expected_max(self) -> float:
        return float(np.interp(self.current_pool_size(), self.pool_sizes, self.expected_maxima))

    def current_exact_mean(self) -> float:
        return float(np.interp(self.current_pool_size(), self.pool_sizes, self.exact_means))

    # ── Act one: the density one survivor's score is drawn from ───────────────────────

    def play_act_one_parent_density(self) -> None:
        title = Text('One axis, and the number the model actually computes',
                     font_size=TITLE_FONT_SIZE, color=WHITE).move_to([0.0, TITLE_Y, 0.0])
        axis = self.build_axis()
        self.play(FadeIn(title), Create(axis), run_time=1.1)
        self.title = title

        caption = build_caption('one surviving player\'s score, from the last scene', CAPTION_Y)
        parent_curve = self.build_density_curve(self.parent_density, PICK_COLOUR)
        self.play(Create(parent_curve), FadeIn(caption), run_time=1.4)
        self.parent_curve = parent_curve
        self.caption = caption

        parent_label = MathTex(
            r'g(t) = 2\,\phi(t)\,\Phi(\alpha t)', font_size=34, color=PICK_COLOUR)
        # Low on the right, where no density this scene draws ever reaches: at M = 100 the
        # best-of-M curve climbs through everything above this height.
        parent_label.move_to([RIGHT_COLUMN_X, 0.45, 0.0])
        parent_note = Text(f'skew-normal, and rho = {self.reference_correlation:.3f}\n'
                           f'is all that sets its shape',
                           font_size=20, color=CAPTION_COLOUR, line_spacing=0.9)
        parent_note.next_to(parent_label, DOWN, buff=0.26)
        self.play(Write(parent_label), FadeIn(parent_note), run_time=1.2)
        self.parent_label = VGroup(parent_label, parent_note)
        self.wait(1.6)

    # ── Act two: the best of M of them ────────────────────────────────────────────────

    def play_act_two_best_of_pool(self) -> None:
        self.swap_caption('the team takes the best of M of them')

        pool_size_number = DecimalNumber(self.current_pool_size(), num_decimal_places=0,
                                         font_size=54, color=WHITE)
        pool_size_number.add_updater(
            lambda number: number.set_value(self.current_pool_size()))
        pool_size_row = VGroup(MathTex(r'M =', font_size=54, color=WHITE), pool_size_number)
        pool_size_row.arrange(RIGHT, buff=0.22)
        pool_size_row.move_to([LEFT_COLUMN_X, 2.45, 0.0])

        max_density_label = MathTex(r'M\,g(t)\,G(t)^{M-1}', font_size=36, color=SURVIVOR_COLOUR)
        max_density_label.move_to([MAX_DENSITY_LABEL_X, 2.55, 0.0])

        # Drawn once as a still so it can be stroked on, then handed over to the redrawing
        # copy and taken off the scene -- leaving the still in place would strand a frozen
        # curve at the starting pool size for the rest of the video.
        max_curve = always_redraw(
            lambda: self.build_density_curve(self.current_max_density(), SURVIVOR_COLOUR))
        still_curve = self.build_density_curve(self.current_max_density(), SURVIVOR_COLOUR)
        self.play(Create(still_curve),
                  FadeIn(pool_size_row), FadeIn(max_density_label), run_time=1.4)
        self.remove(still_curve)
        self.add(max_curve)
        self.max_curve = max_curve
        self.max_density_label = max_density_label
        self.wait(1.2)

        self.swap_caption('widen the window and the best of it piles up against the top')
        # The parent's algebra goes before the sweep starts. At a hundred the best-of-M curve
        # climbs straight through where that label sits, and the label has done its work.
        self.play(FadeOut(self.parent_label), run_time=0.5)
        self.play(self.pool_size.animate.set_value(DEFAULT_POOL_SIZE), run_time=2.6)
        self.wait(0.9)
        self.play(self.pool_size.animate.set_value(LARGE_POOL_SIZE), run_time=2.6)
        self.wait(1.1)
        self.play(self.pool_size.animate.set_value(DEFAULT_POOL_SIZE), run_time=2.0)

        default_note = Text(f'{DEFAULT_POOL_SIZE:.0f} is what the app uses',
                            font_size=20, color=CAPTION_COLOUR)
        default_note.next_to(pool_size_row, DOWN, buff=0.30)
        self.play(FadeIn(default_note), run_time=0.6)
        self.default_note = default_note
        self.wait(1.2)

    # ── Act three: the ledge where the tail is one in M ───────────────────────────────

    def play_act_three_tail_quantile(self) -> None:
        self.swap_caption('find the score only one in M survivors beats')

        tail_shading = always_redraw(self.build_tail_shading)
        quantile_line = always_redraw(self.build_tail_quantile_line)
        quantile_label = always_redraw(self.build_tail_quantile_label)

        # Faded in as stills and then handed over, for the reason differential_base gives:
        # an entrance animation and an always_redraw updater fight each other, and the
        # updater wins halfway through.
        still = VGroup(self.build_tail_shading(), self.build_tail_quantile_line(),
                       self.build_tail_quantile_label())
        self.play(FadeIn(still), run_time=1.2)
        self.remove(still)
        self.add(tail_shading, quantile_line, quantile_label)

        tail_note = MathTex(r'\int_{t_q}^{\infty} g = \frac{1}{M}',
                            font_size=38, color=TAIL_QUANTILE_COLOUR)
        tail_note.move_to([RIGHT_COLUMN_X, 1.35, 0.0])
        self.play(Write(tail_note), run_time=1.0)
        self.tail_note = tail_note
        self.wait(1.8)

    def build_tail_quantile_line(self) -> Line:
        return Line([self.x_of(self.current_tail_quantile()), BASELINE_Y, 0.0],
                    [self.x_of(self.current_tail_quantile()), TAIL_QUANTILE_LINE_TOP, 0.0],
                    color=TAIL_QUANTILE_COLOUR, stroke_width=4)

    def build_tail_quantile_label(self) -> MathTex:
        return MathTex(r't_q', font_size=38, color=TAIL_QUANTILE_COLOUR).move_to(
            [self.x_of(self.current_tail_quantile()) - 0.30,
             TAIL_QUANTILE_LINE_TOP + 0.28, 0.0])

    def build_tail_shading(self) -> Polygon:
        """The sliver of the parent density above the ledge: one draw in M lands there."""
        quantile = self.current_tail_quantile()
        above = [
            self.point_at(score, density)
            for score, density in zip(self.scalar_grid, self.parent_density)
            if quantile <= score <= WINDOW_RIGHT
        ]
        return Polygon(*above,
                       np.array([self.x_of(WINDOW_RIGHT), BASELINE_Y, 0.0]),
                       np.array([self.x_of(quantile), BASELINE_Y, 0.0]),
                       stroke_width=0, fill_color=TAIL_QUANTILE_COLOUR, fill_opacity=0.55)

    # ── Act four: the Gumbel step from the ledge up to the mean ───────────────────────

    def play_act_four_gumbel_step(self) -> None:
        self.swap_caption('the best of M lands a little above that ledge -- and by how much is known')

        expected_max_line = always_redraw(self.build_expected_max_line)
        expected_max_label = always_redraw(self.build_expected_max_label)
        # The step is a fifth of a standard deviation at the app's pool size, which is a
        # hairline on a six-sigma axis. Filling the strip between the two marks is what makes
        # a quantity that small legible at all; the brace and the algebra sit under it.
        step_band = always_redraw(self.build_step_band)
        step_label = always_redraw(self.build_step_label)

        still = VGroup(self.build_step_band(), self.build_expected_max_line(),
                       self.build_expected_max_label())
        self.play(FadeIn(still), run_time=1.0)
        self.remove(still)
        self.add(step_band, expected_max_line, expected_max_label)

        label_still = self.build_step_label()
        self.play(FadeIn(label_still), run_time=0.8)
        self.remove(label_still)
        self.add(step_label)
        self.step_label = step_label
        self.wait(1.0)

        self.play(FadeOut(self.tail_note), run_time=0.4)
        self.play(FadeIn(self.build_numbers_column()), run_time=0.8)
        self.wait(2.0)

    def build_expected_max_line(self) -> Line:
        return Line([self.x_of(self.current_expected_max()), BASELINE_Y, 0.0],
                    [self.x_of(self.current_expected_max()), EXPECTED_MAX_LINE_TOP, 0.0],
                    color=EXPECTED_MAX_COLOUR, stroke_width=4)

    def build_expected_max_label(self) -> MathTex:
        return MathTex(r'e(\rho)', font_size=38, color=EXPECTED_MAX_COLOUR).move_to(
            [self.x_of(self.current_expected_max()) + 0.40,
             EXPECTED_MAX_LINE_TOP + 0.30, 0.0])

    def build_step_band(self) -> Polygon:
        left_x = self.x_of(self.current_tail_quantile())
        right_x = self.x_of(self.current_expected_max())
        band = Polygon(
            [left_x, BASELINE_Y, 0.0], [right_x, BASELINE_Y, 0.0],
            [right_x, TAIL_QUANTILE_LINE_TOP, 0.0], [left_x, TAIL_QUANTILE_LINE_TOP, 0.0],
            stroke_width=0, fill_color=EXPECTED_MAX_COLOUR, fill_opacity=0.30)
        # Behind the densities: the strip is there to measure them, not to wash them out.
        band.set_z_index(-1)
        return band

    def build_step_label(self) -> MathTex:
        """The Gumbel step, parked under the middle of the strip it measures, and moving with it.

        No brace over it. At the app's pool size the strip is a third of a frame unit wide and
        a brace drawn across it is smaller than the gamma sitting beneath it -- it reads as a
        smudge on the axis, and it lands on the numerator.
        """
        midpoint_x = 0.5 * (self.x_of(self.current_tail_quantile())
                            + self.x_of(self.current_expected_max()))
        return MathTex(r'\frac{\gamma}{M\,g(t_q)}', font_size=34, color=EXPECTED_MAX_COLOUR
                       ).move_to([midpoint_x, -2.80, 0.0])

    def build_numbers_column(self) -> VGroup:
        """The ledge, the step and the mean as live numbers, next to the exact answer.

        The exact mean is on screen beside the approximation deliberately. The model does not
        integrate the best-of-M density -- it adds a quantile and a Gumbel correction -- and
        the animation should show what that costs rather than imply the two are one thing.
        """
        rows = VGroup()
        for label, colour, reader in (
            (r't_q', TAIL_QUANTILE_COLOUR, self.current_tail_quantile),
            (r'e(\rho)', EXPECTED_MAX_COLOUR, self.current_expected_max),
            (r'\text{exact}', GREY_B, self.current_exact_mean),
        ):
            number = DecimalNumber(reader(), num_decimal_places=3, font_size=32, color=colour)
            number.add_updater(lambda live, reader=reader: live.set_value(reader()))
            row = VGroup(MathTex(label, font_size=34, color=colour), number)
            row.arrange(RIGHT, buff=0.22)
            rows.add(row)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.26)
        rows.move_to([LEFT_COLUMN_X, 0.75, 0.0])
        return rows

    # ── Act five: what changing the window does to both terms ─────────────────────────

    def play_act_five_pool_size_sweep(self) -> None:
        self.swap_caption('a smaller window: the ledge drops, and the step above it grows')
        self.play(self.pool_size.animate.set_value(STARTING_POOL_SIZE), run_time=2.6)
        self.wait(1.2)
        self.swap_caption('a bigger one: the ledge climbs, and the step shrinks like 1 / M')
        self.play(self.pool_size.animate.set_value(LARGE_POOL_SIZE), run_time=3.2)
        self.wait(1.2)
        self.play(self.pool_size.animate.set_value(DEFAULT_POOL_SIZE), run_time=2.2)
        self.wait(0.8)

    # ── Act six: back out to the value function ───────────────────────────────────────

    def play_act_six_value_function(self) -> None:
        self.swap_caption('and that scalar, scaled by sigma_s, is the whole value function')
        self.play(FadeOut(self.max_density_label), FadeOut(self.step_label), run_time=0.5)

        value_function = MathTex(r'W(w) = \sigma_s \, e(\rho)', font_size=46, color=WEIGHT_COLOUR)
        value_function.move_to([RIGHT_COLUMN_X, 2.45, 0.0])
        gradient_note = Text('the expected pick is its gradient\nin w, taken exactly',
                             font_size=21, color=CAPTION_COLOUR, line_spacing=0.9)
        gradient_note.next_to(value_function, DOWN, buff=0.34)
        self.play(Write(value_function), run_time=1.3)
        self.play(FadeIn(gradient_note), run_time=0.7)
        self.wait(3.0)

    # ── Helpers ───────────────────────────────────────────────────────────────────────

    def swap_caption(
        self
        , message
    ) -> None:
        replacement = build_caption(message, CAPTION_Y)
        self.play(FadeOut(self.caption), FadeIn(replacement), run_time=0.6)
        self.caption = replacement

    def construct(self) -> None:
        self.play_act_one_parent_density()
        self.play_act_two_best_of_pool()
        self.play_act_three_tail_quantile()
        self.play_act_four_gumbel_step()
        self.play_act_five_pool_size_sweep()
        self.play_act_six_value_function()
