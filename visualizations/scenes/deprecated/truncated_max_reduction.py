"""Scene B: the same pool, redrawn in the only two coordinates that matter.

Nine categories -- or two, here -- collapse to a score s = w'z and a value u = v'z, and the
pair is bivariate normal. Replotting the SAME dots in (s, u) is the point: no player has
changed, the axes have. Once they are in that frame the picture is one tilted ellipse cut by
a horizontal line, and the only things the weights can move are the ellipse's width and its
tilt -- sigma_s and rho. That is the reduction the shipped model is built on.

    manim -ql visualizations/scenes/truncated_max_reduction.py TruncatedMaxReduction
"""

from __future__ import annotations

import numpy as np
from manim import (
    Scene, VGroup, Dot, Line, Text, MathTex, DecimalNumber, ValueTracker,
    FadeIn, FadeOut, Create, Write, always_redraw,
    DOWN, LEFT, RIGHT, UP,
    GREY_B, WHITE, YELLOW,
)

from truncated_max_base import (
    CAPTION_COLOUR, DRAFTED_COLOUR, PICK_COLOUR, SURVIVOR_COLOUR, TITLE_FONT_SIZE,
    VALUE_COLOUR, WEIGHT_COLOUR,
    CategoryPlane, build_caption, interpolate_along_angle_grid, load_truncated_max_data,
    weights_at_angle,
)


JOINT_CENTRE = (-3.62, 0.55)
JOINT_SCALE = 0.80            # frame units per standard deviation
JOINT_HALF_EXTENT = 2.8

# The two marginals hang off the joint plot and share its mappings, so a dot and its shadow
# on either margin always line up.
SCORE_MARGINAL_BASELINE_Y = -3.15
VALUE_MARGINAL_BASELINE_X = -6.06
MARGINAL_HEIGHT = 1.6         # frame units per unit of probability density

CAPTION_Y = -3.62
TITLE_Y = 3.55
RIGHT_PANEL_CENTRE_X = 3.25

DOT_RADIUS = 0.075
ELLIPSE_RADIUS = 1.9          # which contour of the joint normal gets drawn, in sigmas
ELLIPSE_SAMPLES = 120

SWEEP_START_DEGREES = 0.0
SWEEP_END_DEGREES = 90.0
NEUTRAL_DEGREES = 45.0        # w parallel to v: score and value become the same variable


class TruncatedMaxReduction(Scene):
    """Fifty players, two coordinates, and the two numbers the weights are allowed to move."""

    def setup(self) -> None:
        self.prepared = load_truncated_max_data()
        self.joint = CategoryPlane(JOINT_CENTRE, JOINT_SCALE, JOINT_HALF_EXTENT)

        self.category_names = self.prepared['categories']
        self.value_direction = np.array(self.prepared['value_direction'])
        self.angle_grid = np.array(self.prepared['angle_grid_degrees'])
        self.score_standard_deviations = np.array(
            self.prepared['score_standard_deviation_by_angle'])
        self.correlations = np.array(self.prepared['correlation_by_angle'])
        self.value_standard_deviation = self.prepared['value_standard_deviation']

        self.drawn_points = np.array(self.prepared['plane']['drawn_points'])
        self.is_survivor = np.array(self.prepared['plane']['is_survivor'])

        reduction = self.prepared['reduction']
        self.score_grid = np.array(reduction['score_grid'])
        self.density_before_conditioning = np.array(
            reduction['score_density_before_conditioning'])
        self.density_after_conditioning = np.array(
            reduction['score_density_after_conditioning'])
        self.value_density = np.array(reduction['value_density'])

        self.weight_angle = ValueTracker(SWEEP_START_DEGREES)

    # ── The two numbers, at the angle the sweep has reached ───────────────────────────

    def current_score_standard_deviation(self) -> float:
        return float(np.interp(self.weight_angle.get_value(), self.angle_grid,
                               self.score_standard_deviations))

    def current_correlation(self) -> float:
        return float(np.interp(self.weight_angle.get_value(), self.angle_grid,
                               self.correlations))

    def standardised_pairs(self) -> np.ndarray:
        """Every drawn player as (s / sigma_s, u / sigma_u).

        Standardised rather than raw, because then the cloud's tilt IS rho and nothing else:
        sigma_s and sigma_u are only the units the two axes are printed in, and dividing them
        out leaves a picture in which the single shape parameter is visible as a shape.
        """
        weights = weights_at_angle(self.weight_angle.get_value())
        scores = (self.drawn_points @ weights) / self.current_score_standard_deviation()
        values = (self.drawn_points @ self.value_direction) / self.value_standard_deviation
        return np.stack([scores, values], axis=1)

    # ── Furniture ─────────────────────────────────────────────────────────────────────

    def build_joint_dots(self) -> VGroup:
        """The pool in (s, u), coloured exactly as scene A coloured it.

        Players whose score runs off the edge of the plotted square are left out rather than
        drawn outside it, where they would land on top of the marginal hanging off that side.
        """
        return VGroup(*[
            Dot(self.joint.point_at(pair), radius=DOT_RADIUS,
                color=SURVIVOR_COLOUR if survived else DRAFTED_COLOUR)
            for pair, survived in zip(self.standardised_pairs(), self.is_survivor)
            if np.abs(pair).max() <= JOINT_HALF_EXTENT
        ])

    def build_contour_ellipse(self) -> VGroup:
        """One contour of the joint normal, drawn from the correlation alone.

        A Cholesky factor turns the unit circle into the contour: rows [1, 0] and
        [rho, sqrt(1 - rho^2)]. At rho = 1 it collapses to the diagonal line, which is not a
        drawing artefact -- it is the statement that score and value have become one variable.
        """
        correlation = self.current_correlation()
        angles = np.linspace(0.0, 2.0 * np.pi, ELLIPSE_SAMPLES)
        circle = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        factor = np.array([[1.0, 0.0],
                           [correlation, np.sqrt(max(1.0 - correlation ** 2, 0.0))]])
        return self.joint.build_curve(ELLIPSE_RADIUS * circle @ factor.T,
                                      WEIGHT_COLOUR, stroke_width=3)

    def build_value_marginal(self) -> VGroup:
        """The standard normal of u, lying on its side against the left edge.

        Split at nought: the half above the bar is drafted and shaded out, the half below is
        the pool. It is the same cut as scene A's shaded half plane, seen end-on.
        """
        heights = [
            np.array([VALUE_MARGINAL_BASELINE_X - density * MARGINAL_HEIGHT,
                      self.joint.point_at([0.0, value])[1], 0.0])
            for value, density in zip(self.score_grid, self.value_density)
            if abs(value) <= JOINT_HALF_EXTENT
        ]
        curve = VGroup(*[
            Line(start, end,
                 # Lighter than the drafted dots: a dot sits on a black ground, this half of
                 # the curve has to stay readable as a curve while still reading as gone.
                 color=GREY_B if start[1] > self.joint.point_at([0.0, 0.0])[1]
                 else VALUE_COLOUR,
                 stroke_width=4)
            for start, end in zip(heights, heights[1:])
        ])
        bottom_y = self.joint.point_at([0.0, -JOINT_HALF_EXTENT])[1]
        top_y = self.joint.point_at([0.0, JOINT_HALF_EXTENT])[1]
        baseline = Line([VALUE_MARGINAL_BASELINE_X, bottom_y, 0.0],
                        [VALUE_MARGINAL_BASELINE_X, top_y, 0.0],
                        color=GREY_B, stroke_width=2)
        return VGroup(baseline, curve)

    def build_score_marginal_points(
        self
        , densities
    ) -> list[np.ndarray]:
        positions = []
        for score, density in zip(self.score_grid, densities):
            if abs(score) > JOINT_HALF_EXTENT:
                continue
            positions.append(np.array([
                self.joint.point_at([score, 0.0])[0],
                SCORE_MARGINAL_BASELINE_Y + density * MARGINAL_HEIGHT,
                0.0,
            ]))
        return positions

    def build_conditioned_score_marginal(self) -> VGroup:
        """The survivor's score density at the current weights: skew-normal, shape -rho/sqrt(1-rho^2)."""
        densities = interpolate_along_angle_grid(
            self.angle_grid, self.density_after_conditioning, self.weight_angle.get_value())
        positions = self.build_score_marginal_points(densities)
        return VGroup(*[
            Line(start, end, color=PICK_COLOUR, stroke_width=4)
            for start, end in zip(positions, positions[1:])
        ])

    def build_readout(self) -> VGroup:
        """The whole of the weights' influence, in two numbers.

        Built once with an updater on each number rather than redrawn whole: the two symbols
        beside them are LaTeX, and recompiling those every frame costs more than the rest of
        the scene put together.
        """
        score_number = DecimalNumber(self.current_score_standard_deviation(),
                                     num_decimal_places=3, font_size=36, color=WEIGHT_COLOUR)
        score_number.add_updater(
            lambda number: number.set_value(self.current_score_standard_deviation()))
        correlation_number = DecimalNumber(self.current_correlation(),
                                           num_decimal_places=3, font_size=36,
                                           color=WEIGHT_COLOUR)
        correlation_number.add_updater(
            lambda number: number.set_value(self.current_correlation()))

        rows = VGroup(
            VGroup(MathTex(r'\sigma_s =', font_size=42, color=WEIGHT_COLOUR), score_number
                   ).arrange(RIGHT, buff=0.20),
            VGroup(MathTex(r'\rho =', font_size=42, color=WEIGHT_COLOUR), correlation_number
                   ).arrange(RIGHT, buff=0.20),
        )
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.36)
        rows.move_to([RIGHT_PANEL_CENTRE_X, 0.50, 0.0])
        return rows

    # ── Act one: the same dots, new axes ──────────────────────────────────────────────

    def play_act_one_change_of_coordinates(self) -> None:
        title = Text('The same pool, in two numbers',
                     font_size=TITLE_FONT_SIZE, color=WHITE).move_to([0.0, TITLE_Y, 0.0])
        category_axes = self.joint.build_axes(self.category_names[0], self.category_names[1])
        category_dots = VGroup(*[
            Dot(self.joint.point_at(point), radius=DOT_RADIUS,
                color=SURVIVOR_COLOUR if survived else DRAFTED_COLOUR)
            for point, survived in zip(self.drawn_points, self.is_survivor)
        ])
        caption = build_caption('the fifty players from before, still in their categories',
                                CAPTION_Y)
        self.play(FadeIn(title), Create(category_axes), run_time=1.0)
        self.play(FadeIn(category_dots), FadeIn(caption), run_time=0.9)
        self.title = title
        self.caption = caption
        self.wait(1.0)

        definitions = VGroup(
            MathTex(r's = w^{\top} z', font_size=44, color=WEIGHT_COLOUR),
            Text('what this team scores them', font_size=20, color=CAPTION_COLOUR),
            MathTex(r'u = v^{\top} z', font_size=44, color=VALUE_COLOUR),
            Text('what everyone scores them', font_size=20, color=CAPTION_COLOUR),
        )
        definitions.arrange(DOWN, buff=0.24)
        definitions[2:].shift(DOWN * 0.30)
        definitions.move_to([RIGHT_PANEL_CENTRE_X, 1.75, 0.0])
        self.play(Write(definitions[0]), FadeIn(definitions[1]), run_time=1.0)
        self.play(Write(definitions[2]), FadeIn(definitions[3]), run_time=1.0)
        self.definitions = definitions
        self.wait(1.2)

        # The move that names the scene: every dot slides to its own (s, u) while the axes
        # underneath are relabelled. Nothing is added and nothing is dropped, which is the
        # only way to make "these are the same players" believable rather than asserted.
        self.swap_caption('give each player those two numbers, and plot THOSE')
        # Each axis in its own standard deviations, said once on screen rather than carried
        # in the axis labels: standardised is what makes the cloud's tilt equal rho, and a
        # label reading "s over sigma s" buries that behind a division.
        reduced_axes = self.joint.build_axes('score  s', 'value  u')
        standardised_note = Text('each axis in its own standard deviations',
                                 font_size=20, color=CAPTION_COLOUR)
        standardised_note.move_to([RIGHT_PANEL_CENTRE_X, -0.55, 0.0])
        self.play(
            *[dot.animate.move_to(self.joint.point_at(pair))
              for dot, pair in zip(category_dots, self.standardised_pairs())],
            FadeOut(category_axes), FadeIn(reduced_axes),
            run_time=2.2,
        )
        self.joint_dots = category_dots
        self.reduced_axes = reduced_axes
        self.standardised_note = standardised_note
        self.play(FadeIn(standardised_note), run_time=0.6)
        self.wait(1.2)

    # ── Act two: the bar, seen end-on ─────────────────────────────────────────────────

    def play_act_two_value_bar(self) -> None:
        self.swap_caption('the value bar is now just a horizontal line')
        bar = Line(self.joint.point_at([-JOINT_HALF_EXTENT, 0.0]),
                   self.joint.point_at([JOINT_HALF_EXTENT, 0.0]),
                   color=VALUE_COLOUR, stroke_width=4)
        shading = self.joint.build_shaded_half_plane(np.array([0.0, 1.0]), DRAFTED_COLOUR,
                                                     opacity=0.45)
        shading.set_z_index(-1)
        self.play(Create(bar), FadeIn(shading), run_time=1.0)
        self.bar = bar
        self.wait(0.8)

        self.swap_caption('u is standard normal, and the pool is the half of it below the bar')
        value_marginal = self.build_value_marginal()
        self.play(Create(value_marginal), run_time=1.4)
        self.value_marginal = value_marginal
        self.wait(1.6)

    # ── Act three: what the truncation does to the score ──────────────────────────────

    def play_act_three_score_marginal(self) -> None:
        self.swap_caption('before the bar, the score is normal too')
        unconditioned_positions = self.build_score_marginal_points(
            self.density_before_conditioning)
        unconditioned = VGroup(*[
            Line(start, end, color=SURVIVOR_COLOUR, stroke_width=3)
            for start, end in zip(unconditioned_positions, unconditioned_positions[1:])
        ])
        baseline = Line([unconditioned_positions[0][0], SCORE_MARGINAL_BASELINE_Y, 0.0],
                        [unconditioned_positions[-1][0], SCORE_MARGINAL_BASELINE_Y, 0.0],
                        color=GREY_B, stroke_width=2)
        self.play(Create(baseline), Create(unconditioned), run_time=1.3)
        self.score_baseline = baseline
        self.wait(1.0)

        # The ghost of the unconditioned curve is kept on screen underneath. The claim is a
        # comparison -- high scores are thinner than they were -- and a comparison needs both
        # curves, not one curve replaced by another.
        self.swap_caption('after it, high scorers are missing: they were valuable, so they went early')
        conditioned = always_redraw(self.build_conditioned_score_marginal)
        self.play(unconditioned.animate.set_stroke(opacity=0.35), run_time=0.5)
        still_conditioned = self.build_conditioned_score_marginal()
        self.play(Create(still_conditioned), run_time=1.3)
        self.remove(still_conditioned)
        self.add(conditioned)
        self.unconditioned_score_marginal = unconditioned
        self.conditioned_score_marginal = conditioned
        self.wait(1.4)

        skew_label = MathTex(
            r'f(t) = 2\,\phi(t)\,\Phi(\alpha t), \quad \alpha = \frac{-\rho}{\sqrt{1-\rho^{2}}}',
            font_size=30, color=PICK_COLOUR)
        skew_label.move_to([RIGHT_PANEL_CENTRE_X, -1.55, 0.0])
        skew_note = Text('skew-normal: one shape parameter', font_size=20, color=CAPTION_COLOUR)
        skew_note.next_to(skew_label, DOWN, buff=0.24)
        self.play(Write(skew_label), FadeIn(skew_note), run_time=1.4)
        self.skew_label = VGroup(skew_label, skew_note)
        self.wait(1.6)

    # ── Act four: the ellipse, and the only two numbers that move ─────────────────────

    def play_act_four_rotate(self) -> None:
        self.swap_caption('the cloud is an ellipse, and its tilt is exactly rho')
        ellipse = always_redraw(self.build_contour_ellipse)
        readout = always_redraw(self.build_readout)
        moving_dots = always_redraw(self.build_joint_dots)
        self.play(FadeOut(self.definitions), FadeOut(self.standardised_note), run_time=0.5)
        # Stroked on as a still, then handed to the redrawing copy and taken away: an
        # always_redraw mobject cannot be animated into existence, and the still left behind
        # would hang there as a second, frozen ellipse for the rest of the scene.
        still_ellipse = self.build_contour_ellipse()
        self.play(Create(still_ellipse), FadeIn(readout), run_time=1.2)
        self.remove(still_ellipse, self.joint_dots)
        self.add(ellipse, moving_dots)
        self.wait(1.4)

        # u is fixed for every player -- the field's ranking does not care what I want -- so
        # the dots slide horizontally and nothing else. Everything on screen that moves is
        # downstream of sigma_s and rho.
        self.swap_caption('turn w: every dot slides sideways, because u never changes')
        self.play(self.weight_angle.animate.set_value(NEUTRAL_DEGREES), run_time=4.5)
        self.wait(0.6)

        neutral_note = Text('w = v: score and value are\nthe same number, rho = 1.\n'
                            'the model clamps just short of it',
                            font_size=21, color=YELLOW, line_spacing=0.9)
        neutral_note.move_to([RIGHT_PANEL_CENTRE_X, 2.1, 0.0])
        self.play(FadeIn(neutral_note), run_time=0.8)
        self.wait(2.0)
        self.play(FadeOut(neutral_note), run_time=0.5)

        self.play(self.weight_angle.animate.set_value(SWEEP_END_DEGREES), run_time=4.5)
        self.wait(1.0)
        self.play(self.weight_angle.animate.set_value(18.0), run_time=3.0)
        self.wait(0.8)

    # ── Act five: the payoff ──────────────────────────────────────────────────────────

    def play_act_five_payoff(self) -> None:
        self.play(FadeOut(self.skew_label), run_time=0.5)
        payoff = VGroup(
            Text('nine categories in,', font_size=24, color=CAPTION_COLOUR),
            MathTex(r'(\sigma_s, \rho)', font_size=52, color=WEIGHT_COLOUR),
            Text('two numbers out', font_size=24, color=CAPTION_COLOUR),
        )
        payoff.arrange(DOWN, buff=0.26)
        payoff.move_to([RIGHT_PANEL_CENTRE_X, -1.95, 0.0])
        self.swap_caption('everything the weights can do, they do through these two')
        self.play(FadeIn(payoff, shift=UP * 0.2), run_time=1.2)
        self.wait(2.6)

    # ── Helpers ───────────────────────────────────────────────────────────────────────

    def swap_caption(
        self
        , message
    ) -> None:
        replacement = build_caption(message, CAPTION_Y)
        self.play(FadeOut(self.caption), FadeIn(replacement), run_time=0.6)
        self.caption = replacement

    def construct(self) -> None:
        self.play_act_one_change_of_coordinates()
        self.play_act_two_value_bar()
        self.play_act_three_score_marginal()
        self.play_act_four_rotate()
        self.play_act_five_payoff()
