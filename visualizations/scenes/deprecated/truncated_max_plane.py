"""Scene A: the whole model, drawn in a plane, with two real categories.

Two categories means a player IS a point, which is the only way to see what the truncated-max
model actually claims: the pool is whoever is left under the value bar, and the weights never
touch a player's numbers -- they only decide which of those survivors gets picked. Rotating w
and watching the pick jump from one dot to another is the entire mechanism.

    manim -ql visualizations/scenes/truncated_max_plane.py TruncatedMaxPlane
"""

from __future__ import annotations

import numpy as np
from manim import (
    Scene, VGroup, Dot, Circle, Arrow, Line, DashedLine, Text, MathTex, ValueTracker,
    FadeIn, FadeOut, Create, Write, always_redraw,
    DOWN, LEFT, UP, UL,
    GREY_B, RED_B, WHITE, YELLOW,
)

from truncated_max_base import (
    CAPTION_COLOUR, DRAFTED_COLOUR, PICK_COLOUR, READOUT_FONT_SIZE, SURVIVOR_COLOUR,
    TITLE_FONT_SIZE, VALUE_COLOUR, WEIGHT_COLOUR,
    CategoryPlane, build_caption, interpolate_along_angle_grid, load_truncated_max_data,
    weights_at_angle,
)


PLANE_CENTRE = (-3.55, -0.12)
PLANE_SCALE = 0.98            # frame units per standard deviation
PLANE_HALF_EXTENT = 3.0       # standard deviations shown either side of the average player

CAPTION_Y = -3.48
TITLE_Y = 3.55
READOUT_ANCHOR = np.array([0.55, 2.60, 0.0])    # upper-left corner of the right-hand column
RIGHT_PANEL_CENTRE_X = 3.55

ARROW_LENGTH = 2.35           # how far the direction arrows reach, in standard deviations
DOT_RADIUS = 0.075
CLOUD_DOT_RADIUS = 0.038

SWEEP_START_DEGREES = 0.0     # all the weight on the first category...
SWEEP_END_DEGREES = 90.0      # ...through to all of it on the second
SWEEP_SECONDS_PER_DEGREE = 0.075
SWEEP_SAMPLES = 1200          # resolution of the scan for where the pick changes hands

# The model's prediction is drawn in its own colour: it lands almost on the value bar, which
# is yellow, and two yellow curves on top of each other would hide the one fact the overlay
# exists to show.
MODEL_COLOUR = RED_B


class TruncatedMaxPlane(Scene):
    """One pool of twenty-five survivors, and the pick walking across it as w turns."""

    def setup(self) -> None:
        self.prepared = load_truncated_max_data()
        self.plane = CategoryPlane(PLANE_CENTRE, PLANE_SCALE, PLANE_HALF_EXTENT)

        self.category_names = self.prepared['categories']
        self.value_direction = np.array(self.prepared['value_direction'])
        self.angle_grid = np.array(self.prepared['angle_grid_degrees'])

        drawn_points = np.array(self.prepared['plane']['drawn_points'])
        is_survivor = np.array(self.prepared['plane']['is_survivor'])
        self.survivors = drawn_points[is_survivor]
        self.drafted = drawn_points[~is_survivor]
        self.draw_count = len(drawn_points)

        self.weight_angle = ValueTracker(SWEEP_START_DEGREES)

    # ── Geometry the scene keeps asking for ───────────────────────────────────────────

    def current_weights(self) -> np.ndarray:
        return weights_at_angle(self.weight_angle.get_value())

    def index_of_pick(self) -> int:
        """Which survivor has the largest score under the weights as they stand."""
        return int((self.survivors @ self.current_weights()).argmax())

    def angles_where_the_pick_changes(self) -> list[float]:
        """Every angle on the sweep at which the pick changes hands.

        Deterministic: the pool is fixed and the sweep is fixed, so these are properties of
        the picture rather than claims about draft night. Scanned rather than asserted
        because the sweep pauses on each of them -- a jump that goes past at full speed is a
        flicker, and the jump is the point of the scene.
        """
        angles = np.linspace(SWEEP_START_DEGREES, SWEEP_END_DEGREES, SWEEP_SAMPLES)
        picks = [int((self.survivors @ weights_at_angle(angle)).argmax()) for angle in angles]
        return [float(angles[position + 1])
                for position, (earlier, later) in enumerate(zip(picks, picks[1:]))
                if earlier != later]

    # ── Furniture ─────────────────────────────────────────────────────────────────────

    def build_direction_arrow(
        self
        , direction
        , colour
    ):
        return Arrow(
            self.plane.point_at([0.0, 0.0]),
            self.plane.point_at(ARROW_LENGTH * np.asarray(direction)),
            buff=0,
            color=colour,
            stroke_width=6,
            max_tip_length_to_length_ratio=0.14,
        )

    def build_weight_arrow(self) -> Arrow:
        return self.build_direction_arrow(self.current_weights(), WEIGHT_COLOUR)

    def build_weight_label(self) -> MathTex:
        """The w beside the arrow's tip, held off the arrow's own line.

        Straight out along w would put the label on the horizontal axis at nought degrees,
        on top of the category name printed there; a nudge to the left of the arrow keeps it
        clear at every angle of the sweep.
        """
        weights = self.current_weights()
        offset = (ARROW_LENGTH + 0.30) * weights + 0.42 * np.array([-weights[1], weights[0]])
        label = MathTex(r'w', font_size=40, color=WEIGHT_COLOUR)
        label.move_to(self.plane.point_at(offset))
        return label

    def build_pick_marker(self) -> VGroup:
        """The ring round the chosen survivor, plus the level line it sits on.

        The level line is what makes the choice look like a decision rather than a colour: the
        pick is the survivor the sliding perpendicular touches first on its way in from the
        far side, and every dot behind that line lost.
        """
        pick = self.survivors[self.index_of_pick()]
        weights = self.current_weights()
        ring = Circle(radius=0.22, color=PICK_COLOUR, stroke_width=4)
        ring.move_to(self.plane.point_at(pick))
        level_line = self.plane.build_clipped_line(
            pick, np.array([-weights[1], weights[0]]), PICK_COLOUR, stroke_width=2, dashed=True)
        return VGroup(level_line, ring)

    def build_readout(self) -> VGroup:
        """The right-hand column: what the weights are, and who they choose."""
        weights = weights_at_angle(self.weight_angle.get_value())
        pick = self.survivors[self.index_of_pick()]

        rows = VGroup(
            MathTex(r's = w^{\top} z', font_size=34, color=WEIGHT_COLOUR),
            Text(f'{self.category_names[0]}   {weights[0]:.2f}',
                 font_size=READOUT_FONT_SIZE, color=GREY_B),
            Text(f'{self.category_names[1]}   {weights[1]:.2f}',
                 font_size=READOUT_FONT_SIZE, color=GREY_B),
            Text('the pick', font_size=READOUT_FONT_SIZE + 3, color=PICK_COLOUR),
            Text(f'{self.category_names[0]}   {pick[0]:+.2f}',
                 font_size=READOUT_FONT_SIZE, color=GREY_B),
            Text(f'{self.category_names[1]}   {pick[1]:+.2f}',
                 font_size=READOUT_FONT_SIZE, color=GREY_B),
        )
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.20)
        # The two blocks are one column with a wider gap between them rather than two
        # separately positioned groups, so the numbers stay aligned as they change width.
        rows[3:].shift(DOWN * 0.36)
        rows.move_to(READOUT_ANCHOR, aligned_edge=UL)
        return rows

    # ── Act one: a player is a point ──────────────────────────────────────────────────

    def play_act_one_scatter(self) -> None:
        title = Text('One pick, drawn in two real categories',
                     font_size=TITLE_FONT_SIZE, color=WHITE).move_to([0.0, TITLE_Y, 0.0])
        axes = self.plane.build_axes(self.category_names[0], self.category_names[1])
        self.play(FadeIn(title), Create(axes), run_time=1.2)
        self.title = title
        self.axes = axes

        # Parked in the empty right-hand column rather than under the title: the title band is
        # already two lines deep with the vertical axis label poking into it.
        correlation_note = Text(
            f'category values as z-scores.\n'
            f'their real correlation is '
            f'{self.prepared["category_correlation"]:+.3f}',
            font_size=20, color=CAPTION_COLOUR, line_spacing=0.9)
        correlation_note.move_to([RIGHT_PANEL_CENTRE_X, 1.1, 0.0])

        self.survivor_dots = VGroup(*[
            Dot(self.plane.point_at(point), radius=DOT_RADIUS, color=SURVIVOR_COLOUR)
            for point in self.survivors
        ])
        self.drafted_dots = VGroup(*[
            Dot(self.plane.point_at(point), radius=DOT_RADIUS, color=SURVIVOR_COLOUR)
            for point in self.drafted
        ])
        caption = build_caption('a player is a point: one number per category', CAPTION_Y)
        self.play(FadeIn(self.survivor_dots, scale=0.4), FadeIn(self.drafted_dots, scale=0.4),
                  FadeIn(caption), run_time=1.4)
        self.caption = caption
        self.play(FadeIn(correlation_note), run_time=0.6)
        self.wait(1.4)
        self.play(FadeOut(correlation_note), run_time=0.5)

    # ── Act two: the value direction, and what projecting onto it means ───────────────

    def play_act_two_value_direction(self) -> None:
        self.swap_caption('every team agrees roughly who is good: that is the direction v')
        self.value_arrow = self.build_direction_arrow(self.value_direction, VALUE_COLOUR)
        value_label = MathTex(r'v', font_size=40, color=VALUE_COLOUR)
        value_label.move_to(self.plane.point_at(
            (ARROW_LENGTH + 0.42) * self.value_direction))
        self.play(Create(self.value_arrow), FadeIn(value_label), run_time=1.0)
        self.value_label = value_label
        self.wait(0.8)

        # Projection demonstrated on the single most valuable player drawn, because the foot of
        # that projection is furthest from the origin and the construction is legible at a
        # glance. Any other point would make the same picture, smaller.
        all_points = np.vstack([self.survivors, self.drafted])
        most_valuable = all_points[(all_points @ self.value_direction).argmax()]
        foot = float(most_valuable @ self.value_direction) * self.value_direction

        dropped = DashedLine(self.plane.point_at(most_valuable), self.plane.point_at(foot),
                             color=VALUE_COLOUR, stroke_width=2, dash_length=0.1)
        projection = Line(self.plane.point_at([0.0, 0.0]), self.plane.point_at(foot),
                          color=VALUE_COLOUR, stroke_width=8)
        highlight = Circle(radius=0.18, color=VALUE_COLOUR, stroke_width=3)
        highlight.move_to(self.plane.point_at(most_valuable))

        value_formula = MathTex(r'u = v^{\top} z', font_size=42, color=VALUE_COLOUR)
        value_formula.move_to([RIGHT_PANEL_CENTRE_X, 1.5, 0.0])
        value_note = Text('how far along v a player sits', font_size=20, color=CAPTION_COLOUR)
        value_note.next_to(value_formula, DOWN, buff=0.3)

        self.play(FadeIn(highlight), Create(dropped), run_time=0.8)
        self.play(Create(projection), Write(value_formula), run_time=1.0)
        self.play(FadeIn(value_note), run_time=0.5)
        self.wait(1.8)
        self.play(FadeOut(highlight), FadeOut(dropped), FadeOut(projection),
                  FadeOut(value_note), run_time=0.7)
        self.value_formula = value_formula

    # ── Act three: the bar, and who is already gone ───────────────────────────────────

    def play_act_three_value_bar(self) -> None:
        self.swap_caption('by the time my pick comes round, everyone above a value bar is gone')
        bar = self.plane.build_clipped_line(
            np.zeros(2), np.array([-self.value_direction[1], self.value_direction[0]]),
            VALUE_COLOUR, stroke_width=4)
        shading = self.plane.build_shaded_half_plane(self.value_direction, DRAFTED_COLOUR,
                                                     opacity=0.55)
        # Behind everything: the shading says "these are gone", and it would be saying it over
        # the axis labels and the arrows if it were drawn on top of them.
        shading.set_z_index(-1)
        bar_label = MathTex(r'u = 0', font_size=32, color=VALUE_COLOUR)
        bar_label.move_to(self.plane.point_at([2.3, 2.35]))

        self.play(Create(bar), FadeIn(shading), FadeIn(bar_label), run_time=1.2)
        self.play(self.drafted_dots.animate.set_color(DRAFTED_COLOUR),
                  FadeOut(self.value_arrow), FadeOut(self.value_label),
                  run_time=0.9)
        self.bar = bar
        self.shading = shading
        self.bar_label = bar_label

        pool_note = Text(
            f'{self.draw_count} drawn,\n'
            f'{len(self.survivors)} still available:\n'
            f'the pool is M = {self.prepared["pick_pool_size"]}',
            font_size=22, color=WHITE, line_spacing=0.9)
        pool_note.move_to([RIGHT_PANEL_CENTRE_X, 1.35, 0.0])
        self.play(FadeOut(self.value_formula), FadeIn(pool_note), run_time=0.8)
        self.wait(1.8)
        self.play(FadeOut(pool_note), run_time=0.5)

    # ── Act four: the weights choose, and only choose ─────────────────────────────────

    def play_act_four_weights_choose(self) -> None:
        self.swap_caption('my own weights w decide which survivor I take')
        self.weight_arrow = always_redraw(self.build_weight_arrow)
        self.weight_label = always_redraw(self.build_weight_label)
        self.pick_marker = always_redraw(self.build_pick_marker)
        self.readout = always_redraw(self.build_readout)

        self.add(self.weight_arrow, self.weight_label)
        self.play(FadeIn(self.readout), run_time=0.7)
        self.wait(0.5)
        self.add(self.pick_marker)
        self.wait(1.8)

        self.swap_caption('the survivor furthest along w -- not the best one, the furthest one')
        self.wait(1.8)

    # ── Act five: the jump ────────────────────────────────────────────────────────────

    def play_act_five_rotate(self) -> None:
        self.swap_caption('now turn w. The dots never move.')
        self.wait(0.6)

        # Pausing just past each change of hands rather than sweeping straight through: the
        # whole claim is that the weights act only by selecting, and a viewer has to be given
        # the half second it takes to notice the ring land somewhere else.
        change_count = 0
        for change_angle in self.angles_where_the_pick_changes():
            self.sweep_to(min(change_angle + 2.0, SWEEP_END_DEGREES))
            self.wait(0.7)
            change_count += 1
        self.sweep_to(SWEEP_END_DEGREES)
        self.wait(0.8)
        self.sweep_to(SWEEP_START_DEGREES, seconds_per_degree=0.045)
        self.wait(0.6)

        verdict = Text(
            f'same pool, same players: the pick changed hands {change_count} times',
            font_size=25, color=YELLOW).move_to([0.0, CAPTION_Y, 0.0])
        self.play(FadeOut(self.caption), FadeIn(verdict, shift=UP * 0.15), run_time=0.9)
        self.caption = verdict
        self.wait(2.2)

    # ── Act six: average over pools, and the centre slides ────────────────────────────

    def play_act_six_expected_pick(self) -> None:
        pool_count = self.prepared['plane']['selection_pool_count']
        self.swap_caption(
            f'draft night is not one pool. Average the pick over {pool_count:,} of them.')

        self.remove(self.pick_marker)
        self.play(FadeOut(self.survivor_dots), FadeOut(self.drafted_dots),
                  FadeOut(self.readout), FadeOut(self.bar_label),
                  self.shading.animate.set_opacity(0.28), run_time=0.8)

        clouds = np.array(self.prepared['plane']['selection_cloud_by_angle'])
        centres = np.array(self.prepared['plane']['selection_mean_by_angle'])
        model_picks = np.array(self.prepared['plane']['model_pick_by_angle'])

        def cloud_positions() -> np.ndarray:
            return interpolate_along_angle_grid(self.angle_grid, clouds,
                                                self.weight_angle.get_value())

        def place_cloud(group: VGroup) -> None:
            for dot, point in zip(group, cloud_positions()):
                dot.move_to(self.plane.point_at(point))
                # Selections past the edge of the plotted square are hidden rather than
                # clamped onto the border, where they would pile up into a fake edge.
                dot.set_opacity(0.45 if np.abs(point).max() <= PLANE_HALF_EXTENT else 0.0)

        cloud_dots = VGroup(*[
            Dot(self.plane.point_at(point), radius=CLOUD_DOT_RADIUS, color=PICK_COLOUR)
            for point in cloud_positions()
        ])
        place_cloud(cloud_dots)
        self.play(FadeIn(cloud_dots), run_time=1.0)
        cloud_dots.add_updater(place_cloud)
        self.wait(0.6)

        def centre_position() -> np.ndarray:
            return interpolate_along_angle_grid(self.angle_grid, centres,
                                                self.weight_angle.get_value())

        centre_marker = always_redraw(
            lambda: Dot(self.plane.point_at(centre_position()), radius=0.13, color=WHITE))
        centre_label = always_redraw(lambda: MathTex(
            r'x(w)', font_size=36, color=WHITE
        ).move_to(self.plane.point_at(centre_position() + np.array([0.66, 0.46]))))
        self.swap_caption('the centre of that cloud is the expected pick, x(w)')
        self.add(centre_marker, centre_label)
        self.wait(1.6)

        # The model's own answer, laid over the simulated one. The two being the same curve is
        # the claim the whole module makes; drawing the prediction as a path the centre then
        # walks along is the cheapest way to let a viewer check it.
        model_path = self.plane.build_curve(model_picks, MODEL_COLOUR, stroke_width=4)
        model_caption = Text('the shipped model,\npredicting the same curve\nin closed form',
                             font_size=21, color=MODEL_COLOUR, line_spacing=0.9)
        model_caption.move_to([RIGHT_PANEL_CENTRE_X, 2.1, 0.0])
        self.play(Create(model_path), FadeIn(model_caption), run_time=1.4)
        self.wait(1.2)

        # What the tilt costs, in the units the bar is drawn in. The path hugging the bar is
        # not the same as sitting on it, and this is the number that says how far below.
        def build_value_cost() -> Text:
            predicted = interpolate_along_angle_grid(
                self.angle_grid, model_picks, self.weight_angle.get_value())
            return Text(
                f'value given up by this pick\n'
                f'{-float(predicted @ self.value_direction):.2f} standard deviations',
                font_size=21, color=VALUE_COLOUR, line_spacing=0.9,
            ).move_to([RIGHT_PANEL_CENTRE_X, -0.6, 0.0])

        value_cost = always_redraw(build_value_cost)
        self.add(value_cost)
        self.wait(0.8)

        self.swap_caption('turning w slides the whole cloud along it -- and pays for the tilt')
        self.sweep_to(SWEEP_END_DEGREES, seconds_per_degree=0.062)
        self.wait(0.8)
        self.sweep_to(45.0, seconds_per_degree=0.055)
        self.wait(0.6)

        cloud_dots.clear_updaters()
        closing = MathTex(
            r'x(w) \;=\; \mathbb{E}\big[\, z \;\big|\; u \le 0, \;\;'
            r's = \max(s_1, \ldots, s_M) \,\big]',
            font_size=36, color=WHITE)
        closing.move_to([0.0, CAPTION_Y, 0.0])
        self.play(FadeOut(self.caption), run_time=0.4)
        self.play(Write(closing), run_time=1.8)
        self.wait(2.6)

    # ── Helpers ───────────────────────────────────────────────────────────────────────

    def sweep_to(
        self
        , angle_degrees
        , seconds_per_degree=SWEEP_SECONDS_PER_DEGREE
    ) -> None:
        """Turn the weights to an angle at a fixed angular rate.

        Rate rather than duration, so that a sweep broken into segments by the pick changing
        hands still turns at one speed throughout instead of hurrying the short legs.
        """
        travel = abs(angle_degrees - self.weight_angle.get_value())
        if travel < 1e-9:
            return
        self.play(self.weight_angle.animate.set_value(angle_degrees),
                  run_time=max(0.3, travel * seconds_per_degree))

    def swap_caption(
        self
        , message
    ) -> None:
        replacement = build_caption(message, CAPTION_Y)
        self.play(FadeOut(self.caption), FadeIn(replacement), run_time=0.6)
        self.caption = replacement

    def construct(self) -> None:
        self.play_act_one_scatter()
        self.play_act_two_value_direction()
        self.play_act_three_value_bar()
        self.play_act_four_weights_choose()
        self.play_act_five_rotate()
        self.play_act_six_expected_pick()
