"""What gradient descent is, and why the algorithm does not run it only once.

Two scenes over two measured surfaces: the H-score of a real first pick as two of its category
weights are varied. X and Y are those two weights, each as a percentage of what the category is
worth in a balanced build, and the height is what the shipped objective says the build is worth.
The axes run past a hundred, so the balanced build is a point inside the picture rather than a
corner of it.

The menu scene uses a slice with two summits, because that is what makes a menu worth having:
one build keeps both categories near balanced, and the other -- worth more -- cuts Threes to
thirty percent while holding Turnovers high. A valley lies between them, which is why the best
build gives up one category and keeps the other, and why where a descent STARTS decides which of
the two it finds. The descent scene uses the simplest slice the search could find instead: one
hill, with its top inside the picture.

This module is the surface itself -- how it is loaded, drawn, labelled and climbed. The two
scenes built on it live beside their own narration, in 8_gradient_descent and 7_seed_menu.

Run `python visualizations/prepare_weight_surface_data.py` first. It builds a 2025-26 session,
lets the self-play bootstrap settle the field, and records this surface and these climbs from the
real objective, from inside the bootstrap itself.

    manim -ql visualizations/8_gradient_descent/gradient_descent.py GradientDescent
    manim -ql visualizations/7_seed_menu/seed_menu.py SeedMenu
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from manim import (
    ThreeDScene, ThreeDAxes, Surface, VGroup, VMobject, Dot3D, DashedLine, Text,
    FadeIn, FadeOut, Create,
    DEGREES, DOWN, LEFT, UL, IN, OUT,
    BLUE_E, TEAL_D, GREEN_C, YELLOW_D, YELLOW, WHITE, GREY_B,
)


# ── Layout ───────────────────────────────────────────────────────────────────────────

# The height axis does NOT start at zero. The whole surface lives inside two percent of H-score,
# and an axis from zero would draw every summit and valley as a single flat sheet. The floor is a
# hair under the lowest measured point so nothing is clipped, and the range is printed on screen,
# so the crop is stated rather than hidden.
Z_FLOOR_MARGIN = 0.0004
Z_CEILING_MARGIN = 0.0006

# Both surface scenes draw at this. It is finer than the 31-step grid the surfaces are
# MEASURED on, deliberately: these objectives have creases in them -- the roster assignment
# switching as the weights cross -- and a fold landing between mesh vertices is drawn as a
# facet edge and reads as a notch punched into the hillside, where a finer mesh ramps across
# it and reads as a fold. The two scenes sit on the same page and are read against each
# other, so they are drawn at the same fineness.
SURFACE_RESOLUTION = 56

# The camera stands where both categories have been given up and looks back along the two weight
# axes, so they radiate from the near corner and the balanced build is the far one. Everything
# worth having is uphill from here, which is the direction both scenes are about. It opens a
# little short of square and drifts through it, so the composition is centred on the whole shot
# rather than only true at the first frame.
CAMERA_PHI, CAMERA_THETA = 50, -143
CAMERA_DRIFT = 0.008

# The picture is drawn a little below the camera's centre: the sheet is high over most of the
# plane and only falls to the floor by the near corner, so centring it on the axes would leave
# the bottom of the frame empty.
SCENE_DROP = 0.7

# How far above the surface a climb is drawn, as a fraction of the drawn height, so that a path
# along a hillside reads as being on it rather than buried in it.
PATH_RISE_FRACTION = 0.04

# How long a single climb takes to draw. These are thirty real Adam iterations across a surface
# the viewer is still reading, so they are drawn at walking pace rather than flicked on.
CLIMB_RUN_TIME = 5.5

READOUT_RISE = 0.62   # how far above a point its H-score is written

TICK_PERCENTS = (50, 100, 150)

_DATA_DIRECTORY = Path(__file__).resolve().parent.parent / 'data'


def load_measurements(filename: str) -> dict:
    data_path = _DATA_DIRECTORY / filename
    if not data_path.exists():
        raise FileNotFoundError(
            f'{data_path} is missing. Run '
            f'`python visualizations/prepare_weight_surface_data.py` first.')
    return json.loads(data_path.read_text(encoding='utf-8'))


class WeightSurfaceScene(ThreeDScene):
    """The measured surface, its axes, and its summits. Subclasses choose what climbs it.

    The two scenes read two different slices. Descent is easier to understand on a surface with
    one hill on it, and the menu only means anything on a surface with several, so the prep
    script measures both and each scene names the one it needs.
    """

    # How finely the mesh is drawn, which is separate from how finely the surface was
    # MEASURED. See SURFACE_RESOLUTION for why it is finer than the data.
    surface_resolution = SURFACE_RESOLUTION

    data_filename = None

    def setup(self) -> None:
        self.measured = load_measurements(self.data_filename)
        self.grid_axis = np.array(self.measured['axis'], dtype=float)
        self.grid = np.array(self.measured['surface'], dtype=float)
        self.axis_top = float(self.grid_axis[-1])
        self.z_low = float(self.grid.min()) - Z_FLOOR_MARGIN
        self.z_high = float(self.grid.max()) + Z_CEILING_MARGIN
        self.path_rise = (self.z_high - self.z_low) * PATH_RISE_FRACTION

    # ── The surface ───────────────────────────────────────────────────────────────────

    def height_at(self, weight_a: float, weight_b: float) -> float:
        """The measured H-score at a point on the plane, bilinear between grid samples."""
        axis = self.grid_axis
        span = axis[-1] - axis[0]
        position_a = np.clip((weight_a - axis[0]) / span, 0.0, 1.0) * (len(axis) - 1)
        position_b = np.clip((weight_b - axis[0]) / span, 0.0, 1.0) * (len(axis) - 1)
        low_a, low_b = int(np.floor(position_a)), int(np.floor(position_b))
        high_a = min(low_a + 1, len(axis) - 1)
        high_b = min(low_b + 1, len(axis) - 1)
        fraction_a, fraction_b = position_a - low_a, position_b - low_b
        return float(
            self.grid[low_a, low_b]     * (1 - fraction_a) * (1 - fraction_b)
            + self.grid[high_a, low_b]  * fraction_a * (1 - fraction_b)
            + self.grid[low_a, high_b]  * (1 - fraction_a) * fraction_b
            + self.grid[high_a, high_b] * fraction_a * fraction_b
        )

    def build_axes(self) -> ThreeDAxes:
        step = TICK_PERCENTS[1] - TICK_PERCENTS[0]
        return ThreeDAxes(
            x_range = [0, self.axis_top, step]
            , y_range = [0, self.axis_top, step]
            , z_range = [self.z_low, self.z_high, (self.z_high - self.z_low) / 3]
            , x_length = 5.9
            , y_length = 5.9
            , z_length = 2.8
            , axis_config = {'color': GREY_B, 'stroke_width': 2, 'include_ticks': True}
            # The height axis is drawn out of the scene: it would stand straight through the
            # middle of the surface, and the colour ramp already says which way is up.
            , z_axis_config = {'stroke_width': 0, 'include_ticks': False, 'include_tip': False}
        ).shift(SCENE_DROP * IN)

    def build_surface(self, axes: ThreeDAxes) -> Surface:
        """The measured plane, coloured by its own height.

        The whole surface spans two percent of H-score, so the shape is the only thing carrying
        the difference between a good build and a bad one. Colouring by height says which way is
        up even where the silhouette is hidden behind a nearer hill.
        """
        surface = Surface(
            lambda u, v: axes.c2p(u, v, self.height_at(u, v))
            , u_range = [0, self.axis_top]
            , v_range = [0, self.axis_top]
            , resolution = (self.surface_resolution, self.surface_resolution)
            , fill_opacity = 0.9
            , stroke_width = 0.5
            , stroke_color = GREY_B
            , checkerboard_colors = False
        )
        span = self.z_high - self.z_low
        surface.set_fill_by_value(axes=axes, colorscale=[
            (BLUE_E,     self.z_low)
            , (TEAL_D,   self.z_low + span * 0.55)
            , (GREEN_C,  self.z_low + span * 0.85)
            , (YELLOW_D, self.z_high)
        ])
        return surface

    def build_corner_posts(self, axes: ThreeDAxes) -> VGroup:
        """Dashed posts from the floor up to the four corners of the plane.

        The sheet is high over most of the plane and falls away only where a category is given
        up, so without them it hangs unattached over its own axes and the drop reads as
        perspective rather than as a real loss of H-score.
        """
        top = self.axis_top
        return VGroup(*[
            DashedLine(
                axes.c2p(weight_a, weight_b, self.z_low)
                , axes.c2p(weight_a, weight_b, self.height_at(weight_a, weight_b))
                , stroke_width = 1.6
                , stroke_opacity = 0.45
                , color = GREY_B
                , dash_length = 0.08
            )
            for weight_a, weight_b in ((0, 0), (top, 0), (0, top), (top, top))
        ])

    def build_axis_labels(self, axes: ThreeDAxes) -> VGroup:
        """What each axis is, and where a hundred percent falls along it.

        The numbers are percentages of the weight a balanced build puts on that category, so the
        point where both read a hundred IS the balanced build, and everything below it is a
        category being given up. They are kept facing the camera, because text lying flat on a
        plane tilted this far is unreadable.
        """
        top = self.axis_top
        labels = VGroup(
            Text(f"{self.measured['category_a']} weight", font_size=22, color=GREY_B)
            .move_to(axes.c2p(top * 0.60, -top * 0.26, self.z_low))
            , Text(f"{self.measured['category_b']} weight", font_size=22, color=GREY_B)
            .move_to(axes.c2p(-top * 0.26, top * 0.60, self.z_low))
            , Text('0', font_size=17, color=GREY_B)
            .move_to(axes.c2p(-top * 0.05, -top * 0.05, self.z_low))
        )
        for percent in TICK_PERCENTS:
            labels.add(
                Text(f'{percent}%', font_size=17, color=GREY_B)
                .move_to(axes.c2p(percent, -top * 0.10, self.z_low))
                , Text(f'{percent}%', font_size=17, color=GREY_B)
                .move_to(axes.c2p(-top * 0.10, percent, self.z_low))
            )
        self.add_fixed_orientation_mobjects(*labels)
        return labels

    def build_caption(self) -> VGroup:
        """The three things the surface cannot say: whose it is, how tall it is, and in what."""
        caption = VGroup(
            Text(f"{self.measured['candidate']}, first pick", font_size=22, color=WHITE)
            , Text(f'H-score  {self.grid.min():.3f} – {self.grid.max():.3f}',
                   font_size=18, color=GREY_B)
            , Text('weights as a percentage of a balanced build', font_size=16, color=GREY_B)
        ).arrange(DOWN, buff=0.18, aligned_edge=LEFT).to_corner(UL, buff=0.45)
        self.add_fixed_in_frame_mobjects(caption)
        return caption

    def mark_point(self, axes: ThreeDAxes, point, colour) -> Dot3D:
        """A ball sitting on the surface at one build, lifted clear of it like the climbs are."""
        return Dot3D(
            axes.c2p(point[0], point[1], self.height_at(point[0], point[1]) + self.path_rise)
            , radius = 0.09
            , color = colour
        )

    def write_score(self, axes: ThreeDAxes, point, score: float, colour) -> Text:
        """What a build is worth, written above where it sits and kept facing the camera.

        The number is the objective at the real nine-dimensional build, which is not the same as
        the surface's height at this point: the surface pins the other seven weights at balanced,
        and a descent moves them. Position is projected; worth is not.
        """
        readout = (Text(f'{score:.5f}', font_size=19, color=colour)
                   .move_to(axes.c2p(point[0], point[1], self.height_at(point[0], point[1]))
                            + READOUT_RISE * OUT))
        self.add_fixed_orientation_mobjects(readout)
        return readout

    def build_climb(self, axes: ThreeDAxes, path: list, colour) -> VMobject:
        trail = VMobject(stroke_color=colour, stroke_width=5)
        trail.set_points_as_corners([
            axes.c2p(point[0], point[1], self.height_at(point[0], point[1]) + self.path_rise)
            for point in path
        ])
        return trail

    def introduce_axes(self, seconds_available: float) -> ThreeDAxes:
        """The camera, the axes and their labels -- the frame, before anything is plotted in it.

        Separate from the surface because the two scenes want them at different moments. Gradient
        descent draws everything inside its opening line. The seed menu holds the axes up first,
        for about a second, so that the line has a frame to start against rather than opening on
        an empty screen -- and then draws the surface itself under the words, where the wait for
        it is spent listening instead of watching a silent picture assemble.
        """
        self.set_camera_orientation(phi=CAMERA_PHI * DEGREES, theta=CAMERA_THETA * DEGREES)
        axes = self.build_axes()
        self.play(Create(axes), FadeIn(self.build_axis_labels(axes)), run_time=seconds_available)
        return axes

    def introduce_surface(self, seconds_available: float) -> ThreeDAxes:
        """Everything both scenes open on: the camera, the axes, the surface and its posts.

        The axes arrive WITH their labels, before the surface does. Both scenes' opening lines
        talk about category weights, and those weights are the axes -- naming them while the
        viewer is looking at an unlabelled frame asks them to hold the word until a label turns
        up to attach it to.

        The steps are stretched to fill the line covering them, so the introduction is still
        being drawn while it is still being described.
        """
        # Four steps, weighted by how much there is to watch in each.
        shares = (0.22, 0.46, 0.14, 0.18)
        durations = [max(0.5, seconds_available * share) for share in shares]
        axes = self.introduce_axes(durations[0])
        self.play(Create(self.build_surface(axes)), run_time=durations[1])
        self.play(Create(self.build_corner_posts(axes)), run_time=durations[2])
        self.play(FadeIn(self.build_caption()), run_time=durations[3])
        return axes

    def play_climb(self, axes: ThreeDAxes, climb: dict, colour) -> Dot3D:
        """Walk one start uphill and leave a ball where it arrives."""
        trail = self.build_climb(axes, climb['path'], colour)
        self.play(Create(trail), run_time=CLIMB_RUN_TIME)
        arrival = self.mark_point(axes, climb['path'][-1], colour)
        self.add(arrival)
        self.wait(0.4)
        return arrival

    def climb_named(self, label: str) -> dict:
        return next(climb for climb in self.measured['climbs'] if climb['label'] == label)


