"""What gradient descent is, and why the algorithm does not run it only once.

Two scenes over two measured surfaces: the H-score of a real first pick as two of its category
weights are varied. X and Y are those two weights, each as a percentage of what the category is
worth in a balanced build, and the height is what the shipped objective says the build is worth.
The axes run past a hundred, so the balanced build is a point inside the picture rather than a
corner of it.

The menu scene uses a slice with two summits, because that is what makes a menu worth having:
one build keeps both categories near balanced, and the other -- worth more -- cuts Threes to
thirty percent while pushing Steals past a hundred and twenty. A valley lies between them, which
is why the best build gives up one category and keeps the other, and why where a descent STARTS
decides which of the two it finds. The descent scene uses the simplest slice the search could
find instead: one hill, with its top inside the picture.

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
    BLUE_E, TEAL_D, GREEN_C, YELLOW_D, YELLOW, WHITE, GREY_B, PURPLE_E,
    Triangle, smooth,
)
from manim.utils.bezier import bezier


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

# How far above the surface a climb and its markers are drawn, as a fraction of the drawn height.
#
# Almost nothing, and deliberately. None of what is drawn on the surface -- the trail, the
# arrowhead, the markers -- is depth-shaded, so manim sorts every one of them in front of the
# sheet whatever their geometry says. The lift buys no visibility at all; all it can do is hold
# them off the hill.
#
# And held off the hill they parallax. At the old 0.04 the trail hung 15 pixels over the surface
# at 1080p and 39 once the camera had pushed in, so panning slid it against the hillside and it
# read as floating in the air rather than lying on the ground. At 0.004 that is about four pixels
# and the curve stays put against the features underneath it.
PATH_RISE_FRACTION = 0.004

# How long a single climb takes to draw. These are thirty real Adam iterations across a surface
# the viewer is still reading, so they are drawn at walking pace rather than flicked on.
#
# The menu scene draws three of these inside one line and keeps the shorter time. The descent
# scene draws one and takes longer over it: it is the scene where the walking itself is the
# subject, and at the shared pace the ball arrives while the viewer is still reading the hill.
# Its climb outlasts its narration by a couple of seconds, which the beat structure allows --
# a beat waits for its animation, and the line it is spoken over is still the one describing it.
CLIMB_RUN_TIME = 5.5
DESCENT_RUN_TIME = 8.0

# The arrowhead riding the front of a climb, as a fraction of the scene. Small enough not to
# hide the summit it is heading for once the camera has pushed in on it.
CLIMB_HEAD_SIZE = 0.09

# How far the camera pushes in over a climb, and how much of the climb it spends getting there.
#
# A descent travels about a quarter of the plane and sits well off centre, so at the opening
# framing it is a short stroke in the upper part of a frame the surface fills -- legible, but far
# too small to read as a considered path. The push-in is aimed at the middle of the climb's own
# bounding box rather than at the frame centre, which is where it would otherwise zoom: the climb
# is not in the middle of the picture, and zooming about the middle walks it out of shot.
#
# The push is front-loaded -- finished a little under halfway through the climb -- because the
# first few iterations are the biggest ones, and a push that eases in the way a camera move
# normally would has not arrived by the time they are drawn.
CLIMB_ZOOM = 2.6
ZOOM_SHARE = 0.4

# How many points are drawn per real iteration. The path itself is thirty Adam steps, and joining
# them with straight segments puts a corner at every one; the steps are not corners, they are
# samples of a continuous trajectory, and the corners are an artefact of drawing.
#
# The smoothing is done in PLANE coordinates and each sample is then lifted onto the surface, not
# the other way round. Smoothing points that have already been lifted rounds the curve through
# the third dimension too, which lifts it off the hillside on the way into a turn and buries it
# on the way out -- it stops reading as a path ON the surface, which is the one thing it has to do.
CLIMB_SMOOTHING_SAMPLES = 10

# How much the drawn climb is allowed to leave its own iterates behind. A curve fitted THROUGH
# every Adam step can only round the corners between them, and the boxy overshoot at the top is
# the shape of the walk rather than a corner, so it survives any amount of that. Each pass here
# pulls every interior point a little toward its neighbours before the curve is fitted, which
# relaxes the overshoot into a curl.
#
# It is approximation, so it is worth saying what it costs: at four passes an iterate moves by at
# most 5.9 of the percentage points the axes are labelled in, against a climb spanning 50.4 of
# them -- 0.99 on average. The two ENDS do not move at all -- they are pinned -- because where the
# descent started and where it finished are claims the scene makes out loud, and the readouts are
# written at them. Everything between is the shape of the journey, which this preserves and a
# straight-segment drawing does not.
CLIMB_BLUR_PASSES = 4

# The marker for one build. It stays this size ON SCREEN however far the camera has pushed in,
# rather than being magnified with the hill.
#
# That is what makes a plain dot enough. Magnified, a dot becomes a faceted sphere hanging in the
# air above the point it marks, and the facets are the first thing the eye finds -- which is the
# problem a ring drawn on the hillside was solving. Held to a constant size there is nothing to
# see facets on, and a point estimate keeps the extent it actually has instead of growing into a
# blob covering several percent of the axis.
MARKER_RADIUS = 0.09          # scene units, at no magnification

# How far from a point its H-score is written, in scene units before magnification. Callers that
# need more room say so; see write_score for what the number means on screen.
READOUT_RISE = 0.62

# The purple a readout is written in -- darker than the climb it labels, which is PURPLE_A.
#
# The surface is coloured by height, so the top of the ramp is the brightest thing in the frame,
# and that is exactly where a climb finishes and its readout gets written. The light purple that
# reads well as a five-wide line on the hillside all but vanishes as text on it. A darker purple
# is still plainly the climb's colour against the white of the seeds the algorithm passed over,
# which is the only distinction the colour is carrying.
READOUT_PURPLE = PURPLE_E

TICK_PERCENTS = (50, 100, 150)

_DATA_DIRECTORY = Path(__file__).resolve().parent.parent / 'prepared_data'


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
            , Text(f'H-score  {self.grid.min() * 100:.1f}% – {self.grid.max() * 100:.1f}%',
                   font_size=18, color=GREY_B)
            , Text('weights as a percentage of a balanced build', font_size=16, color=GREY_B)
        ).arrange(DOWN, buff=0.18, aligned_edge=LEFT).to_corner(UL, buff=0.45)
        self.add_fixed_in_frame_mobjects(caption)
        return caption

    def mark_point(self, axes: ThreeDAxes, point, colour,
                   keep_apparent_size: bool = False) -> Dot3D:
        """A ball sitting on the surface at one build, lifted clear of it like the climbs are.

        `keep_apparent_size` holds it the same size on screen however far the camera has pushed
        in, by scaling it back down as the zoom rises. Only a scene that zooms wants it; without a
        camera move it costs a per-frame updater and does nothing.
        """
        marker = Dot3D(
            axes.c2p(point[0], point[1], self.height_at(point[0], point[1]) + self.path_rise)
            , radius = MARKER_RADIUS
            , color = colour
        )
        if keep_apparent_size:
            unmagnified = marker.width

            def hold_its_size_on_screen(mobject) -> None:
                wanted = unmagnified / self.camera.zoom_tracker.get_value()
                if mobject.width > 1e-9:
                    mobject.scale(wanted / mobject.width)

            marker.add_updater(hold_its_size_on_screen)
        return marker

    def write_score(self, axes: ThreeDAxes, point, score: float, colour,
                    above: bool = True, clear_by: float = READOUT_RISE,
                    sideways: float = 0.0) -> Text:
        """What a build is worth, written above where it sits and kept facing the camera.

        As a percentage, because that is what the number is -- the share of the matchup this build
        expects to win -- and five decimal places of a probability is not something a viewer reads
        off a moving picture.

        The number is the objective at the real build. Whether that is the same as the surface's
        height here depends on the scene: a descent held to the two weights being drawn ends in
        the plane and the two agree, while one free to move all nine ends somewhere the plane does
        not contain, and then its position is a projection and only its worth is exact.

        `above` is which side of the point to write on, and `clear_by` how far. Both exist to get
        out of the way of something: the caption sits in the top corner, and the climb's own line
        runs out of both of its endpoints, so there is usually one side that is free and it is not
        always the same one.

        `sideways` moves it across the frame as well, positive to the right, for a point where
        neither side is free -- the menu's climb finishes under the caption with its own trail
        sweeping back beneath it, and only the room beside it is clear. The direction is taken from
        where the camera is standing, so it stays screen-right as the shot drifts.

        Both distances are divided by the camera's magnification, so a given value means the same
        gap ON SCREEN wherever the camera is standing. Left in scene units they are magnified along
        with the surface, and the value that reads as a label at the opening framing reads as a
        caption adrift by the time the camera has pushed in -- measured, 0.62 is a 33 pixel gap at
        the opening and 75 once the camera has pushed in on the climb.

        Nothing here compensates for the camera's zoom, and nothing should: a fixed-orientation
        mobject is drawn at a fixed size on the frame however far the camera has pushed in, so a
        readout written while zoomed in comes out the same size as one written zoomed out. Scaling
        it down by the zoom -- on the assumption the camera would scale it back up -- renders it at
        a third of its size, sitting on top of the point instead of above it.
        """
        magnification = self.camera.zoom_tracker.get_value()
        rise = clear_by / magnification
        if not above:
            rise = -rise
        theta = self.camera.get_theta()
        across = np.array([-np.sin(theta), np.cos(theta), 0.0]) * sideways / magnification
        readout = (Text(f'{score * 100:.2f}%', font_size=19, color=colour)
                   .move_to(axes.c2p(point[0], point[1], self.height_at(point[0], point[1]))
                            + rise * OUT + across))
        self.add_fixed_orientation_mobjects(readout)
        return readout

    def smooth_plane_path(self, path: list) -> np.ndarray:
        """The iterations, resampled as a smooth curve through them, still in plane coordinates.

        The iterates are first relaxed toward their neighbours (see CLIMB_BLUR_PASSES), then
        manim's own smoothing fits a curve through what is left -- one cubic per step -- and each
        cubic is sampled at a fixed number of points. Both halves are needed: fitting alone rounds
        the corners between steps and leaves the boxy overshoot they trace out, and relaxing alone
        would still be drawn as straight segments.

        Sampling per STEP rather than by arc length is what keeps the pacing honest: every
        iteration keeps the same share of the drawing time, so the small steps at the end read as
        the settling down they are, instead of being hurried through at a constant speed the
        descent never had.
        """
        relaxed = np.array(path, dtype=float)
        for _ in range(CLIMB_BLUR_PASSES):
            interior = (relaxed[:-2] + 2.0 * relaxed[1:-1] + relaxed[2:]) / 4.0
            relaxed = np.vstack([relaxed[0], interior, relaxed[-1]])
        guide = VMobject()
        guide.set_points_smoothly([np.array([point[0], point[1], 0.0]) for point in relaxed])
        samples = []
        for curve in guide.get_cubic_bezier_tuples():
            along = bezier(curve)
            samples.extend(along(step / CLIMB_SMOOTHING_SAMPLES)
                           for step in range(CLIMB_SMOOTHING_SAMPLES))
        samples.append(np.array([path[-1][0], path[-1][1], 0.0]))
        return np.array(samples)

    def build_climb(self, axes: ThreeDAxes, path: list, colour) -> VMobject:
        trail = VMobject(stroke_color=colour, stroke_width=5)
        trail.set_points_as_corners([
            axes.c2p(sample[0], sample[1],
                     self.height_at(sample[0], sample[1]) + self.path_rise)
            for sample in self.smooth_plane_path(path)
        ])
        return trail

    def climb_centre(self, axes: ThreeDAxes, path: list) -> np.ndarray:
        """The middle of the climb's own bounding box, in scene coordinates."""
        points = np.array([
            axes.c2p(point[0], point[1], self.height_at(point[0], point[1]) + self.path_rise)
            for point in path
        ])
        return (points.min(axis=0) + points.max(axis=0)) / 2.0

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

    def build_climb_head(self, colour) -> Triangle:
        """The arrowhead that rides the front of a climb while it is being drawn."""
        return (Triangle(color=colour, fill_opacity=1.0, stroke_width=0)
                .scale(CLIMB_HEAD_SIZE))

    def lay_on_the_hillside(self, axes: ThreeDAxes, scene_point, heading):
        """A rotation that lies a flat shape on the surface at a point, pointing along `heading`.

        Without this the arrowhead is a triangle in the world's horizontal plane: correct over the
        flat top, and plainly hovering everywhere the hillside is steep -- which is most of the
        climb. Turning it into the surface's own tangent plane is what makes it read as lying on
        the ground rather than flying over it.

        The frame is built from the surface itself: two tangents either side of the point give the
        normal, the heading is flattened into the tangent plane, and the third axis completes a
        right-handed set. Columns are where the shape's own x, y and z end up, and a Triangle
        points along its y.
        """
        weight_a, weight_b, _ = axes.p2c(scene_point)
        step = 1.5      # percentage points either side; wide enough to ignore the mesh facets
        across_a = (axes.c2p(weight_a + step, weight_b, self.height_at(weight_a + step, weight_b))
                    - axes.c2p(weight_a - step, weight_b,
                               self.height_at(weight_a - step, weight_b)))
        across_b = (axes.c2p(weight_a, weight_b + step, self.height_at(weight_a, weight_b + step))
                    - axes.c2p(weight_a, weight_b - step,
                               self.height_at(weight_a, weight_b - step)))
        normal = np.cross(across_a, across_b)
        if normal[2] < 0:
            normal = -normal
        length = np.linalg.norm(normal)
        if length < 1e-12:
            return None
        normal = normal / length
        along = heading - np.dot(heading, normal) * normal
        length = np.linalg.norm(along)
        if length < 1e-12:
            return None
        along = along / length
        return np.column_stack([np.cross(along, normal), along, normal])

    def play_climb(self, axes: ThreeDAxes, climb: dict, colour, zoom: float | None = None,
                   fade_while_climbing=None, run_time: float = CLIMB_RUN_TIME) -> Dot3D:
        """Walk one start uphill and leave a ball where it arrives.

        With `zoom`, the camera pushes in on the climb while it is being drawn, and anything in
        `fade_while_climbing` goes out with the same timing -- a readout placed in the scene grows
        with the camera and leaves the frame, so it is retired here rather than left to be dragged
        off the edge.

        The push and the drawing are built as separate animations instead of being handed to
        move_camera, because Scene.play applies its keyword arguments to every animation it is
        given: a rate function passed there to front-load the camera would front-load the climb
        with it, and the line would be finished before the camera arrived to show it.
        """
        trail = self.build_climb(axes, climb['path'], colour)
        head = self.build_climb_head(colour)
        head.move_to(axes.c2p(climb['path'][0][0], climb['path'][0][1],
                              self.height_at(*climb['path'][0]) + self.path_rise))
        flat = head.copy()

        def aim_at_the_front(mobject) -> None:
            drawn = trail.points
            if len(drawn) < 2:
                return
            front = drawn[-1]
            behind = drawn[max(len(drawn) - 4, 0)]
            pose = self.lay_on_the_hillside(axes, front, front - behind)
            if pose is None:
                return
            mobject.become(flat.copy().apply_matrix(pose).move_to(front))

        head.add_updater(aim_at_the_front)
        self.add(head)

        climbing = [Create(trail)]
        if zoom is not None:
            def rushed(alpha: float) -> float:
                return smooth(min(alpha / ZOOM_SHARE, 1.0))

            # The same trackers move_camera animates; there is no public handle on them, and
            # reaching for them here is what buys the camera its own rate function.
            climbing.append(self.camera.zoom_tracker.animate(rate_func=rushed).set_value(zoom))
            climbing.append(self.camera._frame_center.animate(rate_func=rushed)
                            .move_to(self.climb_centre(axes, climb['path'])))
            if fade_while_climbing is not None:
                climbing.append(FadeOut(fade_while_climbing, rate_func=rushed))
        elif fade_while_climbing is not None:
            climbing.append(FadeOut(fade_while_climbing))

        self.play(*climbing, run_time=run_time)
        head.clear_updaters()
        self.remove(head)
        arrival = self.mark_point(axes, climb['path'][-1], colour,
                                  keep_apparent_size=zoom is not None)
        self.add(arrival)
        self.wait(0.4)
        return arrival

    def climb_named(self, label: str) -> dict:
        return next(climb for climb in self.measured['climbs'] if climb['label'] == label)


