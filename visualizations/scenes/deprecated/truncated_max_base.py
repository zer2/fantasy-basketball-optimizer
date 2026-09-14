"""Shared apparatus for the three truncated-max scenes.

The three cuts explain one model from three distances -- the plane a player lives in, the
two numbers that plane collapses to, and the scalar those two numbers are fed through --
so they have to agree on what a dot is, what the value direction is coloured, and where the
axes sit. All of that lives here.

Nothing in this module samples anything. Every number on screen comes out of
`data/truncated_max.json`, written by `visualizations/prepare_truncated_max_data.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from manim import (
    VGroup, Line, DashedLine, Polygon, Text,
    DOWN, LEFT, UP,
    BLUE_B, GREEN_B, GREY_B, GREY_D, WHITE, YELLOW,
)


_DATA_PATH = Path(__file__).resolve().parent.parent / 'data' / 'truncated_max.json'

# ── The palette, fixed across the three scenes ────────────────────────────────────────
# A colour means the same thing in all three: yellow is always the value direction and the
# bar it draws, blue is always a player who is still available, green is always the pick.
VALUE_COLOUR      = YELLOW
WEIGHT_COLOUR     = BLUE_B
SURVIVOR_COLOUR   = WHITE
DRAFTED_COLOUR    = GREY_D
PICK_COLOUR       = GREEN_B
CAPTION_COLOUR    = GREY_B

CAPTION_FONT_SIZE = 23
TITLE_FONT_SIZE   = 30
READOUT_FONT_SIZE = 21


def load_truncated_max_data() -> dict:
    """The pools, selections and curves written by prepare_truncated_max_data.py."""
    if not _DATA_PATH.exists():
        raise FileNotFoundError(
            f'{_DATA_PATH} is missing. Run '
            f'`python visualizations/prepare_truncated_max_data.py` first -- the scenes '
            f'deliberately do no sampling of their own.')
    return json.loads(_DATA_PATH.read_text(encoding='utf-8'))


def interpolate_along_angle_grid(
    angle_grid_degrees
    , values_by_angle
    , angle_degrees
):
    """Linear blend of a per-angle array at an angle between two grid points.

    The scenes sweep the weight angle continuously while the prepared data only knows about
    a grid of them, and the arrays being blended are matched row for row (the same pools,
    the same score grid), so a straight blend is the honest in-between rather than a smear.
    """
    position = float(np.interp(angle_degrees, angle_grid_degrees,
                               np.arange(len(angle_grid_degrees))))
    lower_index = int(np.clip(np.floor(position), 0, len(angle_grid_degrees) - 2))
    fraction = position - lower_index
    lower = np.asarray(values_by_angle[lower_index])
    upper = np.asarray(values_by_angle[lower_index + 1])
    return (1.0 - fraction) * lower + fraction * upper


def weights_at_angle(
    angle_degrees
):
    """The unit weight vector at `angle_degrees` off the first category's axis."""
    angle_radians = np.radians(angle_degrees)
    return np.array([np.cos(angle_radians), np.sin(angle_radians)])


def clip_line_to_box(
    anchor
    , direction
    , half_extent
):
    """Where an infinite line through `anchor` enters and leaves the plotted square.

    Returns None when the line misses the square entirely. Clipping rather than drawing a
    long line and hoping: a level line that runs out past the axes reads as part of the
    frame furniture instead of as a property of the pick it belongs to.
    """
    lowest, highest = -1e9, 1e9
    for axis in (0, 1):
        for sign in (1.0, -1.0):
            # The constraint is sign * coordinate <= half_extent along this axis.
            slope = sign * direction[axis]
            slack = half_extent - sign * anchor[axis]
            if abs(slope) < 1e-12:
                if slack < 0.0:
                    return None
            elif slope > 0.0:
                highest = min(highest, slack / slope)
            else:
                lowest = max(lowest, slack / slope)
    if lowest >= highest:
        return None
    return anchor + lowest * direction, anchor + highest * direction


def clip_square_to_half_plane(
    half_extent
    , normal
    , offset
):
    """The part of the plotted square where normal . z <= offset, as polygon vertices."""
    corners = [np.array([sign_x * half_extent, sign_y * half_extent])
               for sign_x, sign_y in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
    kept = []
    for current, following in zip(corners, corners[1:] + corners[:1]):
        current_inside = normal @ current <= offset
        following_inside = normal @ following <= offset
        if current_inside:
            kept.append(current)
        if current_inside != following_inside:
            step = following - current
            kept.append(current + ((offset - normal @ current) / (normal @ step)) * step)
    return kept


class CategoryPlane:
    """The square of category space a player is plotted in, and how it maps to the frame.

    Both plotted planes in these scenes -- raw categories in scene A, the (s, u) pair in
    scene B -- are the same object with different axis labels, which is the point scene B
    is making, so they share one implementation.
    """

    def __init__(
        self
        , centre
        , units_per_standard_deviation
        , half_extent_standard_deviations
    ):
        self.centre = np.array([centre[0], centre[1], 0.0])
        self.units_per_standard_deviation = units_per_standard_deviation
        self.half_extent = half_extent_standard_deviations

    def point_at(
        self
        , coordinates
    ):
        """Frame position of a point given in standard deviations."""
        return self.centre + np.array([
            coordinates[0] * self.units_per_standard_deviation,
            coordinates[1] * self.units_per_standard_deviation,
            0.0,
        ])

    def build_axes(
        self
        , horizontal_label
        , vertical_label
        , tick_step=1.0
    ):
        """Two crossed axes through the origin, ticked in standard deviations.

        Crossed rather than boxed because the origin is not a corner here: it is the average
        player, the point the value bar passes through and the point every weight vector
        starts from.
        """
        reach = self.half_extent
        axes = VGroup(
            Line(self.point_at([-reach, 0.0]), self.point_at([reach, 0.0]),
                 color=GREY_B, stroke_width=2),
            Line(self.point_at([0.0, -reach]), self.point_at([0.0, reach]),
                 color=GREY_B, stroke_width=2),
        )

        ticks = VGroup()
        tick_positions = np.arange(tick_step, reach, tick_step)
        for position in tick_positions:
            for sign in (1.0, -1.0):
                ticks.add(
                    Line(self.point_at([sign * position, -0.08]),
                         self.point_at([sign * position, 0.08]),
                         color=GREY_B, stroke_width=1.5),
                    Line(self.point_at([-0.08, sign * position]),
                         self.point_at([0.08, sign * position]),
                         color=GREY_B, stroke_width=1.5),
                )
        axes.add(ticks)

        horizontal = Text(horizontal_label, font_size=19, color=GREY_B)
        horizontal.next_to(self.point_at([reach, 0.0]), DOWN, buff=0.16)
        # Nudged inward when the label would otherwise hang off the right of the frame.
        horizontal.shift(LEFT * max(0.0, horizontal.get_right()[0] - 6.85))
        vertical = Text(vertical_label, font_size=19, color=GREY_B)
        vertical.next_to(self.point_at([0.0, reach]), UP, buff=0.14)
        axes.add(horizontal, vertical)
        return axes

    def build_clipped_line(
        self
        , anchor
        , direction
        , colour
        , stroke_width=3
        , dashed=False
    ):
        """An infinite line through `anchor`, cut off at the edge of the plotted square."""
        endpoints = clip_line_to_box(np.asarray(anchor, dtype=float),
                                     np.asarray(direction, dtype=float),
                                     self.half_extent)
        if endpoints is None:
            raise ValueError('the line misses the plotted square entirely')
        start, end = (self.point_at(endpoint) for endpoint in endpoints)
        if dashed:
            return DashedLine(start, end, color=colour, stroke_width=stroke_width,
                              dash_length=0.14)
        return Line(start, end, color=colour, stroke_width=stroke_width)

    def build_shaded_half_plane(
        self
        , normal
        , colour
        , opacity=0.16
    ):
        """The half of the square on the far side of normal . z = 0, filled."""
        vertices = clip_square_to_half_plane(self.half_extent,
                                             -np.asarray(normal, dtype=float), 0.0)
        return Polygon(*[self.point_at(vertex) for vertex in vertices],
                       stroke_width=0, fill_color=colour, fill_opacity=opacity)

    def build_curve(
        self
        , coordinates
        , colour
        , stroke_width=4
    ):
        """A polyline through points given in standard deviations."""
        positions = [self.point_at(coordinate) for coordinate in coordinates]
        return VGroup(*[
            Line(start, end, color=colour, stroke_width=stroke_width)
            for start, end in zip(positions, positions[1:])
        ])


def build_caption(
    message
    , position_y
    , colour=CAPTION_COLOUR
    , font_size=CAPTION_FONT_SIZE
):
    """One line of narration, centred horizontally at a fixed height."""
    return Text(message, font_size=font_size, color=colour).move_to([0.0, position_y, 0.0])
