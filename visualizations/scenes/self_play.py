"""The field works out what it is doing, and the algorithm watches it happen.

Every other scene in this set is about one team's decision. This one is about the fact that the
opposition is solving the same problem, and that the algorithm's picture of them is not assumed
but DISCOVERED -- by running H-scoring against itself until the answers stop moving.

Each row is one of the top players, each column a category, and the colour is the win rate that
player's build buys in that category against the field's average build: green above a coin flip,
red below, dark at parity. A pass re-solves half the field against the averaged builds of the
passes before it, and the grid settles as the builds stop reacting to each other.

Everything on screen is a recording of the shipped bootstrap, not a model of it -- the same loop
that runs at session build, with its per-pass state captured. Run
`python visualizations/prepare_self_play_data.py` first, which re-runs it with the recorder
attached and writes what it saw.

    manim -ql visualizations/scenes/self_play.py SelfPlayLoop
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from manim import (
    Scene, VGroup, Rectangle, Line, Text, DecimalNumber,
    FadeIn, FadeOut, Create,
    DOWN, LEFT, RIGHT,
    GREEN_C, RED_C, YELLOW, WHITE, GREY_B, GREY_D, GREY_E,
    interpolate_color,
)


# ── Layout ───────────────────────────────────────────────────────────────────────────

CELL_WIDTH, CELL_HEIGHT = 0.52, 0.30
CELL_GAP_X, CELL_GAP_Y  = 0.56, 0.335
GRID_LEFT_X   = -4.15       # left edge of the first column
GRID_TOP_Y    = 2.15        # centre of the first row
NAME_GAP      = 0.22        # between a player's name and his first cell

# The win rate a cell is drawn at when it is a coin flip, and the range the colour ramp covers.
# Clipped rather than stretched to the data's own extremes: a fixed scale is what lets pass 1 and
# pass 32 be compared by eye.
PARITY_WIN_RATE = 0.5
COLOUR_SPAN     = 0.18      # win rates this far from parity are fully saturated

DRIFT_LEFT_X, DRIFT_RIGHT_X = 1.15, 6.3
DRIFT_BASE_Y, DRIFT_TOP_Y   = -1.95, 1.35
DRIFT_DECADES = (-1, -4)    # the log10 range the drift axis spans

# Two gears, as in the differential montage: the early passes carry the movement and the rest is
# the field confirming it has stopped, which should not cost the viewer the same time.
EARLY_PASSES      = 8
EARLY_RUN_TIME    = 0.42
LATE_RUN_TIME     = 0.16

_DATA_PATH = Path(__file__).resolve().parent.parent / 'data' / 'self_play.json'


def load_measurements() -> dict:
    if not _DATA_PATH.exists():
        raise FileNotFoundError(
            f'{_DATA_PATH} is missing. Run '
            f'`python visualizations/prepare_self_play_data.py` first.')
    return json.loads(_DATA_PATH.read_text(encoding='utf-8'))


def win_rate_colour(win_rate: float):
    """Red below a coin flip, green above, dark at parity.

    A diverging ramp rather than a brightness one, because the quantity has a meaningful middle:
    the interesting thing about a build is which side of even it puts each category on.
    """
    if not np.isfinite(win_rate):
        return GREY_E
    edge = float(np.clip((win_rate - PARITY_WIN_RATE) / COLOUR_SPAN, -1.0, 1.0))
    if edge >= 0:
        return interpolate_color(GREY_E, GREEN_C, edge)
    return interpolate_color(GREY_E, RED_C, -edge)


class SelfPlayLoop(Scene):
    """Thirty-two passes of the real bootstrap, and the drift that says it has settled."""

    def setup(self) -> None:
        self.measured = load_measurements()
        self.passes = self.measured['passes']
        self.categories = self.measured['categories']
        self.player_names = self.measured['player_names']

    # ── The grid ──────────────────────────────────────────────────────────────────────

    def _cell_position(self, player_index: int, category_index: int) -> np.ndarray:
        return np.array([
            GRID_LEFT_X + category_index * CELL_GAP_X,
            GRID_TOP_Y - player_index * CELL_GAP_Y,
            0.0,
        ])

    def build_grid(self) -> VGroup:
        cells = VGroup()
        for player_index in range(len(self.player_names)):
            for category_index in range(len(self.categories)):
                cells.add(Rectangle(
                    width=CELL_WIDTH, height=CELL_HEIGHT, stroke_width=0,
                    fill_color=GREY_E, fill_opacity=1.0,
                ).move_to(self._cell_position(player_index, category_index)))
        return cells

    def _cell(self, grid: VGroup, player_index: int, category_index: int) -> Rectangle:
        return grid[player_index * len(self.categories) + category_index]

    def build_labels(self) -> VGroup:
        names = VGroup(*[
            Text(name, font_size=13, color=GREY_B)
            .next_to(self._cell_position(player_index, 0), LEFT, buff=NAME_GAP)
            for player_index, name in enumerate(self.player_names)
        ])
        headers = VGroup(*[
            Text(category.replace(' %', '%'), font_size=12, color=GREY_B)
            .rotate(np.pi / 2.6)
            .move_to(self._cell_position(0, category_index) + np.array([0.0, 0.95, 0.0]))
            for category_index, category in enumerate(self.categories)
        ])
        return VGroup(names, headers)

    # ── The drift curve ───────────────────────────────────────────────────────────────

    def _drift_position(self, pass_number: int, drift: float) -> np.ndarray:
        """A pass and its drift, on a log axis: the decay is four decades and a linear axis
        would show the first pass and then a flat line along the bottom."""
        span = max(len(self.passes) - 1, 1)
        top_decade, bottom_decade = DRIFT_DECADES
        height = (np.log10(max(drift, 1e-9)) - bottom_decade) / (top_decade - bottom_decade)
        return np.array([
            DRIFT_LEFT_X + (pass_number / span) * (DRIFT_RIGHT_X - DRIFT_LEFT_X),
            DRIFT_BASE_Y + float(np.clip(height, 0.0, 1.0)) * (DRIFT_TOP_Y - DRIFT_BASE_Y),
            0.0,
        ])

    def build_drift_axes(self) -> VGroup:
        frame = VGroup(
            Line([DRIFT_LEFT_X, DRIFT_BASE_Y, 0], [DRIFT_RIGHT_X, DRIFT_BASE_Y, 0],
                 color=GREY_D, stroke_width=2),
            Line([DRIFT_LEFT_X, DRIFT_BASE_Y, 0], [DRIFT_LEFT_X, DRIFT_TOP_Y, 0],
                 color=GREY_D, stroke_width=2),
        )
        top_decade, bottom_decade = DRIFT_DECADES
        for decade in range(bottom_decade, top_decade + 1):
            position = self._drift_position(0, 10.0 ** decade)
            frame.add(Text(f'1e{decade}', font_size=12, color=GREY_D)
                      .next_to([DRIFT_LEFT_X, position[1], 0], LEFT, buff=0.12))
        frame.add(Text('how far the field moved, per pass', font_size=14, color=GREY_B)
                  .next_to(frame[0], DOWN, buff=0.3))
        return frame

    # ── The scene ─────────────────────────────────────────────────────────────────────

    def construct(self) -> None:
        grid = self.build_grid()
        labels = self.build_labels()
        drift_axes = self.build_drift_axes()

        pass_readout = VGroup(
            Text('pass', font_size=16, color=GREY_B),
            DecimalNumber(0, num_decimal_places=0, font_size=26, color=WHITE),
        ).arrange(RIGHT, buff=0.22).move_to([3.55, 2.35, 0])

        self.play(FadeIn(grid), FadeIn(labels), run_time=1.0)
        self.play(Create(drift_axes), FadeIn(pass_readout), run_time=0.8)
        self.wait(0.8)

        drift_curve = VGroup()
        self.add(drift_curve)
        previous_point = None

        for step, entry in enumerate(self.passes):
            win_rates = np.array(entry['win_rates'], dtype=float)
            run_time = EARLY_RUN_TIME if step < EARLY_PASSES else LATE_RUN_TIME

            recolours = [
                self._cell(grid, player_index, category_index).animate.set_fill(
                    win_rate_colour(win_rates[player_index][category_index]), opacity=1.0)
                for player_index in range(len(self.player_names))
                for category_index in range(len(self.categories))
            ]
            # The drift of the pass BEFORE this one -- the loop reports how far the average field
            # moved to arrive here, so the first pass has nothing to report yet.
            if entry['drift'] is not None:
                point = self._drift_position(step, entry['drift'])
                if previous_point is not None:
                    drift_curve.add(Line(previous_point, point, color=YELLOW, stroke_width=3))
                previous_point = point

            self.play(*recolours,
                      pass_readout[1].animate.set_value(entry['pass_index']),
                      run_time=run_time)

        # The grid has stopped changing and the curve has bottomed out; hold on both together,
        # because "it settled" is a claim about the two of them at once.
        self.wait(3.0)
        self.play(FadeOut(grid), FadeOut(labels), FadeOut(drift_axes),
                  FadeOut(drift_curve), FadeOut(pass_readout), run_time=0.9)
