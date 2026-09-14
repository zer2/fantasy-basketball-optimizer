"""Twelve seats, all reasoning the same way, all watching each other do it.

Every other scene in this set is about one team's decision. This one is about what happens when
the whole league optimises at once: the field's strategies are not assumed, they are discovered,
and the way they are discovered can fail.

Each row is a seat, each column a category, and the brightness of a cell is how much effort that
seat is putting there. A dark cell is a punt. What plays is three runs of the same field under
three update rules, and the difference between them is the point:

    respond to the latest field, searching freely   every seat flips onto the same three punts,
                                                    then flips off them together, forever
    respond to the running average, still freely    the thrash is smaller and does not stop
    respond to the running average, from where      it settles in a few passes, and the punts
    you already are                                 SPREAD across the categories

The third is what the shipped algorithm does -- thirty-two passes, each best-responding to the
running average of the ones before it, each warm-started from the last. The measured drift is on
screen throughout: it is what "settled" means, and in the first run it never falls.

Run `python visualizations/prepare_self_play_data.py` first. Its docstring records a finding
that does not match the note this scene was planned from: averaging alone did not damp the
oscillation in this model -- responding LOCALLY did.

    manim -ql visualizations/scenes/self_play.py SelfPlayLoop
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from manim import (
    Scene, VGroup, Rectangle, Text, DecimalNumber,
    FadeIn, FadeOut,
    DOWN, RIGHT,
    YELLOW, WHITE, GREY_B, GREY_D, BLUE_B,
)


# Which runs play, in order, and the short label each one carries while it does.
RUNS_SHOWN = [
    ('global_latest',   'respond to the latest field'),
    ('global_averaged', 'respond to the running average'),
    ('local_averaged',  'respond to the running average, from where you are'),
]

CELL_WIDTH, CELL_HEIGHT = 0.62, 0.24
CELL_GAP_X, CELL_GAP_Y = 0.70, 0.30
GRID_CENTRE_Y = 0.55
FULL_EFFORT = 2.0             # the effort at which a cell is drawn at full brightness
PASSES_SHOWN = 16
PASS_RUN_TIME = 0.32

_DATA_PATH = Path(__file__).resolve().parent.parent / 'data' / 'self_play.json'


def load_measurements() -> dict:
    if not _DATA_PATH.exists():
        raise FileNotFoundError(
            f'{_DATA_PATH} is missing. Run '
            f'`python visualizations/prepare_self_play_data.py` first.')
    return json.loads(_DATA_PATH.read_text(encoding='utf-8'))


class SelfPlayLoop(Scene):
    """Three update rules, one field, and the drift that says which of them worked."""

    def setup(self) -> None:
        self.measured = load_measurements()
        self.seat_count = self.measured['seat_count']
        self.category_count = self.measured['category_count']

    # ── The board ─────────────────────────────────────────────────────────────────────

    def _cell_position(self, seat: int, category: int) -> np.ndarray:
        return np.array([
            (category - (self.category_count - 1) / 2) * CELL_GAP_X,
            GRID_CENTRE_Y + ((self.seat_count - 1) / 2 - seat) * CELL_GAP_Y,
            0.0,
        ])

    def _brightness(self, effort: float) -> float:
        return float(np.clip(effort / FULL_EFFORT, 0.0, 1.0))

    def build_grid(self) -> VGroup:
        cells = VGroup()
        for seat in range(self.seat_count):
            for category in range(self.category_count):
                cells.add(Rectangle(
                    width=CELL_WIDTH, height=CELL_HEIGHT, stroke_width=0,
                    fill_color=BLUE_B, fill_opacity=self._brightness(1.0),
                ).move_to(self._cell_position(seat, category)))
        return cells

    def build_category_labels(self) -> VGroup:
        top = self._cell_position(0, 0)[1] + 0.35
        return VGroup(*[
            Text(name.replace(' %', '%'), font_size=13, color=GREY_B)
            .rotate(np.pi / 2.6)
            .move_to([self._cell_position(0, index)[0], top + 0.42, 0.0])
            for index, name in enumerate(self.measured['category_names'])
        ])

    def _cell(self, grid: VGroup, seat: int, category: int) -> Rectangle:
        return grid[seat * self.category_count + category]

    # ── One run ───────────────────────────────────────────────────────────────────────

    def play_run(self, grid: VGroup, key: str, label_text: str) -> None:
        run = self.measured['runs'][key]
        weights = run['weights']
        drift = run['drift']

        label = Text(label_text, font_size=22, color=WHITE).move_to([0.0, -2.55, 0.0])
        readout = VGroup(
            Text('drift', font_size=18, color=GREY_B),
            DecimalNumber(drift[0], num_decimal_places=3, font_size=32, color=YELLOW),
        ).arrange(RIGHT, buff=0.3).move_to([0.0, -3.15, 0.0])
        self.play(FadeIn(label), FadeIn(readout), run_time=0.7)

        # Repaint the whole board each pass rather than animating the differences: what a viewer
        # is reading is the PATTERN, and a pattern that redraws wholesale is what makes twelve
        # seats flipping together look like twelve seats flipping together.
        for pass_index in range(1, min(PASSES_SHOWN, len(weights))):
            self.play(
                *[self._cell(grid, seat, category).animate.set_fill(
                    opacity=self._brightness(weights[pass_index][seat][category]))
                  for seat in range(self.seat_count)
                  for category in range(self.category_count)],
                readout[1].animate.set_value(drift[pass_index - 1]),
                run_time=PASS_RUN_TIME,
            )
        self.wait(1.6)

        # How the punts ended up distributed: one number per category, which is where herding
        # and spreading look different even when the board has stopped moving.
        final = np.array(weights[min(PASSES_SHOWN, len(weights)) - 1])
        punts = (final < 0.05).sum(axis=0)
        counts = VGroup(*[
            Text(str(int(count)), font_size=20,
                 color=YELLOW if count >= self.seat_count else GREY_B)
            .move_to([self._cell_position(0, index)[0],
                      self._cell_position(self.seat_count - 1, 0)[1] - 0.45, 0.0])
            for index, count in enumerate(punts)
        ])
        self.play(FadeIn(counts), run_time=0.8)
        self.wait(2.4)
        self.play(FadeOut(label), FadeOut(readout), FadeOut(counts), run_time=0.6)

    def construct(self) -> None:
        grid = self.build_grid()
        labels = self.build_category_labels()
        seats = VGroup(*[
            Text(f'{seat + 1}', font_size=14, color=GREY_D)
            .move_to(self._cell_position(seat, 0) + np.array([-0.72, 0.0, 0.0]))
            for seat in range(self.seat_count)
        ])
        self.play(FadeIn(grid), FadeIn(labels), FadeIn(seats), run_time=1.2)
        self.wait(1.0)

        for key, label_text in RUNS_SHOWN:
            self.play_run(grid, key, label_text)
            # Back to a flat field between runs, so each one starts from the same picture.
            self.play(*[cell.animate.set_fill(opacity=self._brightness(1.0)) for cell in grid],
                      run_time=0.8)
            self.wait(0.4)

        self.play(FadeOut(grid), FadeOut(labels), FadeOut(seats), run_time=0.9)
