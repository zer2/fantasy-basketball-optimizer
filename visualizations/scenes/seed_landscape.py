"""Where you start decides which answer you get.

Gradient descent finds a local optimum, and in this objective every choice of which categories
to abandon is one. So the optimiser needs somewhere to start, and `algorithm_agents.py` does not
give it just one: the seed menu is a gentle punt in each category, plus the balanced build, plus
each candidate's own converged build from the previous pass. Everything climbs, the best answer
wins. A comment there records the payoff -- challenger seeds win 24% of bootstrap solves.

The picture is a plane through weight space containing three of the optima. Brightness is score.
The feasible region is a triangle, and its three CORNERS are the three answers: abandon the
first three categories, abandon the last three, abandon the middle three. All three score 4.6247
where perfect balance scores 4.5000, and perfect balance sits in the dark valley between them.

Which makes the cold start the whole argument. Balanced is a stationary point of this objective
by symmetry -- the gradient there is exactly zero, every direction toward a peak is matched by
an identical one away from it -- so an optimiser started there does not climb slowly, it does
not move AT ALL. It reports 4.5000 and stops, and every challenger seed beats it.

Run `python visualizations/prepare_seed_landscape_data.py` first; it paints the surface and
runs every climb.

    manim -ql visualizations/scenes/seed_landscape.py SeedLandscape
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from manim import (
    Scene, VGroup, VMobject, ImageMobject, Dot, Text,
    FadeIn, FadeOut, Create,
    RIGHT,
    YELLOW, WHITE, GREY_B, GREEN_B, RED_C,
)


SEED_COLOURS = {'cold': RED_C, 'menu': GREEN_B, 'warm': YELLOW}

SURFACE_CENTRE = np.array([-2.35, 0.15, 0.0])
SURFACE_SIZE = 5.9            # frame units across the whole slice
READOUT_LEFT_X = 1.4
READOUT_TOP_Y = 2.2
READOUT_STEP_Y = 0.62

_VISUALIZATIONS_DIR = Path(__file__).resolve().parent.parent
_DATA_PATH = _VISUALIZATIONS_DIR / 'data' / 'seed_landscape.json'


def load_measurements() -> dict:
    if not _DATA_PATH.exists():
        raise FileNotFoundError(
            f'{_DATA_PATH} is missing. Run '
            f'`python visualizations/prepare_seed_landscape_data.py` first.')
    return json.loads(_DATA_PATH.read_text(encoding='utf-8'))


class SeedLandscape(Scene):
    """The surface, the cold start that cannot move, and the seeds that can."""

    def setup(self) -> None:
        self.measured = load_measurements()

    def to_screen(self, along_first: float, along_second: float) -> np.ndarray:
        """Slice coordinates to frame coordinates, matching how the image was painted."""
        low, high = self.measured['slice_low'], self.measured['slice_high']
        unit = SURFACE_SIZE / (high - low)
        return SURFACE_CENTRE + np.array([
            (along_first - (low + high) / 2) * unit,
            (along_second - (low + high) / 2) * unit,
            0.0,
        ])

    def build_surface(self) -> ImageMobject:
        """The painted objective.

        Never dimmed with set_opacity, which replaces per-pixel alpha wholesale -- any dimming
        this needs is baked into the file by the prep script.
        """
        image = ImageMobject(str(_VISUALIZATIONS_DIR / self.measured['image']))
        image.height = SURFACE_SIZE
        image.move_to(SURFACE_CENTRE)
        return image

    def build_climb_trace(self, entry: dict) -> VMobject:
        trace = VMobject(stroke_color=SEED_COLOURS[entry['kind']], stroke_width=4)
        trace.set_points_as_corners([self.to_screen(*point) for point in entry['path']])
        return trace

    def build_readout(self, entry: dict, row: int) -> VGroup:
        readout = VGroup(
            Text(entry['label'], font_size=20, color=SEED_COLOURS[entry['kind']]),
            Text(f'{entry["score"]:.4f}', font_size=22, color=WHITE),
            Text(f'{len(entry["path"]) - 1} steps', font_size=17, color=GREY_B),
        ).arrange(RIGHT, buff=0.35)
        readout.move_to([READOUT_LEFT_X + readout.width / 2, READOUT_TOP_Y - row * READOUT_STEP_Y, 0])
        return readout

    def play_seed(self, entry: dict, row: int) -> VGroup:
        colour = SEED_COLOURS[entry['kind']]
        start = Dot(self.to_screen(*entry['start']), radius=0.09, color=colour)
        self.play(FadeIn(start, scale=2.0), run_time=0.6)

        pieces = VGroup(start)
        if len(entry['path']) > 1:
            trace = self.build_climb_trace(entry)
            landing = Dot(self.to_screen(*entry['final']), radius=0.07, color=colour)
            self.play(Create(trace), run_time=2.0)
            self.play(FadeIn(landing), run_time=0.4)
            pieces.add(trace, landing)
        else:
            # Nothing to draw, because nothing happened -- which is the point of this seed.
            self.wait(1.8)

        self.play(FadeIn(self.build_readout(entry, row)), run_time=0.6)
        pieces.add(self.build_readout(entry, row))
        self.wait(1.0)
        return pieces

    def construct(self) -> None:
        surface = self.build_surface()
        self.add(surface)
        self.play(FadeIn(surface), run_time=1.2)
        self.wait(1.2)

        # The three corners, named by their scores rather than by a sentence about them.
        corners = VGroup(*[
            Dot(self.to_screen(*position), radius=0.06, color=WHITE)
            for position in ((1.0, 0.0), (0.0, 1.0), (-1.0, -1.0))
        ])
        best = Text(f'{self.measured["optimum_score"]:.4f}', font_size=20, color=WHITE)
        best.next_to(corners[0], RIGHT, buff=0.15)
        self.play(FadeIn(corners), FadeIn(best), run_time=1.0)
        self.wait(1.6)

        everything = VGroup(corners, best)
        for row, entry in enumerate(self.measured['climbs']):
            everything.add(self.play_seed(entry, row))
        self.wait(3.0)

        self.play(FadeOut(everything), FadeOut(surface), run_time=1.0)
