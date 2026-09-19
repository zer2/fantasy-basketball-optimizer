"""Three formats, three appetites for punting -- and one answer they all agree on.

Each Category, Most Categories and Rotisserie differ only in the shape of the payoff. Put the
same nine identical categories and the same nine units of effort under each, and run the same
search, and the result is not the one the planning note predicted.

All three land on the SAME optimum: abandon three categories, split the budget six ways. What
changes is how much that detour is worth. Most Categories gains 9.5% over perfect balance,
Each Category 2.8%, Rotisserie 1.1% -- a nine-fold spread in the value of punting, on an
identical board, from nothing but the payoff curve underneath.

The reason is in the first act. Most Categories pays all-or-nothing on winning five of nine, so
its single-category payoff is the steepest thing here: near the decision point a little margin
is worth a lot and far from it nothing is worth anything, which is exactly the condition that
rewards trading sure ground for contested ground. Rotisserie places you among eleven independent
rivals rather than against one opponent's total, so its margin carries the spread of two draws
instead of one, its payoff is the flattest, and nothing is ever cheap enough to give away.

Every number here is measured, not asserted. Run
`python visualizations/prepare_payoff_formats_data.py` first; it runs the three searches and
writes what they found.

    manim -ql visualizations/scenes/payoff_formats.py ThreeFormats
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from manim import (
    Scene, VGroup, Axes, Dot, Line, Text, DashedLine,
    FadeIn, FadeOut, Create,
    DOWN, UP, RIGHT,
    YELLOW, WHITE, GREY_B, GREY_D, BLUE_B, GREEN_B,
)


FORMAT_COLOURS = {
    'each_category':   BLUE_B,
    'most_categories': YELLOW,
    'rotisserie':      GREEN_B,
}

_DATA_PATH = Path(__file__).resolve().parent.parent / 'data' / 'payoff_formats.json'


def load_measurements() -> dict:
    if not _DATA_PATH.exists():
        raise FileNotFoundError(
            f'{_DATA_PATH} is missing. Run '
            f'`python visualizations/prepare_payoff_formats_data.py` first.')
    return json.loads(_DATA_PATH.read_text(encoding='utf-8'))


def gain_over_balance(entry: dict) -> np.ndarray:
    """A format's score along the drain path, as a percentage of what perfect balance scores.

    Not rescaled to a common range, which was the first attempt and threw the scene away: each
    curve normalised to its own peak lands on top of the others, since all three peak at three
    punts. Percent-of-balance is a real shared unit -- the three payoffs are in categories, a
    probability and rank points, but "how much better than splitting evenly" means the same
    thing in all of them, and it is the quantity that actually differs.
    """
    balanced = abs(entry['balanced'])
    return 100.0 * (np.array(entry['drain_path']['score']) - entry['balanced']) / balanced


class ThreeFormats(Scene):
    """The payoff curves, the searches they drive, and how much each one gains."""

    def setup(self) -> None:
        self.measured = load_measurements()

    # ── Act one: what one category is worth, under each payoff ───────────────────────

    def play_act_one_payoff_shapes(self) -> VGroup:
        panels = VGroup()
        for index, entry in enumerate(self.measured['formats']):
            axes = Axes(
                x_range=[-3, 3, 1], y_range=[0, 1.05, 0.5],
                x_length=3.6, y_length=2.2,
                axis_config={'color': GREY_D, 'stroke_width': 2, 'include_ticks': False},
            )
            curve = entry['single_category_payoff']
            graph = axes.plot_line_graph(
                x_values=curve['margins'], y_values=curve['payoff'],
                line_color=FORMAT_COLOURS[entry['key']], add_vertex_dots=False,
                stroke_width=4,
            )
            title = Text(entry['label'], font_size=22, color=FORMAT_COLOURS[entry['key']])
            title.next_to(axes, UP, buff=0.3)
            caption = Text('margin in one category', font_size=14, color=GREY_D)
            caption.next_to(axes, DOWN, buff=0.22)
            panel = VGroup(axes, graph, title, caption)
            # Positioned as a whole after it is built: the axes sit at the BOTTOM of a panel
            # whose y-range starts at zero, so centring the axes leaves the panel top-heavy.
            panel.move_to([(index - 1) * 4.4, 0.3, 0])
            panels.add(panel)

        self.play(FadeIn(panels[0][0]), FadeIn(panels[1][0]), FadeIn(panels[2][0]), run_time=0.8)
        for panel in panels:
            self.play(FadeIn(panel[2]), FadeIn(panel[3]), Create(panel[1]), run_time=1.2)
            self.wait(0.9)
        self.wait(1.8)
        return panels

    # ── Act two: the same search, run under each ──────────────────────────────────────

    def play_act_two_drain_paths(self) -> VGroup:
        axes = Axes(
            x_range=[0, 4, 1], y_range=[-2, 11, 2],
            x_length=9.0, y_length=5.0,
            axis_config={'color': GREY_D, 'stroke_width': 2},
            x_axis_config={'include_numbers': True, 'font_size': 22},
            y_axis_config={'include_ticks': True, 'include_numbers': True, 'font_size': 18},
        )
        axes.move_to([0.3, -0.2, 0])
        captions = VGroup(
            Text('categories abandoned', font_size=18, color=GREY_B).next_to(axes, DOWN, buff=0.3),
            Text('% better than splitting the budget evenly', font_size=17, color=GREY_B)
            .next_to(axes, UP, buff=0.3),
        )
        zero_line = Line(axes.c2p(0, 0), axes.c2p(4, 0), color=GREY_D, stroke_width=2)
        self.play(Create(axes), FadeIn(captions), Create(zero_line), run_time=1.2)

        everything = VGroup(axes, captions, zero_line)
        for entry in self.measured['formats']:
            gains = gain_over_balance(entry)
            graph = axes.plot_line_graph(
                x_values=entry['drain_path']['punted'], y_values=gains,
                line_color=FORMAT_COLOURS[entry['key']], add_vertex_dots=False, stroke_width=4,
            )
            peak = float(gains.max())
            dot = Dot(axes.c2p(3.0, peak), radius=0.07, color=FORMAT_COLOURS[entry['key']])
            # Labelled at its own peak rather than in a shared legend: the peaks are what the
            # scene is comparing, and three labels stacked at the right edge collided anyway.
            label = Text(f'{entry["label"]}   +{peak:.1f}%', font_size=20,
                         color=FORMAT_COLOURS[entry['key']])
            label.next_to(dot, RIGHT, buff=0.18)
            self.play(Create(graph), run_time=1.4)
            self.play(FadeIn(dot), FadeIn(label), run_time=0.6)
            everything.add(graph, dot, label)
            self.wait(1.0)

        # One shared answer, three different appetites for it. The vertical says the agreement;
        # the gap between the curves says the disagreement.
        marker = DashedLine(axes.c2p(3.0, -2), axes.c2p(3.0, 11),
                            color=WHITE, stroke_width=2, dash_length=0.12)
        self.play(Create(marker), run_time=1.0)
        everything.add(marker)
        self.wait(3.4)
        return everything

    def construct(self) -> None:
        shapes = self.play_act_one_payoff_shapes()
        self.play(FadeOut(shapes), run_time=0.8)

        paths = self.play_act_two_drain_paths()
        self.play(FadeOut(paths), run_time=0.8)
