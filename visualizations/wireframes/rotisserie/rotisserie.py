"""Rotisserie: why the algorithm wants your season to be uncertain.

WIREFRAME -- a first draft to find out whether the argument lands, drafted against gTTS rather
than the shipped voice. See wireframes/README.md.

The counter-intuitive claim the docs make in one sentence: because winning a league needs an
aberrant result, VARIANCE is worth having, and a Bernoulli's variance p(1-p) is largest at
p = 0.5 -- so the algorithm holds categories near fifty-fifty instead of punting them.

The argument in three moves:

    One, two curves on one axis: the total needed to win the league, and your own total. You win
    on the overlap, and the overlap is small. That smallness is what makes the rest follow.

    Two, widen your curve without moving its centre. The left tail buys nothing -- losing badly
    and losing narrowly are the same outcome -- while the right tail reaches into the win. So
    spread is worth something the mean is not.

    Three, where spread comes from. Nine coin flips vary a great deal; nine near-certainties
    barely vary at all. Same expected total, different width, and the first wins more leagues.

The distributions here are illustrative rather than measured -- the point is the shape of the
argument. A prep script against the real Rotisserie objective comes once the beats settle.

    manim -ql visualizations/wireframes/rotisserie/rotisserie.py Rotisserie
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import (
    VGroup, Line, Polygon, Text, MathTex,
    FadeIn, FadeOut, Create, Write, Transform,
    DOWN, UP,
    BLUE_D, BLUE_B, RED_B, GREY_B, GREY_D, YELLOW, WHITE,
)
from manim_voiceover import VoiceoverScene

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.draft_voice import DraftVoice                 # noqa: E402
from shared.narration_timing import wait_until_phrase     # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from narration import NARRATION                           # noqa: E402


# ── Layout ───────────────────────────────────────────────────────────────────────────

BASELINE_Y = -2.35
AXIS_HALF_WIDTH = 6.0
CURVE_HEIGHT = 2.5          # height of the TALLEST curve drawn; area is what is held constant
CURVE_SAMPLES = 220

# In the scene's own units, where a season total sits on the axis. The bar to beat is high and
# fairly tight; the team is centred below it, which is what makes winning unlikely.
WINNING_BAR_CENTRE, WINNING_BAR_SPREAD = 2.0, 1.0
TEAM_CENTRE = -0.3
TEAM_NARROW_SPREAD, TEAM_WIDE_SPREAD = 0.85, 1.75

# Nine categories as coin flips against nine as near-certainties. Same expected wins, wildly
# different spread -- which is the whole third act.
BALANCED_CHANCES  = (0.5,) * 9
COMMITTED_CHANCES = (0.97, 0.96, 0.95, 0.94, 0.93, 0.06, 0.05, 0.04, 0.02)


class Rotisserie(VoiceoverScene):
    """The bar to beat, the value of width, and where width comes from."""

    def construct(self) -> None:
        self.set_speech_service(DraftVoice())
        self.play_the_bar()
        self.play_widening()
        self.play_two_builds()

    # ── Shared apparatus ──────────────────────────────────────────────────────────────

    def build_axis(self) -> VGroup:
        axis = Line([-AXIS_HALF_WIDTH, BASELINE_Y, 0.0], [AXIS_HALF_WIDTH, BASELINE_Y, 0.0],
                    color=GREY_B, stroke_width=3)
        caption = Text('season total', font_size=24, color=GREY_B)
        caption.move_to([0.0, BASELINE_Y - 0.5, 0.0])
        return VGroup(axis, caption)

    def build_curve(self, centre: float, spread: float, colour, opacity: float = 0.35,
                    tallest_spread: float | None = None) -> Polygon:
        """A Normal curve as a filled shape, drawn so that AREA is what stays constant.

        Height is scaled by tallest_spread/spread rather than fixed, because a curve that got
        wider without getting shorter would be claiming more total probability -- which is
        exactly the thing the widening act must not appear to do.
        """
        reference = tallest_spread if tallest_spread is not None else spread
        peak = CURVE_HEIGHT * reference / spread
        xs = np.linspace(-AXIS_HALF_WIDTH, AXIS_HALF_WIDTH, CURVE_SAMPLES)
        # 1.15 rather than 1.6: at the wider setting a curve's tail reached past the end
        # of the axis and was cut off flat there, which reads as a bar rather than a tail.
        ys = peak * np.exp(-0.5 * ((xs - self.to_scene_x(centre)) / (spread * 1.15)) ** 2)
        points = [[x, BASELINE_Y + y, 0.0] for x, y in zip(xs, ys)]
        points = [[xs[0], BASELINE_Y, 0.0]] + points + [[xs[-1], BASELINE_Y, 0.0]]
        return Polygon(*points, color=colour, fill_color=colour,
                       fill_opacity=opacity, stroke_width=2.5)

    def to_scene_x(self, value: float) -> float:
        """Season-total units onto the drawn axis."""
        return value * 1.55

    # ── Act one: the bar, and how far under it you are ────────────────────────────────

    def play_the_bar(self) -> None:
        with self.voiceover(text=NARRATION['the_bar']):
            self.axis = self.build_axis()
            self.play(Create(self.axis), run_time=0.9)
            self.wait(0.8)

        with self.voiceover(text=NARRATION['winning_total']):
            self.bar = self.build_curve(WINNING_BAR_CENTRE, WINNING_BAR_SPREAD, RED_B,
                                        tallest_spread=WINNING_BAR_SPREAD)
            bar_label = Text('what it takes to win the league', font_size=22, color=RED_B)
            bar_label.move_to([self.to_scene_x(WINNING_BAR_CENTRE), 1.75, 0.0])
            self.play(Create(self.bar), FadeIn(bar_label), run_time=1.2)
            self.bar_label = bar_label
            self.wait(0.8)

        with self.voiceover(text=NARRATION['your_total']):
            self.team = self.build_curve(TEAM_CENTRE, TEAM_NARROW_SPREAD, BLUE_D,
                                         tallest_spread=WINNING_BAR_SPREAD)
            team_label = Text('your team', font_size=22, color=BLUE_B)
            team_label.move_to([self.to_scene_x(TEAM_CENTRE) - 1.4, 1.35, 0.0])
            self.play(Create(self.team), FadeIn(team_label), run_time=1.2)
            self.team_label = team_label
            self.wait(0.6)

        with self.voiceover(text=NARRATION['the_overlap']):
            # PLACEHOLDER: the win region wants shading where the team's right tail passes the
            # bar's left tail, which is a proper overlap integral rather than a clipped curve.
            # Left as a marker so the beat exists and can be timed.
            marker = Text('you win up here', font_size=20, color=YELLOW)
            marker.move_to([self.to_scene_x(WINNING_BAR_CENTRE) - 0.4, BASELINE_Y + 0.45, 0.0])
            self.play(FadeIn(marker), run_time=0.7)
            self.overlap_marker = marker
            self.wait(1.2)

    # ── Act two: width is worth something the mean is not ─────────────────────────────

    def play_widening(self) -> None:
        with self.voiceover(text=NARRATION['widen']) as tracker:
            wait_until_phrase(self, tracker, 'wider')
            wide = self.build_curve(TEAM_CENTRE, TEAM_WIDE_SPREAD, BLUE_D,
                                    tallest_spread=WINNING_BAR_SPREAD)
            centre_mark = Line([self.to_scene_x(TEAM_CENTRE), BASELINE_Y, 0.0],
                               [self.to_scene_x(TEAM_CENTRE), BASELINE_Y + 2.1, 0.0],
                               color=YELLOW, stroke_width=3)
            self.play(Transform(self.team, wide), Create(centre_mark), run_time=1.6)
            self.centre_mark = centre_mark
            self.wait(0.8)

        with self.voiceover(text=NARRATION['why_wide']):
            # PLACEHOLDER: the two tails want opposite treatment -- the left greyed out as
            # "losing either way", the right picked out as the part that gained. Needs the
            # overlap shading above to exist first.
            self.wait(2.2)
            self.play(FadeOut(VGroup(self.bar, self.team, self.bar_label, self.team_label,
                                     self.overlap_marker, self.centre_mark)), run_time=0.8)

    # ── Act three: where width comes from ─────────────────────────────────────────────

    def play_two_builds(self) -> None:
        with self.voiceover(text=NARRATION['two_builds']):
            self.wait(1.0)

        with self.voiceover(text=NARRATION['coin_flips']):
            balanced = self.build_category_row(BALANCED_CHANCES, 1.55, 'nine coin flips')
            self.play(FadeIn(balanced), run_time=1.0)
            self.balanced_row = balanced
            self.wait(1.0)

        with self.voiceover(text=NARRATION['certainties']):
            committed = self.build_category_row(COMMITTED_CHANCES, -0.35,
                                                'five locked in, four abandoned')
            self.play(FadeIn(committed), run_time=1.0)
            self.committed_row = committed
            self.wait(1.0)

        with self.voiceover(text=NARRATION['conclusion']):
            # The two spreads written out, which is the arithmetic the whole scene rests on:
            # a sum of Bernoulli variances, largest when every p is a half.
            spreads = VGroup(
                self.spread_readout(BALANCED_CHANCES, 'coin flips', BLUE_B),
                self.spread_readout(COMMITTED_CHANCES, 'certainties', GREY_B),
            ).arrange(DOWN, buff=0.4).move_to([0.0, -2.1, 0.0])
            self.play(FadeIn(spreads), run_time=1.0)
            self.wait(2.0)
        self.wait(0.6)

    def build_category_row(self, chances, y: float, caption: str) -> VGroup:
        """Nine categories as win-probability bars, with a caption."""
        bars = VGroup()
        for index, chance in enumerate(chances):
            x = -4.4 + index * 1.1
            full = Line([x, y - 0.5, 0.0], [x, y + 0.5, 0.0], color=GREY_D, stroke_width=9)
            filled = Line([x, y - 0.5, 0.0], [x, y - 0.5 + chance, 0.0],
                          color=BLUE_D, stroke_width=9)
            bars.add(VGroup(full, filled))
        label = Text(caption, font_size=22, color=GREY_B).move_to([0.0, y + 1.0, 0.0])
        return VGroup(bars, label)

    def spread_readout(self, chances, name: str, colour) -> VGroup:
        """The standard deviation of the number of categories won, written out."""
        variance = float(sum(p * (1 - p) for p in chances))
        return VGroup(
            Text(f'{name}:  ', font_size=24, color=colour),
            MathTex(rf'\sigma = {variance ** 0.5:.2f}', font_size=34, color=colour),
        ).arrange(buff=0.2)
