"""SAVOR: what a player is worth once you can drop them.

WIREFRAME -- a first draft to find out whether the argument lands, drafted against gTTS rather
than the shipped voice. See wireframes/README.md.

The whole adjustment is one picture: a projection is a Normal, there is a floor under it because
a player who busts gets dropped for a free agent, and the alternative you are bidding against is
itself a free player with the same floor.

    value = E[max(mu + noise, 0)] - E[max(noise, 0)]

which is exactly the formula in the docs,

    mu * Phi(mu/sigma) - (sigma / sqrt(2 pi)) * (1 - exp(-mu^2 / (2 sigma^2)))

The two terms are the two things drawn: a Normal truncated at the replacement line, minus a
half-normal for the dollar flyer. The scene's job is to make the asymmetry obvious -- the cut
takes almost nothing from a star and a great deal from a marginal player -- which is why money
concentrates at the top of an auction.

Both halves are computed here in closed form rather than measured, because they ARE closed form;
what a prep script would add is real dollar values from a session to land the last beat on.

    manim -ql visualizations/wireframes/savor/savor.py Savor
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import (
    VGroup, Line, Polygon, DashedLine, Text, MathTex,
    FadeIn, FadeOut, Create, Write, Transform,
    DOWN, UP, RIGHT,
    BLUE_D, BLUE_B, RED_B, GREEN_C, GREY_B, GREY_D, YELLOW, WHITE,
)
from manim_voiceover import VoiceoverScene

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.draft_voice import DraftVoice                 # noqa: E402
from shared.narration_timing import wait_until_phrase     # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from narration import NARRATION                           # noqa: E402


# ── Layout ───────────────────────────────────────────────────────────────────────────

BASELINE_Y = -2.3
AXIS_HALF_WIDTH = 6.2
CURVE_PEAK = 1.9
CURVE_SAMPLES = 260

# Value in units of sigma above replacement. The replacement line sits at zero by definition:
# it is the value of a player you can have for nothing.
REPLACEMENT = 0.0
STAR_VALUE = 2.2
MARGINAL_VALUE = 0.55
NOISE_SPREAD = 1.0

# Kept small enough that a curve centred on the star still has both tails inside the axis;
# wider and the right tail is cut off flat at the end of the line.
SCENE_UNITS_PER_VALUE = 1.15


class Savor(VoiceoverScene):
    """A projection with a floor under it, and the free player it has to beat."""

    def construct(self) -> None:
        self.set_speech_service(DraftVoice())
        self.play_the_floor()
        self.play_two_players()
        self.play_the_flyer()

    # ── Shared apparatus ──────────────────────────────────────────────────────────────

    def to_scene_x(self, value: float) -> float:
        return value * SCENE_UNITS_PER_VALUE

    def build_axis(self) -> VGroup:
        axis = Line([-AXIS_HALF_WIDTH, BASELINE_Y, 0.0], [AXIS_HALF_WIDTH, BASELINE_Y, 0.0],
                    color=GREY_B, stroke_width=3)
        caption = Text('value delivered over the season', font_size=23, color=GREY_B)
        caption.move_to([0.0, BASELINE_Y - 0.52, 0.0])
        return VGroup(axis, caption)

    def build_replacement_line(self) -> VGroup:
        line = DashedLine([self.to_scene_x(REPLACEMENT), BASELINE_Y, 0.0],
                          [self.to_scene_x(REPLACEMENT), BASELINE_Y + 3.1, 0.0],
                          color=YELLOW, stroke_width=3, dash_length=0.12)
        label = Text('replacement level', font_size=21, color=YELLOW)
        label.next_to(line, UP, buff=0.12)
        return VGroup(line, label)

    def normal_points(self, centre: float, spread: float, low: float, high: float):
        xs = np.linspace(low, high, CURVE_SAMPLES)
        ys = CURVE_PEAK * np.exp(-0.5 * ((xs - self.to_scene_x(centre))
                                         / (spread * SCENE_UNITS_PER_VALUE)) ** 2)
        return xs, ys

    def build_curve(self, centre: float, colour, opacity: float = 0.30) -> Polygon:
        """The full projection distribution, floor ignored."""
        xs, ys = self.normal_points(centre, NOISE_SPREAD, -AXIS_HALF_WIDTH, AXIS_HALF_WIDTH)
        points = ([[xs[0], BASELINE_Y, 0.0]]
                  + [[x, BASELINE_Y + y, 0.0] for x, y in zip(xs, ys)]
                  + [[xs[-1], BASELINE_Y, 0.0]])
        return Polygon(*points, color=colour, fill_color=colour,
                       fill_opacity=opacity, stroke_width=2.5)

    def build_lost_tail(self, centre: float) -> Polygon:
        """The part of the distribution below replacement -- the outcomes you never take."""
        cut = self.to_scene_x(REPLACEMENT)
        xs, ys = self.normal_points(centre, NOISE_SPREAD, -AXIS_HALF_WIDTH, cut)
        points = ([[xs[0], BASELINE_Y, 0.0]]
                  + [[x, BASELINE_Y + y, 0.0] for x, y in zip(xs, ys)]
                  + [[cut, BASELINE_Y, 0.0]])
        return Polygon(*points, color=GREY_D, fill_color=GREY_D,
                       fill_opacity=0.55, stroke_width=0)

    def savor_value(self, mean: float, spread: float = NOISE_SPREAD) -> float:
        """The docs' formula, written as the two expectations it actually is.

        E[max(mu + noise, 0)] - E[max(noise, 0)], the second being the dollar flyer. Expanding
        the first gives mu*Phi(mu/sigma) + sigma*phi(mu/sigma) and the second sigma/sqrt(2 pi),
        which is the published closed form.
        """
        from math import erf, exp, pi, sqrt
        standardised = mean / spread
        cumulative = 0.5 * (1.0 + erf(standardised / sqrt(2.0)))
        density = exp(-0.5 * standardised ** 2) / sqrt(2.0 * pi)
        return mean * cumulative + spread * density - spread / sqrt(2.0 * pi)

    # ── Act one: there is a floor ─────────────────────────────────────────────────────

    def play_the_floor(self) -> None:
        with self.voiceover(text=NARRATION['projection_is_a_guess']):
            self.axis = self.build_axis()
            self.play(Create(self.axis), run_time=0.8)
            self.curve = self.build_curve(STAR_VALUE, BLUE_D)
            self.play(Create(self.curve), run_time=1.2)
            self.wait(0.6)

        with self.voiceover(text=NARRATION['the_floor']):
            self.replacement = self.build_replacement_line()
            self.play(Create(self.replacement), run_time=1.0)
            self.wait(1.2)

        with self.voiceover(text=NARRATION['truncation']):
            # PLACEHOLDER: the collapse itself -- the sub-replacement area sliding onto the line
            # as a spike -- is the single most explanatory moment in the scene and wants a
            # proper animation rather than a fade. Marked so the beat exists and can be timed.
            lost = self.build_lost_tail(STAR_VALUE)
            self.play(FadeIn(lost), run_time=0.8)
            self.wait(1.4)
            self.play(FadeOut(VGroup(self.curve, lost)), run_time=0.7)

    # ── Act two: the same floor, two very different players ───────────────────────────

    def play_two_players(self) -> None:
        with self.voiceover(text=NARRATION['two_players']):
            self.star = self.build_curve(STAR_VALUE, BLUE_D)
            self.marginal = self.build_curve(MARGINAL_VALUE, GREEN_C)
            star_label = Text('star', font_size=22, color=BLUE_B)
            star_label.move_to([self.to_scene_x(STAR_VALUE), BASELINE_Y + 2.25, 0.0])
            marginal_label = Text('marginal player', font_size=22, color=GREEN_C)
            marginal_label.move_to([self.to_scene_x(MARGINAL_VALUE) - 2.3,
                                    BASELINE_Y + 1.75, 0.0])
            self.play(Create(self.star), FadeIn(star_label), run_time=1.0)
            self.play(Create(self.marginal), FadeIn(marginal_label), run_time=1.0)
            self.labels = VGroup(star_label, marginal_label)
            self.wait(0.6)

        with self.voiceover(text=NARRATION['star_unaffected']):
            star_lost = self.build_lost_tail(STAR_VALUE)
            self.play(FadeIn(star_lost), run_time=0.8)
            self.wait(1.6)

        with self.voiceover(text=NARRATION['marginal_shrinks']):
            marginal_lost = self.build_lost_tail(MARGINAL_VALUE)
            self.play(FadeIn(marginal_lost), run_time=0.8)
            self.lost_tails = VGroup(star_lost, marginal_lost)
            self.wait(1.8)

    # ── Act three: the dollar flyer, and the subtraction ──────────────────────────────

    def play_the_flyer(self) -> None:
        with self.voiceover(text=NARRATION['the_flyer']):
            self.play(FadeOut(VGroup(self.star, self.marginal, self.labels, self.lost_tails)),
                      run_time=0.7)
            flyer = self.build_curve(REPLACEMENT, GREY_B, opacity=0.22)
            flyer_lost = self.build_lost_tail(REPLACEMENT)
            flyer_label = Text('a one dollar flyer', font_size=22, color=GREY_B)
            flyer_label.move_to([self.to_scene_x(REPLACEMENT) - 2.0, BASELINE_Y + 2.3, 0.0])
            self.play(Create(flyer), FadeIn(flyer_lost), FadeIn(flyer_label), run_time=1.2)
            self.flyer_group = VGroup(flyer, flyer_lost, flyer_label)
            self.wait(1.4)

        with self.voiceover(text=NARRATION['the_subtraction']):
            formula = MathTex(
                r'\mu\,\Phi\!\left(\tfrac{\mu}{\sigma}\right)'
                r'-\tfrac{\sigma}{\sqrt{2\pi}}\left(1 - e^{-\mu^2 / 2\sigma^2}\right)',
                font_size=44, color=WHITE,
            ).move_to([0.0, 2.55, 0.0])
            self.play(Write(formula), run_time=1.4)
            self.formula = formula
            self.wait(1.2)

        with self.voiceover(text=NARRATION['conclusion']):
            self.play(FadeOut(VGroup(self.flyer_group, self.replacement, self.axis)),
                      run_time=0.7)
            # What the adjustment does to the two players, side by side: the star keeps almost
            # all of its projection, the marginal player keeps a fraction of its edge.
            rows = VGroup(*[
                self.value_row(name, value)
                for name, value in (('star', STAR_VALUE), ('marginal player', MARGINAL_VALUE))
            ]).arrange(DOWN, buff=0.62).move_to([0.0, -0.3, 0.0])
            self.play(FadeIn(rows), run_time=1.0)
            self.wait(2.0)
        self.wait(0.6)

    def value_row(self, name: str, projected: float) -> VGroup:
        """Projected value against what it is worth once the floor and the flyer are applied."""
        kept = self.savor_value(projected)
        return VGroup(
            Text(f'{name}', font_size=26, color=GREY_B),
            MathTex(rf'{projected:.2f} \;\rightarrow\; {kept:.2f}', font_size=36, color=YELLOW),
            Text(f'keeps {kept / projected:.0%}', font_size=24, color=GREY_B),
        ).arrange(RIGHT, buff=0.55)
