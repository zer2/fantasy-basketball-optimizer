"""What a Z-score misses, and what a G-score adds.

A Z-score standardises a player against the spread of OTHER PLAYERS. A G-score widens that
denominator to include how much a player swings from week to week. This is the pair of scenes
that shows the difference is real and says how large it is.

Three simulations, all on 2025-26, all in points-per-week, each isolating one source:

    who you drafted, at their weekly average      sd  86   <- all a Z-score sees
    one fixed matchup, replayed in real weeks     sd 104   <- all a Z-score ignores
    both varying                                  sd 137   <- what a G-score prices

The two sources are independent, so their variances add rather than their spreads:
sqrt(86^2 + 104^2) = 135, against a measured 137 -- agreement to about one percent, which is
sampling noise at ten thousand draws. That is the arithmetic VarianceQuadrature draws.

VarianceQuadrature is NARRATED, through manim-voiceover and gTTS. It needs a network connection
on a first render to synthesise its lines, after which they are cached under `media/voiceovers`.

    manim -ql visualizations/scenes/z_versus_g.py VarianceQuadrature
    manim -ql visualizations/scenes/z_versus_g.py FixedMatchupFull
"""

from __future__ import annotations

import json

import numpy as np
from manim import (
    VGroup, Line, Polygon, Text, MathTex,
    FadeIn, FadeOut, Create, Write,
    DOWN, LEFT,
    YELLOW, WHITE, GREY_B, BLUE_B, RED_B,
)
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

from differential_base import DifferentialSceneBase, _DATA_DIR


# ── The spoken track ─────────────────────────────────────────────────────────────────
# PLACEHOLDERS. Every line below is written to be replaced; each says which beat it covers so
# the brief travels with the slot. Edit only this block -- the scene reads these by key and
# times every animation off however long the audio turns out to be, so a rewritten line
# retimes the picture rather than desynchronising it.
#
# Keep an eye on length. These placeholders are roughly the duration the beat wants, so a
# finished line that is much shorter or longer will visibly change the pacing of that beat.

NARRATION = {
    'opening':
        'Placeholder. Say what a Z-score measures a player against, and that a G-score widens '
        'that denominator to take in something a Z-score leaves out.',
    'cross_player':
        'Placeholder. This curve is the first source of variation: which players you happened '
        'to draft, with every one of them performing at exactly their weekly average. Eighty-six '
        'points.',
    'week_to_week':
        'Placeholder. This curve holds the draft still and changes only the week. The same '
        'twenty-six players, a different week of basketball. A hundred and four.',
    'both':
        'Placeholder. This curve lets both vary, and comes out at a hundred and thirty-seven. '
        'That is the spread a G-score prices, and a Z-score sees only the first of the three.',
    'the_question':
        'Placeholder. Point out that eighty-six and a hundred and four do not add up to a '
        'hundred and thirty-seven.',
    'right_angle':
        'Placeholder. Say that the two combine at a right angle because they are independent: '
        'which draft you got tells you nothing about which weeks your players then had.',
    'squares':
        'Placeholder. The squares are what add. The areas are variances. Standard deviations do '
        'not add; variances do.',
}


class FixedMatchupDifferential(DifferentialSceneBase):
    """One draft, replayed ten thousand times in different weeks.

    The middle term of the three, and the only one that isolates week-to-week variation: the
    same twenty-six players every time, so nothing varies except which week they played. It is
    also the only one not centred on zero -- one of these two teams really is better -- which is
    exactly the question H-scoring exists to answer.
    """

    data_filename      = 'matchup_2025_26.json'
    differential_limit = 420          # the same axis as the weekly scene, so the two compare
    bin_width          = 20
    axis_tick_step     = 140
    total_caption      = 'Points in the week'
    spread_caption     = 'standard deviation: {spread:.0f} points in the week'
    axis_caption       = 'the same two teams, a different week'

    def contribution_values(self, simulation_index: int) -> np.ndarray:
        """Each player's dealt week, apportioned so act one's totals climb as faces land.

        The prepared data stores team totals rather than the individual weeks behind them, so the
        split across a side is reconstructed in proportion to weekly averages. Both sides still
        finish on exactly the stored total.
        """
        roster = self.prepared['rosters'][simulation_index]
        averages = np.array(
            [player[self.prepared['value_key']] for player in self.prepared['pool']])
        drawn = averages[roster]

        contributions = np.empty(len(roster))
        for side_index in range(2):
            side = slice(side_index * self.team_size, (side_index + 1) * self.team_size)
            share = drawn[side] / drawn[side].sum()
            contributions[side] = share * self.prepared['totals'][simulation_index][side_index]
        return contributions


class FixedMatchupFull(FixedMatchupDifferential):
    """The renderable cut of the fixed matchup, all five acts."""

    def construct(self) -> None:
        self.play_all_acts()


class FixedMatchupActOne(FixedMatchupDifferential):
    """Act one alone, for tuning without re-rendering the whole thing."""

    def construct(self) -> None:
        self.build_static_frame()
        self.play_act_one_single_draw()


# ── The payoff: how the two spreads combine ──────────────────────────────────────────

CURVE_BASELINE_Y = -1.15     # where the three comparison bells stand
CURVE_HEIGHT     = 1.75      # height of the WIDEST bell; narrower ones stand taller
CURVE_HALF_WIDTH = 5.6       # half-width of the widest bell, in scene units
WIDEST_SPREAD    = 137.0     # the scale everything is drawn against

TRIANGLE_SCALE   = 0.021     # scene units per point of standard deviation


def _measured_spreads() -> dict[str, float]:
    """The three standard deviations, read back out of the prepared simulations.

    Read rather than written down so the scene cannot drift from the data: re-run the prep
    script with a different seed or season and these follow.
    """
    spreads = {}
    for label, filename in (('cross_player', 'pool_2025_26.json'),
                            ('week_to_week', 'matchup_2025_26.json'),
                            ('both',         'weekly_2025_26.json')):
        totals = np.array(json.loads(
            (_DATA_DIR / filename).read_text(encoding='utf-8'))['simulation_totals'])
        spreads[label] = float((totals[:, 0] - totals[:, 1]).std())
    return spreads


class VarianceQuadrature(VoiceoverScene):
    """Three spreads, and why the first two make the third by squares rather than by sums.

    Narrated. The scene carries numbers but no prose: what each curve IS gets said rather than
    captioned, and every animation is timed off the length of the line being spoken, so
    rewriting a sentence retimes the picture instead of desynchronising it.
    """

    def setup(self) -> None:
        self.spreads = _measured_spreads()
        # gTTS: no key, no account, and good enough to cut against. This is the only line that
        # has to change to swap in a better voice later, and the timings follow whatever the
        # service returns rather than being pinned to this one.
        self.set_speech_service(GTTSService(lang='en'))

    def _bell(self, spread: float, colour, opacity: float) -> VGroup:
        """One Normal curve, all three drawn to a common scale so widths are comparable."""
        # Width scales with the spread and height inversely with it, so all three enclose the
        # same area. Drawing them at equal height would be the more obvious choice and the wrong
        # one: these are densities, and the whole point is that the same total probability is
        # spread over more ground, not that there is more of it.
        half_width = CURVE_HALF_WIDTH * (spread / WIDEST_SPREAD)
        height = CURVE_HEIGHT * (WIDEST_SPREAD / spread)
        offsets = np.linspace(-3.2, 3.2, 200)
        points = [
            np.array([offset * half_width / 3.2,
                      CURVE_BASELINE_Y + height * np.exp(-0.5 * offset ** 2),
                      0.0])
            for offset in offsets
        ]
        return VGroup(*[
            Line(start, end, color=colour, stroke_width=4, stroke_opacity=opacity)
            for start, end in zip(points, points[1:])
        ])

    def construct(self) -> None:
        cross_player = self.spreads['cross_player']
        week_to_week = self.spreads['week_to_week']
        both = self.spreads['both']

        baseline = Line([-6.4, CURVE_BASELINE_Y, 0], [6.4, CURVE_BASELINE_Y, 0],
                        color=GREY_B, stroke_width=2)

        with self.voiceover(text=NARRATION['opening']):
            self.play(Create(baseline), run_time=0.8)

        # Each spread arrives with the question it answers, narrowest first, so the widening
        # is what the eye follows. The line spoken over each one is what identifies it --
        # there are no headings, because a heading and a voice saying the same thing is one
        # of them too many.
        entries = [
            (cross_player, BLUE_B, NARRATION['cross_player']),
            (week_to_week, RED_B, NARRATION['week_to_week']),
            (both, YELLOW, NARRATION['both']),
        ]
        drawn_curves = []
        for spread, colour, line in entries:
            curve = self._bell(spread, colour, 1.0)
            value = Text(f'sd {spread:.0f}', font_size=26, color=colour).move_to(
                [0, CURVE_BASELINE_Y - 0.55, 0])
            with self.voiceover(text=line):
                self.play(Create(curve), FadeIn(value), run_time=1.3)
                self.wait(0.6)
                self.play(FadeOut(value), run_time=0.4)
            drawn_curves.append(curve)
            # Earlier curves stay, dimmed, so all three widths are on screen at once by the end.
            for earlier in drawn_curves[:-1]:
                earlier.set_stroke(opacity=0.35)

        with self.voiceover(text=NARRATION['the_question']):
            self.wait(0.9)
            self.play(*[FadeOut(curve) for curve in drawn_curves], FadeOut(baseline),
                      run_time=0.9)

        corner = np.array([-2.55, -1.05, 0.0])
        horizontal_leg = cross_player * TRIANGLE_SCALE
        vertical_leg = week_to_week * TRIANGLE_SCALE
        along = corner + np.array([horizontal_leg, 0.0, 0.0])
        up = corner + np.array([0.0, vertical_leg, 0.0])

        triangle = VGroup(
            Line(corner, along, color=BLUE_B, stroke_width=6),
            Line(corner, up, color=RED_B, stroke_width=6),
            Line(along, up, color=YELLOW, stroke_width=6),
        )
        right_angle = Polygon(
            corner, corner + np.array([0.32, 0, 0]),
            corner + np.array([0.32, 0.32, 0]), corner + np.array([0, 0.32, 0]),
            stroke_color=GREY_B, stroke_width=2, fill_opacity=0,
        )
        leg_labels = VGroup(
            Text(f'{cross_player:.0f}', font_size=26, color=BLUE_B)
            .next_to(Line(corner, along), DOWN, buff=0.2),
            Text(f'{week_to_week:.0f}', font_size=26, color=RED_B)
            .next_to(Line(corner, up), LEFT, buff=0.2),
            Text(f'{both:.0f}', font_size=30, color=YELLOW)
            .move_to((along + up) / 2 + np.array([0.85, 0.55, 0.0])),
        )

        with self.voiceover(text=NARRATION['right_angle']):
            self.play(Create(triangle), Create(right_angle), run_time=1.5)
            self.play(FadeIn(leg_labels), run_time=0.8)

        squares = VGroup(
            Polygon(corner, along, along + np.array([0, -horizontal_leg, 0]),
                    corner + np.array([0, -horizontal_leg, 0]),
                    stroke_color=BLUE_B, stroke_width=3, fill_color=BLUE_B, fill_opacity=0.25),
            Polygon(corner, up, up + np.array([-vertical_leg, 0, 0]),
                    corner + np.array([-vertical_leg, 0, 0]),
                    stroke_color=RED_B, stroke_width=3, fill_color=RED_B, fill_opacity=0.25),
        )
        arithmetic = MathTex(
            rf'{cross_player:.0f}^2 + {week_to_week:.0f}^2 = '
            rf'{np.hypot(cross_player, week_to_week):.0f}^2',
            font_size=46, color=WHITE,
        ).move_to([3.15, 0.55, 0])

        with self.voiceover(text=NARRATION['squares']):
            self.play(Create(squares), run_time=1.4)
            self.play(Write(arithmetic), run_time=1.3)
            self.wait(1.0)
