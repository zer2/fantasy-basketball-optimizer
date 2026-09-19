"""The week is not settled by the draft -- and the score that admits it.

The second of the two scenes about what a score has to measure. The first (team_differential.py)
dealt two random teams, gave every player his season average, and read a Z-score off the height
of the resulting bell. That scene made an assumption it never examined: a player contributes the
same number every time, so the ONLY thing varying between matchups is who was drafted.

This scene takes the assumption away, in two simulations and a piece of arithmetic.

    Part one holds the draft completely still and replays the same twenty-six players in real
    weeks. The outcome swings anyway -- wider, in fact, than drafting differently made it -- so
    a matchup is not decided by the player base.

    Part two lets both vary, which is the honest case, and lands wider still.

    Part three puts the three numbers together. The two sources are independent, so their
    VARIANCES add rather than their spreads, and the three form a right triangle. That wider
    denominator is what a G-score divides by and a Z-score does not.

Every spread on screen is read back out of the prepared simulations rather than written down, so
re-running the prep script with a different season or seed moves the scene with the data.

    python visualizations/prepare_season_data.py      # writes all three datasets
    manim -ql visualizations/scenes/g_score.py GScoreFull
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from manim import (
    Group, VGroup, Line, Polygon, Text, MathTex,
    FadeIn, FadeOut, Create, Write,
    DOWN, LEFT,
    BLUE_B, RED_B, YELLOW, WHITE, GREY_B,
)
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.differential_base import DifferentialSceneBase   # noqa: E402


from narration import NARRATION   # noqa: E402


# ── The two simulations ──────────────────────────────────────────────────────────────

FIXED_MATCHUP_DATA = 'matchup_2025_26.json'
BOTH_VARY_DATA     = 'weekly_2025_26.json'

_DATA_DIR = Path(__file__).resolve().parent.parent / 'data'


class WeeklyUnitDifferential(DifferentialSceneBase):
    # One week of one category, counted in made baskets and rebounds: whole numbers.
    decimal_places = 0

    """Both simulations in this scene, which differ only in which dataset is dealt from.

    One axis, one binning, one caption: the whole argument is that the second spread is wider
    than the first, and a chart that changed its own scale between them could not make it.
    """

    data_filename      = FIXED_MATCHUP_DATA
    # Four and a half times the axis of the averages scene, because the spread really is that
    # much larger: weekly totals run about three times a per-game number, and real weeks add
    # half again on top. Three standard deviations either way, as there.
    differential_limit = 420
    bin_width          = 20
    axis_tick_step     = 140
    total_caption      = 'Points in the week'
    spread_caption     = 'standard deviation: {spread:.0f} points in the week'
    axis_caption       = 'the same two teams, a different week'
    # Once the dealing is done the faces have nothing left to say, so they clear out and the
    # chart takes the full frame for the curve.
    dismiss_rosters_after_montage = True

    def contribution_values(self, simulation_index: int) -> np.ndarray:
        """Each player's dealt week, apportioned so act one's totals climb as faces land.

        The prepared data stores team totals rather than the individual weeks behind them, so the
        split across a side is reconstructed in proportion to season averages. Act one uses this
        only to animate the totals climbing; both sides still finish on exactly the stored total,
        so nothing downstream can disagree with the histogram.
        """
        roster = self.prepared['rosters'][simulation_index]
        season_averages = np.array(
            [player[self.prepared['value_key']] for player in self.prepared['pool']])
        drawn_averages = season_averages[roster]

        contributions = np.empty(len(roster))
        for side_index in range(2):
            side = slice(side_index * self.team_size, (side_index + 1) * self.team_size)
            share = drawn_averages[side] / drawn_averages[side].sum()
            contributions[side] = share * self.prepared['totals'][simulation_index][side_index]
        return contributions


# ── The payoff: how the two spreads combine ──────────────────────────────────────────

CURVE_BASELINE_Y = -2.20     # where the three comparison bells stand
CURVE_HEIGHT     = 1.15      # height of the WIDEST bell; narrower ones stand taller
CURVE_HALF_WIDTH = 5.6       # half-width of the widest bell, in scene units
# The scale everything is drawn against: the widest of the three, whatever it turns out to be.

TRIANGLE_SCALE   = 0.021     # scene units per point of standard deviation


def _differential_spread(filename: str) -> float:
    totals = np.array(json.loads(
        (_DATA_DIR / filename).read_text(encoding='utf-8'))['simulation_totals'])
    return float((totals[:, 0] - totals[:, 1]).std())


def measured_spreads() -> dict[str, float]:
    """The three standard deviations the closing arithmetic is built from.

    Read rather than written down so the scene cannot drift from the data: re-run the prep
    script with a different seed or season and these follow.

    The week-to-week one is NOT the spread of the fixed-matchup simulation, even though that is
    the simulation the scene has just shown. That run replays ONE pair of randomly drafted teams,
    so its spread is a single draw from the distribution of possible matchups -- fine as the
    answer to "what does one matchup do across weeks", which is what that act asks, and wrong as
    a general quantity to carry into the arithmetic.

    The general one is available exactly. A differential is the sum of twenty-six independent
    player-weeks, so the week-to-week variance of a random matchup, averaged over matchups, is
    twenty-six times the mean of the players' own week-to-week variances. Measured on 2025-26 it
    comes to 105, against 104 for the single matchup that happened to be drawn.
    """
    weekly = json.loads((_DATA_DIR / BOTH_VARY_DATA).read_text(encoding='utf-8'))
    player_variances = np.array([np.var(weeks) for weeks in weekly['weekly_values']])
    return {
        'cross_player': _differential_spread('pool_2025_26.json'),
        'week_to_week': float(np.sqrt(2 * weekly['team_size'] * player_variances.mean())),
        'both':         _differential_spread(BOTH_VARY_DATA),
    }


class GScoreFull(VoiceoverScene, WeeklyUnitDifferential):
    """Both simulations and the arithmetic that joins them, as one narrated scene."""

    def setup(self) -> None:
        super().setup()
        self.spreads = measured_spreads()
        # gTTS: no key, no account, and good enough to cut against. The only line to change to
        # swap in a better voice, and the timings follow whatever the service returns.
        self.set_speech_service(GTTSService())

    # ── Between the parts ─────────────────────────────────────────────────────────────

    def clear_frame(self) -> None:
        """Take everything off screen between simulations.

        The two parts share one apparatus -- same axis, same histogram, same roster slots -- so
        the second cannot simply draw over the first: the first's always_redraw histogram is
        still bound to the draw counter they both use. Clearing drops those updaters along with
        the mobjects carrying them.
        """
        if self.mobjects:
            self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)
        self.clear()

    def play_simulation(
        self
        , data_filename: str
        , axis_caption: str
        , opening_line: str
        , result_line: str
    ) -> None:
        """One full simulation: deal, fill the histogram, and draw the curve over it."""
        self.load_dataset(data_filename)
        self.axis_caption = axis_caption

        # The frame goes up first, in silence: the opening line talks about the two teams, so it
        # starts as they are being dealt rather than over an empty set of roster slots.
        self.build_static_frame()
        with self.voiceover(text=opening_line):
            self.play_act_one_single_draw()

        # And the simulation itself waits for the line that calls it one. Repeated draws under
        # the opening line had the thing running well before it was named.
        with self.voiceover(text=result_line):
            self.play_act_two_repeated_draws()
            self.play_act_three_montage()
            self.play_act_four_normal_curve()

    # ── The three bells and the triangle ──────────────────────────────────────────────

    def _bell(self, spread: float, colour, opacity: float) -> VGroup:
        """One Normal curve, all three drawn to a common scale so widths are comparable."""
        # Width scales with the spread and height inversely with it, so all three enclose the
        # same area. Drawing them at equal height would be the more obvious choice and the wrong
        # one: these are densities, and the whole point is that the same total probability is
        # spread over more ground, not that there is more of it.
        widest = max(self.spreads.values())
        half_width = CURVE_HALF_WIDTH * (spread / widest)
        height = CURVE_HEIGHT * (widest / spread)
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

    def play_quadrature(self) -> None:
        """Three spreads, and the triangle that says how the first two make the third."""
        cross_player = self.spreads['cross_player']
        week_to_week = self.spreads['week_to_week']
        both = self.spreads['both']

        baseline = Line([-6.4, CURVE_BASELINE_Y, 0], [6.4, CURVE_BASELINE_Y, 0],
                        color=GREY_B, stroke_width=2)

        # The axis never appears on its own: an empty pair of axes drawn under a line of
        # narration is a held breath, so the baseline arrives with the first curve on it.
        first_curve = self._bell(cross_player, BLUE_B, 1.0)
        with self.voiceover(text=NARRATION['opening']):
            self.play(Create(baseline), Create(first_curve), run_time=1.4)

        # Each spread arrives with the question it answers, narrowest first, so the widening
        # is what the eye follows. The line spoken over each one is what identifies it --
        # there are no headings, because a heading and a voice saying the same thing is one
        # of them too many.
        entries = [
            (cross_player, BLUE_B, NARRATION['cross_player'], first_curve),
            (week_to_week, RED_B, NARRATION['week_to_week'], None),
            (both, YELLOW, NARRATION['both'], None),
        ]
        drawn_curves = []
        for spread, colour, line, existing in entries:
            curve = existing if existing is not None else self._bell(spread, colour, 1.0)
            value = Text(f'sd {spread:.0f}', font_size=26, color=colour).move_to(
                [0, CURVE_BASELINE_Y - 0.55, 0])
            with self.voiceover(text=line):
                if existing is None:
                    self.play(Create(curve), FadeIn(value), run_time=1.3)
                else:
                    self.play(FadeIn(value), run_time=0.5)
                self.wait(0.6)
                self.play(FadeOut(value), run_time=0.4)
            drawn_curves.append(curve)
            # Earlier curves stay, dimmed, so all three widths are on screen at once by the end.
            for earlier in drawn_curves[:-1]:
                earlier.set_stroke(opacity=0.35)

        # One beat for the whole close. The triangle is offered as an ANALOGY -- the two
        # spreads behave like the legs of a right triangle, so the answer behaves like its
        # hypotenuse -- which is a thing a viewer already knows the shape of. The earlier
        # version built the right angle first and explained it afterwards, which asked the
        # viewer to accept the picture before being told what it was for.
        #
        # It is built ABOVE the curves rather than in place of them: the legs are the two
        # widths still on screen, in their own colours, so the claim can be checked against the
        # thing it is a claim about instead of being remembered from a frame ago.
        corner = np.array([-3.40, 0.45, 0.0])
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
        arithmetic = MathTex(
            rf'\sqrt{{{cross_player:.0f}^2 + {week_to_week:.0f}^2}} = '
            rf'{np.hypot(cross_player, week_to_week):.0f}',
            font_size=46, color=WHITE,
        ).move_to([2.55, 1.75, 0])

        with self.voiceover(text=NARRATION['the_question']):
            self.play(Create(triangle), Create(right_angle), run_time=1.5)
            self.play(FadeIn(leg_labels), run_time=0.8)
            self.play(Write(arithmetic), run_time=1.3)
            self.wait(1.2)

    def construct(self) -> None:
        # One draft, many weeks: the assumption the Z-score scene made, taken away.
        self.play_simulation(
            FIXED_MATCHUP_DATA, 'the same two teams, a different week',
            NARRATION['fixed_matchup'], NARRATION['fixed_matchup_result'])
        self.clear_frame()

        # Both varying, which is what a real matchup is.
        self.play_simulation(
            BOTH_VARY_DATA, 'a different draft, in a different week',
            NARRATION['both_vary'], NARRATION['both_vary_result'])
        self.clear_frame()

        self.play_quadrature()


class GScoreFixedMatchupActOne(WeeklyUnitDifferential):
    """Act one of the first simulation alone, for tuning without rendering the whole thing."""

    def construct(self) -> None:
        self.build_static_frame()
        self.play_act_one_single_draw()
