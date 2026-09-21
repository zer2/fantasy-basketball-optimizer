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
    Group, VGroup, Line, Polygon, Text, MathTex, Brace,
    FadeIn, FadeOut, Create, Write, GrowFromCenter,
    DOWN, UP, LEFT,
    BLUE_B, RED_B, YELLOW, WHITE, GREY_B,
)
from manim_voiceover import VoiceoverScene

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.differential_base import (                              # noqa: E402
    DifferentialSceneBase, TEAM_LABELS, DISMISSAL_SECONDS,
)


from shared.narration_timing import (                                # noqa: E402
    seconds_remaining_until_phrase, wait_until_phrase,
)
from shared.narration_voice import NarrationVoice   # noqa: E402
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
    spread_caption     = 'σ = {spread:.0f} points in the week'
    axis_caption       = 'the same two teams, a different week'
    # This montage is allowed to deal faster than its natural pace to land with the sentence it
    # plays under. That line is shorter than the dealing takes, and the alternative is the line
    # finishing and the draws carrying on in silence.
    montage_minimum_stretch = 0.5
    # Once the dealing is done the faces have nothing left to say, so they clear out and the
    # chart takes the full frame for the curve.
    dismiss_rosters_after_montage = True

# ── The payoff: how the two spreads combine ──────────────────────────────────────────

CURVE_BASELINE_Y = -2.20     # where the three comparison bells stand
CURVE_HEIGHT     = 1.15      # height of the WIDEST bell; narrower ones stand taller
CURVE_HALF_WIDTH = 5.6       # half-width of the widest bell, in scene units
# The scale everything is drawn against: the widest of the three, whatever it turns out to be.

TRIANGLE_SCALE   = 0.021     # scene units per point of standard deviation

# The closing formula, once the working that produced it has been cleared away.
FORMULA_CENTRE       = [0.0, 1.35, 0.0]
WORKING_FADE_SECONDS = 0.8
# Measured from the rendered expression: x, minus, mu, the fraction rule, the radical hook, its
# overbar, then sigma, exponent, plus, sigma, exponent.
FORMULA_GLYPH_COUNT  = 11
FIRST_SIGMA_GLYPH    = 6
SECOND_SIGMA_GLYPH   = 9

# Clear of the fitted curve, which peaks just above the tallest bar the brace is drawn around.
BRACKET_CLEARANCE = 0.45
# The three spreads stand here for the rest of the scene, off to the right of everything.
SIGMA_COLUMN_X   = 5.15
SIGMA_COLUMN_Y   = 0.25
SIGMA_COLUMN_GAP = 0.52


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
        self.set_speech_service(NarrationVoice())

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
        , curve_phrase: str
        , win_rate_line: str | None
        , deal_during_opening: bool
    ) -> None:
        """One full simulation: deal, fill the histogram, and draw the curve over it.

        `curve_phrase` is the moment in the result line that the dealing should be finishing on,
        and `deal_during_opening` says which side of the line break the dealing starts on. The
        two simulations want different answers because their opening lines are about different
        things: the first one argues that weeks vary, which is not yet a thing to watch, while
        the second one describes the mechanism being shown and should be shown it while talking.
        """
        self.load_dataset(data_filename)
        self.axis_caption = axis_caption

        # The frame goes up first, in silence: the opening line talks about the two teams, so it
        # starts as they are being dealt rather than over an empty set of roster slots.
        self.build_static_frame()

        if deal_during_opening:
            # Everything except the rosters leaving runs under the opening line, which spends
            # fifteen seconds describing the very mechanism the board is demonstrating. Held
            # back until the result line, the board sat still through all of it and then had to
            # race; and the result line went on to call the curve wide while the curve was still
            # being built, which is a claim about a picture that was not there to check.
            with self.voiceover(text=opening_line) as tracker:
                self.play_act_one_single_draw()
                self.drop_first_bar()
                self.play_act_two_repeated_draws()
                self.play_act_three_montage(
                    seconds_available=tracker.get_remaining_duration(),
                    finish_the_frame=False)

            # The rosters leave ON the phrase rather than after it: they are what the histogram
            # was a record of, and their going is the moment it becomes a shape in its own right.
            with self.voiceover(text=result_line) as tracker:
                wait_until_phrase(self, tracker, curve_phrase,
                                  lead_seconds=DISMISSAL_SECONDS)
                self.dismiss_rosters()
                self.play_act_four_normal_curve()
            return

        with self.voiceover(text=opening_line):
            self.play_act_one_single_draw()

        # The simulation waits for the line that calls it one -- including its first bar, which
        # is the moment the simulating starts. It is then paced to END on the words that announce
        # what it produced: ten thousand draws landing and the rosters leaving IS "the result",
        # so it should be finishing as the result is named. Left to its own pace it was over well
        # before the sentence got there, and the line described something already stopped.
        with self.voiceover(text=result_line) as tracker:
            self.drop_first_bar()
            self.play_act_two_repeated_draws()
            self.play_act_three_montage(
                seconds_available=seconds_remaining_until_phrase(self, tracker, curve_phrase))
            self.play_act_four_normal_curve()

        # The claim in the next line is about a share of the board, so the board says it: the
        # bars the left team won turn its own colour, and the number they come to is written
        # where they are. Said without showing it, "just a majority" is a figure the viewer has
        # to take on trust from a picture that is right there.
        if win_rate_line is not None:
            self.play_win_rate(win_rate_line)

    def play_win_rate(self, line: str) -> None:
        """Bracket the winning half of the histogram and write what share of it that is.

        A bracket rather than a colour: yellow already means "the region you win" in the punting
        scene, and a second colour language for the same idea would have the set contradicting
        itself. A bracket says "these ones" without claiming any of that.
        """
        winning = self.bars_above_zero()
        share = self.win_rate_for_left_team()

        # Above the bars, not below them. Below is where the axis lives -- ticks, the caption
        # naming the axis, and the draw counter under that -- and a brace with a two-line
        # readout hanging off it landed straight on all three.
        bracket = Brace(winning, direction=UP, color=GREY_B, buff=BRACKET_CLEARANCE)
        readout = VGroup(
            Text(f'{share * 100:.0f}%', font_size=34, color=WHITE),
            Text(f'of weeks won by {TEAM_LABELS[0]}', font_size=17, color=GREY_B),
        ).arrange(DOWN, buff=0.10).next_to(bracket, UP, buff=0.18)

        with self.voiceover(text=line):
            self.play(GrowFromCenter(bracket), run_time=0.8)
            self.play(FadeIn(readout), run_time=0.6)
            self.wait(1.4)

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
        # Each spread arrives with its curve and then STAYS, in that curve's colour, in a column
        # off to the right. Flashing each one up and taking it away again meant the arithmetic
        # at the end argued about three numbers of which none were on screen, and asked the
        # viewer to have remembered them. The colour is what says which curve a number belongs
        # to, so no number needs a name next to it.
        #
        # Sigma rather than "sd", here and wherever else the scene names a spread: the closing
        # arithmetic is written in symbols, and a label that says one thing while the equation
        # beside it says another reads as two quantities.
        sigma_labels = VGroup(*[
            MathTex(rf'\sigma = {spread:.0f}', font_size=36, color=colour)
            for spread, colour, _, _ in entries
        ]).arrange(DOWN, buff=SIGMA_COLUMN_GAP, aligned_edge=LEFT).move_to(
            [SIGMA_COLUMN_X, SIGMA_COLUMN_Y, 0])

        drawn_curves = []
        for label_index, (spread, colour, line, existing) in enumerate(entries):
            curve = existing if existing is not None else self._bell(spread, colour, 1.0)
            value = sigma_labels[label_index]
            with self.voiceover(text=line):
                if existing is None:
                    self.play(Create(curve), FadeIn(value), run_time=1.3)
                else:
                    self.play(FadeIn(value), run_time=0.5)
                self.wait(0.6)
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
        # No labels on the legs. Each leg is drawn in its curve's colour and that colour's
        # sigma is standing in the column to the right, so labelling them here would have put
        # the same three numbers on screen twice -- and the hypotenuse three times, counting
        # the arithmetic below.
        # Each number wears its curve's colour, so the equation can be read against the three
        # curves standing under it without a word joining them up.
        blue_text = f'{cross_player:.0f}'
        red_text = f'{week_to_week:.0f}'
        total_text = f'{np.hypot(cross_player, week_to_week):.0f}'
        arithmetic = MathTex(
            rf'\sqrt{{{blue_text}^2 + {red_text}^2}} = {total_text}',
            font_size=46, color=WHITE,
            substrings_to_isolate=[blue_text, red_text, total_text],
        ).move_to([2.55, 1.75, 0])
        arithmetic.set_color_by_tex(blue_text, BLUE_B)
        arithmetic.set_color_by_tex(red_text, RED_B)
        arithmetic.set_color_by_tex(total_text, YELLOW)

        # Held on the scene so the closing beat can clear exactly what this one built.
        self.quadrature_working = VGroup(triangle, right_angle, arithmetic, sigma_labels)

        with self.voiceover(text=NARRATION['the_question']):
            self.play(Create(triangle), Create(right_angle), run_time=1.5)
            self.play(Write(arithmetic), run_time=1.3)
            self.wait(1.2)

        # The working stays up through "this new standard deviation", which is the thing it
        # worked out, and goes as the sentence reaches what is made of it. Cleared at the top of
        # the line instead, the quantity being named would vanish on the words naming it.
        with self.voiceover(text=NARRATION['g_scores']) as tracker:
            wait_until_phrase(self, tracker, 'we get G-scores',
                              lead_seconds=WORKING_FADE_SECONDS)
            self.play_g_score_formula()

    def play_g_score_formula(self) -> None:
        """Clear the working and leave the formula the scene was for.

        Everything that led here goes: the triangle, the arithmetic and the three measured
        numbers have all been said, and leaving them up while a new expression arrives asks the
        viewer to read four things at once.

        The curves stay. The sigmas in the formula are coloured rather than named, which only
        works while the things they are coloured AFTER are still on screen -- take the curves
        away and the colours stop referring to anything.
        """
        self.play(
            FadeOut(self.quadrature_working),
            run_time=WORKING_FADE_SECONDS,
        )

        formula = MathTex(
            r'\frac{x - \mu}{\sqrt{\sigma^2 + \sigma^2}}',
            font_size=64, color=WHITE,
        ).move_to(FORMULA_CENTRE)

        # Coloured a glyph at a time rather than by substring. Isolating the sigma makes Manim
        # re-render the expression in pieces, and the pieces are not brace-balanced: it dropped
        # the second exponent and put the colours on the radical and the plus sign. Glyph order
        # is stable for a fixed expression, so the two sigmas are addressed by position -- and
        # checked, because a changed expression would otherwise colour the wrong thing quietly.
        glyphs = formula[0]
        matching_pair = (len(glyphs) == FORMULA_GLYPH_COUNT
                         and abs(glyphs[FIRST_SIGMA_GLYPH].width
                                 - glyphs[SECOND_SIGMA_GLYPH].width) < 0.02)
        if not matching_pair:
            raise ValueError(
                f'the G-score formula rendered {len(glyphs)} glyphs rather than '
                f'{FORMULA_GLYPH_COUNT}, or glyphs {FIRST_SIGMA_GLYPH} and '
                f'{SECOND_SIGMA_GLYPH} are no longer the matching pair of sigmas. The colours '
                f'go on by position, so a changed expression has to be re-measured here.')
        glyphs[FIRST_SIGMA_GLYPH].set_color(BLUE_B)
        glyphs[SECOND_SIGMA_GLYPH].set_color(RED_B)

        self.play(Write(formula), run_time=1.4)
        # Short: the line running over this beat carries on past the formula, into what G-scores
        # are and are not, and the voiceover block holds the frame for as long as that takes.
        self.wait(0.8)

    def construct(self) -> None:
        # One draft, many weeks: the assumption the Z-score scene made, taken away.
        self.play_simulation(
            FIXED_MATCHUP_DATA, 'the same two teams, a different week',
            NARRATION['fixed_matchup'], NARRATION['fixed_matchup_result'],
            'The result is another bell curve',
            NARRATION['fixed_matchup_win_rate'], deal_during_opening=False)
        self.clear_frame()

        # Both varying, which is what a real matchup is.
        # No win-rate beat here: with the draft varying too, neither side is a team that could
        # have a record, so there is no share of weeks to point at.
        self.play_simulation(
            BOTH_VARY_DATA, 'a different draft, in a different week',
            NARRATION['both_vary'], NARRATION['both_vary_result'],
            'quite wide', None, deal_during_opening=True)
        self.clear_frame()

        self.play_quadrature()


class GScoreFixedMatchupActOne(WeeklyUnitDifferential):
    """Act one of the first simulation alone, for tuning without rendering the whole thing."""

    def construct(self) -> None:
        self.build_static_frame()
        self.play_act_one_single_draw()
