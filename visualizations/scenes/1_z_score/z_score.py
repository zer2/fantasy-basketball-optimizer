"""Why a category is worth winning: two random teams, and the gap between them.

Deals two thirteen-player rosters at random out of the pool a standard league drafts, totals
each side's per-game Points, and drops the difference into a histogram in the middle. Repeated
ten thousand times the histogram is a bell -- which is the fact every later scene leans on,
since it is what makes a category something you can be favoured or unfavoured to win rather
than a number you simply accumulate.

Everything shown is real: 2025-26 per-game scoring for the 156 players a twelve-team league
drafts, chosen by G-score. Run `python visualizations/shared/prepare_season_data.py` to build the data
and headshots this reads; the draws are seeded there, so a re-render reproduces the same video.

Every player contributes the same number in every simulation here -- their season average. The
companion scene in weekly_differential.py deals each player a real week instead, which is what
that average is hiding.

Render one act while tuning it, the whole thing when it is right:

    manim -ql visualizations/1_z_score/z_score.py ActOneSingleDraw
    manim -qh visualizations/1_z_score/z_score.py TeamDifferentialFull
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import (
    Scene, Group, VGroup, Circle, Line, Text, MathTex,
    FadeIn, FadeOut, Write,
    DOWN, UP, RIGHT, BLACK, WHITE, GREY_B, YELLOW,
)
from manim_voiceover import VoiceoverScene

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.differential_base import DifferentialSceneBase   # noqa: E402
from shared.narration_timing import wait_until_phrase   # noqa: E402
from shooting_percentage import ShootingPercentageAct   # noqa: E402
from shared.narration_voice import NarrationVoice   # noqa: E402
from narration import NARRATION   # noqa: E402


# ── The opening: the whole draftable pool, and two teams pulled out of it ─────────────

POOL_GRID_COLUMNS = 13        # 13 x 12 lays the 156 drafted players out exactly
POOL_CHIP_SPACING_X = 0.72
POOL_CHIP_SPACING_Y = 0.60
POOL_CHIP_DIAMETER = 0.48
POOL_GRID_CENTRE_Y = 0.15
POOL_VEIL_OPACITY = 0.78      # how far an unchosen player is dimmed behind his veil
POOL_FLASH_SECONDS = 1.5      # one roster picked out, held long enough to read, and dimmed again
# How many hand-dealt draws come before the montage. Fewer than the other scene's six, and the
# reason is a deadline: "begins to look like a bell curve" is said 12.7s into the
# line, and the montage's fast early gear is the only part that puts a bell on screen. Every
# second spent re-dealing rosters pushes that gear later. Three draws still show the repetition
# -- deal, total, drop a bar, again -- and leave the montage well under way when the words land.
REPEATED_DRAWS = 3

# -- What a few more points are worth, highlighted on the histogram ------------------
# The closing pair of formulas is written in silence, then held a moment before the last line
# starts, so the verdict lands on a finished picture instead of arriving alongside it. The whole
# silence is this plus the writing -- about two and a half seconds, which is what the two
# formulas need to appear; shortening it further means the line starts over a formula still
# being drawn, which is the thing it was moved to avoid.
CONCLUSION_SETTLE_SECONDS = 0.2
# What the closing line holds on afterwards. The line is longer than anything left to animate,
# so this only has to exist; the voiceover block runs until the sentence is finished.
CONCLUSION_HOLD_SECONDS = 1.4
# And how long the finished frame is held after the last word, before the video ends.
FINAL_FRAME_HOLD_SECONDS = 1.0

SHADE_CAPTION = 'a few more points'
SHADE_CAPTION_X = 2.1         # how far to the side the label sits, so its leader reads as one


class SeasonAverageDifferential(DifferentialSceneBase):
    """The averages cut: a player is worth their weekly average, every single time."""

    data_filename      = 'pool_2025_26.json'
    # Three standard deviations either way, as in the weekly scene. Both scenes count the same
    # thing in the same unit -- points in a week -- so the two axes can be read against each
    # other: 270 here against 420 there is the whole difference real weeks make.
    differential_limit = 270
    bin_width          = 15
    axis_tick_step     = 90
    spread_caption     = 'σ = {spread:.0f} points in the week'
    # This scene keeps the markers standing long after its caption has gone -- through the
    # density formula and into the Z-score itself -- so they are named where they stand. Sigma
    # is what the algebra that follows is entirely about, and two unexplained grey lines are
    # not an argument for it.
    spread_marker_labels = ('−σ', '+σ')

    # ── Act zero: the pool everyone is drafting out of ────────────────────────────────

    def _pool_grid_position(self, pool_index: int) -> np.ndarray:
        row, column = divmod(pool_index, POOL_GRID_COLUMNS)
        rows = -(-len(self.prepared['pool']) // POOL_GRID_COLUMNS)
        return np.array([
            (column - (POOL_GRID_COLUMNS - 1) / 2) * POOL_CHIP_SPACING_X,
            POOL_GRID_CENTRE_Y + ((rows - 1) / 2 - row) * POOL_CHIP_SPACING_Y,
            0.0,
        ])

    def reveal_pool(self) -> None:
        """Put every drafted player on screen, before anything is said about them.

        Selection is shown by dimming everyone else, and the dimming is done with a dark disc laid
        OVER each portrait rather than by fading the portrait itself. The headshots are circular
        PNGs with transparent corners, and Manim's set_opacity replaces per-pixel alpha with a
        single value -- fading one directly would square it off into a grey block. A veil is a
        vector object and dims exactly as expected.
        """
        self.pool_chips, self.pool_veils = Group(), VGroup()
        for pool_index in range(len(self.prepared['pool'])):
            position = self._pool_grid_position(pool_index)
            portrait = self._headshot(pool_index)
            portrait.height = POOL_CHIP_DIAMETER
            portrait.move_to(position)
            self.pool_chips.add(portrait)
            self.pool_veils.add(Circle(
                radius=POOL_CHIP_DIAMETER / 2, stroke_width=0,
                fill_color=BLACK, fill_opacity=POOL_VEIL_OPACITY,
            ).move_to(position))

        self.add(self.pool_chips, self.pool_veils)
        self.play(FadeIn(self.pool_chips), FadeIn(self.pool_veils), run_time=1.2)
        self.wait(0.6)

    def flash_pool_draws(self, seconds_available: float) -> None:
        """Pull a random thirteen out of the pool, over and over, for as long as the clause runs.

        Thirteen, not twenty-six: this is one roster being chosen, and the second team is what
        act one introduces. The draws are the ones the rest of the scene goes on to use, so the
        team dealt in act one is one the viewer has already watched being picked.

        The count is not fixed. A fixed three ran out with half the line still to go, leaving
        the pool gone through the part of the sentence still describing it being drawn from.
        However many flashes fit are played instead, each stretched a little so the last one
        lands on the last word rather than short of it.
        """
        flashes = max(1, round(seconds_available / POOL_FLASH_SECONDS))
        seconds_each = seconds_available / flashes
        for simulation_index in range(flashes):
            drawn = self.prepared['rosters'][simulation_index][:self.team_size]
            self.play(*[self.pool_veils[int(pool_index)].animate.set_opacity(0.0)
                        for pool_index in drawn], run_time=seconds_each * 0.30)
            self.wait(seconds_each * 0.46)
            self.play(self.pool_veils.animate.set_opacity(POOL_VEIL_OPACITY),
                      run_time=seconds_each * 0.24)

    def dismiss_pool(self) -> None:
        """Take the pool away, once the line that was about it has finished saying so."""
        self.play(FadeOut(self.pool_chips), FadeOut(self.pool_veils), run_time=0.9)

    # -- What one more point is worth -------------------------------------------------

    def highlight_first_bar(self) -> None:
        """Fill the whole first bar to the right of a dead heat.

        Every matchup standing in that bar is one a handful of points carries from a loss into a
        win, and the bar is tall because the middle of the distribution is where the matchups
        are. Which is the claim the line makes, and the reason the height in the middle is the
        number the next two beats go after.

        A whole bar rather than a single point of ground. One point is a four-hundredth of this
        axis: drawn honestly it is two pixels wide, and a two-pixel sliver cannot carry a point
        about area. A bin is fifteen points, wide enough to see and still narrow enough that its
        height is the height in the middle.
        """
        # Found by where it stands rather than by index: empty bins are never built, so a bar's
        # position in the group says nothing about which bin it holds.
        bin_centre = self._x_of_differential(self.bin_width / 2)
        first_bar = min(self.frozen_bars,
                        key=lambda bar: abs(bar.get_center()[0] - bin_centre))
        self.first_bar_shading = (first_bar.copy()
                                  .set_fill(WHITE, opacity=0.9)
                                  .set_stroke(WHITE, width=2))

        # Set off to the side with a leader running back to it, rather than directly above:
        # above, the leader and the bar line up into one vertical stroke and the label stops
        # looking like a label.
        caption = Text(SHADE_CAPTION, font_size=20, color=WHITE)
        caption.move_to([SHADE_CAPTION_X, self.curve_peak_y + 1.05, 0])
        leader = Line([SHADE_CAPTION_X - 0.30, self.curve_peak_y + 0.86, 0],
                      [first_bar.get_center()[0], first_bar.get_top()[1] + 0.10, 0],
                      color=GREY_B, stroke_width=2)

        self.play(FadeIn(self.first_bar_shading), run_time=0.5)
        self.play(FadeIn(caption), Write(leader), run_time=0.6)
        self.first_bar_label = VGroup(caption, leader)
        self.wait(1.2)


class ActOneSingleDraw(SeasonAverageDifferential):
    def construct(self) -> None:
        self.build_static_frame()
        self.play_act_one_single_draw()


class ActTwoRepeatedDraws(SeasonAverageDifferential):
    def construct(self) -> None:
        self.build_static_frame()
        self.play_act_one_single_draw()
        self.play_act_two_repeated_draws()


class ActThreeMontage(SeasonAverageDifferential):
    def construct(self) -> None:
        self.build_static_frame()
        self.play_act_one_single_draw()
        self.play_act_two_repeated_draws()
        self.play_act_three_montage()


class ShootingFrame(ShootingPercentageAct, Scene):
    """The shooting apparatus alone, silent, for tuning what it looks like.

    Separate from the narrated scene because the act is built before its narration is written:
    a frame can be judged without spending a line on it.
    """

    def setup(self) -> None:
        super().setup()
        self.setup_shooting()

    def construct(self) -> None:
        self.play_shooting_frame()
        self.play_zoom_tanks()
        self.play_pull_comparison()
        self.play_volume_formula()
        self.play_volume_factor_emphasis()
        self.wait(0.5)


class ActZeroPool(SeasonAverageDifferential):
    """The opening alone, for tuning the pool grid without re-rendering the whole scene."""

    def construct(self) -> None:
        self.reveal_pool()
        self.flash_pool_draws(seconds_available=11.7)
        self.dismiss_pool()


class TeamDifferentialFull(VoiceoverScene, ShootingPercentageAct, SeasonAverageDifferential):
    """The Z-score scene, narrated.

    Each act plays inside the voiceover block that covers it, rather than every animation being
    timed to a clause. That keeps the mapping between line and act obvious, and means a rewritten
    line stretches or shortens only its own act -- if the audio outlasts the animation the last
    frame is held, and if the animation outlasts the audio it simply plays on.
    """

    def setup(self) -> None:
        super().setup()
        self.setup_shooting()
        self.set_speech_service(NarrationVoice())

    def play_act_six_z_score(self) -> None:
        """From the height of the curve to the formula everyone already knows.

        The density at a dead heat is one over sigma root two pi. Root two pi is a constant, so
        the only thing in it that varies between categories is sigma -- which means a category's
        importance is inversely proportional to its spread. Writing that down with an empty
        numerator and then filling the numerator in is the whole derivation of a Z-score, and it
        arrives as a consequence of the simulation rather than as a definition handed down.
        """
        self.play(
            FadeOut(self.density_equation),
            FadeOut(self.normal_curve),
            FadeOut(self.spread_markers),
            FadeOut(self.spread_marker_names),
            FadeOut(self.axis),
            FadeOut(self.peak_marker),
            FadeOut(self.first_bar_shading),
            FadeOut(self.first_bar_label),
            run_time=1.0,
        )

        # The fraction is assembled from separate pieces rather than typeset as one expression,
        # so the numerator can be WRITTEN INTO the empty space above a bar and a sigma that never
        # move. Transforming a whole fraction into another one would slide and morph the parts
        # that are meant to be standing still, and the point of the beat is that only the top
        # changes.
        vinculum = Line([-0.95, 0.0, 0.0], [0.95, 0.0, 0.0], color=YELLOW, stroke_width=5)
        denominator = MathTex(r'\sigma', font_size=96, color=YELLOW)
        denominator.next_to(vinculum, DOWN, buff=0.28)

        with self.voiceover(text=NARRATION['inverse_sigma']):
            self.play(Write(VGroup(vinculum, denominator)), run_time=1.2)
            self.wait(0.8)

        with self.voiceover(text=NARRATION['numerator']):
            numerator = MathTex(r'x - \mu', font_size=96, color=YELLOW)
            numerator.next_to(vinculum, UP, buff=0.28)
            self.play(Write(numerator), run_time=1.4)
            self.wait(0.6)

        with self.voiceover(text=NARRATION['z_score']):
            self.wait(1.2)

    def construct(self) -> None:
        # The pool is on screen before a word is said about it. Starting the line over the fade-in
        # meant the first sentence played to an empty frame.
        self.reveal_pool()

        # The grid then holds still through the sentence that is only about there being a pool,
        # and starts drawing teams out of it on the clause that says players are chosen at
        # random -- which is the first moment the picture has anything to add.
        with self.voiceover(text=NARRATION['pool']) as tracker:
            wait_until_phrase(self, tracker, 'that players are chosen randomly')
            self.flash_pool_draws(tracker.get_remaining_duration())
        self.dismiss_pool()

        self.build_static_frame()
        # One line, and the dealing follows straight on from it rather than waiting for a cue.
        #
        # The sentence to serve is "the margin looks more and more like a bell curve", which is
        # about a shape ARRIVING -- so the montage, which is the only part where it arrives, has
        # to be running while those words are said. Measured, the phrase lands at 11.1s. Held
        # back until "What you will notice" at 7.1s, the first bar and act two ate until 14.8s
        # and the montage began three and a half seconds AFTER the line said it was happening,
        # over six bars that looked like nothing at all.
        #
        # Starting as soon as the teams are dealt, and keeping the montage at its own pace, puts
        # several hundred draws on the board by the time the phrase arrives. Stretching it to
        # fill the line was tried and is worse: a slower montage means FEWER draws landed by any
        # given moment, so the words still arrive before the shape does.
        #
        # What the finished curve then does is stand through the central-limit explanation,
        # which is the picture that explanation is about.
        with self.voiceover(text=NARRATION['two_teams']) as tracker:
            self.play_act_one_single_draw()
            self.drop_first_bar()
            self.play_act_two_repeated_draws(through_simulation=REPEATED_DRAWS)
            self.play_act_three_montage()

        # The curve arrives on the sentence about the height in the middle; the shading arrives
        # on the sentence about moving right through it, which is the sentence that says what
        # the height is worth.
        with self.voiceover(text=NARRATION['bar_height']) as tracker:
            self.play_act_four_normal_curve()
            wait_until_phrase(self, tracker, 'If we move over to the right')
            self.highlight_first_bar()
        # The bars go as soon as the line starts, because the sentence about the height in the
        # middle is about the curve; the formula waits for the sentence that promises a formula.
        with self.voiceover(text=NARRATION['simple_expression']) as tracker:
            self.clear_for_the_formula()
            wait_until_phrase(self, tracker, 'Fortunately, there is a simple expression')
            self.play_act_five_density_at_zero()
        self.play_act_six_z_score()
        self.play_shooting_act()

    def play_shooting_act(self) -> None:
        """The percentages, which the formula just derived does not handle on its own.

        One line covers the whole act, so its three clauses are what the three beats run on:
        the apparatus arrives as percentages are raised, the tanks stretch as the constant-volume
        approximation is stated, the two players are compared as the change is described, the
        formula lands on the sentence that promises it, and its volume factor is pointed at as
        the last clause names it.
        """
        # Whatever act six left standing goes first: the Z-score formula has been reached and
        # the next thing said is that it is not the whole story. Cleared INSIDE the line rather
        # than before it -- the fade and a trailing wait between the two acts added two seconds
        # of silence after "a counting statistic like points", which is a long time to look at a
        # finished formula nobody is talking about.
        with self.voiceover(text=NARRATION['ratio_statistics']) as tracker:
            self.play(FadeOut(Group(*self.mobjects)), run_time=0.6)
            self.clear()
            self.play_shooting_frame()
            wait_until_phrase(self, tracker, 'Approximating that the total volume')
            self.play_zoom_tanks()
            wait_until_phrase(self, tracker, 'an additional player changes')
            self.play_pull_comparison()
            wait_until_phrase(self, tracker, 'That means we can use the same math')
            self.play_volume_formula()
            wait_until_phrase(self, tracker, 'This lines up with the Z-score formula')
            self.play_volume_factor_emphasis()

        self.play_conclusion()

    def play_conclusion(self) -> None:
        """Both formulas at once, side by side, with nothing else on the screen.

        The scene has derived them a long way apart -- the counting one out of the histogram, the
        percentage one out of the tanks -- and they have never been seen together. Put beside
        each other the difference is one factor wide, which is the whole claim: a percentage is
        not a different kind of quantity, it is the same one with volume in front of it.
        """
        counting = VGroup(
            Text('counting statistic', font_size=20, color=GREY_B),
            MathTex(r'\frac{x - \mu}{\sigma}', font_size=60, color=WHITE),
        ).arrange(DOWN, buff=0.34)
        percentage = VGroup(
            Text('percentage statistic', font_size=20, color=GREY_B),
            MathTex(r'\frac{v}{\bar{v}} \cdot \frac{x - \mu}{\sigma}',
                    font_size=60, color=WHITE),
        ).arrange(DOWN, buff=0.34)

        # Their formulas aligned rather than their captions: the two expressions are what is
        # being compared, and a caption that is a word longer should not shift the thing under it.
        both = VGroup(counting, percentage).arrange(RIGHT, buff=1.70)
        percentage[1].align_to(counting[1], DOWN)
        both.move_to([0.0, 0.0, 0.0])

        # Written BEFORE the closing line rather than under it. The line opens by summing the
        # scene up -- "so Z-scoring does have a justification" -- and it is summing up these two
        # formulas, so they have to be finished and standing when it starts. Written underneath
        # it, the verdict arrived while the thing being judged was still appearing.
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.6)
        self.clear()
        self.play(Write(counting), run_time=0.9)
        self.play(Write(percentage), run_time=1.0)
        self.wait(CONCLUSION_SETTLE_SECONDS)

        with self.voiceover(text=NARRATION['conclusion']):
            self.wait(CONCLUSION_HOLD_SECONDS)

        # The last thing the video does is sit still on the two formulas. Without this it cuts
        # on the final syllable, which reads as the file ending rather than the point landing.
        self.wait(FINAL_FRAME_HOLD_SECONDS)
