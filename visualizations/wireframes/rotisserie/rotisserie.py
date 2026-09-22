"""Rotisserie: why the algorithm wants your season to be uncertain.

WIREFRAME -- a first draft to find out whether the argument lands, drafted against gTTS rather
than the shipped voice. See wireframes/README.md.

The docs put the claim in one sentence: winning a league needs an aberrant result, so variance
is worth having, and a Bernoulli's variance p(1-p) is largest at p = 0.5 -- which is why the
Rotisserie algorithm holds categories near fifty-fifty instead of punting them.

The scene argues it by simulation rather than by algebra, because the probability of winning a
league is a comparison between your season total and the BEST of eleven others -- a convolution
against an order statistic, which is not a shape anyone reads off a picture. Simulated seasons
is shaded instead: the column chart is how often you finish on each total, and the yellow laid
over it is how often that total was enough to win. The visible yellow mass IS the win
probability -- the convolution drawn rather than asserted, with nothing left to take on trust.

The result it lands on is stronger than "same mean, more spread", and it is measured here rather
than asserted:

    everything 50/50        total 58.5 +/- 10.3    wins  7.8%
    five locked, four out   total 64.0 +/-  0.5    wins  0.2%

The committed team scores MORE on average and wins the league about forty times less often. It
is pinned too tightly to its own average ever to reach a bar that sits around 75.

    manim -ql visualizations/wireframes/rotisserie/rotisserie.py Rotisserie
"""

from __future__ import annotations

import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np
from manim import (
    VGroup, Line, DashedLine, Rectangle, Text, Dot,
    FadeIn, FadeOut, Create, Write, Transform,
    DOWN, UP, RIGHT,
    BLUE_B, GREEN_C, RED_B, GREY_B, GREY_D, YELLOW,
)
from manim_voiceover import VoiceoverScene

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.draft_voice import DraftVoice                 # noqa: E402
from shared.narration_timing import wait_until_phrase     # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from narration import NARRATION                           # noqa: E402


# -- The league ----------------------------------------------------------------------

TEAMS = 12
CATEGORIES = 9
MAX_POINTS = TEAMS * CATEGORIES            # 108: first in every category
AVERAGE_POINTS = (TEAMS + 1) / 2 * CATEGORIES
# The reachable ends and the middle. NOT zero: a team that finishes last in all nine categories
# still scores nine, because the worst place in a category pays one point rather than none.
MIN_POINTS = CATEGORIES * 1
# The middle of 9..108 is 58.5, which is ALSO what an average team scores: points 1..12 are
# symmetric about 6.5, so nine categories of them centre on 9 x 6.5. The midpoint of the scale
# and the mean of the distribution are the same number here.
AXIS_TICKS = (MIN_POINTS, AVERAGE_POINTS, MAX_POINTS)

# Nothing is drawn per season any more -- the seasons only supply the distribution and
# the conditional win rate -- so this can be large enough to make both smooth.
SEASONS_DRAWN = 60_000
SIMULATION_SEED = 7

# The two builds. Values are per-category strengths for the focal team; every rival draws around
# zero, so a strength of zero is a coin flip against each of them.
BALANCED_BUILD = ([0.0] * CATEGORIES, 1.0)
COMMITTED_BUILD = ([3.0] * 5 + [-3.0] * 4, 0.30)

# -- Layout ---------------------------------------------------------------------------

AXIS_Y = -2.9
AXIS_LEFT_X = -6.0
AXIS_RIGHT_X = 6.0
CLOUD_HEIGHT = 4.2         # height of the most common total; the rest scale against it
# The winning bar is drawn shorter, so it reads as background against the team's own curve.
THRESHOLD_HEIGHT = 2.4


class Rotisserie(VoiceoverScene):
    """What a Rotisserie score is, how hard the bar is, and why spread beats a better average."""

    def construct(self) -> None:
        self.set_speech_service(DraftVoice())
        self.rng = np.random.default_rng(SIMULATION_SEED)
        self.play_the_scale()
        self.play_the_bar()
        self.play_widening()
        self.play_two_builds()

    # -- Shared apparatus --------------------------------------------------------------

    def to_scene_x(self, points: float) -> float:
        return AXIS_LEFT_X + (AXIS_RIGHT_X - AXIS_LEFT_X) * points / MAX_POINTS

    def build_axis(self) -> VGroup:
        """The season-points scale, with its ends and midpoint written on it.

        Marked rather than bare: the scene is entirely about where a total sits relative to a
        bar, and an unlabelled line gives a viewer nothing to place either against. It also used
        to sit on screen alone under a long sentence, which is what made the opening feel empty.
        """
        axis = Line([AXIS_LEFT_X, AXIS_Y, 0.0], [AXIS_RIGHT_X, AXIS_Y, 0.0],
                    color=GREY_B, stroke_width=3)
        marks = VGroup()
        for points in AXIS_TICKS:
            x = self.to_scene_x(points)
            marks.add(Line([x, AXIS_Y - 0.12, 0.0], [x, AXIS_Y + 0.12, 0.0],
                           color=GREY_B, stroke_width=3))
            written = f'{points:g}'
            marks.add(Text(written, font_size=22, color=GREY_B)
                      .move_to([x, AXIS_Y - 0.38, 0.0]))
        caption = Text('Fantasy points', font_size=23, color=GREY_B)
        caption.move_to([0.0, AXIS_Y - 0.82, 0.0])
        return VGroup(axis, marks, caption)

    def simulate(self, build) -> tuple[np.ndarray, np.ndarray]:
        """Season totals for the focal team and for the best rival, by the real scoring rule.

        Every team draws a value per category and the ranks pay 1..12 points. That IS Rotisserie
        scoring, so the distributions the scene draws come out of the rule rather than being
        fitted to it -- including the fact that the bar is the MAXIMUM of eleven others, which is
        why it sits so far above average.
        """
        means, spread = build
        rivals = self.rng.standard_normal((SEASONS_DRAWN, TEAMS - 1, CATEGORIES))
        focal = (self.rng.standard_normal((SEASONS_DRAWN, 1, CATEGORIES)) * spread
                 + np.array(means))
        board = np.concatenate([focal, rivals], axis=1)
        ranks = board.argsort(axis=1).argsort(axis=1) + 1
        totals = ranks.sum(axis=2)
        return totals[:, 0], totals[:, 1:].max(axis=1)

    def build_win_shaded_distribution(self, totals, bar, colour) -> VGroup:
        """Your season totals as a column chart, each column tinted by how often it won.

        The column height is how often you finish on that total. The yellow laid over it is
        P(win | total = x) -- taken from the seasons that actually landed there, so the shading
        is a measured conditional rather than a curve fitted to look right.

        That conditioning matters. Your total and the bar are NOT independent: ranks are shared,
        so a season where you do well is one where the rivals were pushed down. Multiplying your
        density by the rivals' CDF as if they were independent gives 6.3% here against a true
        7.8%, wrong by a fifth. Reading the conditional off the same seasons keeps it exact --
        the shaded mass integrates to 7.75% against 7.77% counted directly.
        """
        lowest, highest = int(totals.min()), int(totals.max())
        counts = np.bincount(totals - lowest, minlength=highest - lowest + 1).astype(float)
        tallest = counts.max()
        width = (self.to_scene_x(1) - self.to_scene_x(0)) * 0.92

        columns = VGroup()
        for offset, count in enumerate(counts):
            if count <= 0:
                continue
            total = lowest + offset
            landed = totals == total
            enough = float(np.mean(bar[landed] < total)) if landed.sum() else 0.0
            height = CLOUD_HEIGHT * count / tallest
            centre = [self.to_scene_x(total), AXIS_Y + height / 2, 0.0]
            columns.add(VGroup(
                Rectangle(width=width, height=height, stroke_width=0,
                          fill_color=colour, fill_opacity=0.50).move_to(centre),
                Rectangle(width=width, height=height, stroke_width=0,
                          fill_color=YELLOW, fill_opacity=enough).move_to(centre),
            ))
        return columns

    def build_threshold_distribution(self, bar, height: float = THRESHOLD_HEIGHT,
                                     opacity: float = 0.32) -> VGroup:
        """What it took to win, as its own curve, sitting behind the team's.

        The bar is the best of eleven rivals, so it is a distribution in its own right and a
        different one every season. Drawn as a single dashed line it looked like a fixed target,
        which is the one thing it is not -- and it is why a modest season can still win a weak
        year.
        """
        lowest, highest = int(bar.min()), int(bar.max())
        counts = np.bincount(bar - lowest, minlength=highest - lowest + 1).astype(float)
        tallest = counts.max()
        width = (self.to_scene_x(1) - self.to_scene_x(0)) * 0.92

        columns = VGroup()
        for offset, count in enumerate(counts):
            if count <= 0:
                continue
            column_height = height * count / tallest
            columns.add(Rectangle(width=width, height=column_height, stroke_width=0,
                                  fill_color=RED_B, fill_opacity=opacity)
                        .move_to([self.to_scene_x(lowest + offset),
                                  AXIS_Y + column_height / 2, 0.0]))
        return columns

    def win_rate(self, totals, bar) -> float:
        """How often the season beat that year's bar."""
        return float(np.mean(totals > bar))

    # -- Act one: what a Rotisserie point is -------------------------------------------

    def play_the_scale(self) -> None:
        """Straight onto the axis. Anyone watching a Rotisserie explainer knows what Rotisserie
        is, so the scoring is stated in one sentence rather than taught in an act."""
        with self.voiceover(text=NARRATION['the_scale']) as tracker:
            self.axis = self.build_axis()
            self.play(Create(self.axis), run_time=1.2)

            # What a category pays, shown while the line says it. Without this the axis sat
            # alone for twelve seconds under a running sentence, which is the fault this scene
            # has already been rewritten once to remove.
            wait_until_phrase(self, tracker, 'twelve points')
            paid = VGroup(
                Text('1st in a category', font_size=24, color=GREY_B),
                Text('12 points', font_size=26, color=BLUE_B),
                Text('last in a category', font_size=24, color=GREY_B),
                Text('1 point', font_size=26, color=GREY_D),
            ).arrange_in_grid(rows=2, cols=2, buff=(0.6, 0.35))
            paid.move_to([0.0, 1.1, 0.0])
            self.play(FadeIn(paid), run_time=1.0)

            wait_until_phrase(self, tracker, 'about fifty eight')
            self.play(FadeOut(paid), run_time=0.5)
            average = DashedLine([self.to_scene_x(AVERAGE_POINTS), AXIS_Y, 0.0],
                                 [self.to_scene_x(AVERAGE_POINTS), AXIS_Y + 1.1, 0.0],
                                 color=GREY_B, stroke_width=2, dash_length=0.1)
            label = Text('an average team', font_size=21, color=GREY_B)
            label.next_to(average, UP, buff=0.1)
            self.play(Create(average), FadeIn(label), run_time=0.9)
            self.average_mark = VGroup(average, label)
            self.wait(0.6)

    # -- Act two: the bar, and how rarely anyone clears it -----------------------------

    def play_the_bar(self) -> None:
        with self.voiceover(text=NARRATION['the_bar']) as tracker:
            self.mine, self.bar = self.simulate(BALANCED_BUILD)
            wait_until_phrase(self, tracker, 'lands around')
            # Introduced at full height and explained on its own. It only becomes background
            # once the team's curve arrives to sit in front of it -- coming up already faint
            # would make it scenery before anyone had been told what it is.
            self.bar_line = self.build_threshold_distribution(
                self.bar, height=CLOUD_HEIGHT, opacity=0.55)
            bar_label = Text('what it took to win', font_size=22, color=RED_B)
            bar_label.move_to([self.to_scene_x(self.bar.mean()) + 1.6,
                               AXIS_Y + CLOUD_HEIGHT + 0.25, 0.0])
            self.play(FadeOut(self.average_mark), run_time=0.4)
            self.play(FadeIn(self.bar_line), FadeIn(bar_label), run_time=1.0)
            self.bar_label = bar_label
            self.wait(1.0)

        with self.voiceover(text=NARRATION['simulate']) as tracker:
            # Now it goes to the back, shorter and fainter, and the team's own curve takes the
            # front of the frame.
            receded = self.build_threshold_distribution(self.bar)
            self.play(Transform(self.bar_line, receded),
                      self.bar_label.animate.move_to(
                          [self.to_scene_x(self.bar.mean()) + 1.6,
                           AXIS_Y + THRESHOLD_HEIGHT + 0.3, 0.0]),
                      run_time=1.0)
            self.cloud = self.build_win_shaded_distribution(self.mine, self.bar, BLUE_B)
            self.play(FadeIn(self.cloud), run_time=max(1.4, tracker.duration * 0.35))
            self.wait(0.6)

        with self.voiceover(text=NARRATION['its_hard']):
            won = self.win_rate(self.mine, self.bar)
            self.win_readout = Text(f'won the league in {won:.1%} of seasons',
                                    font_size=28, color=BLUE_B)
            self.win_readout.move_to([0.0, 3.1, 0.0])
            self.play(Write(self.win_readout), run_time=1.0)
            self.wait(1.6)

    # -- Act three: spread is worth something the average is not -----------------------

    def play_widening(self) -> None:
        with self.voiceover(text=NARRATION['widen']):
            self.wait(2.0)

        with self.voiceover(text=NARRATION['why_wide']):
            # PLACEHOLDER: the two tails want opposite treatment -- the left greyed as "losing
            # either way", the right picked out as the part that gained. Wants the cloud redrawn
            # at a wider spread with the mean pinned, which is a second simulation.
            self.wait(2.4)
            self.play(FadeOut(VGroup(self.cloud, self.win_readout)), run_time=0.7)

    # -- Act four: two builds, their tables, and what they actually win ----------------

    def play_two_builds(self) -> None:
        # The first table comes up DURING this line rather than after it. Clearing the screen
        # and then waiting out the sentence left ten seconds of black with a voice over it,
        # which is the same fault the old opening had.
        table = self.build_matchup_table(BALANCED_BUILD, 'every category a coin flip', BLUE_B)
        with self.voiceover(text=NARRATION['two_builds']) as tracker:
            self.play(FadeOut(VGroup(self.bar_line, self.bar_label, self.axis)), run_time=0.6)
            self.play(FadeIn(table), run_time=1.2)

        with self.voiceover(text=NARRATION['coin_flips']):
            self.wait(2.0)

        with self.voiceover(text=NARRATION['certainties']):
            committed = self.build_matchup_table(
                COMMITTED_BUILD, 'five locked in, four abandoned', GREEN_C)
            self.play(FadeOut(table), run_time=0.4)
            self.play(FadeIn(committed), run_time=1.0)
            self.committed_table = committed

        with self.voiceover(text=NARRATION['the_result']) as tracker:
            self.axis = self.build_axis()
            self.play(FadeOut(self.committed_table), run_time=0.4)
            self.play(Create(self.axis), run_time=0.9)
            clouds, self.summaries = VGroup(), []
            for build, colour in ((BALANCED_BUILD, BLUE_B), (COMMITTED_BUILD, GREEN_C)):
                mine, bar = self.simulate(build)
                clouds.add(self.build_win_shaded_distribution(mine, bar, colour))
                self.summaries.append((float(mine.mean()), float(mine.std()),
                                       self.win_rate(mine, bar)))
            self.clouds = clouds
            self.play(FadeIn(clouds), run_time=1.6)
            wait_until_phrase(self, tracker, 'more points on average')
            self.wait(1.4)

        with self.voiceover(text=NARRATION['conclusion']):
            rows = VGroup(*[
                VGroup(
                    Text(name, font_size=23, color=colour),
                    Text(f'{mean:.1f} points, spread {spread:.1f}', font_size=21, color=GREY_B),
                    Text(f'wins {won:.1%}', font_size=25, color=colour),
                ).arrange(RIGHT, buff=0.45)
                for name, colour, (mean, spread, won) in (
                    ('every category a coin flip', BLUE_B, self.summaries[0]),
                    ('five locked, four abandoned', GREEN_C, self.summaries[1]),
                )
            ]).arrange(DOWN, buff=0.4).move_to([0.0, 2.9, 0.0])
            self.play(FadeIn(rows), run_time=1.2)
            self.wait(2.4)
        self.wait(0.6)

    def build_matchup_table(self, build, heading: str, colour) -> VGroup:
        """Chance of beating each rival in each category: nine rows, eleven opponents.

        A Rotisserie team plays everyone at once, so what settles a category is not one number
        but a row of them. A table is the honest shape for that; a single bar would be the head
        to head picture wearing Rotisserie's name.
        """
        means, spread = build
        title = Text(heading, font_size=26, color=colour).move_to([0.0, 2.9, 0.0])
        grid = VGroup()
        for category in range(CATEGORIES):
            # P(this team beats a neutral rival in this category), for strengths drawn around
            # these means against rivals drawn around zero.
            chance = 0.5 * (1.0 + erf(means[category] / sqrt(2.0 * (1.0 + spread ** 2))))
            row_y = 1.75 - category * 0.38
            for opponent in range(TEAMS - 1):
                grid.add(Rectangle(width=0.44, height=0.30, stroke_width=0,
                                   fill_color=colour, fill_opacity=max(0.05, chance))
                         .move_to([-3.6 + opponent * 0.50, row_y, 0.0]))
            grid.add(Text(f'{chance:.0%}', font_size=19, color=GREY_B)
                     .move_to([2.5, row_y, 0.0]))
        legend = Text('one row per category, one column per opponent',
                      font_size=20, color=GREY_D).move_to([0.0, -1.6, 0.0])
        return VGroup(title, grid, legend)
