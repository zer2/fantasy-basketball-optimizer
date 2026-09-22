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
    DOWN, UP, LEFT, RIGHT,
    BLUE_B, BLUE_D, GREEN_C, RED_B, GREY_B, GREY_D, GREY_E, YELLOW,
    interpolate_color,
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
# 90% per category, not 100%. Two reasons, both measured. A build at 100/0 has almost no spread
# left (sigma 0.5), so its curve is 22x taller than the balanced one and the two cannot share a
# vertical scale -- and without a shared scale their areas are not comparable, though both are
# distributions with mass one. And a MILD tilt is not punished at all: at 70% per category the
# team wins MORE often than the balanced one (9.2% against 7.8%), because it gains mean faster
# than it loses spread. The penalty only appears once commitment is severe. 90% is where the
# argument is true and the picture is still drawable: 5.4% against 7.8%, and 2.2x the height.
_COMMITTED_STRENGTH = 1.8124      # Phi(0.90) x sqrt(2): beats a neutral rival 90% of the time
COMMITTED_BUILD = ([_COMMITTED_STRENGTH] * 5 + [-_COMMITTED_STRENGTH] * 4, 1.0)

# The two levers the widening act demonstrates, measured: a better team (58.5 -> 70.1 points,
# 7.7% -> 35.0%), and a wilder one at exactly the same expectation (sigma 10.3 -> 13.8, and
# 7.7% -> 14.5% on nothing but spread).
RICHER_BUILD = ([0.42] * CATEGORIES, 1.0)
WILDER_BUILD = ([0.0] * CATEGORIES, 2.6)

# -- Layout ---------------------------------------------------------------------------

AXIS_Y = -2.9
AXIS_LEFT_X = -6.0
AXIS_RIGHT_X = 6.0
CLOUD_HEIGHT = 4.2         # height of the most common total; the rest scale against it
# A finished season's standings: twelve teams down, nine categories across, totals on the right.
STANDINGS_TOP_Y = 2.55
STANDINGS_ROW_GAP = 0.40
STANDINGS_LEFT_X = -5.2
STANDINGS_COLUMN_GAP = 0.62
STANDINGS_TOTAL_X = 0.9
# A build's own season, shown beside its table: small, above the grid, on the same vertical
# scale as the overlay that follows so the two readings agree.
INSET_CURVE_SCALE = 0.42

# The build heat map: a row per category, a column per opponent, on a diverging scale.
CATEGORY_NAMES = ('Field Goal %', 'Free Throw %', 'Threes', 'Points', 'Rebounds',
                  'Assists', 'Steals', 'Blocks', 'Turnovers')
HEAT_TOP_Y = 2.1
HEAT_ROW_GAP = 0.44
HEAT_LEFT_X = -1.9
HEAT_CELL_WIDTH = 0.42
HEAT_CELL_HEIGHT = 0.36
HEAT_WEAK = RED_B        # losing the category
HEAT_NEUTRAL = GREY_E    # a coin flip
HEAT_STRONG = BLUE_D     # winning it
INSET_CURVE_CENTRE = [0.0, -2.75, 0.0]


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

    def build_win_shaded_distribution(self, totals, bar, colour,
                                      reference_peak: float | None = None,
                                      show_win_shading: bool = True) -> VGroup:
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
        probabilities = counts / counts.sum()
        # Height is probability against a SHARED reference, not against this curve's own peak.
        # Scaling each to its own peak makes two distributions that both integrate to one look
        # like different amounts of stuff, which is exactly the wrong thing to imply.
        reference = reference_peak if reference_peak is not None else probabilities.max()
        width = (self.to_scene_x(1) - self.to_scene_x(0)) * 0.92

        columns = VGroup()
        for offset, count in enumerate(counts):
            if count <= 0:
                continue
            total = lowest + offset
            landed = totals == total
            enough = float(np.mean(bar[landed] < total)) if landed.sum() else 0.0
            height = CLOUD_HEIGHT * probabilities[offset] / reference
            centre = [self.to_scene_x(total), AXIS_Y + height / 2, 0.0]
            bar_column = VGroup(Rectangle(width=width, height=height, stroke_width=0,
                                          fill_color=colour, fill_opacity=0.50).move_to(centre))
            if show_win_shading:
                bar_column.add(Rectangle(width=width, height=height, stroke_width=0,
                                         fill_color=YELLOW, fill_opacity=enough).move_to(centre))
            columns.add(bar_column)
        return columns

    def peak_probability(self, totals) -> float:
        """The tallest single total's share, which sets the shared vertical scale."""
        counts = np.bincount(totals - int(totals.min())).astype(float)
        return float(counts.max() / counts.sum())

    def build_threshold_distribution(self, bar, reference_peak: float,
                                     opacity: float = 0.32) -> VGroup:
        """What it took to win, as its own curve, sitting behind the team's.

        The bar is the best of eleven rivals, so it is a distribution in its own right and a
        different one every season. Drawn as a single dashed line it looked like a fixed target,
        which is the one thing it is not -- and it is why a modest season can still win a weak
        year.
        """
        lowest, highest = int(bar.min()), int(bar.max())
        counts = np.bincount(bar - lowest, minlength=highest - lowest + 1).astype(float)
        probabilities = counts / counts.sum()
        # The SAME probability-to-height scale the team's curve uses. Both are distributions
        # with mass one; drawn at their own peaks and different heights the bar looked like a
        # far smaller quantity than the team, which it is not.
        width = (self.to_scene_x(1) - self.to_scene_x(0)) * 0.92

        columns = VGroup()
        for offset, count in enumerate(counts):
            if count <= 0:
                continue
            column_height = CLOUD_HEIGHT * probabilities[offset] / reference_peak
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
        """A finished season's standings, which is the thing the rest of the scene abstracts.

        The act used to open on twelve labelled dots, which showed nothing: it said the league
        had twelve teams and stopped there. A real table says what the scene needs said -- that
        every team scores in every category, that the totals spread a long way, and that the one
        at the top is far above the middle rather than a little above it.
        """
        with self.voiceover(text=NARRATION['the_scale']) as tracker:
            board, totals = self.simulate_one_season()
            table = self.build_standings_table(board, totals)
            self.play(FadeIn(table['grid']), run_time=1.4)

            wait_until_phrase(self, tracker, 'scores 58.5 points')
            self.play(FadeIn(table['average']), run_time=0.8)
            self.wait(0.6)

            wait_until_phrase(self, tracker, 'far above that')
            self.play(FadeIn(table['winner']), run_time=0.9)
            self.wait(1.2)
            self.play(FadeOut(VGroup(table['grid'], table['average'], table['winner'])),
                      run_time=0.6)

    def simulate_one_season(self):
        """One finished season: every team's points in every category, sorted by total."""
        board = self.rng.standard_normal((TEAMS, CATEGORIES))
        ranks = board.argsort(axis=0).argsort(axis=0) + 1
        totals = ranks.sum(axis=1)
        order = np.argsort(-totals)
        return ranks[order], totals[order]

    def build_standings_table(self, board, totals) -> dict:
        """Twelve teams down, nine categories across, totals on the right, best first."""
        grid = VGroup()
        for column in range(CATEGORIES):
            grid.add(Text(f'C{column + 1}', font_size=17, color=GREY_D)
                     .move_to([STANDINGS_LEFT_X + column * STANDINGS_COLUMN_GAP,
                               STANDINGS_TOP_Y + 0.42, 0.0]))
        grid.add(Text('total', font_size=19, color=GREY_B)
                 .move_to([STANDINGS_TOTAL_X, STANDINGS_TOP_Y + 0.42, 0.0]))

        for row in range(TEAMS):
            y = STANDINGS_TOP_Y - row * STANDINGS_ROW_GAP
            leader = row == 0
            grid.add(Text(f'Team {row + 1}', font_size=18,
                          color=RED_B if leader else GREY_D)
                     .move_to([STANDINGS_LEFT_X - 1.5, y, 0.0], aligned_edge=LEFT))
            for column in range(CATEGORIES):
                points = int(board[row, column])
                grid.add(Text(str(points), font_size=18,
                              color=BLUE_B if points >= TEAMS - 2 else GREY_D)
                         .move_to([STANDINGS_LEFT_X + column * STANDINGS_COLUMN_GAP, y, 0.0]))
            grid.add(Text(str(int(totals[row])), font_size=21,
                          color=RED_B if leader else GREY_B)
                     .move_to([STANDINGS_TOTAL_X, y, 0.0]))

        average = VGroup(
            Text(f'an average team scores {AVERAGE_POINTS:.1f}', font_size=22, color=GREY_B),
        ).move_to([STANDINGS_TOTAL_X + 2.6, STANDINGS_TOP_Y - 5.5 * STANDINGS_ROW_GAP, 0.0])
        winner = VGroup(
            Text(f'the league is won on {int(totals[0])}', font_size=24, color=RED_B),
        ).move_to([STANDINGS_TOTAL_X + 2.6, STANDINGS_TOP_Y, 0.0])
        return {'grid': grid, 'average': average, 'winner': winner}

    # -- Act two: the bar, and how rarely anyone clears it -----------------------------

    def play_the_bar(self) -> None:
        with self.voiceover(text=NARRATION['the_bar']) as tracker:
            self.mine, self.bar = self.simulate(BALANCED_BUILD)
            # One scale for both curves, set by whichever peaks higher -- the bar is the
            # narrower of the two, so it is the one that sets it.
            self.reference = max(self.peak_probability(self.mine),
                                 self.peak_probability(self.bar))
            self.axis = self.build_axis()
            self.play(Create(self.axis), run_time=0.9)
            wait_until_phrase(self, tracker, 'cannot know exactly')
            # Introduced at full height and explained on its own. It only becomes background
            # once the team's curve arrives to sit in front of it -- coming up already faint
            # would make it scenery before anyone had been told what it is.
            self.bar_line = self.build_threshold_distribution(
                self.bar, self.reference, opacity=0.55)
            bar_label = Text('what it took to win', font_size=22, color=RED_B)
            bar_label.move_to([self.to_scene_x(self.bar.mean()) + 1.6,
                               AXIS_Y + CLOUD_HEIGHT + 0.25, 0.0])
            self.play(FadeIn(self.bar_line), FadeIn(bar_label), run_time=1.0)
            self.bar_label = bar_label
            self.wait(1.0)

        with self.voiceover(text=NARRATION['simulate']) as tracker:
            # Now it goes to the back, shorter and fainter, and the team's own curve takes the
            # front of the frame.
            # It recedes by going fainter, not by shrinking: shrinking it would break the
            # equal-area reading the shared scale exists to give.
            receded = self.build_threshold_distribution(self.bar, self.reference)
            self.play(Transform(self.bar_line, receded),
                      self.bar_label.animate.move_to(
                          [self.to_scene_x(self.bar.mean()) + 1.9,
                           AXIS_Y + CLOUD_HEIGHT + 0.25, 0.0]),
                      run_time=1.0)
            self.cloud = self.build_win_shaded_distribution(self.mine, self.bar, BLUE_B,
                                                            self.reference)
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
        """The two levers the line names, each done to the curve that is already on screen.

        Shifting right and widening are different moves with different costs, and the scene can
        show that they are BOTH worth something here -- which is the part that is specific to
        Rotisserie. Measured on these builds: the shift takes the win rate from 7.7% to 35.0%,
        and the widening takes it to 14.5% without improving the team at all.
        """
        with self.voiceover(text=NARRATION['widen']) as tracker:
            wait_until_phrase(self, tracker, 'increasing our expected value')
            self.play(Transform(self.cloud, self.variant_cloud(RICHER_BUILD)),
                      Transform(self.win_readout, self.win_line(RICHER_BUILD)),
                      run_time=1.6)
            self.wait(0.8)
            self.play(Transform(self.cloud, self.variant_cloud(BALANCED_BUILD)),
                      Transform(self.win_readout, self.win_line(BALANCED_BUILD)),
                      run_time=0.9)

            wait_until_phrase(self, tracker, 'increasing variance')
            self.play(Transform(self.cloud, self.variant_cloud(WILDER_BUILD)),
                      Transform(self.win_readout, self.win_line(WILDER_BUILD)),
                      run_time=1.6)
            self.wait(max(0.6, tracker.get_remaining_duration() - 0.9))
            self.play(FadeOut(VGroup(self.cloud, self.win_readout, self.bar_line,
                                     self.bar_label, self.axis)), run_time=0.8)

    def variant_cloud(self, build) -> VGroup:
        """The same team's season under one of the two levers, on the established scale."""
        mine, bar = self.simulate(build)
        return self.build_win_shaded_distribution(mine, bar, BLUE_B, self.reference)

    def win_line(self, build) -> Text:
        mine, bar = self.simulate(build)
        return Text(f'won the league in {self.win_rate(mine, bar):.1%} of seasons',
                    font_size=28, color=BLUE_B).move_to([0.0, 3.1, 0.0])

    # -- Act four: two builds, each with the season it produces ------------------------

    def play_two_builds(self) -> None:
        """Each build gets its table AND the distribution that table produces, then both are
        laid over one axis together.

        Introducing a build without showing what it lands on left the tables as assertions --
        the whole claim is about the shape of the season each one produces, so that shape
        belongs on screen while the build is being described.
        """
        runs = [self.simulate(build) for build in (BALANCED_BUILD, COMMITTED_BUILD)]
        # A single vertical scale across every distribution the act draws, set by the tallest,
        # so the two are comparable here and remain comparable when overlaid.
        reference = max(self.peak_probability(mine) for mine, _ in runs)
        self.summaries = [(float(mine.mean()), float(mine.std()), self.win_rate(mine, bar))
                          for mine, bar in runs]

        def build_panel(build, colour, heading, run):
            table = self.build_matchup_table(build, heading, colour)
            mine, bar = run
            # No win shading here. These panels are introducing a BUILD -- what the yellow
            # means belongs to the comparison that follows, and shown beside a table of win
            # probabilities it reads as a second, unexplained quantity.
            curve = self.build_win_shaded_distribution(mine, bar, colour, reference,
                                                       show_win_shading=False)
            curve.scale(INSET_CURVE_SCALE).move_to(INSET_CURVE_CENTRE)
            # A baseline under it, so the little curve reads as a distribution rather than as a
            # shape floating below the grid.
            floor = Line([-3.2, INSET_CURVE_CENTRE[1] - 0.62, 0.0],
                         [3.2, INSET_CURVE_CENTRE[1] - 0.62, 0.0],
                         color=GREY_D, stroke_width=2)
            caption = Text('the season it produces', font_size=19, color=GREY_D)
            caption.move_to([0.0, INSET_CURVE_CENTRE[1] - 0.92, 0.0])
            return table, VGroup(curve, floor, caption)

        first_table, first_curve = build_panel(
            BALANCED_BUILD, BLUE_B, 'every fantasy point a coin flip', runs[0])
        with self.voiceover(text=NARRATION['two_builds']):
            # Up from the first word. Anchored half way through the line instead, the first
            # twelve seconds of it played over a black screen.
            self.play(FadeIn(first_table), run_time=1.0)

        with self.voiceover(text=NARRATION['coin_flips']):
            self.play(FadeIn(first_curve), run_time=0.9)
            self.wait(1.2)

        with self.voiceover(text=NARRATION['certainties']):
            second_table, second_curve = build_panel(
                COMMITTED_BUILD, GREEN_C, 'most nearly won, the rest nearly lost', runs[1])
            self.play(FadeOut(VGroup(first_table, first_curve)), run_time=0.4)
            self.play(FadeIn(second_table), run_time=0.8)
            self.play(FadeIn(second_curve), run_time=0.8)
            self.second_panel = VGroup(second_table, second_curve)

        with self.voiceover(text=NARRATION['conclusion']) as tracker:
            # The committed panel is cleared HERE rather than at the end of its own line, which
            # left the rest of that sentence running over nothing.
            self.play(FadeOut(self.second_panel), run_time=0.5)
            self.axis = self.build_axis()
            self.play(Create(self.axis), run_time=0.9)
            clouds = VGroup(*[
                self.build_win_shaded_distribution(mine, bar, colour, reference)
                for (mine, bar), colour in zip(runs, (BLUE_B, GREEN_C))
            ])
            self.play(FadeIn(clouds), run_time=1.4)

            wait_until_phrase(self, tracker, 'too concentrated')
            rows = VGroup(*[
                VGroup(
                    Text(name, font_size=23, color=colour),
                    Text(f'{mean:.1f} points, spread {spread:.1f}', font_size=21, color=GREY_B),
                    Text(f'wins {won:.1%}', font_size=25, color=colour),
                ).arrange(RIGHT, buff=0.45)
                for name, colour, (mean, spread, won) in (
                    ('every point a coin flip', BLUE_B, self.summaries[0]),
                    ('most won, the rest lost', GREEN_C, self.summaries[1]),
                )
            ]).arrange(DOWN, buff=0.4).move_to([0.0, 2.9, 0.0])
            self.play(FadeIn(rows), run_time=1.2)
            self.wait(2.0)
        self.wait(0.6)

    def build_matchup_table(self, build, heading: str, colour) -> VGroup:
        """Chance of beating each rival in each category, as a heat map.

        A Rotisserie team plays everyone at once, so what settles a category is a row of numbers
        rather than one. Drawn as flat squares at varying opacity it read as a grid of the same
        colour repeated; on a diverging scale the shape of a build is legible at a glance -- a
        balanced one is a single flat tone, a committed one splits into two blocks.

        The scale is the scenes' own convention rather than a new one: red for a category being
        lost, blue for one being won, and the neutral ground between them for a coin flip.
        """
        means, spread = build
        title = Text(heading, font_size=26, color=colour).move_to([0.0, 2.95, 0.0])
        rows = VGroup()
        for category in range(CATEGORIES):
            chance = 0.5 * (1.0 + erf(means[category] / sqrt(2.0 * (1.0 + spread ** 2))))
            row_y = HEAT_TOP_Y - category * HEAT_ROW_GAP
            rows.add(Text(CATEGORY_NAMES[category], font_size=19, color=GREY_B)
                     .move_to([HEAT_LEFT_X - 0.55, row_y, 0.0], aligned_edge=RIGHT))
            for opponent in range(TEAMS - 1):
                rows.add(Rectangle(width=HEAT_CELL_WIDTH, height=HEAT_CELL_HEIGHT,
                                   stroke_width=0, fill_opacity=1.0,
                                   fill_color=self.heat_colour(chance))
                         .move_to([HEAT_LEFT_X + opponent * HEAT_CELL_WIDTH, row_y, 0.0]))
            rows.add(Text(f'{chance:.0%}', font_size=20, color=self.heat_colour(chance))
                     .move_to([HEAT_LEFT_X + (TEAMS - 1) * HEAT_CELL_WIDTH + 0.6, row_y, 0.0]))

        columns = Text('eleven opponents', font_size=19, color=GREY_D)
        columns.move_to([HEAT_LEFT_X + (TEAMS - 2) * HEAT_CELL_WIDTH / 2,
                         HEAT_TOP_Y + 0.45, 0.0])
        return VGroup(title, columns, rows)

    def heat_colour(self, chance: float):
        """Red where a category is being lost, blue where it is being won, neutral at a flip."""
        if chance >= 0.5:
            return interpolate_color(HEAT_NEUTRAL, HEAT_STRONG, (chance - 0.5) * 2.0)
        return interpolate_color(HEAT_NEUTRAL, HEAT_WEAK, (0.5 - chance) * 2.0)
