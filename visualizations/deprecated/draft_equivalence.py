"""Why drafting at random is not a cop-out.

The obvious objection to the Z-score scene is that nobody drafts at random. The answer is not
that it is a rough approximation -- it is that a value-ordered snake draft REDUCES to random
drafting, exactly, under one assumption about how value is spread across picks.

Assume total value is linear in pick order: the first pick is worth 4, the second 3.9, the third
3.8. On top of that, every player carries a category tilt drawn from a random process R which
moves value BETWEEN categories without changing the total. That value-neutrality is load-bearing
rather than decorative: because a tilt cannot change how good a player is, drafting by value is
exactly drafting in pick order, so the k-th pick really is the player carrying the k-th baseline.

Then the snake does the work. Every seat's pick numbers pair up to the same six totals -- seat one
takes picks 1 and 24, seat twelve takes 12 and 13, and both pairs come to 25 -- so every drafter
ends up with the same baseline and differs only in their draws of R. Which is the two-random-teams
model, arrived at rather than assumed.

The last beat concedes where it strains, with numbers from
`prepare_draft_equivalence_data.py`. Run that first.

    manim -ql visualizations/scenes/draft_equivalence.py DraftEquivalence
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from manim import (
    VGroup, Rectangle, Line, Text, MathTex,
    FadeIn, FadeOut, Create, Write, Transform, Indicate,
    DOWN, UP, LEFT, RIGHT,
    YELLOW, WHITE, GREY_B, GREY_D, BLUE_B, RED_B, GREEN_B,
)
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService


# ── The spoken track ─────────────────────────────────────────────────────────────────
# Edit only this block. Each step plays inside the line covering it, so a rewritten line
# retimes its own step rather than desynchronising everything after it.

NARRATION = {
    # The opening is split at clause boundaries rather than sentence ones, because the phrases
    # that matter are inside the sentences: "linear in pick order" is the baselines stepping
    # down, "with some random process" is R being drawn for the first time, and "we will call R"
    # is the letter itself arriving on screen. One long block let all three drift.
    'premise':
        'The random drafting setup can be equated to a more complex and realistic setup.',
    'linear':
        'We just need to assume that value per category is on average linear in pick order,',
    'random_process':
        'with some random process on top of that, with standard deviations proportional to the '
        'linear bonus.',
    'call_it_R':
        'We start with a baseline level of each category for the pick, then use a random process '
        'we will call R to add an individual category-level profile.',
    # Split so each half gets its own picture: the first states what a team IS, which wants an
    # equation, and the second goes looking at the board, which wants the board.
    'team_formula':
        'This means that team stats can be calculated as several draws of R, plus the baseline '
        'stats for their picks.',
    'plot_the_draft':
        'If we plot out a full snake draft, we will notice that the total of each drafter is '
        'pick numbers is the same. That means that their baseline stats will all cancel out, '
        'leaving each team with only their draws of R.',
    'scaling':
        'The standard deviation of a category for the whole player pool comes from R, plus a '
        'small amount from the linear adjustment. If R and the linear adjustment are '
        'proportional, as we assumed, the linear adjustment does not matter. The standard '
        'deviation of a category is still proportional to the standard deviation of R, '
        'validating Z-scores.',
    'caveat':
        'In reality this is not necessarily true: some categories have more inter-round variance '
        'than others. This approximation holds reasonably well for most categories, though there '
        'is a bit of an issue with points, since points tends to be so highly correlated with '
        'draft position that it tends to cancel out. But we are looking for an approximation '
        'here.',
    'dynamic':
        'This still is not quite right. But we should keep in mind that no static ranking system '
        'possibly can be. Fantasy basketball is inherently dynamic.',
}


# ── Layout ───────────────────────────────────────────────────────────────────────────

LADDER_SLOTS = 5              # how many picks the opening ladder spells out
LADDER_TOP_VALUE = 4.0
LADDER_STEP = 0.1

# Twelve rounds, not thirteen. The cancellation is exact only for an EVEN number of rounds: at
# thirteen the pick-number totals ladder from 1015 to 1026 and the first seat keeps an edge the
# snake never pays back. The rest of the series uses thirteen, so this scene says twelve out loud.
BOARD_SEATS = 12
BOARD_ROUNDS = 12

# Three picks side by side, so nine bars each have to fit in a third of the frame.
GROUPS_SHOWN = 3
GROUP_SPACING = 4.45
CATEGORY_BAR_WIDTH = 0.28
CATEGORY_BAR_GAP = 0.42
BASELINE_TOP = 0.62           # the first pick's per-category baseline
BASELINE_STEP = 0.07          # how much it steps down each pick
TILT_SCALE = 0.40             # R's spread, the same at every slot
BASELINE_AXIS_Y = 0.95
TILT_AXIS_Y = -1.70
PLUS_Y = -0.35                # the plus sits in the empty band between the two rows
R_LABEL_Y = -2.95             # clear of the deepest tilt bar

CAVEAT_BASELINE_Y = -2.0
CAVEAT_SCALE = 3.6            # frame units per unit of residual spread fraction
# Five, because the closing line runs about eleven seconds and three draws left the bars frozen
# under the half of it that is about nothing being frozen.
DYNAMIC_UNSETTLED_DRAWS = 5
DYNAMIC_WOBBLE_SPREAD = 0.09

_DATA_PATH = Path(__file__).resolve().parent.parent / 'data' / 'draft_equivalence.json'


def load_measurements() -> dict:
    if not _DATA_PATH.exists():
        raise FileNotFoundError(
            f'{_DATA_PATH} is missing. Run '
            f'`python visualizations/prepare_draft_equivalence_data.py` first.')
    return json.loads(_DATA_PATH.read_text(encoding='utf-8'))


def snake_board() -> list[list[int]]:
    """Overall pick numbers, [round][seat], snaking back and forth."""
    board, overall = [], 1
    for round_index in range(BOARD_ROUNDS):
        row = [0] * BOARD_SEATS
        seats = range(BOARD_SEATS) if round_index % 2 == 0 else reversed(range(BOARD_SEATS))
        for seat in seats:
            row[seat] = overall
            overall += 1
        board.append(row)
    return board


class DraftEquivalence(VoiceoverScene):
    """The ladder, the tilt, the snake, the cancellation, and where it strains."""

    def setup(self) -> None:
        self.measured = load_measurements()
        self.board = snake_board()
        self.set_speech_service(GTTSService(lang='en'))

    # ── Step one: value is linear in pick order ───────────────────────────────────────

    def play_step_one_ladder(self) -> VGroup:
        slots = VGroup()
        for index in range(LADDER_SLOTS):
            value = LADDER_TOP_VALUE - index * LADDER_STEP
            column = VGroup(
                Text(f'pick {index + 1}', font_size=20, color=GREY_B),
                Text(f'{value:.1f}', font_size=40, color=YELLOW),
            ).arrange(DOWN, buff=0.18)
            column.move_to([(index - (LADDER_SLOTS - 1) / 2) * 2.15, 1.15, 0])
            slots.add(column)

        # Deliberately unhurried. This is the assumption the whole argument rests on, and it is
        # five numbers -- a viewer needs a moment to notice they are evenly spaced.
        self.play(FadeIn(slots, lag_ratio=0.28), run_time=2.4)
        self.wait(1.3)

        # A straight line through the values, so the assumption is the visible thing rather than
        # a claim buried in the numbers.
        ladder_line = Line(
            slots[0][1].get_center() + np.array([-0.55, -0.42, 0]),
            slots[-1][1].get_center() + np.array([0.55, -0.42, 0]),
            color=GREY_D, stroke_width=3,
        )
        self.play(Create(ladder_line), run_time=1.1)
        self.wait(1.0)
        return VGroup(slots, ladder_line)

    # ── Step two: the baseline spreads into categories, and R tilts it ────────────────

    def _category_bars(self, heights, centre, colours) -> VGroup:
        """One player's category profile, as bars from a common baseline."""
        bars = VGroup()
        for index, (height, colour) in enumerate(zip(heights, colours)):
            bar = Rectangle(
                width=CATEGORY_BAR_WIDTH, height=max(abs(height), 1e-3),
                fill_color=colour, fill_opacity=0.85, stroke_width=0,
            )
            offset_x = (index - (len(heights) - 1) / 2) * CATEGORY_BAR_GAP
            bar.move_to([centre[0] + offset_x, centre[1] + height / 2, 0])
            bars.add(bar)
        return bars

    def _group_centre_x(self, group_index: int) -> float:
        return (group_index - (GROUPS_SHOWN - 1) / 2) * GROUP_SPACING

    def _row_half_width(self) -> float:
        return (len(self.categories) / 2) * CATEGORY_BAR_GAP

    def play_step_two_baselines(self) -> None:
        """Three consecutive picks, each a step lower than the one before it.

        Three rather than one, because the argument is about a WHOLE TEAM of picks: the baselines
        step down as the draft goes on while the tilts to come stay the same size, and it is that
        contrast -- an ordered part and a disorderly part of fixed magnitude -- that the
        cancellation later depends on. One player cannot show it.
        """
        self.categories = [row['category'] for row in self.measured['categories']]
        self.turnover_index = self.categories.index('Turnovers')
        self.category_colours = [RED_B if index == self.turnover_index else BLUE_B
                                 for index in range(len(self.categories))]
        self.step_two_mobjects = VGroup()

        headings, baseline_axes = VGroup(), VGroup()
        for group_index in range(GROUPS_SHOWN):
            centre_x = self._group_centre_x(group_index)
            headings.add(Text(f'pick {group_index + 1}', font_size=22, color=GREY_B)
                         .move_to([centre_x, 2.30, 0]))
            baseline_axes.add(Line([centre_x - self._row_half_width(), BASELINE_AXIS_Y, 0],
                                   [centre_x + self._row_half_width(), BASELINE_AXIS_Y, 0],
                                   color=GREY_B, stroke_width=2))
        self.play(FadeIn(headings), Create(baseline_axes), run_time=0.9)
        self.step_two_mobjects.add(headings, baseline_axes)

        # The ladder from step one, now spread across categories and stepping down pick by pick.
        # Turnovers point downward, since accumulating them is the bad direction.
        self.baseline_bars = VGroup()
        for group_index in range(GROUPS_SHOWN):
            height = BASELINE_TOP - group_index * BASELINE_STEP
            profile = np.full(len(self.categories), height)
            profile[self.turnover_index] = -(height * 0.72)
            bars = self._category_bars(
                profile,
                np.array([self._group_centre_x(group_index), BASELINE_AXIS_Y, 0]),
                self.category_colours)
            self.play(FadeIn(bars, lag_ratio=0.06), run_time=0.9)
            self.baseline_bars.add(bars)
            self.wait(0.5)
        self.step_two_mobjects.add(self.baseline_bars)

    def _draw_of_R(self) -> VGroup:
        """One tilt per pick: the same spread at every slot, and value-neutral at every slot."""
        draw = VGroup()
        for group_index in range(GROUPS_SHOWN):
            tilt = self.tilt_generator.normal(scale=TILT_SCALE, size=len(self.categories))
            tilt -= tilt.mean()          # a tilt redistributes value; it never adds any
            draw.add(self._category_bars(
                tilt, np.array([self._group_centre_x(group_index), TILT_AXIS_Y, 0]),
                [GREEN_B if height >= 0 else RED_B for height in tilt]))
        return draw

    def play_step_two_first_draw(self) -> None:
        """The plus, the second axis, and R appearing beneath it for the first time.

        This runs under "with some random process on top of that", so the bars for R have to
        start being drawn on those words rather than several seconds later. The plus between the
        rows is what makes the lower row read as something ADDED to the baseline instead of a
        second, unrelated chart.
        """
        self.tilt_generator = np.random.default_rng(1971)

        pluses, tilt_axes = VGroup(), VGroup()
        for group_index in range(GROUPS_SHOWN):
            centre_x = self._group_centre_x(group_index)
            pluses.add(MathTex('+', font_size=56, color=WHITE).move_to([centre_x, PLUS_Y, 0]))
            tilt_axes.add(Line([centre_x - self._row_half_width(), TILT_AXIS_Y, 0],
                               [centre_x + self._row_half_width(), TILT_AXIS_Y, 0],
                               color=GREY_B, stroke_width=2))
        self.play(FadeIn(pluses), Create(tilt_axes), run_time=0.8)

        # These do NOT shrink with pick order: the same random process at every slot.
        self.tilt_bars = self._draw_of_R()
        self.play(FadeIn(self.tilt_bars, lag_ratio=0.06), run_time=1.6)
        self.wait(1.2)
        # A second draw under "standard deviations proportional to the linear bonus", which is a
        # sentence about R's SPREAD -- something only visible once the bars have moved once.
        self.play(Transform(self.tilt_bars, self._draw_of_R()), run_time=0.9)
        self.wait(2.2)
        self.step_two_mobjects.add(pluses, tilt_axes, self.tilt_bars)

    def play_step_two_name_R(self) -> None:
        """Point at the baseline again, then give the lower row its letter.

        The letter lands on "we will call R" -- a name is worth showing at the moment it is
        given, and every later step writes R as if the viewer has already met it.
        """
        self.play(Indicate(self.baseline_bars, scale_factor=1.05, color=YELLOW), run_time=1.6)
        self.wait(1.6)

        # Two more draws while the sentence is still on "a random process". R has to look like a
        # process before it is handed a name, and the name has to wait for the words that give it.
        for _ in range(2):
            self.play(Transform(self.tilt_bars, self._draw_of_R()), run_time=0.9)
            self.wait(0.6)

        labels = VGroup(*[
            MathTex('R', font_size=48, color=GREEN_B)
            .move_to([self._group_centre_x(group_index), R_LABEL_Y, 0])
            for group_index in range(GROUPS_SHOWN)
        ])
        self.play(Write(labels), run_time=1.0)
        self.step_two_mobjects.add(labels)
        self.wait(1.2)

        # One more draw, now that the lower row has a name to be a draw OF.
        self.play(Transform(self.tilt_bars, self._draw_of_R()), run_time=0.9)
        self.wait(1.6)

    # ── Step two and a half: what a team is, written down ─────────────────────────────

    def play_step_two_and_a_half_formula(self) -> VGroup:
        """The claim the whole argument turns on, as one line of algebra.

        It arrives before the board rather than after it. The sentence being spoken states what a
        team IS -- draws of R plus baselines -- and that is an equation; the board is the evidence
        for what happens to the second term, and it belongs with the sentence that goes looking
        for it.
        """
        formula = MathTex(
            r'\text{team} \;=\; \underbrace{R + R + \cdots + R}_{\text{one draw per pick}}'
            r'\;+\;\underbrace{b_{k_1} + b_{k_2} + \cdots + b_{k_{12}}}_{\text{baseline of each pick}}',
            font_size=40, color=WHITE,
        )
        self.play(Write(formula), run_time=2.0)
        self.wait(1.6)
        return formula

    # ── Step three: the snake, paired ─────────────────────────────────────────────────

    def _board_cell_position(self, round_index: int, seat: int) -> np.ndarray:
        return np.array([
            (seat - (BOARD_SEATS - 1) / 2) * 1.06,
            2.55 - round_index * 0.44,
            0.0,
        ])

    def play_step_three_board(self) -> VGroup:
        cells = VGroup()
        for round_index, row in enumerate(self.board):
            for seat, pick in enumerate(row):
                cells.add(Text(str(pick), font_size=15, color=GREY_B)
                          .move_to(self._board_cell_position(round_index, seat)))
        self.play(FadeIn(cells, lag_ratio=0.01), run_time=2.2)
        self.wait(0.5)

        # Pair each odd round with the one below it. Every pair closes to the same number for
        # every seat -- which is the whole proof, and it is watchable rather than assertable.
        pair_totals = VGroup()
        for pair_index in range(0, 4, 2):
            highlight = VGroup(*[
                Rectangle(width=BOARD_SEATS * 1.06 + 0.3, height=0.42,
                          stroke_color=YELLOW, stroke_width=2, fill_opacity=0)
                .move_to([0, self._board_cell_position(pair_index + offset, 0)[1], 0])
                for offset in (0, 1)
            ])
            self.play(Create(highlight), run_time=0.6)

            sums = VGroup(*[
                Text(str(self.board[pair_index][seat] + self.board[pair_index + 1][seat]),
                     font_size=16, color=YELLOW)
                .move_to([self._board_cell_position(0, seat)[0], -3.05, 0])
                for seat in range(BOARD_SEATS)
            ])
            self.play(FadeIn(sums), run_time=0.7)
            self.wait(1.0)
            self.play(FadeOut(highlight), FadeOut(sums), run_time=0.4)
            pair_totals.add(sums)

        totals = VGroup(*[
            Text(str(sum(row[seat] for row in self.board)), font_size=18, color=YELLOW)
            .move_to([self._board_cell_position(0, seat)[0], -3.05, 0])
            for seat in range(BOARD_SEATS)
        ])
        self.play(FadeIn(totals), run_time=0.9)
        self.wait(1.4)
        return VGroup(cells, totals)

    # ── Step four: the baselines cancel ───────────────────────────────────────────────

    def play_step_four_cancellation(self) -> VGroup:
        first_seat = [row[0] for row in self.board]
        last_seat = [row[BOARD_SEATS - 1] for row in self.board]
        total = sum(first_seat)

        lines = VGroup(
            MathTex(rf'\text{{seat 1}}: \quad \sum R + {total}\,\delta',
                    font_size=44, color=BLUE_B),
            MathTex(rf'\text{{seat 12}}: \quad \sum R + {total}\,\delta',
                    font_size=44, color=RED_B),
        ).arrange(DOWN, buff=0.6).move_to([0, 1.2, 0])
        self.play(Write(lines), run_time=1.6)
        self.wait(0.8)

        difference = MathTex(
            rf'\left(\sum R + {total}\,\delta\right) - '
            rf'\left(\sum R + {total}\,\delta\right)',
            font_size=42, color=WHITE,
        ).move_to([0, -0.6, 0])
        self.play(Write(difference), run_time=1.4)
        self.wait(0.9)

        cancelled = MathTex(r'\sum R \;-\; \sum R', font_size=52, color=YELLOW)
        cancelled.move_to([0, -0.6, 0])
        self.play(Transform(difference, cancelled), run_time=1.2)
        self.wait(1.6)
        return VGroup(lines, difference)

    # ── Step four and a half: why the pool's spread can stand in for R's ──────────────

    def play_step_four_and_a_half_scaling(self) -> VGroup:
        """The quadrature that lets a Z-score divide by the pool's spread instead of R's.

        Steps one to four settle the DIFFERENTIAL: baselines cancel, so what separates two teams
        is their draws of R, and a category's importance goes as one over R's spread. But a
        Z-score does not divide by R's spread -- it divides by the POOL's, which carries the
        ladder as well. The two line up only if the ladder's spread is proportional to R's, and
        this is that argument in four lines: add the two sources in quadrature, substitute the
        assumption, factor, and read off what is left. The factor multiplying sigma R is one
        constant shared by every category, and a score that only ever compares categories against
        each other cannot see a constant they all share.
        """
        quadrature = MathTex(
            r'\sigma_{\text{pool}} \;=\; \sqrt{\sigma_R^2 \;+\; \sigma_{\text{ladder}}^2}',
            font_size=52, color=WHITE,
        ).move_to([0, 0.9, 0])
        self.play(Write(quadrature), run_time=1.8)
        self.wait(2.4)

        # The assumption from step one, in the units this line needs it in.
        assumption = MathTex(r'\sigma_{\text{ladder}} \;=\; c\,\sigma_R',
                             font_size=40, color=GREY_B).move_to([0, -0.9, 0])
        self.play(Write(assumption), run_time=1.4)
        self.wait(2.2)

        substituted = MathTex(
            r'\sigma_{\text{pool}} \;=\; \sqrt{\sigma_R^2 \;+\; c^2\sigma_R^2}',
            font_size=52, color=WHITE,
        ).move_to([0, 0.9, 0])
        self.play(Transform(quadrature, substituted), run_time=1.4)
        self.wait(2.4)

        factored = MathTex(
            r'\sigma_{\text{pool}} \;=\; '
            # The label is set small deliberately: at full size it is wider than the term it
            # braces, and the extra width shoves sigma R far enough right to read as detached.
            r'\underbrace{\sqrt{1 + c^2}}_{\text{\scriptsize same in every category}} \,\sigma_R',
            font_size=52, color=YELLOW,
        ).move_to([0, 0.9, 0])
        self.play(Transform(quadrature, factored), FadeOut(assumption), run_time=1.6)
        self.wait(2.6)

        # And the payoff: dividing by the pool's spread is dividing by R's, off by a constant no
        # comparison between categories can detect. Which is a Z-score.
        payoff = MathTex(
            r'\frac{1}{\sigma_{\text{pool}}} \;\propto\; \frac{1}{\sigma_R}',
            font_size=56, color=YELLOW,
        ).move_to([0, -1.1, 0])
        self.play(Write(payoff), run_time=1.6)
        self.wait(3.0)
        return VGroup(quadrature, payoff)

    # ── Step five: how far the last assumption holds ──────────────────────────────────

    def _caveat_bars(self, fractions) -> VGroup:
        """One bar per category, at whatever heights are handed in."""
        bars = VGroup()
        for index, (row, fraction) in enumerate(zip(self.measured['categories'], fractions)):
            bars.add(Rectangle(
                width=0.68, height=max(fraction, 1e-3) * CAVEAT_SCALE,
                fill_color=YELLOW if row['category'] == 'Points' else BLUE_B,
                fill_opacity=0.85, stroke_width=0,
            ).move_to([self._caveat_bar_x(index),
                       CAVEAT_BASELINE_Y + fraction * CAVEAT_SCALE / 2, 0]))
        return bars

    def _caveat_bar_x(self, index: int) -> float:
        return (index - (len(self.measured['categories']) - 1) / 2) * 1.16

    def play_step_five_caveat(self) -> VGroup:
        rows = self.measured['categories']
        self.caveat_fractions = np.array([row['residual_spread_fraction'] for row in rows])

        axis = Line([-5.4, CAVEAT_BASELINE_Y, 0], [5.4, CAVEAT_BASELINE_Y, 0],
                    color=GREY_B, stroke_width=2)
        self.caveat_bars = self._caveat_bars(self.caveat_fractions)
        labels = VGroup(*[
            Text(row['category'].replace(' %', '%'), font_size=11,
                 color=YELLOW if row['category'] == 'Points' else GREY_B)
            .rotate(np.pi / 2.6)
            .move_to([self._caveat_bar_x(index), CAVEAT_BASELINE_Y - 0.72, 0])
            for index, row in enumerate(rows)
        ])

        self.play(Create(axis), run_time=0.5)
        self.play(FadeIn(self.caveat_bars, lag_ratio=0.08), FadeIn(labels), run_time=1.5)

        # The eye should read "flat, with one exception". A line across the cluster makes the
        # exception the only thing that stands out.
        level = float(np.median(self.caveat_fractions)) * CAVEAT_SCALE
        cluster_line = Line([-5.4, CAVEAT_BASELINE_Y + level, 0],
                            [5.4, CAVEAT_BASELINE_Y + level, 0],
                            color=GREY_D, stroke_width=3)
        self.play(Create(cluster_line), run_time=0.9)
        self.wait(2.4)
        return VGroup(axis, labels, cluster_line)

    # ── Step six: and no fixed set of bars was ever going to be the answer ─────────────

    def play_step_six_dynamic(self) -> None:
        """The measured bars, refusing to hold still.

        The closing line is that no static ranking can be right, and the honest way to show it is
        to let the picture the scene just finished defending keep moving. Nothing is said on
        screen -- the bars simply will not settle, which is the point.
        """
        generator = np.random.default_rng(4409)
        for _ in range(DYNAMIC_UNSETTLED_DRAWS):
            wobbled = np.clip(
                self.caveat_fractions
                + generator.normal(scale=DYNAMIC_WOBBLE_SPREAD, size=len(self.caveat_fractions)),
                0.05, 1.15)
            self.play(Transform(self.caveat_bars, self._caveat_bars(wobbled)), run_time=1.1)
            self.wait(0.5)
        self.wait(1.2)

    def construct(self) -> None:
        with self.voiceover(text=NARRATION['premise']):
            ladder = self.play_step_one_ladder()

        with self.voiceover(text=NARRATION['linear']):
            self.play(FadeOut(ladder), run_time=0.6)
            self.play_step_two_baselines()
        with self.voiceover(text=NARRATION['random_process']):
            self.play_step_two_first_draw()
        with self.voiceover(text=NARRATION['call_it_R']):
            self.play_step_two_name_R()

        self.play(FadeOut(self.step_two_mobjects), run_time=0.7)
        with self.voiceover(text=NARRATION['team_formula']):
            formula = self.play_step_two_and_a_half_formula()

        self.play(FadeOut(formula), run_time=0.5)
        with self.voiceover(text=NARRATION['plot_the_draft']):
            board = self.play_step_three_board()
            self.play(FadeOut(board), run_time=0.7)
            algebra = self.play_step_four_cancellation()

        self.play(FadeOut(algebra), run_time=0.7)
        with self.voiceover(text=NARRATION['scaling']):
            scaling = self.play_step_four_and_a_half_scaling()
        self.play(FadeOut(scaling), run_time=0.7)

        with self.voiceover(text=NARRATION['caveat']):
            caveat_frame = self.play_step_five_caveat()
        with self.voiceover(text=NARRATION['dynamic']):
            self.play_step_six_dynamic()
        self.play(FadeOut(caveat_frame), FadeOut(self.caveat_bars), run_time=0.9)
