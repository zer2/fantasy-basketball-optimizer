"""Most Categories: every way the week can go, and the shortcut that makes it computable.

WIREFRAME -- a first draft to find out whether the argument lands, drafted against gTTS rather
than the shipped voice. See wireframes/README.md.

The argument in four moves:

    One, a real week: nine categories, a winner marked against each, and a verdict. Then the
    same two teams in a different week, going the other way. A landslide last, worth exactly
    what the narrow majority was worth -- which is the payoff the whole format turns on.

    Two, every combination as a row of a table, scrolled. What is being computed is a sum over
    rows, so rows are what to show; the length of the scroll is the cost being objected to.

    Three, the same outcomes redrawn as a walk: a category won steps up, one lost steps down,
    and the majority is simply finishing above the line. Two paths reaching the same height are
    worth the same from there on, so the paths are thrown away and one column of heights is
    carried forward instead -- each category translating that column into the next in a single
    pass. That is the dynamic programme the docs describe in prose.

    Four, the same column answers the question worth asking: how often is THIS category the one
    that decides the matchup? It decides exactly when the other eight leave the walk level, and
    that probability is the gradient the algorithm steps on -- and the reason Most Categories
    punts hardest, since a category already certain either way can never tip anything.

Nothing here is measured against the real objective yet. The win probabilities are stand-ins
chosen to make the shape legible; a prep script comes once the beats are settled.

    manim -ql visualizations/wireframes/most_categories/most_categories.py MostCategories
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import (
    VGroup, VMobject, Line, DashedLine, Rectangle, Text, MathTex, Dot,
    FadeIn, FadeOut, Create, Write, Transform, linear,
    DOWN, UP, LEFT, RIGHT,
    BLUE_D, BLUE_B, RED_D, RED_B, GREEN_C, GREY_B, GREY_D, YELLOW, WHITE,
)
from manim_voiceover import VoiceoverScene

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.draft_voice import DraftVoice                 # noqa: E402
from shared.narration_timing import wait_until_phrase     # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from narration import NARRATION                           # noqa: E402


# ── The board ────────────────────────────────────────────────────────────────────────

CATEGORIES = ('Field Goal %', 'Free Throw %', 'Threes', 'Points', 'Rebounds',
              'Assists', 'Steals', 'Blocks', 'Turnovers')
# Stand-in win probabilities, not measured: two the team is nearly certain of, two nearly lost,
# and five genuinely in the balance. The shape is what the last beat needs -- the certain ones
# have to be visibly unable to tip anything.
# A team that has committed: five categories close to locked, four given up. Chosen because the
# last act needs the nine tipping points to DIFFER. On a perfectly balanced team they are
# identical by symmetry (27.34% each), and on a mildly tilted one they span only 1.15x -- nine
# dots at the same height, which would say every category is equally decisive. This build spans
# 1.87x, and the categories it kept are the decisive ones, which is the point.
WIN_CHANCES = (0.95, 0.92, 0.88, 0.85, 0.82, 0.12, 0.09, 0.06, 0.04)
MAJORITY = len(CATEGORIES) // 2 + 1

TEAM_LABELS = ('Team 1', 'Team 2')

# The opening scoreboard: categories down the middle, a team either side.
BOARD_TOP_Y = 2.35
BOARD_ROW_GAP = 0.46
BOARD_HALF_WIDTH = 2.6
BOARD_BOTTOM_Y = -2.6

# Three weeks between the same two teams: a majority, the other way, and a landslide worth
# exactly the same as the majority.
FIRST_WEEK     = (1, 1, 0, 1, 1, 0, 1, 0, 1)   # six of nine
SECOND_WEEK    = (0, 1, 0, 0, 1, 1, 0, 0, 1)   # four of nine
LANDSLIDE_WEEK = (1, 1, 1, 1, 1, 1, 1, 1, 1)   # all nine, still one win

# The table of outcomes. Only the first rows are built -- enough to scroll convincingly without
# drawing all 512, which would cost minutes of render for no extra argument.
TABLE_ROWS_BUILT = 46
TABLE_VISIBLE_ROWS = 11
TABLE_CELL = 0.28
TABLE_ROW_GAP = 0.40
TABLE_TOP_Y = 3.4

# The walk the tree collapses into: one column per category, net position up for a category won
# and down for one lost. Nine steps is odd, so the walk can never finish level -- it is above the
# line exactly when five or more were won, which is what makes "majority" a side of the picture.
WALK_LEFT_X = -5.2
WALK_RIGHT_X = 5.2
WALK_CENTRE_Y = -0.35
WALK_UNIT_Y = 0.30
# Where the cut opens first, and where it slides to -- one kept category, then two abandoned.
CUT_CATEGORY = 2
SLIDE_TO = (5, 0)


class MostCategories(VoiceoverScene):
    """The 512-leaf tree, its collapse into a tally, and the tipping point that falls out."""

    def construct(self) -> None:
        self.set_speech_service(DraftVoice())
        self.play_a_week()
        self.play_the_table()
        self.play_walk()
        self.play_tipping_point()

    # -- Act one: a week, and who took it ----------------------------------------------

    def play_a_week(self) -> None:
        """One matchup, category by category, so the thing being counted is concrete first.

        The scene used to open on an empty frame while the first line was spoken. It opens on a
        scoreboard instead: nine categories, a winner marked against each, and a verdict. Two
        different weeks between the same teams make the point that the same teams do not produce
        the same result -- which is what there is a probability OF.
        """
        with self.voiceover(text=NARRATION['a_week']) as tracker:
            self.board = self.build_scoreboard()
            self.play(FadeIn(self.board), run_time=1.0)
            wait_until_phrase(self, tracker, 'goes to whichever team')
            self.play(*[FadeIn(mark) for mark in self.mark_week(FIRST_WEEK)],
                      lag_ratio=0.12, run_time=2.2)
            self.wait(0.5)

        with self.voiceover(text=NARRATION['count_them']):
            self.verdict = self.build_verdict(FIRST_WEEK)
            self.play(FadeIn(self.verdict), run_time=0.9)
            self.wait(1.0)

        with self.voiceover(text=NARRATION['another_week']):
            self.play(FadeOut(self.week_marks), FadeOut(self.verdict), run_time=0.5)
            self.play(*[FadeIn(mark) for mark in self.mark_week(SECOND_WEEK)],
                      lag_ratio=0.10, run_time=1.6)
            self.verdict = self.build_verdict(SECOND_WEEK)
            self.play(FadeIn(self.verdict), run_time=0.8)
            self.wait(0.8)

        with self.voiceover(text=NARRATION['all_or_nothing']) as tracker:
            wait_until_phrase(self, tracker, 'exactly the same')
            self.play(FadeOut(self.week_marks), FadeOut(self.verdict), run_time=0.5)
            self.play(*[FadeIn(mark) for mark in self.mark_week(LANDSLIDE_WEEK)],
                      lag_ratio=0.06, run_time=1.2)
            self.verdict = self.build_verdict(LANDSLIDE_WEEK)
            self.play(FadeIn(self.verdict), run_time=0.8)
            self.wait(1.4)
            self.play(FadeOut(self.board), FadeOut(self.week_marks),
                      FadeOut(self.verdict), run_time=0.7)

    def build_scoreboard(self) -> VGroup:
        """Two team headings and the nine categories between them."""
        rows = VGroup()
        for index, name in enumerate(CATEGORIES):
            y = BOARD_TOP_Y - index * BOARD_ROW_GAP
            rows.add(Text(name, font_size=22, color=GREY_B).move_to([0.0, y, 0.0]))
        headings = VGroup(
            Text(TEAM_LABELS[0], font_size=26, color=BLUE_B)
                .move_to([-BOARD_HALF_WIDTH, BOARD_TOP_Y + 0.65, 0.0]),
            Text(TEAM_LABELS[1], font_size=26, color=RED_B)
                .move_to([BOARD_HALF_WIDTH, BOARD_TOP_Y + 0.65, 0.0]),
        )
        return VGroup(rows, headings)

    def mark_week(self, won_by_first) -> VGroup:
        """A marker against each category on the side of whoever took it."""
        self.week_marks = VGroup()
        for index, first_won in enumerate(won_by_first):
            y = BOARD_TOP_Y - index * BOARD_ROW_GAP
            x = -BOARD_HALF_WIDTH if first_won else BOARD_HALF_WIDTH
            self.week_marks.add(Dot(point=[x, y, 0.0], radius=0.15,
                                    color=BLUE_D if first_won else RED_D))
        return self.week_marks

    def build_verdict(self, won_by_first) -> VGroup:
        """The tally and who it gives the matchup to."""
        taken = sum(won_by_first)
        first_wins = taken > len(CATEGORIES) / 2
        winner = TEAM_LABELS[0] if first_wins else TEAM_LABELS[1]
        colour = BLUE_B if first_wins else RED_B
        return VGroup(
            Text(f'{taken} - {len(CATEGORIES) - taken}', font_size=40, color=colour),
            Text(f'{winner} wins  ->  1 win', font_size=26, color=colour),
        ).arrange(DOWN, buff=0.22).move_to([0.0, BOARD_BOTTOM_Y, 0.0])

    # -- Act two: everything that could happen, as rows --------------------------------

    def play_the_table(self) -> None:
        """The 512 combinations as a table, scrolled, so the sum being asked for is visible.

        This replaces a binary tree. The tree showed the same 512 outcomes but spent its length
        on the branching, which is not the thing being computed -- what is being computed is a
        sum over rows, so rows are what to show.
        """
        with self.voiceover(text=NARRATION['the_table']) as tracker:
            self.table = self.build_outcome_table()
            self.play(FadeIn(self.table[:TABLE_VISIBLE_ROWS]), run_time=1.2)
            wait_until_phrase(self, tracker, 'either a win for you')
            self.wait(1.0)

        with self.voiceover(text=NARRATION['how_many_rows']) as tracker:
            # Scrolled rather than paged: the length of it is the argument, and a scroll is the
            # only way to show length without drawing 512 rows.
            travel = TABLE_ROW_GAP * (len(self.table) - TABLE_VISIBLE_ROWS)
            self.play(self.table.animate.shift(UP * travel),
                      run_time=max(2.0, tracker.duration * 0.55), rate_func=linear)
            counter = MathTex(r'2^{9} = 512 \;	ext{rows}', font_size=52, color=YELLOW)
            counter.move_to([0.0, 0.0, 0.0])
            self.play(FadeOut(self.table), FadeIn(counter), run_time=0.8)
            self.wait(0.8)
            self.play(FadeOut(counter), run_time=0.5)

    def build_outcome_table(self) -> VGroup:
        """One row per outcome: nine cells of won-or-lost, and the verdict it produces.

        Only the first few dozen combinations are built. They are generated in binary order, so
        what scrolls past is genuinely the enumeration rather than a decorative sample.
        """
        rows = VGroup()
        for number in range(TABLE_ROWS_BUILT):
            bits = [(number >> shift) & 1 for shift in range(len(CATEGORIES) - 1, -1, -1)]
            cells = VGroup(*[
                Rectangle(width=TABLE_CELL, height=TABLE_CELL,
                          fill_color=BLUE_D if bit else GREY_D,
                          fill_opacity=1.0, stroke_width=0)
                for bit in bits
            ]).arrange(RIGHT, buff=0.08)
            taken = sum(bits)
            first_wins = taken > len(CATEGORIES) / 2
            verdict = Text('win' if first_wins else 'loss', font_size=20,
                           color=BLUE_B if first_wins else GREY_B)
            rows.add(VGroup(cells, verdict).arrange(RIGHT, buff=0.42))
        rows.arrange(DOWN, buff=TABLE_ROW_GAP - TABLE_CELL)
        rows.move_to([0.0, TABLE_TOP_Y, 0.0], aligned_edge=UP)
        return rows

    # ── Act three: the tree is a walk, and the walk is one column ─────────────────────

    def play_walk(self) -> None:
        """Re-draw the same 512 outcomes as a walk, then stop following paths at all.

        A win steps up, a loss steps down, so after nine odd-numbered steps the walk finishes
        above zero exactly when five or more categories were won. The majority stops being a
        counting rule and becomes a side of the picture, which is what lets the next move land:
        two paths that reach the same height are worth the same from there on, so the algorithm
        can throw the paths away and keep one column of heights.
        """
        with self.voiceover(text=NARRATION['dynamic']) as tracker:
            self.lattice = self.build_lattice()
            self.play(Create(self.lattice), run_time=1.0)

            wait_until_phrase(self, tracker, 'a step up')
            self.all_paths = self.build_all_paths()
            self.play(FadeIn(self.all_paths), run_time=1.8)

            wait_until_phrase(self, tracker, 'above where you started')
            self.play(FadeIn(self.build_win_region()), run_time=0.8)
            self.wait(0.8)

        with self.voiceover(text=NARRATION['collapse']) as tracker:
            wait_until_phrase(self, tracker, 'a single column')
            self.play_column_sweep()
            self.wait(0.8)

    # The walk: ten columns (before any category, then after each of the nine) against net
    # position, which runs from -9 to +9 but only ever reaches values of the step's own parity.
    def walk_x(self, step: int) -> float:
        return WALK_LEFT_X + step * (WALK_RIGHT_X - WALK_LEFT_X) / len(CATEGORIES)

    def walk_y(self, net: int) -> float:
        return WALK_CENTRE_Y + net * WALK_UNIT_Y

    def build_lattice(self) -> VGroup:
        """The axis the walk happens on: the zero line, and what each side of it means."""
        zero = Line([WALK_LEFT_X - 0.3, self.walk_y(0), 0.0],
                    [WALK_RIGHT_X + 0.3, self.walk_y(0), 0.0],
                    color=GREY_B, stroke_width=2)
        start = Text('start', font_size=20, color=GREY_B)
        start.next_to([WALK_LEFT_X - 0.3, self.walk_y(0), 0.0], LEFT, buff=0.25)
        return VGroup(zero, start)

    def build_win_region(self) -> VGroup:
        """Everything above the line, which is every way of taking the majority."""
        band = Rectangle(
            width=WALK_RIGHT_X - WALK_LEFT_X + 0.6, height=9 * WALK_UNIT_Y,
            fill_color=BLUE_D, fill_opacity=0.12, stroke_width=0,
        ).move_to([(WALK_LEFT_X + WALK_RIGHT_X) / 2,
                   self.walk_y(0) + 4.5 * WALK_UNIT_Y, 0.0])
        label = Text('majority', font_size=21, color=BLUE_B)
        label.next_to(band, RIGHT, buff=0.15)
        return VGroup(band, label)

    def build_all_paths(self) -> VGroup:
        """Every one of the 512 ways the week can go, drawn at once.

        Each is faint, so where many paths share an edge the strokes stack and that edge comes
        out brighter. The density in the picture is therefore real rather than styled -- it is
        how many outcomes pass through, which is exactly what the dots go on to quantify.
        """
        paths = VGroup()
        for number in range(2 ** len(CATEGORIES)):
            net, points = 0, [[self.walk_x(0), self.walk_y(0), 0.0]]
            for step in range(len(CATEGORIES)):
                net += 1 if (number >> step) & 1 else -1
                points.append([self.walk_x(step + 1), self.walk_y(net), 0.0])
            path = VMobject(stroke_color=GREY_D, stroke_width=1.0, stroke_opacity=0.13)
            path.set_points_as_corners(points)
            paths.add(path)
        return paths

    def play_column_sweep(self) -> None:
        """Advance the distribution one category at a time, translating column into column.

        The whole point of the act is that this loop is the algorithm: one pass per category
        over a column of ten numbers, rather than a walk over 512 paths.
        """
        # The paths stay, dimmed. The line says the algorithm never follows them, but they are
        # what the dots are a summary OF -- taking them away would leave the density looking
        # like a chosen decoration rather than a count of what passes through.
        self.play(self.all_paths.animate.set_stroke(opacity=0.04), run_time=0.5)

        self.forward = [{0: 1.0}]
        for chance in WIN_CHANCES:
            self.forward.append(self.step_distribution(self.forward[-1], chance))

        # The backward table, built now and drawn later: suffix[i] is what categories i onward
        # contribute, so suffix[n] is the empty walk and suffix[0] is all nine.
        self.backward = [None] * (len(CATEGORIES) + 1)
        self.backward[len(CATEGORIES)] = {0: 1.0}
        for index in range(len(CATEGORIES) - 1, -1, -1):
            self.backward[index] = self.step_distribution(
                self.backward[index + 1], WIN_CHANCES[index])

        self.forward_columns = VGroup()
        for step, distribution in enumerate(self.forward):
            column = self.build_column(distribution, step)
            self.forward_columns.add(column)
            self.play(FadeIn(column), run_time=0.5 if step == 0 else 0.42)

    def step_distribution(self, distribution: dict, chance: float) -> dict:
        """One category folded into a column: every height sends its mass up and down.

        This is the whole dynamic programme -- and it is used for BOTH sweeps, because a walk
        run backwards over the same independent categories obeys the same recurrence.
        """
        stepped = {}
        for net, probability in distribution.items():
            stepped[net + 1] = stepped.get(net + 1, 0.0) + probability * chance
            stepped[net - 1] = stepped.get(net - 1, 0.0) + probability * (1.0 - chance)
        return stepped

    def build_column(self, distribution: dict, step: int, forward: bool = True) -> VGroup:
        """One column of a sweep: a dot per reachable height, sized by its probability.

        The backward sweep is drawn a little to the right of the forward one and in its own
        colour, so the two sit side by side at the same position rather than on top of each
        other -- the cut in the next beat needs both to be readable at once.
        """
        offset = 0.0 if forward else 0.13
        dots = VGroup()
        for net, probability in distribution.items():
            dots.add(Dot(
                point=[self.walk_x(step) + offset, self.walk_y(net), 0.0],
                radius=0.05 + 0.18 * probability ** 0.5,
                color=(BLUE_B if net > 0 else GREY_D) if forward else GREEN_C,
            ).set_opacity(0.30 + 0.60 * probability ** 0.5))
        return dots

    # ── Act four: two sweeps, and a cut that slides ───────────────────────────────────

    def play_tipping_point(self) -> None:
        """How the algorithm actually values a category: one forward sweep, one backward, and a
        gap slid along between them.

        This is what backend/math/algorithm_helpers.py does, not a restatement of it. It builds
        a prefix table sweeping one way and a suffix table sweeping the other, then for each
        category convolves the prefix that stops just before it with the suffix that starts just
        after -- `_leave_one_out_probability`, whose docstring is "P(the categories either side
        of the excluded one contribute exactly target_points)".

        The animation has to be the cheap version rather than the obvious one. Re-running an
        eight-step walk per category would look like the same answer and teach the opposite
        lesson: the point of the two tables is that nine categories cost two sweeps, not nine.
        """
        with self.voiceover(text=NARRATION['tipping']):
            self.wait(1.4)

        with self.voiceover(text=NARRATION['backward']):
            self.backward_columns = VGroup()
            for step in range(len(CATEGORIES), -1, -1):
                column = self.build_column(self.backward[step], step, forward=False)
                self.backward_columns.add(column)
                self.play(FadeIn(column), run_time=0.30)
            self.wait(0.5)

        with self.voiceover(text=NARRATION['the_cut']) as tracker:
            wait_until_phrase(self, tracker, 'cut the walk open')
            self.show_cut_at(CUT_CATEGORY, first_time=True)
            self.wait(1.0)

        with self.voiceover(text=NARRATION['convolution']) as tracker:
            wait_until_phrase(self, tracker, 'pair every height')
            self.play_meeting(CUT_CATEGORY)
            self.wait(1.2)

        with self.voiceover(text=NARRATION['slide']):
            for category in SLIDE_TO:
                self.show_cut_at(category)
                self.play_meeting(category, quickly=True)
            self.wait(0.8)

        with self.voiceover(text=NARRATION['punting']):
            # PLACEHOLDER for the beat that earns the punting claim: the nine tipping points
            # listed against the categories, the kept ones visibly the decisive ones. Wants the
            # real gradient from the objective, so it waits for a prep script.
            self.wait(2.0)
        self.wait(0.6)

    def show_cut_at(self, category: int, first_time: bool = False) -> None:
        """Open a gap where one category sits, keeping only what reaches it from either side."""
        gap_x = (self.walk_x(category) + self.walk_x(category + 1)) / 2
        marker = DashedLine([gap_x, self.walk_y(-9) - 0.2, 0.0],
                            [gap_x, self.walk_y(9) + 0.2, 0.0],
                            color=YELLOW, stroke_width=3, dash_length=0.14)
        label = Text(CATEGORIES[category], font_size=21, color=YELLOW)
        label.move_to([gap_x, self.walk_y(9) + 0.45, 0.0])

        # Only the forward column that stops at the cut and the backward column that starts
        # after it are wanted; everything else is what those two already contain.
        keep_forward, keep_backward = category, category + 1
        fades = [self.forward_columns[index].animate.set_opacity(0.12)
                 for index in range(len(self.forward_columns)) if index != keep_forward]
        fades += [self.backward_columns[len(CATEGORIES) - index].animate.set_opacity(0.12)
                  for index in range(len(CATEGORIES) + 1) if index != keep_backward]

        if first_time:
            self.cut_marker, self.cut_label = marker, label
            self.play(Create(marker), FadeIn(label), *fades, run_time=1.0)
        else:
            self.play(Transform(self.cut_marker, marker),
                      Transform(self.cut_label, label), *fades, run_time=0.8)

    def play_meeting(self, category: int, quickly: bool = False) -> None:
        """Pair each height on the left with the opposite height on the right, and total it.

        A height of +h before the category and -h after it sum to level, which is the only way
        the category can be the one that decides the matchup. The arcs are those pairings and
        the readout is their total -- the convolution, drawn.
        """
        before, after = self.forward[category], self.backward[category + 1]
        arcs, total = VGroup(), 0.0
        for height, probability in before.items():
            partner = after.get(-height)
            if partner is None:
                continue
            total += probability * partner
            arcs.add(Line(
                [self.walk_x(category), self.walk_y(height), 0.0],
                [self.walk_x(category + 1), self.walk_y(-height), 0.0],
                color=YELLOW, stroke_width=1.0 + 7.0 * (probability * partner) ** 0.5,
            ).set_opacity(0.35 + 0.65 * (probability * partner) ** 0.5))

        readout = Text(f'{CATEGORIES[category]} decides it {total:.1%} of the time',
                       font_size=26, color=YELLOW)
        readout.move_to([0.0, self.walk_y(-9) - 0.55, 0.0])

        if quickly:
            self.play(Transform(self.meeting_arcs, arcs),
                      Transform(self.meeting_readout, readout), run_time=0.9)
        else:
            self.meeting_arcs, self.meeting_readout = arcs, readout
            self.play(Create(arcs), run_time=1.2)
            self.play(Write(readout), run_time=0.9)
