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
    Group, VGroup, VMobject, Line, DashedLine, Rectangle, Text, MathTex, Dot,
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
# Weeks between the same two teams, cycled at the top of the scene. One is not enough to show
# what there is a probability OF -- the same teams produce a different result every week, and
# the majority falls either way, sometimes by one category and sometimes by three.
ROTATING_WEEKS = (
    (1, 1, 0, 1, 1, 0, 1, 0, 1),   # six of nine
    (0, 1, 0, 0, 1, 1, 0, 0, 1),   # four of nine
    (1, 0, 1, 1, 0, 1, 0, 1, 0),   # five of nine, the narrowest majority there is
    (0, 0, 1, 0, 1, 0, 0, 1, 0),   # three of nine
    (1, 1, 1, 0, 1, 1, 0, 1, 0),   # six of nine again, arrived at differently
)
# The pair that makes the payoff argument: a sweep and the narrowest possible majority, worth
# exactly the same. Ending on the narrow one is the point -- it is the outcome the algorithm
# actually plays for, and the sweep is only there to be matched by it.
LANDSLIDE_WEEK = (1, 1, 1, 1, 1, 1, 1, 1, 1)   # all nine
NARROW_WEEK    = (1, 0, 1, 1, 0, 1, 0, 1, 0)   # five of nine, and worth the same

# What one week of the rotation gets, once the first has been shown deliberately.
ROTATION_SECONDS = 1.5
# How long the board takes to clear once its line has finished.
BOARD_CLEAR_SECONDS = 0.7

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

# The closing example: five categories held, four given up. Which one is taken away decides
# whether the remaining eight are level, and that is the whole punting argument.
# The three cases the toggle panel steps through: the other eight level, then already won,
# then already lost. Only the first leaves the ninth category mattering at all.
LEVEL_CASE   = (1, 1, 1, 1, 0, 0, 0, 0)
DECIDED_WON  = (1, 1, 1, 1, 1, 1, 0, 0)
DECIDED_LOST = (1, 1, 0, 0, 0, 0, 0, 0)

CONTESTED_COUNT = 5
PUNT_BAR_HEIGHT = 1.9
# How many of the 256 scenarios are drawn before the table is summarised rather than continued.
SCENARIO_ROWS_DRAWN = 14


class MostCategories(VoiceoverScene):
    """The 512-leaf tree, its collapse into a tally, and the tipping point that falls out."""

    def construct(self) -> None:
        self.set_speech_service(DraftVoice())
        self.play_a_week()
        self.play_the_table()
        self.play_walk()
        self.play_what_tipping_means()
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

            wait_until_phrase(self, tracker, 'five out of nine')
            # The first week is dealt out category by category; the rest cycle, because by then
            # the viewer knows how to read the board and what is worth showing is that the
            # answer keeps changing.
            self.play(*[FadeIn(mark) for mark in self.mark_week(ROTATING_WEEKS[0])],
                      lag_ratio=0.12, run_time=2.0)
            self.verdict = self.build_verdict(ROTATING_WEEKS[0])
            self.play(FadeIn(self.verdict), run_time=0.7)

            for week in ROTATING_WEEKS[1:]:
                if tracker.get_remaining_duration() < ROTATION_SECONDS:
                    break
                self.play(FadeOut(self.week_marks), FadeOut(self.verdict), run_time=0.25)
                self.play(*[FadeIn(mark) for mark in self.mark_week(week)],
                          lag_ratio=0.04, run_time=0.55)
                self.verdict = self.build_verdict(week)
                self.play(FadeIn(self.verdict), run_time=0.35)
                self.wait(max(0.1, ROTATION_SECONDS - 1.15))

        with self.voiceover(text=NARRATION['all_or_nothing']) as tracker:
            wait_until_phrase(self, tracker, 'aiming to win that majority')
            self.play(FadeOut(self.week_marks), FadeOut(self.verdict), run_time=0.5)
            self.play(*[FadeIn(mark) for mark in self.mark_week(LANDSLIDE_WEEK)],
                      lag_ratio=0.06, run_time=1.2)
            self.verdict = self.build_verdict(LANDSLIDE_WEEK)
            self.play(FadeIn(self.verdict), run_time=0.8)
            self.wait(1.2)

            # And back down to the narrowest majority there is, which pays exactly the same.
            # Ending on the sweep left the last thing on screen being the outcome that is NOT
            # worth chasing.
            self.play(FadeOut(self.week_marks), FadeOut(self.verdict), run_time=0.3)
            self.play(*[FadeIn(mark) for mark in self.mark_week(NARROW_WEEK)],
                      lag_ratio=0.05, run_time=0.8)
            self.verdict = self.build_verdict(NARROW_WEEK)
            self.play(FadeIn(self.verdict), run_time=0.7)

            # Held to the end of the line. The sentence runs on to say what the objective is
            # NOT, and taking the board away before that lands leaves the rest of it spoken
            # over an empty frame.
            self.wait(max(0.5, tracker.get_remaining_duration() - BOARD_CLEAR_SECONDS))
            self.play(FadeOut(self.board), FadeOut(self.week_marks),
                      FadeOut(self.verdict), run_time=BOARD_CLEAR_SECONDS)

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
            wait_until_phrase(self, tracker, 'each of these scenarios')
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
        # Counted DOWN from every category won. Counting up from zero opened the table on a row
        # of nine losses, which is a strange thing to lead with when the subject is winning a
        # majority -- the eye should start on the outcome the viewer is hoping for.
        highest = 2 ** len(CATEGORIES) - 1
        for index in range(TABLE_ROWS_BUILT):
            number = highest - index
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
            wait_until_phrase(self, tracker, 'keeping track of the distribution')
            self.lattice = self.build_lattice()
            self.play(Create(self.lattice), run_time=1.0)
            self.wait(1.2)

        with self.voiceover(text=NARRATION['collapse']) as tracker:
            wait_until_phrase(self, tracker, 'one level upwards')
            self.all_paths = self.build_all_paths()
            self.play(FadeIn(self.all_paths), run_time=1.6)

            wait_until_phrase(self, tracker, 'above the middle line')
            self.win_region = self.build_win_region()
            self.play(FadeIn(self.win_region), run_time=0.8)

            wait_until_phrase(self, tracker, 'walk forward though all')
            self.play_column_sweep()
            self.wait(0.6)

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
            wait_until_phrase(self, tracker, 'Take any of the categories')
            self.show_cut_at(CUT_CATEGORY, first_time=True)
            self.wait(1.0)

        with self.voiceover(text=NARRATION['convolution']) as tracker:
            wait_until_phrase(self, tracker, 'multiply the opposing numbers')
            self.play_meeting(CUT_CATEGORY)
            self.wait(1.2)

        with self.voiceover(text=NARRATION['slide']):
            for category in SLIDE_TO:
                self.show_cut_at(category)
                self.play_meeting(category, quickly=True)
            self.wait(0.8)

        with self.voiceover(text=NARRATION['punting']) as tracker:
            # Everything currently drawn, rather than a list of names that has to be kept in
            # step by hand -- the previous list had fallen behind and the board came up on top
            # of the scenario table and the leftover sweep.
            self.play(FadeOut(Group(*self.mobjects)), run_time=0.7)
            self.clear()
            board = self.build_punt_board()
            self.play(FadeIn(board['bars']), run_time=1.2)

            wait_until_phrase(self, tracker, 'excluding one of them')
            self.play(FadeIn(board['contested']), run_time=1.0)
            self.wait(1.4)

            wait_until_phrase(self, tracker, 'For the punted categories')
            self.play(FadeIn(board['punted']), run_time=1.0)
            self.wait(2.0)
        self.wait(0.6)

    def build_punt_board(self) -> dict:
        """Five categories held high and four given up, and what taking one away leaves.

        The two cases are the whole punting argument and they differ by one column. Drop a held
        category and the remaining eight split four-four: level, so the one dropped decides the
        matchup. Drop a given-up one and the rest sit five-three, already settled, so it decides
        nothing. Same board, opposite conclusions.
        """
        bars = VGroup()
        for index in range(len(CATEGORIES)):
            held = index < CONTESTED_COUNT
            x = -4.6 + index * 1.15
            outline = Rectangle(width=0.78, height=PUNT_BAR_HEIGHT, stroke_width=2,
                                stroke_color=GREY_D, fill_opacity=0.0)
            outline.move_to([x, 0.9, 0.0])
            filled_height = PUNT_BAR_HEIGHT * (0.88 if held else 0.12)
            filled = Rectangle(width=0.78, height=filled_height, stroke_width=0,
                               fill_color=BLUE_D if held else GREY_D, fill_opacity=0.9)
            filled.move_to([x, 0.9 - PUNT_BAR_HEIGHT / 2 + filled_height / 2, 0.0])
            bars.add(VGroup(outline, filled))
        caption = Text(f'{CONTESTED_COUNT} held, {len(CATEGORIES) - CONTESTED_COUNT} given up',
                       font_size=24, color=GREY_B).move_to([0.0, 2.35, 0.0])

        def verdict(title, split, note, colour, y):
            return VGroup(
                Text(title, font_size=23, color=colour),
                Text(split, font_size=27, color=colour),
                Text(note, font_size=22, color=GREY_B),
            ).arrange(RIGHT, buff=0.5).move_to([0.0, y, 0.0])

        return {
            'bars': VGroup(caption, bars),
            'contested': verdict('take away a held category', 'the rest sit 4 - 4',
                                 'level, so it decides the matchup', YELLOW, -1.1),
            'punted': verdict('take away a given-up category', 'the rest sit 5 - 3',
                              'already settled, so it decides nothing', GREY_B, -2.1),
        }

    def play_what_tipping_means(self) -> None:
        """Why the gradient is a probability: the other eight settle it unless they are level.

        Eight toggles for the other categories, the chance of taking the one in question, and
        the objective beside them. Flip the toggles and the objective is 1 or 0 whatever that
        chance is -- the matchup is already decided. Set them level and the objective becomes
        the chance itself. So the slope is the probability of landing in that middle case, and
        the scenario table is that probability written out: the level rows are the only ones
        that respond, and every other row is flat.
        """
        with self.voiceover(text=NARRATION['tipping_point_probability']) as tracker:
            self.play(FadeOut(VGroup(self.forward_columns, self.all_paths,
                                     self.lattice, self.win_region)), run_time=0.7)
            panel = self.build_tipping_panel()
            self.play(FadeIn(panel['frame']), run_time=1.0)

            wait_until_phrase(self, tracker, 'precisely even')
            self.play(*self.set_toggles(panel, LEVEL_CASE), run_time=0.9)
            self.play(Transform(panel['objective'], self.objective_readout(LEVEL_CASE)),
                      run_time=0.6)
            self.wait(1.2)

            wait_until_phrase(self, tracker, 'not even')
            for others in (DECIDED_WON, DECIDED_LOST):
                self.play(*self.set_toggles(panel, others), run_time=0.7)
                self.play(Transform(panel['objective'], self.objective_readout(others)),
                          run_time=0.5)
                self.wait(0.9)

            wait_until_phrase(self, tracker, 'we call this a tipping point')
            self.play(FadeOut(panel['frame']), FadeOut(panel['objective']), run_time=0.5)
            table = self.build_scenario_table()
            self.play(FadeIn(table), run_time=1.4)
            self.wait(max(0.6, tracker.get_remaining_duration() - 1.0))

            # Handed back. This act borrows the frame from the walk, and the cut that follows
            # reaches for the forward sweep and the lattice again -- left cleared, the cut would
            # open on a backward sweep with nothing to meet.
            self.play(FadeOut(table), run_time=0.5)
            self.play(FadeIn(VGroup(self.lattice, self.all_paths, self.win_region,
                                    self.forward_columns)), run_time=0.7)

    def build_tipping_panel(self) -> dict:
        """Eight toggles, the chance of taking the ninth, and the objective beside them."""
        self.toggles = VGroup()
        for index in range(len(CATEGORIES) - 1):
            row, column = divmod(index, 4)
            self.toggles.add(Rectangle(
                width=0.86, height=0.46, stroke_width=2, stroke_color=GREY_D,
                fill_color=GREY_D, fill_opacity=0.25,
            ).move_to([-4.3 + column * 1.0, 1.5 - row * 0.62, 0.0]))
        heading = Text('the other eight', font_size=22, color=GREY_B)
        heading.move_to([-2.8, 2.3, 0.0])

        mine = VGroup(
            Text('this category', font_size=22, color=GREY_B),
            Text('p', font_size=34, color=YELLOW),
        ).arrange(DOWN, buff=0.18).move_to([0.9, 1.2, 0.0])

        objective = self.objective_readout(None)
        return {'frame': VGroup(heading, self.toggles, mine), 'objective': objective}

    def objective_readout(self, others) -> VGroup:
        """What the majority objective comes to, given what the other eight did."""
        if others is None:
            value = '?'
        else:
            won = sum(others)
            value = 'p' if won == MAJORITY - 1 else ('1' if won >= MAJORITY else '0')
        return VGroup(
            Text('chance of the majority', font_size=22, color=GREY_B),
            Text(value, font_size=46, color=YELLOW if value == 'p' else GREY_B),
        ).arrange(DOWN, buff=0.22).move_to([4.2, 1.2, 0.0])

    def set_toggles(self, panel, others):
        """Light the toggles for a given outcome of the other eight."""
        return [
            toggle.animate.set_fill(BLUE_D if won else GREY_D,
                                    opacity=0.9 if won else 0.25)
            for toggle, won in zip(self.toggles, others)
        ]

    def build_scenario_table(self) -> VGroup:
        """Every way the other eight can land, marked by whether this category still matters.

        Rows that sit level are the only ones where the answer depends on p at all; the rest
        are already decided, so they contribute nothing to the slope. The share of probability
        in the level rows IS the tipping point.
        """
        rows = VGroup()
        level_chance = 0.0
        for number in range(2 ** (len(CATEGORIES) - 1)):
            bits = [(number >> shift) & 1 for shift in range(len(CATEGORIES) - 2, -1, -1)]
            chance = 1.0
            for bit, probability in zip(bits, WIN_CHANCES[:-1]):
                chance *= probability if bit else (1.0 - probability)
            if sum(bits) == MAJORITY - 1:
                level_chance += chance
            if number >= SCENARIO_ROWS_DRAWN:
                continue
            cells = VGroup(*[
                Rectangle(width=0.3, height=0.24, stroke_width=0,
                          fill_color=BLUE_D if bit else GREY_D, fill_opacity=1.0)
                for bit in bits
            ]).arrange(RIGHT, buff=0.06)
            level = sum(bits) == MAJORITY - 1
            verdict = Text('p' if level else ('1' if sum(bits) >= MAJORITY else '0'),
                           font_size=20, color=YELLOW if level else GREY_D)
            rows.add(VGroup(cells, verdict).arrange(RIGHT, buff=0.4))
        rows.arrange(DOWN, buff=0.1).move_to([-1.4, 0.0, 0.0])

        summary = VGroup(
            Text('only the level rows respond to p', font_size=24, color=YELLOW),
            Text(f'their total probability: {level_chance:.1%}', font_size=26, color=YELLOW),
            Text('that is the tipping point', font_size=22, color=GREY_B),
        ).arrange(DOWN, buff=0.28).move_to([3.7, 0.0, 0.0])
        return VGroup(rows, summary)

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
