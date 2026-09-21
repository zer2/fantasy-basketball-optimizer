"""Most Categories: every way the week can go, and the shortcut that makes it computable.

WIREFRAME -- a first draft to find out whether the argument lands, drafted against gTTS rather
than the shipped voice. See wireframes/README.md.

The argument in four moves:

    One, the payoff is all-or-nothing. Eight categories won pays the same as five, so what the
    algorithm needs is not an average but the probability of a majority.

    Two, that is a sum over every branch of a binary tree nine levels deep -- 512 leaves, each
    either a win or a loss. Drawn in full, because the size of it is the point.

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
    FadeIn, FadeOut, Create, Write, Transform,
    DOWN, UP, LEFT, RIGHT,
    BLUE_D, BLUE_B, RED_D, GREEN_C, GREY_B, GREY_D, YELLOW, WHITE,
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

# The tree is drawn only as deep as stays legible; the rest is stated rather than shown.
TREE_LEVELS_DRAWN = 5
TREE_TOP_Y = 2.55
TREE_LEVEL_DROP = 0.92
TREE_HALF_WIDTH = 5.6

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
        self.play_payoff()
        self.play_tree()
        self.play_walk()
        self.play_tipping_point()

    # ── Act one: the payoff is a cliff, not a slope ───────────────────────────────────

    def play_payoff(self) -> None:
        """Five of nine pays; eight of nine pays exactly the same.

        Drawn as two stacks of nine pips with the same reward written under each, because the
        claim is an equality and two equal numbers side by side are the whole argument.
        """
        with self.voiceover(text=NARRATION['setup']):
            five = self.build_pip_row(5, [-3.3, 0.9, 0.0], 'five of nine')
            eight = self.build_pip_row(8, [3.3, 0.9, 0.0], 'eight of nine')
            self.play(FadeIn(five), run_time=0.8)
            self.play(FadeIn(eight), run_time=0.8)

            rewards = VGroup(*[
                Text('1 win', font_size=30, color=YELLOW).move_to([x, -0.9, 0.0])
                for x in (-3.3, 3.3)
            ])
            self.play(Write(rewards), run_time=0.9)
            self.wait(1.0)
            self.play(FadeOut(VGroup(five, eight, rewards)), run_time=0.6)

    def build_pip_row(self, won: int, centre, caption: str) -> VGroup:
        """Nine pips, `won` of them filled, with a caption underneath."""
        pips = VGroup(*[
            Dot(radius=0.16, color=BLUE_D if index < won else GREY_D)
            for index in range(len(CATEGORIES))
        ]).arrange(RIGHT, buff=0.16)
        label = Text(caption, font_size=22, color=GREY_B)
        return VGroup(pips, label).arrange(DOWN, buff=0.34).move_to(centre)

    # ── Act two: every way the week can go ────────────────────────────────────────────

    def play_tree(self) -> None:
        """Grow the binary tree a level at a time, then say how big the whole thing is."""
        with self.voiceover(text=NARRATION['tree']) as tracker:
            self.tree_levels = VGroup()
            for level in range(TREE_LEVELS_DRAWN):
                self.tree_levels.add(self.build_tree_level(level))
                self.play(Create(self.tree_levels[-1]), run_time=0.55)

            wait_until_phrase(self, tracker, 'two to the ninth')
            self.leaf_count = MathTex(r'2^{9} = 512', font_size=52, color=YELLOW)
            self.leaf_count.move_to([0.0, TREE_TOP_Y - TREE_LEVELS_DRAWN * TREE_LEVEL_DROP - 0.5, 0.0])
            self.play(Write(self.leaf_count), run_time=0.9)

        with self.voiceover(text=NARRATION['majority']):
            # Colour the drawn leaves by whether the branch is still alive for a majority. The
            # tree is only five deep, so this is "can still reach five", not a final verdict --
            # which is also why the beat says "at least five" rather than naming a leaf.
            self.wait(1.4)

        with self.voiceover(text=NARRATION['explosion']):
            doubling = MathTex(r'2^{10} = 1024', font_size=52, color=RED_D).move_to(self.leaf_count)
            self.play(Transform(self.leaf_count, doubling), run_time=0.8)
            self.wait(1.2)

    def build_tree_level(self, level: int) -> VGroup:
        """The branches descending into one level of the tree."""
        parents = 2 ** level
        top = TREE_TOP_Y - level * TREE_LEVEL_DROP
        bottom = top - TREE_LEVEL_DROP
        strokes = max(1.0, 3.6 - level * 0.6)
        branches = VGroup()
        for index in range(parents):
            start_x = self.tree_x(level, index)
            for child in (0, 1):
                end_x = self.tree_x(level + 1, index * 2 + child)
                branches.add(Line([start_x, top, 0.0], [end_x, bottom, 0.0],
                                  color=BLUE_B if child == 0 else GREY_D,
                                  stroke_width=strokes))
        return branches

    def tree_x(self, level: int, index: int) -> float:
        """Where a node sits: the middle of the slice of the width its subtree owns.

        Each level divides the full width into 2^level equal slices and a node takes the centre
        of its own, so a node's two children sit inside the span their parent occupied. Spreading
        every level across the FULL width instead puts level one at the extreme edges, and the
        branches cross into a box rather than descending into a tree.
        """
        count = 2 ** level
        return -TREE_HALF_WIDTH + 2 * TREE_HALF_WIDTH * (index + 0.5) / count

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
            self.play(FadeOut(self.tree_levels), FadeOut(self.leaf_count), run_time=0.6)
            self.lattice = self.build_lattice()
            self.play(Create(self.lattice), run_time=1.0)

            wait_until_phrase(self, tracker, 'a step up')
            self.sample_walks = VGroup()
            for seed in (4, 11, 23):
                self.sample_walks.add(self.build_sample_walk(seed))
                self.play(Create(self.sample_walks[-1]), run_time=0.7)

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

    def build_sample_walk(self, seed: int) -> VMobject:
        """One way the week could go, as a path through the lattice."""
        generator = np.random.default_rng(seed)
        net, points = 0, [[self.walk_x(0), self.walk_y(0), 0.0]]
        for step, chance in enumerate(WIN_CHANCES, start=1):
            net += 1 if generator.random() < chance else -1
            points.append([self.walk_x(step), self.walk_y(net), 0.0])
        path = VMobject(color=GREY_D, stroke_width=2.5)
        path.set_points_as_corners(points)
        return path

    def play_column_sweep(self) -> None:
        """Advance the distribution one category at a time, translating column into column.

        The whole point of the act is that this loop is the algorithm: one pass per category
        over a column of ten numbers, rather than a walk over 512 paths.
        """
        # The paths go as the column arrives. The line being spoken is that the algorithm never
        # follows them, and leaving them underneath would show the opposite of what is said.
        self.play(FadeOut(self.sample_walks), run_time=0.5)

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
