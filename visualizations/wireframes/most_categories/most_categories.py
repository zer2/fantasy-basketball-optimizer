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
    VGroup, VMobject, Line, Rectangle, Text, MathTex, Dot,
    FadeIn, FadeOut, Create, Write, Transform,
    DOWN, UP, LEFT, RIGHT,
    BLUE_D, BLUE_B, RED_D, GREY_B, GREY_D, YELLOW, WHITE,
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
WIN_CHANCES = (0.93, 0.88, 0.62, 0.55, 0.50, 0.46, 0.41, 0.12, 0.07)
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

        distribution = {0: 1.0}
        column = self.build_column(distribution, 0)
        self.play(FadeIn(column), run_time=0.5)
        self.columns = VGroup(column)

        for step, chance in enumerate(WIN_CHANCES, start=1):
            nxt = {}
            for net, probability in distribution.items():
                nxt[net + 1] = nxt.get(net + 1, 0.0) + probability * chance
                nxt[net - 1] = nxt.get(net - 1, 0.0) + probability * (1.0 - chance)
            distribution = nxt
            column = self.build_column(distribution, step)
            self.columns.add(column)
            self.play(FadeIn(column), run_time=0.42)
        self.final_distribution = distribution

    def build_column(self, distribution: dict, step: int) -> VGroup:
        """One column of the sweep: a dot per reachable height, sized by its probability."""
        dots = VGroup()
        for net, probability in distribution.items():
            dots.add(Dot(
                point=[self.walk_x(step), self.walk_y(net), 0.0],
                radius=0.06 + 0.20 * probability ** 0.5,
                color=BLUE_B if net > 0 else GREY_D,
            ).set_opacity(0.35 + 0.65 * probability ** 0.5))
        return dots

    # ── Act four: which category decides it ───────────────────────────────────────────

    def play_tipping_point(self) -> None:
        """The category is worth whatever chance the other eight have of leaving you level.

        Measured over the OTHER eight rather than all nine, which is what makes the height that
        matters zero: eight steps land on an even net, and a net of zero is four-four with the
        ninth category holding the casting vote.
        """
        with self.voiceover(text=NARRATION['tipping']) as tracker:
            self.play(FadeOut(self.columns), run_time=0.6)

            wait_until_phrase(self, tracker, 'exactly level')
            others = {0: 1.0}
            for chance in WIN_CHANCES[:-1]:
                nxt = {}
                for net, probability in others.items():
                    nxt[net + 1] = nxt.get(net + 1, 0.0) + probability * chance
                    nxt[net - 1] = nxt.get(net - 1, 0.0) + probability * (1.0 - chance)
                others = nxt

            level = Dot(point=[self.walk_x(len(CATEGORIES) - 1), self.walk_y(0), 0.0],
                        radius=0.20, color=YELLOW)
            readout = Text(f'{others.get(0, 0.0):.1%} of the time', font_size=26, color=YELLOW)
            readout.next_to(level, UP, buff=0.35)
            self.play(FadeIn(level, scale=2.0), Write(readout), run_time=1.0)
            self.wait(1.6)

        with self.voiceover(text=NARRATION['punting']):
            # PLACEHOLDER for the beat that earns the punting claim: the nine categories with
            # their tipping-point weights beside them, the near-certain ones visibly at nothing.
            # Wants the real gradient from the objective, so it waits for a prep script.
            self.wait(2.0)
        self.wait(0.6)
