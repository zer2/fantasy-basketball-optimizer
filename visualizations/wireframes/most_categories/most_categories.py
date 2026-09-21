"""Most Categories: every way the week can go, and the shortcut that makes it computable.

WIREFRAME -- a first draft to find out whether the argument lands, drafted against gTTS rather
than the shipped voice. See wireframes/README.md.

The argument in four moves:

    One, the payoff is all-or-nothing. Eight categories won pays the same as five, so what the
    algorithm needs is not an average but the probability of a majority.

    Two, that is a sum over every branch of a binary tree nine levels deep -- 512 leaves, each
    either a win or a loss. Drawn in full, because the size of it is the point.

    Three, the tree collapses. Nothing about a branch matters except how many categories it has
    won, so 512 leaves fold into ten running totals and each new category is one pass over them.
    This is the dynamic programme the docs describe in prose.

    Four, the same tally answers the question worth asking: how often is THIS category the one
    that decides the matchup? That is the tipping point probability, and it is the gradient the
    algorithm actually steps on -- and the reason Most Categories punts hardest, since a
    category already certain either way can never tip anything.

Nothing here is measured against the real objective yet. The win probabilities are stand-ins
chosen to make the shape legible; a prep script comes once the beats are settled.

    manim -ql visualizations/wireframes/most_categories/most_categories.py MostCategories
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import (
    VGroup, Line, Rectangle, Text, MathTex, Dot,
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

TALLY_BASELINE_Y = -2.1
TALLY_HEIGHT = 3.0
TALLY_BAR_WIDTH = 0.52
TALLY_GAP = 0.22


class MostCategories(VoiceoverScene):
    """The 512-leaf tree, its collapse into a tally, and the tipping point that falls out."""

    def construct(self) -> None:
        self.set_speech_service(DraftVoice())
        self.play_payoff()
        self.play_tree()
        self.play_collapse()
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

    # ── Act three: the tree collapses into a tally ────────────────────────────────────

    def play_collapse(self) -> None:
        """Fold the branches into ten bars: the chance of having won exactly k categories."""
        with self.voiceover(text=NARRATION['dynamic']):
            self.wait(1.2)

        with self.voiceover(text=NARRATION['collapse']):
            distribution = self.won_category_distribution(WIN_CHANCES)
            self.tally = self.build_tally(distribution)
            self.play(FadeOut(self.tree_levels), FadeOut(self.leaf_count), run_time=0.7)
            self.play(Create(self.tally), run_time=1.4)
            self.wait(1.0)

    def won_category_distribution(self, chances) -> np.ndarray:
        """P(exactly k categories won), by the same one-pass recurrence the algorithm uses.

        This IS the dynamic programme the scene is about, so it is written out rather than
        imported: each category folds into the running distribution once, which is the whole
        claim that 512 leaves were never needed.
        """
        distribution = np.array([1.0])
        for chance in chances:
            lost = np.append(distribution * (1.0 - chance), 0.0)
            won = np.insert(distribution * chance, 0, 0.0)
            distribution = lost + won
        return distribution

    def build_tally(self, distribution: np.ndarray) -> VGroup:
        """Ten bars, one per possible number of categories won, majority ones picked out."""
        tallest = float(distribution.max())
        bars = VGroup()
        span = len(distribution) * (TALLY_BAR_WIDTH + TALLY_GAP)
        for won, probability in enumerate(distribution):
            height = TALLY_HEIGHT * probability / tallest
            x = -span / 2 + won * (TALLY_BAR_WIDTH + TALLY_GAP) + TALLY_BAR_WIDTH / 2
            bar = Rectangle(
                width=TALLY_BAR_WIDTH, height=max(height, 0.01),
                fill_color=BLUE_D if won >= MAJORITY else GREY_D,
                fill_opacity=1.0, stroke_width=0,
            ).move_to([x, TALLY_BASELINE_Y + height / 2, 0.0])
            label = Text(str(won), font_size=20, color=GREY_B)
            label.move_to([x, TALLY_BASELINE_Y - 0.3, 0.0])
            bars.add(VGroup(bar, label))
        return bars

    # ── Act four: which category decides it ───────────────────────────────────────────

    def play_tipping_point(self) -> None:
        """The boundary bar, and what it means for a category to be worth anything."""
        with self.voiceover(text=NARRATION['tipping']) as tracker:
            # The tally so far counts all nine. The tipping point is a statement about the OTHER
            # eight: this category decides the matchup exactly when they land on four, one short
            # of the majority. So the bars are recomputed without the category being valued, and
            # the bar that matters is four-of-eight rather than the majority bar of the nine.
            wait_until_phrase(self, tracker, 'exactly on the boundary')
            others = self.won_category_distribution(WIN_CHANCES[:-1])
            without = self.build_tally(others)
            self.play(Transform(self.tally, without), run_time=1.0)

            decisive = self.build_tally(others)[MAJORITY - 1]
            decisive.set_color(YELLOW)
            self.play(FadeIn(decisive), run_time=0.6)
            self.wait(1.4)

        with self.voiceover(text=NARRATION['punting']):
            # PLACEHOLDER for the beat that earns the punting claim: the nine categories listed
            # with their tipping-point weights beside them, the near-certain ones visibly at
            # nothing. Needs the real gradient from the objective, so it waits for a prep script.
            self.wait(2.0)
        self.wait(0.6)
