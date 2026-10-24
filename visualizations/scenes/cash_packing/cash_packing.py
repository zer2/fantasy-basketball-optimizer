"""Cash packing: why a team's values rise when it cannot spend its money.

Voiced with the shipped voice (shared/narration_voice.py). Each beat's pictures are cued to the phrases of its narration
line (shared/narration_timing.wait_until_phrase), so rewording a line can move or break a cue: the phrases each beat
waits on are listed in its method.

The argument, in seven beats:

  packing             thirteen slots and $200; a shelf of players, each a block as wide as his value, with his position.
                      Three tries at thirteen come to $180, $210 (over the budget) and $196: none is exactly $200;
  fewer_options       other teams take players: the shelf empties, and there are fewer ways to make it work;
  ratio               two slots and $40, with the best players left worth $20 and $10 (and $9, $8, ... behind them): the
                      best two cannot spend the money;
  simple_scaling      scale them up by the cash ratio, $40 / $30, so they spend it -- max(1, cash ratio);
  smooth_scaling      1 + ln(1 + e^(x - 1)) is about 1 for small x and about x for large; calibrated with a softness
                      s = 0.2 it hugs the corner, and starts before it;
  feedback_process    your dot slides right as your cash builds, then snaps back left and lower when you win a bid;
  other_modification  the same two slots and $40: losing the $20 player costs $11 (you settle for $10 + $9), losing the
                      $10 player only $1 ($20 + $9) -- so the boost goes as value squared, +$8 to +$2, not +$6.7 to +$3.3.

The numbers illustrate (they are chosen, not measured), which is all this argument needs.

    manim -qh visualizations/scenes/cash_packing/cash_packing.py CashPacking
"""

from __future__ import annotations

import sys
from math import exp, log
from pathlib import Path

from manim import (
    VGroup, Rectangle, Text, MathTex, Axes, Dot, DashedLine, Circle, Line,
    FadeIn, FadeOut, Create, Write, Transform, Indicate, Flash, ValueTracker, always_redraw,
    UP, DOWN, LEFT, RIGHT, UL, UR,
    BLUE_B, YELLOW, GREY_B, GREY_C, GREY_D, GREEN_C, RED_B, WHITE,
)
from manim_voiceover import VoiceoverScene

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.narration_voice import NarrationVoice         # noqa: E402
from shared.narration_timing import wait_until_phrase     # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from narration import NARRATION                           # noqa: E402


# ── Blocks and bars ──────────────────────────────────────────────────────────────────────

BAR_HEIGHT = 1.0
# The gap between packed blocks, taken out of each block's own width, so blocks worth exactly the bar fill it exactly.
BLOCK_GAP = 0.04
# Every bar is centred on this height; the shelf of options sits above it.
BAR_Y = -1.9
SHELF_TOP_Y = 2.6
SHELF_ROW_GAP = 1.15
# The $40 example's single row of options, lower, clear of the slot count and the boosts written over it.
LATE_SHELF_Y = 1.0
SHELF_GAP = 0.12
PLAYER_COLOUR = BLUE_B

# ── The opening: thirteen slots and $200 ─────────────────────────────────────────────────

BUDGET_DOLLARS = 200
OPENING_SLOTS = 13
# Scene units per dollar: the $200 bar spans most of the frame's width.
UNITS_PER_DOLLAR = 0.058
# (value, position). The thirteen you end up with come to $196, a mix of positions; the rest are the players you pass
# over.
TAKEN_PLAYERS = (
    (38, 'C'), (27, 'PG'), (22, 'SF'), (18, 'PF'), (16, 'SG'), (14, 'C'),
    (12, 'PG'), (11, 'F'), (10, 'G'), (9, 'C'), (8, 'SF'), (6, 'PG'),
    (5, 'PF'),
)
PASSED_PLAYERS = (
    (55, 'PG'), (41, 'C'), (30, 'SG'), (24, 'PF'), (19, 'SF'), (15, 'C'),
    (13, 'SG'), (9, 'PF'), (7, 'G'),
)
# Three tries at thirteen players, as positions in TAKEN_PLAYERS + PASSED_PLAYERS: $180 (two stars swapped for cheaper
# players), $210 (over the budget: it will not fit), and $196, the one you keep.
COMBINATIONS = (
    (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 17),
    (0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 19),
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
)
# The shelf's order: taken and passed players mixed, so which thirteen to take is not obvious.
SHELF_ORDER = (0, 13, 1, 14, 2, 15, 3, 16, 4, 17, 5, 18, 6, 19, 7, 20, 8, 21, 9, 10, 11, 12)
SHELF_ROWS = 3
# The passed players other teams take when the options run low: all but the last two.
TAKEN_BY_OTHERS_COUNT = len(PASSED_PLAYERS) - 2

# ── The late example: two slots and $40 ──────────────────────────────────────────────────

LATE_CASH = 40
LATE_SLOTS = 2
LATE_UNITS_PER_DOLLAR = 0.2
# The players left, best first: the best two are worth $30 together, and the rest are no help.
LATE_PLAYERS = ((20, 'SF'), (10, 'C'), (9, 'PG'), (8, 'C'), (6, 'SG'), (5, 'PF'))
LATE_BEST_COUNT = 2

# ── The chart ────────────────────────────────────────────────────────────────────────────

# The app's softness (parameters.yaml, cash_share_softness).
SOFTNESS = 0.2
# Wide enough on the left for the raw expression to settle towards 1; the cash ratio rarely gets near 2.
X_RANGE = (-1.0, 2.0, 0.5)
Y_RANGE = (0.0, 2.5, 0.5)
# Where the curves of the scaling rules are drawn: the ratio is never negative.
RULE_X_RANGE = (0.0, 2.0)
# The feedback beat: you are close to the corner, slightly overpay to win a player, and the cash that costs you bounces
# you back to where the ratio is no problem at all.
FEEDBACK_START_RATIO = 0.75
FEEDBACK_PEAK_RATIO = 0.9
FEEDBACK_AFTER_WIN_RATIO = 0.4
# Where the 'far right zone' starts: past the corner, where nothing has eased the ratio back.
FAR_RIGHT_RATIO = 1.5


def compute_raw_expression(x: float) -> float:
    """1 + ln(1 + e^(x - 1)): about 1 for small x, about x for large x."""
    return 1.0 + log(1.0 + exp(x - 1.0))


def compute_smooth_scale(ratio: float) -> float:
    """1 + s ln(1 + e^((cash ratio - 1) / s)), as backend/services/auction_pricing.smooth_spending_scale."""
    return 1.0 + SOFTNESS * log(1.0 + exp((ratio - 1.0) / SOFTNESS))


def compute_simple_scale(ratio: float) -> float:
    """max(1, cash ratio): simple scaling."""
    return max(1.0, ratio)


def format_dollars(dollars: float) -> str:
    """$28, or $26.7 where the cents matter."""
    return f'${dollars:.0f}' if abs(dollars - round(dollars)) < 0.05 else f'${dollars:.1f}'


class CashPacking(VoiceoverScene):
    """Player values packed into a budget, the cash ratio that says when they no longer can be, and the two ways of
    scaling them up."""

    def construct(self) -> None:
        self.set_speech_service(NarrationVoice())
        self.play_packing()
        self.play_fewer_options()
        self.play_ratio()
        self.play_simple_scaling()
        self.play_smooth_scaling()
        self.play_feedback_process()
        self.play_other_modification()

    # ── Shared apparatus ─────────────────────────────────────────────────────────────────

    def build_budget_bar(
        self
        , dollars: float
        , units_per_dollar: float
    ) -> VGroup:
        """An empty bar this many dollars wide at this scale, centred, labelled with its size."""
        frame = Rectangle(width=dollars * units_per_dollar, height=BAR_HEIGHT, color=GREY_B, stroke_width=3)
        frame.move_to([0.0, BAR_Y, 0.0])
        label = Text(f'${dollars}', font_size=26, color=GREY_B).next_to(frame, DOWN, buff=0.2)
        return VGroup(frame, label)

    def build_player_block(
        self
        , dollars: float
        , position: str
        , units_per_dollar: float
    ) -> VGroup:
        """One player: a block as wide as his value at this scale, his value and position on it."""
        block = Rectangle(width=dollars * units_per_dollar - BLOCK_GAP, height=BAR_HEIGHT * 0.88, color=PLAYER_COLOUR,
                          fill_color=PLAYER_COLOUR, fill_opacity=0.35, stroke_width=2)
        font_size = 16 if block.width < 0.6 else 20
        label = VGroup(Text(format_dollars(dollars), font_size=font_size, color=WHITE),
                       Text(position, font_size=font_size - 4, color=GREY_B)).arrange(DOWN, buff=0.06)
        return VGroup(block, label.move_to(block))

    def place_on_shelf(
        self
        , blocks: list[VGroup]
        , rows: int
        , top_y: float
    ) -> None:
        """The options side by side above the bar, in this many rows of about equal width (split by width, not count:
        a row of the expensive players ran off the frame), each row centred."""
        target_width = (sum(block[0].width for block in blocks) + SHELF_GAP * len(blocks)) / rows
        row_lists, current, width = [], [], 0.0
        for block in blocks:
            if current and width + block[0].width > target_width and len(row_lists) < rows - 1:
                row_lists.append(current)
                current, width = [], 0.0
            current.append(block)
            width += block[0].width + SHELF_GAP
        row_lists.append(current)
        for row, row_blocks in enumerate(row_lists):
            VGroup(*row_blocks).arrange(RIGHT, buff=SHELF_GAP).move_to([0.0, top_y - row * SHELF_ROW_GAP, 0.0])

    def place_packed(self, blocks: list[VGroup]) -> None:
        """Puts these blocks where they sit packed into the current bar, from its left end, in order."""
        x = self.bar[0].get_left()[0]
        for block in blocks:
            block.move_to([x + BLOCK_GAP / 2 + block[0].width / 2, BAR_Y, 0.0])
            x += block[0].width + BLOCK_GAP

    def pack_into_bar(self, blocks: list[VGroup]) -> list:
        """The animations that slide these blocks into the current bar from its left end, in order."""
        animations, x = [], self.bar[0].get_left()[0]
        for block in blocks:
            width = block[0].width
            animations.append(block.animate.move_to([x + BLOCK_GAP / 2 + width / 2, BAR_Y, 0.0]))
            x += width + BLOCK_GAP
        return animations

    def build_unspent_label(self, dollars: float) -> Text:
        """What the team cannot spend, over the bar's right end."""
        label = Text(f'{format_dollars(dollars)} unspent', font_size=24, color=RED_B)
        return label.next_to(self.bar, UP, buff=0.25).align_to(self.bar, RIGHT)

    def build_slots_label(self, slots: int) -> Text:
        """How many roster slots are open, just above the current bar's left end, so it reads as part of the budget
        (at the top of the frame it looked unrelated to the bar)."""
        label = Text(f'Slots: {slots}', font_size=26, color=WHITE)
        return label.next_to(self.bar[0], UP, buff=0.25).align_to(self.bar[0], LEFT)

    # ── Beat 1: packing ──────────────────────────────────────────────────────────────────

    def play_packing(self) -> None:
        """Cues: 'thirteen players' (the open slots), 'worth 200 dollars combined', 'positionally balanced' and
        'coherent punt build' (one try at thirteen players each: $180, $210 -- over the budget, so it overflows the bar
        -- and $196), 'not obvious how to get the full value' (the last try's leftover)."""
        players = TAKEN_PLAYERS + PASSED_PLAYERS
        with self.voiceover(text=NARRATION['packing']) as tracker:
            self.bar = self.build_budget_bar(BUDGET_DOLLARS, UNITS_PER_DOLLAR)
            self.shelf = [self.build_player_block(value, position, UNITS_PER_DOLLAR) for value, position in players]
            self.place_on_shelf([self.shelf[index] for index in SHELF_ORDER], SHELF_ROWS, SHELF_TOP_Y)
            shelf_spots = [block.get_center() for block in self.shelf]
            self.play(Create(self.bar), FadeIn(VGroup(*self.shelf), lag_ratio=0.04), run_time=1.6)

            wait_until_phrase(self, tracker, 'thirteen players')
            self.slots_label = self.build_slots_label(OPENING_SLOTS)
            self.play(FadeIn(self.slots_label), run_time=0.6)

            packed, total_label = [], None
            for phrase, combination in zip(('worth 200 dollars combined', 'positionally balanced', 'coherent punt build'),
                                           COMBINATIONS):
                wait_until_phrase(self, tracker, phrase)
                if packed:
                    self.play(*[block.animate.move_to(shelf_spots[self.shelf.index(block)]) for block in packed],
                              FadeOut(total_label), run_time=0.5)
                packed = [self.shelf[index] for index in sorted(combination, key=lambda index: -players[index][0])]
                total = sum(players[index][0] for index in combination)
                self.play(*self.pack_into_bar(packed), run_time=1.0)
                over = total > BUDGET_DOLLARS
                total_label = Text(f'${total}' + (': over budget' if over else ''), font_size=24,
                                   color=RED_B if over else WHITE)
                total_label.next_to(self.bar, UP, buff=0.25).align_to(self.bar, RIGHT)
                self.play(FadeIn(total_label), run_time=0.4)
            self.taken = packed

            wait_until_phrase(self, tracker, 'not obvious how to get the full value')
            self.leftover = self.build_unspent_label(BUDGET_DOLLARS - sum(value for value, _ in TAKEN_PLAYERS))
            self.play(Transform(total_label, self.leftover), run_time=0.6)
            self.leftover = total_label

    # ── Beat 2: fewer options ────────────────────────────────────────────────────────────

    def play_fewer_options(self) -> None:
        """Cues: 'decreases' (other teams take most of the players you passed over), 'fewer ways' (the few left)."""
        passed = self.shelf[len(TAKEN_PLAYERS):]
        with self.voiceover(text=NARRATION['fewer_options']) as tracker:
            wait_until_phrase(self, tracker, 'decreases')
            self.play(*[FadeOut(block, shift=UP * 0.5) for block in passed[:TAKEN_BY_OTHERS_COUNT]], run_time=1.0)

            wait_until_phrase(self, tracker, 'fewer ways')
            self.play(*[Indicate(block, color=WHITE) for block in passed[TAKEN_BY_OTHERS_COUNT:]], run_time=0.9)
        self.play(FadeOut(VGroup(self.bar, *self.taken, *passed[TAKEN_BY_OTHERS_COUNT:], self.leftover,
                                 self.slots_label)), run_time=0.6)

    # ── Beat 3: the ratio ────────────────────────────────────────────────────────────────

    def play_ratio(self) -> None:
        """Cues: 'worth less to you than' (the best two of the players left pack into the $40 bar), 'impossible to
        spend' (what is left over: the $9, $8 and the rest are no help)."""
        with self.voiceover(text=NARRATION['ratio']) as tracker:
            self.bar = self.build_budget_bar(LATE_CASH, LATE_UNITS_PER_DOLLAR)
            self.slots_label = self.build_slots_label(LATE_SLOTS)
            self.late_shelf = [self.build_player_block(value, position, LATE_UNITS_PER_DOLLAR)
                               for value, position in LATE_PLAYERS]
            self.place_on_shelf(self.late_shelf, 1, LATE_SHELF_Y)
            self.play(Create(self.bar), FadeIn(self.slots_label), FadeIn(VGroup(*self.late_shelf), lag_ratio=0.1),
                      run_time=1.2)

            wait_until_phrase(self, tracker, 'worth less to you than')
            self.late_best = self.late_shelf[:LATE_BEST_COUNT]
            self.play(*self.pack_into_bar(self.late_best), run_time=1.2)

            wait_until_phrase(self, tracker, 'impossible to spend')
            best_value = sum(value for value, _ in LATE_PLAYERS[:LATE_BEST_COUNT])
            self.leftover = self.build_unspent_label(LATE_CASH - best_value)
            self.play(FadeIn(self.leftover), run_time=0.6)

    # ── Beat 4: simple scaling ───────────────────────────────────────────────────────────

    def play_simple_scaling(self) -> None:
        """Cues: 'Scaling up your valuation' (the best two grow to fill the bar), 'ratio of your remaining
        cash' (the cash ratio written), 'is below 1' (the chart), 'maximum of one and' (the corner drawn,
        with the example on it)."""
        best_value = sum(value for value, _ in LATE_PLAYERS[:LATE_BEST_COUNT])
        ratio = LATE_CASH / best_value
        with self.voiceover(text=NARRATION['simple_scaling']) as tracker:
            wait_until_phrase(self, tracker, 'Scaling up your valuation')
            # The grown blocks are built where they will sit and the old ones transform into them: moving and growing
            # in one step (a Transform alongside a move) lets the move undo the growth.
            scaled = [self.build_player_block(value * ratio, position, LATE_UNITS_PER_DOLLAR)
                      for value, position in LATE_PLAYERS[:LATE_BEST_COUNT]]
            self.place_packed(scaled)
            self.play(FadeOut(self.leftover), *[Transform(old, new) for old, new in zip(self.late_best, scaled)],
                      run_time=1.4)

            wait_until_phrase(self, tracker, 'ratio of your remaining cash')
            self.ratio_label = MathTex(r'\text{cash ratio} = \frac{\$' + str(LATE_CASH) + r'}{\$' + str(best_value)
                                       + r'} \approx ' + f'{ratio:.2f}', color=YELLOW).to_edge(UP, buff=0.5)
            self.play(Write(self.ratio_label), run_time=1.2)

            wait_until_phrase(self, tracker, 'is below 1')
            self.axes = Axes(x_range=X_RANGE, y_range=Y_RANGE, x_length=8.0, y_length=5.2,
                             axis_config={'color': GREY_B, 'include_numbers': True, 'font_size': 24}).shift(DOWN * 0.9)
            axis_labels = VGroup(Text('cash ratio', font_size=24, color=GREY_B).next_to(self.axes.x_axis, RIGHT, buff=0.25),
                                 Text('multiplier', font_size=24, color=GREY_B).next_to(self.axes.y_axis, UP, buff=0.2))
            self.play(FadeOut(VGroup(self.bar, *self.late_shelf, self.slots_label)),
                      FadeOut(self.ratio_label),   # the first example's ratio; the chart is about any ratio
                      Create(self.axes), Write(axis_labels), run_time=1.5)

            wait_until_phrase(self, tracker, 'maximum of one and')
            self.simple_curve = self.axes.plot(compute_simple_scale, x_range=list(RULE_X_RANGE), color=YELLOW,
                                               use_smoothing=False)
            simple_label = MathTex(r'\max(1,\ \text{cash ratio})', color=YELLOW, font_size=32)
            # Under the corner, where no curve passes and clear of the far-right zone: beside the rising segment, the
            # line and the curves ran through it.
            simple_label.move_to(self.axes.c2p(1.0, 0.75))
            self.play(Create(self.simple_curve), FadeIn(simple_label), run_time=1.8)
            self.simple_dot = Dot(self.axes.c2p(ratio, compute_simple_scale(ratio)), color=YELLOW)
            self.play(FadeIn(self.simple_dot), run_time=0.5)

    # ── Beat 5: the smooth version ───────────────────────────────────────────────────────

    def play_smooth_scaling(self) -> None:
        """Cues: 'smoothing out the sharp angle' (the corner marked), 'This expression' (1 + ln(1 + e^(x-1)) and its
        curve), 'approximately 1 when x is small' and 'approximately x when x is large' (its two limits as guides),
        'Calibrating that' (divided by s = 0.2: the curve tightens onto the corner), 'We can scale player values by it'
        (the example on it)."""
        axes = self.axes
        ratio = LATE_CASH / sum(value for value, _ in LATE_PLAYERS[:LATE_BEST_COUNT])
        with self.voiceover(text=NARRATION['smooth_scaling']) as tracker:
            wait_until_phrase(self, tracker, 'smoothing out the sharp angle')
            corner = Circle(radius=0.32, color=RED_B, stroke_width=3).move_to(axes.c2p(1.0, 1.0))
            self.play(Create(corner), run_time=0.8)

            wait_until_phrase(self, tracker, 'This expression')
            # Top right: centred, it ran into the cash ratio's definition at the top left.
            formula = MathTex(r'1 + \ln\left(1 + e^{\,x - 1}\right)', color=GREY_C, font_size=40).to_corner(UR, buff=0.5)
            raw_curve = axes.plot(compute_raw_expression, x_range=list(X_RANGE[:2]), color=GREY_C)
            self.play(FadeOut(corner), Write(formula), Create(raw_curve), run_time=1.6)

            wait_until_phrase(self, tracker, 'approximately 1 when x is small')
            flat_guide = DashedLine(axes.c2p(X_RANGE[0], 1.0), axes.c2p(X_RANGE[1], 1.0), color=GREY_D)
            self.play(Create(flat_guide), run_time=0.8)

            wait_until_phrase(self, tracker, 'approximately x when x is large')
            diagonal_guide = DashedLine(axes.c2p(0.0, 0.0), axes.c2p(2.0, 2.0), color=GREY_D)
            self.play(Create(diagonal_guide), run_time=0.8)

            wait_until_phrase(self, tracker, 'Calibrating that')
            calibrated = MathTex(r'1 + s\ln\left(1 + e^{\,(x - 1)/s}\right)', r',\quad s = 0.2', color=BLUE_B,
                                 font_size=40).to_corner(UR, buff=0.5)
            smooth_curve = axes.plot(compute_smooth_scale, x_range=list(RULE_X_RANGE), color=BLUE_B)
            self.play(Transform(formula, calibrated), Transform(raw_curve, smooth_curve),
                      FadeOut(VGroup(flat_guide, diagonal_guide)), run_time=2.0)

            wait_until_phrase(self, tracker, 'We can scale player values by it')
            self.smooth_dot = Dot(axes.c2p(ratio, compute_smooth_scale(ratio)), color=BLUE_B)
            self.play(FadeIn(self.smooth_dot), run_time=0.6)

    # ── Beat 6: the feedback process ─────────────────────────────────────────────────────

    def play_feedback_process(self) -> None:
        """Cues: 'the more cash you have on hand' (your dot slides right, close to the corner), 'slightly overpay and win
        a bid' (it wins, overpaying by about a dollar), 'pushing the cash ratio down' (it bounces back to 0.4, where the
        ratio is no problem), 'far right zone' (the region it never reaches, where the overpay would be about ten
        dollars)."""
        axes = self.axes
        position = ValueTracker(FEEDBACK_START_RATIO)
        you = always_redraw(lambda: Dot(axes.c2p(position.get_value(), compute_smooth_scale(position.get_value())),
                                        color=GREEN_C, radius=0.11))
        you_label = always_redraw(lambda: Text('you', font_size=22, color=GREEN_C).next_to(you, UP + LEFT, buff=0.05))
        with self.voiceover(text=NARRATION['feedback_process']) as tracker:
            self.play(FadeOut(VGroup(self.smooth_dot, self.simple_dot)), FadeIn(you), FadeIn(you_label), run_time=0.6)

            wait_until_phrase(self, tracker, 'the more cash you have on hand')
            self.play(position.animate.set_value(FEEDBACK_PEAK_RATIO), run_time=2.0)

            wait_until_phrase(self, tracker, 'slightly overpay and win a bid')
            # Below the curves: above them it ran into the max(1, cash ratio) label and the axis numbers.
            small_overpay = Text('overpay by ~$1', font_size=24, color=GREEN_C)
            small_overpay.move_to(axes.c2p(FEEDBACK_PEAK_RATIO, 0.6))
            self.play(Flash(you, color=GREEN_C), FadeIn(small_overpay), run_time=0.9)

            wait_until_phrase(self, tracker, 'pushing the cash ratio down')
            self.play(position.animate.set_value(FEEDBACK_AFTER_WIN_RATIO), FadeOut(small_overpay), run_time=1.0)

            wait_until_phrase(self, tracker, 'far right zone')
            far_right = Rectangle(width=axes.c2p(X_RANGE[1], 0)[0] - axes.c2p(FAR_RIGHT_RATIO, 0)[0],
                                  height=axes.y_length, stroke_width=0, fill_color=RED_B, fill_opacity=0.15)
            far_right.move_to(axes.c2p((FAR_RIGHT_RATIO + X_RANGE[1]) / 2, (Y_RANGE[0] + Y_RANGE[1]) / 2))
            # Two lines set apart (one Text with a line break squeezed its letters), above the curve and ending at the
            # zone's right edge: the zone is narrower than the words.
            large_overpay = VGroup(Text('forced to overpay', font_size=22, color=RED_B),
                                   Text('by ~$10', font_size=22, color=RED_B)).arrange(DOWN, buff=0.1)
            large_overpay.move_to(axes.c2p(1.6, 2.3)).align_to(far_right, RIGHT)
            self.play(FadeIn(far_right), FadeIn(large_overpay), run_time=0.8)
        # Everything drawn goes; the ValueTracker behind your dot is not drawn, so it cannot be grouped with the rest.
        self.play(*[FadeOut(mobject) for mobject in self.mobjects if not isinstance(mobject, ValueTracker)],
                  run_time=0.8)

    # ── Beat 7: the squared boost ────────────────────────────────────────────────────────

    def play_other_modification(self) -> None:
        """Cues: 'inflates all values' (every option's boost under one multiplier: +$6.7 for the $20 player, +$3.3 for
        the $10), 'lose the twenty dollar player' (you settle for $10 + $9 = $19), 'Losing the ten dollar player'
        ($20 + $9 = $29: only $1 lost), 'secure the highest-value' (the $20 player marked), 'square of player value'
        (the squared boosts: +$8, +$2, +$1.6, ...), 'four times the boost' (the two marked). Drawn at the simple scale,
        a multiplier of 4/3, the clean numbers of the docs' example."""
        best_value = sum(value for value, _ in LATE_PLAYERS[:LATE_BEST_COUNT])
        excess = LATE_CASH / best_value - 1
        weighted_average = sum(value ** 2 for value, _ in LATE_PLAYERS[:LATE_BEST_COUNT]) / best_value
        uniform_boosts = [excess * value for value, _ in LATE_PLAYERS]
        squared_boosts = [excess * value ** 2 / weighted_average for value, _ in LATE_PLAYERS]

        def build_boost_labels(blocks, boosts) -> VGroup:
            return VGroup(*[Text('+' + format_dollars(boost), font_size=22, color=YELLOW).next_to(block, UP, buff=0.15)
                            for block, boost in zip(blocks, boosts)])

        def build_team_value(players) -> Text:
            return Text(f'team: {format_dollars(sum(value for value, _ in players))}', font_size=26,
                        color=WHITE).next_to(self.bar, RIGHT, buff=0.3)

        with self.voiceover(text=NARRATION['other_modification']) as tracker:
            self.bar = self.build_budget_bar(LATE_CASH, LATE_UNITS_PER_DOLLAR)
            slots_label = self.build_slots_label(LATE_SLOTS)
            shelf = [self.build_player_block(value, position, LATE_UNITS_PER_DOLLAR)
                     for value, position in LATE_PLAYERS]
            self.place_on_shelf(shelf, 1, LATE_SHELF_Y)
            self.play(Create(self.bar), FadeIn(slots_label), FadeIn(VGroup(*shelf)), run_time=1.0)

            wait_until_phrase(self, tracker, 'inflates all values')
            boosts = build_boost_labels(shelf, uniform_boosts)
            self.play(FadeIn(boosts, lag_ratio=0.1), run_time=1.2)

            twenty, ten, nine = shelf[0], shelf[1], shelf[2]
            wait_until_phrase(self, tracker, 'lose the twenty dollar player')
            cross = VGroup(Line(twenty.get_corner(UL), twenty.get_corner(DOWN + RIGHT), color=RED_B, stroke_width=5),
                           Line(twenty.get_corner(DOWN + LEFT), twenty.get_corner(UR), color=RED_B, stroke_width=5))
            ten_copy, nine_copy = ten.copy(), nine.copy()
            team = build_team_value((LATE_PLAYERS[1], LATE_PLAYERS[2]))
            self.play(Create(cross), *self.pack_into_bar([ten_copy, nine_copy]), FadeIn(team), run_time=1.4)

            wait_until_phrase(self, tracker, 'Losing the ten dollar player')
            twenty_copy, second_nine = twenty.copy(), nine.copy()
            other_team = build_team_value((LATE_PLAYERS[0], LATE_PLAYERS[2]))
            self.play(FadeOut(VGroup(cross, ten_copy, nine_copy)), *self.pack_into_bar([twenty_copy, second_nine]),
                      Transform(team, other_team), run_time=1.4)

            wait_until_phrase(self, tracker, 'secure the highest-value')
            self.play(FadeOut(VGroup(twenty_copy, second_nine, team)), Indicate(twenty, color=YELLOW), run_time=1.0)

            wait_until_phrase(self, tracker, 'square of player value')
            self.play(Transform(boosts, build_boost_labels(shelf, squared_boosts)), run_time=1.4)

            wait_until_phrase(self, tracker, 'four times the boost')
            self.play(Indicate(boosts[0], color=YELLOW), run_time=0.8)
            self.play(Indicate(boosts[1], color=YELLOW), run_time=0.8)
        self.wait(1.0)
