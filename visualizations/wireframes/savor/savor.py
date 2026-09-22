"""SAVOR: what a player is worth once you can drop them.

WIREFRAME -- a first draft to find out whether the argument lands, drafted against gTTS rather
than the shipped voice. See wireframes/README.md.

The whole adjustment is one picture: a projection is a Normal, there is a floor under it because
a player who busts gets dropped for a free agent, and the alternative you are bidding against is
itself a free player with the same floor.

    value = E[max(mu + noise, 0)] - E[max(noise, 0)]

which is exactly the formula in the docs,

    mu * Phi(mu/sigma) - (sigma / sqrt(2 pi)) * (1 - exp(-mu^2 / (2 sigma^2)))

The two terms are the two things drawn: a Normal truncated at the replacement line, minus a
half-normal for the dollar flyer. The scene's job is to make the asymmetry obvious -- the cut
takes almost nothing from a star and a great deal from a marginal player -- which is why money
concentrates at the top of an auction.

Both halves are computed here in closed form rather than measured, because they ARE closed form;
what a prep script would add is real dollar values from a session to land the last beat on.

    manim -ql visualizations/wireframes/savor/savor.py Savor
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import (
    VGroup, Line, Polygon, DashedLine, Text, MathTex,
    FadeIn, FadeOut, Create, Write, Transform,
    DOWN, UP, RIGHT,
    BLUE_D, BLUE_B, RED_B, GREEN_C, GREY_B, GREY_D, YELLOW, WHITE,
)
from manim_voiceover import VoiceoverScene

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.draft_voice import DraftVoice                 # noqa: E402
from shared.narration_timing import wait_until_phrase     # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from narration import NARRATION                           # noqa: E402


# ── Layout ───────────────────────────────────────────────────────────────────────────

BASELINE_Y = -2.3
AXIS_HALF_WIDTH = 6.2
CURVE_PEAK = 1.9
CURVE_SAMPLES = 260

# Value in units of sigma above replacement. The replacement line sits at zero by definition:
# it is the value of a player you can have for nothing.
REPLACEMENT = 0.0
NOISE_SPREAD = 1.0

# All three on one axis, which is what makes the comparison legible. Their marginal value is
# Phi(mu/sigma) -- the share of them above the line -- so the flyer at the line keeps half of any
# improvement, the starter about seven tenths, and the star essentially all of it.
PLAYERS = (
    {'name': 'star',        'value': 2.20, 'colour': BLUE_B,
     'label_shift':  0.00, 'label_height': 2.45},
    # Shifted up and to the right of the replacement line: centred near it, the word ran
    # across the dashed line itself.
    {'name': 'starter',     'value': 0.55, 'colour': GREEN_C,
     'label_shift':  0.62, 'label_height': 2.28},
    {'name': 'dollar flyer','value': 0.00, 'colour': GREY_B,
     'label_shift': -1.95, 'label_height': 1.35},
)
# How far a mean is pushed to ask what the improvement is worth. Small on purpose: the claim is
# about the MARGINAL unit, and a large shift would be answering a different question.
NUDGE = 0.45

# The opening: one projection with a question mark against it, then the values a season could
# actually return, scattered around it.
GUESS_DOLLARS = 45
# Offset in dollars, then where it sits. Held well clear of the projection row in the middle --
# at half these distances the values crowded the text they are meant to be scattering around.
POSSIBLE_OUTCOMES = (
    (-14, -4.6,  1.35), (-8, -3.0, -1.30), (-3, -1.6,  1.25),
    (  2,  1.6, -1.25), (  6,  3.0,  1.30), (12,  4.6, -1.35),
)

# The auction board the last act works on. Dollar values a drafter would recognise, against a
# one dollar replacement player, and S-sigma at the sidebar default of 10.
PROJECTED_DOLLARS = (60, 44, 32, 23, 16, 10, 7, 4, 3, 1)   # one team's $200 auction budget
REPLACEMENT_DOLLARS = 1
S_SIGMA = 10.0
# Ticks in dollars above replacement. The curve act draws in units of S-sigma, so a tick every
# twenty dollars is every two units, and the axis reaches forty without running past its end.
AXIS_DOLLAR_TICKS = (-20, 0, 20, 40)
TABLE_TOP_Y = 2.55
TABLE_ROW_GAP = 0.46
# How long the nudged curve takes to settle back, once its line has finished.
REVERT_SECONDS = 0.8

# Kept small enough that a curve centred on the star still has both tails inside the axis;
# wider and the right tail is cut off flat at the end of the line.
SCENE_UNITS_PER_VALUE = 1.15


class Savor(VoiceoverScene):
    """A projection with a floor under it, and the free player it has to beat."""

    def construct(self) -> None:
        self.set_speech_service(DraftVoice())
        self.play_three_players()
        self.play_the_nudge()
        self.play_the_calculation()

    # ── Shared apparatus ──────────────────────────────────────────────────────────────

    def to_scene_x(self, value: float) -> float:
        return value * SCENE_UNITS_PER_VALUE

    def build_axis(self) -> VGroup:
        """The value axis, ticked in dollars rather than left as a bare line.

        A projection's spread is S-sigma, which ships at ten dollars, so the scene's unit IS a
        ten dollar step and the axis can simply say so. That also lets the replacement line be
        labelled with what it is worth -- nothing -- and connects this act to the dollar table
        the scene ends on.
        """
        axis = Line([-AXIS_HALF_WIDTH, BASELINE_Y, 0.0], [AXIS_HALF_WIDTH, BASELINE_Y, 0.0],
                    color=GREY_B, stroke_width=3)
        marks = VGroup()
        for dollars in AXIS_DOLLAR_TICKS:
            x = self.to_scene_x(dollars / S_SIGMA)
            marks.add(Line([x, BASELINE_Y - 0.11, 0.0], [x, BASELINE_Y + 0.11, 0.0],
                           color=GREY_B, stroke_width=2))
            marks.add(Text(f'${dollars}' if dollars >= 0 else f'-${abs(dollars)}',
                           font_size=20, color=GREY_D)
                      .move_to([x, BASELINE_Y - 0.36, 0.0]))
        caption = Text('value delivered over the season', font_size=23, color=GREY_B)
        caption.move_to([0.0, BASELINE_Y - 0.78, 0.0])
        return VGroup(axis, marks, caption)

    def build_replacement_line(self) -> VGroup:
        line = DashedLine([self.to_scene_x(REPLACEMENT), BASELINE_Y, 0.0],
                          [self.to_scene_x(REPLACEMENT), BASELINE_Y + 3.1, 0.0],
                          color=YELLOW, stroke_width=3, dash_length=0.12)
        # Above the line it names, which is where it belongs. It sits clear of the players'
        # own labels because the tallest of those is lower than the top of this line.
        label = Text('replacement level', font_size=21, color=YELLOW)
        label.move_to([self.to_scene_x(REPLACEMENT), BASELINE_Y + 3.38, 0.0])
        return VGroup(line, label)

    def normal_points(self, centre: float, spread: float, low: float, high: float):
        xs = np.linspace(low, high, CURVE_SAMPLES)
        ys = CURVE_PEAK * np.exp(-0.5 * ((xs - self.to_scene_x(centre))
                                         / (spread * SCENE_UNITS_PER_VALUE)) ** 2)
        return xs, ys

    def build_curve(self, centre: float, colour, opacity: float = 0.26) -> Polygon:
        """The full projection distribution, floor ignored."""
        xs, ys = self.normal_points(centre, NOISE_SPREAD, -AXIS_HALF_WIDTH, AXIS_HALF_WIDTH)
        points = ([[xs[0], BASELINE_Y, 0.0]]
                  + [[x, BASELINE_Y + y, 0.0] for x, y in zip(xs, ys)]
                  + [[xs[-1], BASELINE_Y, 0.0]])
        return Polygon(*points, color=colour, fill_color=colour,
                       fill_opacity=opacity, stroke_width=2.5)

    def build_lost_tail(self, centre: float) -> Polygon:
        """The part of the distribution below replacement -- the outcomes you never take."""
        cut = self.to_scene_x(REPLACEMENT)
        xs, ys = self.normal_points(centre, NOISE_SPREAD, -AXIS_HALF_WIDTH, cut)
        points = ([[xs[0], BASELINE_Y, 0.0]]
                  + [[x, BASELINE_Y + y, 0.0] for x, y in zip(xs, ys)]
                  + [[cut, BASELINE_Y, 0.0]])
        return Polygon(*points, color=GREY_D, fill_color=GREY_D,
                       fill_opacity=0.55, stroke_width=0)

    def savor_value(self, mean: float, spread: float = NOISE_SPREAD) -> float:  # noqa: D401
        """The docs' formula, written as the two expectations it actually is.

        E[max(mu + noise, 0)] - E[max(noise, 0)], the second being the dollar flyer. Expanding
        the first gives mu*Phi(mu/sigma) + sigma*phi(mu/sigma) and the second sigma/sqrt(2 pi),
        which is the published closed form.
        """
        from math import erf, exp, pi, sqrt
        standardised = mean / spread
        cumulative = 0.5 * (1.0 + erf(standardised / sqrt(2.0)))
        density = exp(-0.5 * standardised ** 2) / sqrt(2.0 * pi)
        return mean * cumulative + spread * density - spread / sqrt(2.0 * pi)

    # -- Act one: three players, one floor --------------------------------------------

    def play_three_players(self) -> None:
        """All three on the same axis from the start, because the scene is a comparison.

        Drawn at once rather than one at a time: what the viewer has to hold is where each sits
        RELATIVE to the replacement line, and that only exists once all three are up.
        """
        with self.voiceover(text=NARRATION['three_players']) as tracker:
            # A projection with a question mark against it, then the question mark replaced by
            # what the season could actually return. The axis alone used to hold this line until
            # eighty percent of the way through it.
            wait_until_phrase(self, tracker, 'we do not know exactly')
            player = Text('a player', font_size=30, color=GREY_B)
            projection = Text(f'${GUESS_DOLLARS}', font_size=44, color=WHITE)
            query = Text('?', font_size=52, color=YELLOW)
            guess = VGroup(player, projection, query).arrange(RIGHT, buff=0.7)
            guess.move_to([0.0, 0.9, 0.0])
            self.play(FadeIn(guess), run_time=0.9)
            self.wait(0.6)

            wait_until_phrase(self, tracker, 'well above or below')
            outcomes = VGroup(*[
                Text(f'${GUESS_DOLLARS + offset}', font_size=30, color=GREY_B)
                .move_to([shift, 0.9 + rise, 0.0])
                for offset, shift, rise in POSSIBLE_OUTCOMES
            ])
            self.play(FadeOut(query), run_time=0.35)
            self.play(FadeIn(outcomes, lag_ratio=0.12), run_time=1.5)
            self.wait(0.8)

            wait_until_phrase(self, tracker, 'Normally distributed')
            self.axis = self.build_axis()
            self.play(FadeOut(VGroup(player, projection, outcomes)), run_time=0.5)
            self.play(Create(self.axis), run_time=0.8)

            self.curves, self.names = VGroup(), VGroup()
            for entry in PLAYERS:
                self.curves.add(self.build_curve(entry['value'], entry['colour']))
                label = Text(entry['name'], font_size=22, color=entry['colour'])
                label.move_to([self.to_scene_x(entry['value']) + entry['label_shift'],
                               BASELINE_Y + entry['label_height'], 0.0])
                self.names.add(label)
            wait_until_phrase(self, tracker, 'Here are some potential distributions')
            self.play(FadeIn(self.curves), FadeIn(self.names), run_time=1.6)
            self.wait(0.8)

        with self.voiceover(text=NARRATION['the_floor']) as tracker:
            wait_until_phrase(self, tracker, 'below the replacement level')
            self.replacement = self.build_replacement_line()
            self.play(Create(self.replacement), run_time=1.0)
            wait_until_phrase(self, tracker, 'below the line do not actually help')
            self.lost = VGroup(*[self.build_lost_tail(player['value']) for player in PLAYERS])
            self.play(FadeIn(self.lost), run_time=1.2)
            self.wait(1.2)

    # -- Act two: what one more unit of projection buys -------------------------------

    def play_the_nudge(self) -> None:
        """Shift each mean by the same amount and show how much of it survives the floor.

        This is the derivative of the SAVOR value, and it is exactly the share of the player
        above replacement -- verified against the closed form: 50.0% at the line, 70.9% for the
        starter, 98.6% for the star. So "half their distribution" is not a loose way of speaking;
        it is the number.
        """
        for key, index in (('the_flyer_half', 2), ('the_star_whole', 0)):
            with self.voiceover(text=NARRATION[key]) as tracker:
                self.play_one_nudge(index, tracker)


    def play_one_nudge(self, index: int, tracker) -> None:
        """Move one player up a little, keeping the others where they are.

        The part of the shift that lands below the line is drawn as kept back, because that is
        the whole asymmetry: improving an outcome you were going to discard buys nothing.
        """
        player = PLAYERS[index]
        shifted = self.build_curve(player['value'] + NUDGE, player['colour'])
        arrow = Line([self.to_scene_x(player['value']), BASELINE_Y + 2.25, 0.0],
                     [self.to_scene_x(player['value'] + NUDGE), BASELINE_Y + 2.25, 0.0],
                     color=YELLOW, stroke_width=4)

        kept = self.share_above_replacement(player['value'])
        readout = Text(f"{player['name']}:  worth {kept:.0%} of the improvement",
                       font_size=26, color=YELLOW)
        readout.move_to([0.0, BASELINE_Y - 1.05, 0.0])

        self.play(Transform(self.curves[index], shifted),
                  Create(arrow), run_time=1.1)
        self.play(Write(readout), run_time=0.9)

        # Held for the rest of the line rather than reverted straight away. The sentence that
        # names the number -- "about fifty cents" -- arrives near the END of the line, and at
        # the animation's own pace the shifted curve and its readout were long gone by then,
        # leaving the claim spoken over a static picture of the unshifted player.
        self.wait(max(0.5, tracker.get_remaining_duration() - REVERT_SECONDS))
        self.play(Transform(self.curves[index], self.build_curve(player['value'],
                                                                 player['colour'])),
                  FadeOut(arrow), FadeOut(readout), run_time=REVERT_SECONDS)

    def share_above_replacement(self, mean: float, spread: float = NOISE_SPREAD) -> float:
        """Phi(mu/sigma) -- and, exactly, the derivative of the SAVOR value at that mean."""
        from math import erf, sqrt
        return 0.5 * (1.0 + erf((mean - REPLACEMENT) / (spread * sqrt(2.0))))

    # -- Act three: the calculation itself --------------------------------------------

    def play_the_calculation(self) -> None:
        """The adjustment as it is actually applied: project, translate, scale back up.

        The middle column is the SAVOR value -- what a player is worth once the floor and the
        flyer are taken into account -- and it is SMALLER than the projection for everyone,
        because every player loses the outcomes below replacement. Summed, the board no longer
        adds up to the money in the room, so the whole column is scaled back up until it does.

        That last step is what turns an across-the-board haircut into a redistribution. The
        scaling is uniform, but the haircut was not, so the players who lost least to the floor
        come out ahead and the ones who lost most come out behind.
        """
        with self.voiceover(text=NARRATION['the_general_rule']) as tracker:
            self.play(FadeOut(VGroup(self.curves, self.names, self.lost,
                                     self.replacement, self.axis)),
                      run_time=0.7)

            table = self.build_value_table()
            self.play(FadeIn(table['frame']), run_time=0.8)
            self.play(FadeIn(table['projected']), run_time=0.9)

            wait_until_phrase(self, tracker, 'relative to flyers')
            self.play(FadeIn(table['raw']), run_time=1.0)
            self.play(FadeIn(table['raw_total']), run_time=0.6)

            wait_until_phrase(self, tracker, 'scales the values back up')
            self.play(FadeIn(table['scaling']), run_time=0.8)
            self.play(FadeIn(table['final']), FadeIn(table['final_total']), run_time=1.0)
            self.table = table
            self.wait(0.8)

        with self.voiceover(text=NARRATION['concentration']) as tracker:
            wait_until_phrase(self, tracker, 'concentrates value')
            self.play(FadeIn(table['change']), run_time=1.0)
            self.wait(2.0)
        self.wait(0.6)

    def build_value_table(self) -> dict:
        """Projected dollars, their SAVOR values, the scale-up, and what each player ends on."""
        # Every column in TOTAL auction dollars. The adjustment is defined on value above
        # replacement, so that is what gets transformed and scaled -- but the replacement dollar
        # is added back before anything is printed. Shown without it, the middle column sat on a
        # different baseline from its neighbours and the last row read "$1 becomes $0 becomes
        # $1", which is not a thing that happens to a player.
        above = [value - REPLACEMENT_DOLLARS for value in PROJECTED_DOLLARS]
        exact_raw = [self.savor_value(margin, S_SIGMA) for margin in above]
        scaling = sum(above) / sum(exact_raw)

        # Rounded to what is actually PRINTED, and every total summed from those same rounded
        # numbers. Printing whole dollars against changes carrying a decimal made the table
        # visibly wrong: $33 went to $33.5, which showed as "$33, change +0.5", and the final
        # column's rows added to $205 under a total reading $206. The underlying arithmetic was
        # right -- the changes cancel exactly and both totals are 206.0 -- so the fix is to show
        # a decimal everywhere and to add up what is on screen rather than what is behind it.
        raw = [round(value + REPLACEMENT_DOLLARS, 1) for value in exact_raw]
        final = [round(value * scaling + REPLACEMENT_DOLLARS, 1) for value in exact_raw]
        change = [round(end - start, 1) for end, start in zip(final, PROJECTED_DOLLARS)]

        def column(values, x, colour, money=True):
            entries = VGroup()
            for row, value in enumerate(values):
                # A value that rounds to zero is written as zero: the replacement player's SAVOR
                # value is a hair below it and came out as "$-0".
                shown = 0.0 if abs(value) < 0.05 else value
                # Gains and losses are the whole point of the last column, so they are not one
                # colour: money moves from the bottom of the board to the top.
                tint = colour if money else (GREEN_C if shown > 0 else GREY_B)
                entries.add(
                    Text(f'${shown:.1f}' if money else f'{shown:+.1f}',
                         font_size=24, color=tint)
                    .move_to([x, TABLE_TOP_Y - row * TABLE_ROW_GAP, 0.0]))
            return entries

        headings = VGroup(
            Text('projected', font_size=22, color=GREY_B).move_to([-3.4, TABLE_TOP_Y + 0.6, 0]),
            Text('SAVOR value', font_size=22, color=GREY_B).move_to([-0.6, TABLE_TOP_Y + 0.6, 0]),
            Text('scaled back up', font_size=22, color=GREY_B).move_to([2.4, TABLE_TOP_Y + 0.6, 0]),
        )
        rule = Line([-4.6, TABLE_TOP_Y + 0.35, 0.0], [4.9, TABLE_TOP_Y + 0.35, 0.0],
                    color=GREY_D, stroke_width=2)
        bottom = TABLE_TOP_Y - len(PROJECTED_DOLLARS) * TABLE_ROW_GAP
        return {
            'frame': VGroup(headings, rule),
            'projected': VGroup(
                *[Text(f'${value:.0f}', font_size=24, color=WHITE)
                  .move_to([-3.4, TABLE_TOP_Y - row * TABLE_ROW_GAP, 0.0])
                  for row, value in enumerate(PROJECTED_DOLLARS)],
                # The budget is shown, not just asserted, so the column the scaling restores
                # can be read against the one it started from.
                Line([-4.4, bottom + 0.22, 0.0], [-2.4, bottom + 0.22, 0.0],
                     color=GREY_D, stroke_width=2),
                Text(f'${sum(PROJECTED_DOLLARS):.1f}', font_size=24, color=WHITE)
                .move_to([-3.4, bottom - 0.08, 0.0]),
                Text('the budget', font_size=19, color=GREY_D)
                .move_to([-3.4, bottom - 0.45, 0.0]),
            ),
            'raw': column(raw, -0.6, GREY_B),
            'raw_total': VGroup(
                Line([-1.6, bottom + 0.22, 0.0], [0.4, bottom + 0.22, 0.0],
                     color=GREY_D, stroke_width=2),
                Text(f'${sum(raw):.1f}', font_size=24, color=GREY_B)
                .move_to([-0.6, bottom - 0.08, 0.0]),
                Text('short of the budget', font_size=19, color=GREY_D)
                .move_to([-0.6, bottom - 0.45, 0.0]),
            ),
            'scaling': Text(f'x {scaling:.2f}', font_size=28, color=YELLOW)
                       .move_to([1.0, bottom - 0.08, 0.0]),
            'final': column(final, 2.4, YELLOW),
            'final_total': VGroup(
                Line([1.4, bottom + 0.22, 0.0], [3.4, bottom + 0.22, 0.0],
                     color=GREY_D, stroke_width=2),
                Text(f'${sum(final):.1f}', font_size=24, color=YELLOW)
                .move_to([2.4, bottom - 0.08, 0.0]),
                Text('back to the budget', font_size=19, color=GREY_D)
                .move_to([2.4, bottom - 0.45, 0.0]),
            ),
            'change': column(change, 4.5, GREEN_C, money=False),
        }
