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
half-normal for the flyer. The scene's job is to make the asymmetry obvious -- the cut
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
    Group, VGroup, Line, Polygon, DashedLine, Circle, Text, MathTex, ImageMobject,
    FadeIn, FadeOut, Create, Write, Transform,
    DOWN, UP, LEFT, RIGHT,
    BLUE_D, BLUE_B, RED_B, GREEN_C, GREY_B, GREY_D, GREY_E, YELLOW, WHITE, BLACK,
)
from manim_voiceover import VoiceoverScene

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.narration_voice import NarrationVoice         # noqa: E402
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
#
# The star IS the player the scene opens on: 2.2 sigma above replacement, which S_SIGMA prices
# at forty dollars. That position is fixed by the narration rather than chosen -- Phi(2.2) is
# 0.986, the 'about ninety nine cents' the line quotes -- so the dollar scale was moved to meet
# it rather than the curve being moved to meet the dollars.
# All three names sit at ONE height, so they read as a row of three players rather than as
# three captions each stuck to its own curve at whatever height happened to be free. Only the
# horizontal shifts differ, and only enough to keep each name off its neighbours and off the
# replacement line the middle of the picture is built around.
PLAYER_LABEL_HEIGHT = 2.45
# The nudge arrow rides just over the peak of the curve it is moving, which is what says WHICH
# curve is moving. Level with the names it read as an underline on whichever name it landed
# nearest -- and that was never the player being nudged.
NUDGE_ARROW_HEIGHT = 2.05
PLAYERS = (
    {'name': 'star',        'value': 2.20, 'colour': BLUE_B, 'label_shift':  0.00},
    # Shifted to the right of the replacement line: centred on its own mean, the word ran
    # across the dashed line itself.
    {'name': 'starter',     'value': 0.55, 'colour': GREEN_C, 'label_shift':  0.62},
    {'name': 'flyer',       'value': 0.00, 'colour': GREY_B, 'label_shift': -1.95},
)
# How far a mean is pushed to ask what the improvement is worth. Small on purpose: the claim is
# about the MARGINAL unit, and a large shift would be answering a different question.
NUDGE = 0.45
# How far the starter slides when the line says they could dip. Enough to take most of
# what they had above the line, which is the point being made.
FLOOR_DIP = 0.45

# The opening: a real player, the season nobody can call yet, one projection with a question
# mark against it, and the values that season could actually return scattered around it.
#
# A real player rather than the words 'a player'. The act opened on an abstraction and a bare
# number, when the thing a viewer already knows how to picture is a person and a stat line.
_HEADSHOT_DIR = Path(__file__).resolve().parent.parent.parent / 'prepared_assets' / 'headshots'
OPENING_PLAYER_ID = 1626164
OPENING_PLAYER_NAME = 'Devin Booker'
# What nobody knows in September. Written as questions because that is what they are.
OPENING_STAT_LINE = ('27 points?', '7 assists?', '3 turnovers?')
PORTRAIT_HEIGHT = 2.1
PORTRAIT_CENTRE = [-4.3, 0.55, 0.0]
# Where the stat line, then the projection, then the scatter all sit -- one place, so each
# replaces the last rather than the frame rearranging itself three times.
VALUE_CENTRE = [1.6, 0.55, 0.0]

# What a star actually goes for in a normal auction.
GUESS_DOLLARS = 40
# Offset in dollars, then where it sits relative to VALUE_CENTRE. Spread to match: a forty
# dollar projection that could return twenty-four or fifty-seven is the point being made, and
# the same dollar either way would not be. Held well clear of the projection in the middle --
# at half these distances the values crowded the text they are meant to be scattering around.
POSSIBLE_OUTCOMES = (
    (-16, -3.3,  1.45), (-9, -2.2, -1.40), (-4, -0.9,  1.35),
    (   4,  1.0, -1.35), (10,  2.2,  1.40), (17,  3.4, -1.45),
)

# The auction board the last act works on. Dollar values a drafter would recognise, against a
# replacement player worth nothing.
# A $100 pot, every figure being value ABOVE REPLACEMENT -- which is what the
# adjustment is defined on and what the axis of the curve act is already ticked in. Replacement
# level is zero by definition: it is what a freely available player is worth. A one dollar
# minimum bid is a different idea altogether, and mixing the two is what made the bottom of this
# table read as a player going to nothing and back.
#
# A hundred rather than two hundred, so nobody reads these as one team's budget: the adjustment
# redistributes across the whole pot, not within a single roster.
# A hundred and fifty rather than a hundred, which is what having more than one winner costs.
# A forty dollar star inside a hundred dollar pot is forty percent of everything, and a pot that
# top-heavy has a high average SAVOR share, which means a small scale-up factor, which pushes
# the break-even point UP -- to twenty dollars, above every other player in it. So exactly one
# player gained and the other nine all gave back, which reads as a rule about the best player
# rather than about the top of the auction. With room for a real auction curve the break-even
# falls where it should, four players sit in the eighteen-to-forty band, and three of them gain.
#
# Still not a number anyone reads as a team budget, which is what mattered about avoiding 200.
#
# Chosen so the scaled column adds to exactly $150.0 as PRINTED. The adjustment conserves the
# pot to the last cent, but the table shows one decimal place, and a total that rounds to $149.9
# under a caption reading 'back to the pot' looks like an arithmetic error rather than a
# rounding one.
PROJECTED_DOLLARS = (40, 28, 23, 18, 14, 10, 7, 5, 3, 2)
# A sigma is worth eighteen dollars, which is what puts the star at forty -- thirty was low for
# the best player in a pot. The curves themselves are drawn in SIGMA and do not move: the star
# sits 2.2 sigma above replacement either way, because that is what makes the line's 'about
# ninety nine cents' true (Phi(2.2) = 98.6%). All this constant does is say what a sigma is
# worth, and it is the only place the scene turns that geometry into money -- so the opening's
# forty dollar player, the star curve and the top of the pot are now one number rather than
# three unrelated ones. Nothing outside this scene reads it.
S_SIGMA = 40.0 / 2.20
# Ticks in dollars above replacement. The axis reaches about ninety-eight dollars either side,
# so forty-dollar steps span it without running past its end.
AXIS_DOLLAR_TICKS = (-40, 0, 40, 80)
TABLE_TOP_Y = 2.55
TABLE_ROW_GAP = 0.46
# How long the nudged curve takes to settle back, once its line has finished.
REVERT_SECONDS = 0.8
# Where the nudge's answer is written: clear above the replacement line's own label, in the
# band of the frame the curve act otherwise leaves empty.
READOUT_Y = 2.6

# Kept small enough that a curve centred on the star still has both tails inside the axis;
# wider and the right tail is cut off flat at the end of the line.
SCENE_UNITS_PER_VALUE = 1.15


class Savor(VoiceoverScene):
    """A projection with a floor under it, and the free player it has to beat."""

    def construct(self) -> None:
        self.set_speech_service(NarrationVoice())
        self.play_three_players()
        self.play_the_nudge()
        self.play_the_calculation()

    # ── Shared apparatus ──────────────────────────────────────────────────────────────

    def to_scene_x(self, value: float) -> float:
        return value * SCENE_UNITS_PER_VALUE

    def build_opening_portrait(self) -> dict:
        """One real player to ask the question about, and the veil that reveals them.

        Revealed by lifting a disc rather than by fading the portrait itself. These are circular
        PNGs with transparent corners, and Manim's set_opacity replaces per-pixel alpha with a
        single value, so fading one directly squares it off into a grey block. A disc is a
        vector object and dims exactly as expected -- the same trick the z-score pool uses.
        """
        portrait = ImageMobject(str(_HEADSHOT_DIR / f'{OPENING_PLAYER_ID}.png'))
        portrait.height = PORTRAIT_HEIGHT
        portrait.move_to(PORTRAIT_CENTRE)
        # NBA headshots are cut out against a dark background, so on this scene's black ground
        # an unringed portrait has no edge at all.
        backing = Circle(radius=PORTRAIT_HEIGHT / 2, fill_color=GREY_E, fill_opacity=1.0,
                         stroke_color=GREY_B, stroke_width=2.5).move_to(PORTRAIT_CENTRE)
        name = Text(OPENING_PLAYER_NAME, font_size=27, color=WHITE)
        name.move_to([PORTRAIT_CENTRE[0], PORTRAIT_CENTRE[1] - PORTRAIT_HEIGHT / 2 - 0.45, 0.0])
        # Wider than the backing, so the ring does not peek out from under the veil.
        veil = Circle(radius=PORTRAIT_HEIGHT / 2 + 0.06, stroke_width=0,
                      fill_color=BLACK, fill_opacity=1.0).move_to(PORTRAIT_CENTRE)
        return {'chip': Group(backing, portrait), 'name': name, 'veil': veil}

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

    def build_kept_head(self, centre: float, colour) -> Polygon:
        """The part of the distribution above replacement -- the only part worth anything."""
        cut = self.to_scene_x(REPLACEMENT)
        xs, ys = self.normal_points(centre, NOISE_SPREAD, cut, AXIS_HALF_WIDTH)
        points = ([[cut, BASELINE_Y, 0.0]]
                  + [[x, BASELINE_Y + y, 0.0] for x, y in zip(xs, ys)]
                  + [[xs[-1], BASELINE_Y, 0.0]])
        return Polygon(*points, color=colour, fill_color=colour,
                       fill_opacity=0.55, stroke_width=0)

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

        E[max(mu + noise, 0)] - E[max(noise, 0)], the second being the flyer. Expanding
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
            # A real player to hang the question on, and the season nobody can call yet written
            # as the questions they are. The act used to open on the words 'a player' and a bare
            # number, which is an abstraction standing where the viewer already has a picture.
            opening = self.build_opening_portrait()
            self.add(opening['chip'], opening['veil'])
            self.play(FadeOut(opening['veil']), FadeIn(opening['name']), run_time=0.9)

            wait_until_phrase(self, tracker, 'we do not know exactly')
            questions = VGroup(*[
                Text(line, font_size=32, color=GREY_B) for line in OPENING_STAT_LINE
            ]).arrange(DOWN, buff=0.55, aligned_edge=LEFT).move_to(VALUE_CENTRE)
            self.play(FadeIn(questions, lag_ratio=0.5), run_time=2.2)

            # The stat line and the dollar value occupy the same place, so the second replaces
            # the first rather than the frame rearranging itself around both.
            wait_until_phrase(self, tracker, 'some level of value')
            projection = Text(f'${GUESS_DOLLARS}?', font_size=54, color=YELLOW)
            projection.move_to(VALUE_CENTRE)
            self.play(FadeOut(questions), FadeIn(projection), run_time=0.9)

            wait_until_phrase(self, tracker, 'well above or below')
            # Each one carries a question mark of its own. Nothing on screen has established a
            # distribution yet -- that is what the next beat is for -- so at this moment every
            # value really is still a question, not a reading off a curve.
            outcomes = VGroup(*[
                Text(f'${GUESS_DOLLARS + offset}?', font_size=30, color=GREY_B)
                .move_to([VALUE_CENTRE[0] + shift, VALUE_CENTRE[1] + rise, 0.0])
                for offset, shift, rise in POSSIBLE_OUTCOMES
            ])
            self.play(FadeIn(outcomes, lag_ratio=0.12), run_time=1.6)
            self.wait(0.5)

            wait_until_phrase(self, tracker, 'Normally distributed')
            self.axis = self.build_axis()
            # Veiled on the way out for the same reason it was veiled on the way in.
            self.play(FadeIn(opening['veil']),
                      FadeOut(VGroup(opening['name'], projection, outcomes)), run_time=0.6)
            self.remove(opening['chip'], opening['veil'])
            self.play(Create(self.axis), run_time=0.8)

            self.curves, self.names = VGroup(), VGroup()
            for entry in PLAYERS:
                self.curves.add(self.build_curve(entry['value'], entry['colour']))
                label = Text(entry['name'], font_size=22, color=entry['colour'])
                label.move_to([self.to_scene_x(entry['value']) + entry['label_shift'],
                               BASELINE_Y + PLAYER_LABEL_HEIGHT, 0.0])
                self.names.add(label)

            # On the words themselves: the curves when the line says their values ARE Normal
            # distributions, and the names when it says here are some. Held back to 'here are
            # some potential distributions' instead, the axis sat alone for nine seconds while
            # the sentence that describes the curves went by.
            wait_until_phrase(self, tracker, 'are Normal distributions')
            self.play(FadeIn(self.curves), run_time=1.3)

            wait_until_phrase(self, tracker, 'Here are some potential distributions')
            self.play(FadeIn(self.names), run_time=0.9)
            self.wait(0.6)

        with self.voiceover(text=NARRATION['the_floor']) as tracker:
            # Drawn on 'if we draw a line', not on the line's first mention of the replacement
            # level a clause earlier: the sentence says what is about to happen and the drawing
            # should be what happens.
            wait_until_phrase(self, tracker, 'if we draw a line')
            self.replacement = self.build_replacement_line()
            self.play(Create(self.replacement), run_time=1.0)
            wait_until_phrase(self, tracker, 'to the left of the line do not actually help')
            self.lost = VGroup(*[self.build_lost_tail(player['value']) for player in PLAYERS])
            self.play(FadeIn(self.lost), run_time=1.2)

            # The last clause of this line is about one player in particular -- the one just
            # above the line, banking on a good outcome -- and it used to be spoken over a
            # picture that had stopped moving. The dip is that sentence, drawn.
            starter = PLAYERS[1]
            wait_until_phrase(self, tracker, 'banking on their positive outcome')
            self.kept = self.build_kept_head(starter['value'], starter['colour'])
            self.play(FadeIn(self.kept), run_time=0.8)

            wait_until_phrase(self, tracker, 'dip a little')
            dipped = starter['value'] - FLOOR_DIP
            self.play(Transform(self.curves[1], self.build_curve(dipped, starter['colour'])),
                      Transform(self.lost[1], self.build_lost_tail(dipped)),
                      Transform(self.kept, self.build_kept_head(dipped, starter['colour'])),
                      run_time=1.2)
            self.wait(0.4)

            # Put back before the next act, which nudges these same curves from where they
            # started and would otherwise begin with a jump.
            self.play(Transform(self.curves[1],
                                self.build_curve(starter['value'], starter['colour'])),
                      Transform(self.lost[1], self.build_lost_tail(starter['value'])),
                      FadeOut(self.kept), run_time=0.8)

    # -- Act two: what one more unit of projection buys -------------------------------

    def play_the_nudge(self) -> None:
        """Shift each mean by the same amount and show how much of it survives the floor.

        This is the derivative of the SAVOR value, and it is exactly the share of the player
        above replacement -- verified against the closed form: 50.0% at the line, 70.9% for the
        starter, 98.6% for the star. So "half their distribution" is not a loose way of speaking;
        it is the number.
        """
        # Two anchors per nudge. The first is the clause that asks for the shift; the second
        # is the clause that states what it was worth. Both used to hang off the first alone,
        # so the answer appeared within two seconds of the dollar being added -- a third of the
        # way into the line, while the sentence was still setting the question up.
        for key, index, shift_anchor, answer_anchor in (
            ('the_flyer_half', 2, 'Imagine starting from', 'so we only get to capture'),
            ('the_star_whole', 0, 'increasing the value of a star',
             'A dollar of projection for a star'),
        ):
            with self.voiceover(text=NARRATION[key]) as tracker:
                wait_until_phrase(self, tracker, shift_anchor)
                self.play_one_nudge(index, tracker, answer_anchor)


    def play_one_nudge(self, index: int, tracker, answer_anchor: str) -> None:
        """Move one player up a little, keeping the others where they are.

        The part of the shift that lands below the line is drawn as kept back, because that is
        the whole asymmetry: improving an outcome you were going to discard buys nothing.
        """
        player = PLAYERS[index]
        shifted = self.build_curve(player['value'] + NUDGE, player['colour'])
        arrow = Line([self.to_scene_x(player['value']), BASELINE_Y + NUDGE_ARROW_HEIGHT, 0.0],
                     [self.to_scene_x(player['value'] + NUDGE),
                      BASELINE_Y + NUDGE_ARROW_HEIGHT, 0.0],
                     color=YELLOW, stroke_width=4)

        kept = self.share_above_replacement(player['value'])
        # Above the curves, not below them. Under the axis it landed on top of the axis's own
        # caption -- both lines sat within a tenth of a unit of each other and overlapped on
        # screen -- and the top of the frame was empty the whole act.
        readout = Text(f"{player['name']}:  worth {kept:.0%} of the improvement",
                       font_size=26, color=YELLOW)
        readout.move_to([0.0, READOUT_Y, 0.0])

        self.play(Transform(self.curves[index], shifted),
                  Create(arrow), run_time=1.1)

        # The shifted curve holds on its own until the line says what the shift bought.
        wait_until_phrase(self, tracker, answer_anchor)
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
        # All three columns on one baseline: value above replacement. There is no baseline to
        # add back, because replacement is zero -- so a player projected at replacement is worth
        # nothing above a free one, in every column, which is the whole point of the flyer.
        exact_raw = [self.savor_value(value, S_SIGMA) for value in PROJECTED_DOLLARS]
        scaling = sum(PROJECTED_DOLLARS) / sum(exact_raw)

        # Rounded to what is actually PRINTED, and every total summed from those same rounded
        # numbers. Printing whole dollars against changes carrying a decimal made the table
        # visibly wrong: $33 went to $33.5, which showed as "$33, change +0.5", and the final
        # column's rows added to $205 under a total reading $206. The underlying arithmetic was
        # right -- the changes cancel exactly and both totals are 206.0 -- so the fix is to show
        # a decimal everywhere and to add up what is on screen rather than what is behind it.
        raw = [round(value, 1) for value in exact_raw]
        final = [round(value * scaling, 1) for value in exact_raw]
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
                Text('the pot', font_size=19, color=GREY_D)
                .move_to([-3.4, bottom - 0.45, 0.0]),
            ),
            'raw': column(raw, -0.6, GREY_B),
            'raw_total': VGroup(
                Line([-1.6, bottom + 0.22, 0.0], [0.4, bottom + 0.22, 0.0],
                     color=GREY_D, stroke_width=2),
                Text(f'${sum(raw):.1f}', font_size=24, color=GREY_B)
                .move_to([-0.6, bottom - 0.08, 0.0]),
                Text('short of the pot', font_size=19, color=GREY_D)
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
                Text('back to the pot', font_size=19, color=GREY_D)
                .move_to([2.4, bottom - 0.45, 0.0]),
            ),
            'change': column(change, 4.5, GREEN_C, money=False),
        }
