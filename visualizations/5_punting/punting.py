"""Why giving up on categories wins more of them.

Nine categories, nine units of effort, and an opponent who spreads theirs evenly. Put one unit
into each and every category is a coin flip. Then take the effort out of one category and spread
it over the rest, and the expected haul goes UP -- the ground lost in a category you have
abandoned is cheaper than the ground gained in one you are contesting.

The optimum is to abandon three outright and split the budget six ways, worth 4.625 categories
against the 4.500 that perfect balance gets. That is not a figure chosen to make a point: an
optimiser handed all nine weights and told only to keep them non-negative and summing to nine
finds exactly [0, 0, 0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5].

Reading a panel: the bell never moves -- it is the distribution of how the category comes out,
and it is the same distribution whatever you do. What moves is the threshold. You win the
category when the result lands to the LEFT of the bar, so pushing the bar right is buying win
probability. Yellow is the half you get for showing up, green is what the effort bought on top
of it, red is what abandoning the category gave back.

The scene carries no words; see `visualizations/narration_notes.md` for what it is saying.

    manim -ql visualizations/5_punting/punting.py PuntingSearch
    manim -qh visualizations/5_punting/punting.py PuntingSearch
"""

from __future__ import annotations

from math import erf

import numpy as np
from manim import (
    Scene, VGroup, Line, Polygon, Text, DecimalNumber, ValueTracker,
    Create, FadeIn, always_redraw,
    DOWN, RIGHT,
    YELLOW, WHITE, GREY_B, GREEN_B, RED_C,
)
from manim_voiceover import VoiceoverScene

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.narration_timing import seconds_until_phrase, wait_until_phrase   # noqa: E402
from shared.narration_voice import NarrationVoice   # noqa: E402
from narration import NARRATION


CATEGORY_NAMES = [
    'Field Goal %', 'Free Throw %', 'Threes',
    'Points', 'Rebounds', 'Assists',
    'Steals', 'Blocks', 'Turnovers',
]
CATEGORY_COUNT = 9
EFFORT_BUDGET = 9.0
OPPONENT_EFFORT = 1.0     # what the opponent puts into every category, so parity is a coin flip

# The opening, before a word is said: the panels name themselves, then the curves draw
# themselves, then the frame settles. Silence that buys a picture arriving at a readable pace.
TITLES_FADE_SECONDS = 0.7
CURVES_DRAW_SECONDS = 2.6
SHADING_FADE_SECONDS = 1.1     # the yellow regions arriving, all nine together

# The order categories are abandoned in. Free Throw % first is not arbitrary -- it is the punt
# the algorithm reaches for most often, and the one fantasy players will recognise.
ABANDON_ORDER = [1, 7, 2, 0]

# How long a punt is left on screen after the slider stops, before the next line starts.
SETTLE_AFTER_A_PUNT = 1.0

# One panel of the three-by-three grid.
PANEL_WIDTH, PANEL_HEIGHT = 4.15, 1.95
PANEL_ORIGIN_Y = 1.75
CURVE_HALF_WIDTH = 1.72   # horizontal half-extent of a bell inside its panel
CURVE_HEIGHT = 0.95       # height of a bell at its peak
CURVE_SPAN = 2.8          # how many standard deviations either side of centre are drawn
_SHADE_RESOLUTION = 120   # points along a shaded region's curved edge


def weights_at(progress: float, reallocated: float) -> np.ndarray:
    """The nine efforts once `progress` categories have been drained.

    Draining is continuous and one at a time: the category currently being emptied gives its
    effort up smoothly, and whatever it releases is shared equally among the categories still
    being contested. Equal shares are not a simplification -- for identical categories the
    optimum genuinely splits the contested budget evenly, because at a maximum every contested
    category must have the same marginal value, and a Normal density hits a given value at only
    one point on a given side of the threshold.

    `reallocated` is how much of the freed effort has actually been handed over, none of it
    through all of it. A punt is two things at once -- giving a category up costs you, and
    spending what it freed pays you back -- and at one, which is what every punt but the first
    uses, they happen together and this is the plain redistribution above. Held at zero, the
    contested categories stay exactly where they were and the freed effort sits unspent, which
    is the half of the trade the first punt stops to show on its own.
    """
    weights = np.zeros(CATEGORY_COUNT)
    abandoned = int(progress)
    fraction = progress - abandoned

    emptied = list(ABANDON_ORDER[:abandoned])
    # Only a category part-way through being given up is draining. At a whole number of punts
    # nothing is mid-drain, and the next category on the list is still an ordinary contested one
    # -- which is what keeps the weights continuous across the moment a punt completes.
    draining = None
    if fraction > 0.0 and abandoned < len(ABANDON_ORDER):
        draining = ABANDON_ORDER[abandoned]

    contested = [index for index in range(CATEGORY_COUNT)
                 if index not in emptied and index != draining]

    if draining is not None:
        # It falls from the level it stood at when this punt started, not from parity: by the
        # third punt the categories still being contested are each carrying more than a ninth.
        start_level = EFFORT_BUDGET / (CATEGORY_COUNT - len(emptied))
        weights[draining] = start_level * (1.0 - fraction)

    parity = EFFORT_BUDGET / CATEGORY_COUNT
    shared = (EFFORT_BUDGET - weights.sum()) / len(contested)
    weights[contested] = parity + reallocated * (shared - parity)
    return weights


def win_probability(weight: float) -> float:
    """The chance of taking a category, given the effort put into it against an even opponent."""
    return 0.5 * (1.0 + erf((weight - OPPONENT_EFFORT) / np.sqrt(2.0)))


def categories_won(weights: np.ndarray) -> float:
    return float(sum(win_probability(weight) for weight in weights))


class PuntingSearch(VoiceoverScene):
    """Walk the punt one category at a time and watch the expected haul rise, then fall."""

    def setup(self) -> None:
        self.progress = ValueTracker(0.0)
        # How much of the freed effort has been handed to the categories still being contested.
        # Starts at none of it, which is what the first punt walks up from -- and which changes
        # nothing before then, since at parity there is no freed effort to hand over. Every punt
        # after the first leaves it at one, where draining and reallocating happen together.
        self.reallocation = ValueTracker(0.0)

        # The best-so-far readout ratchets, and it keeps a running maximum of what the scene has
        # actually reached rather than a table sampled over `progress`. A table cannot describe
        # this path any more: the first punt moves through states that share a `progress` with
        # states of a quite different score, and reading a best-so-far off progress alone would
        # post a number during the drain that had not been reached yet.
        self._best_seen = 0.0

    # ── Geometry ──────────────────────────────────────────────────────────────────────

    def _panel_centre(self, category_index: int) -> np.ndarray:
        row, column = divmod(category_index, 3)
        return np.array([(column - 1) * PANEL_WIDTH,
                         PANEL_ORIGIN_Y - row * PANEL_HEIGHT,
                         0.0])

    def _point_on_curve(self, category_index: int, offset: float) -> np.ndarray:
        """A point on the bell, `offset` standard deviations from its centre."""
        return self._panel_centre(category_index) + np.array([
            offset * (CURVE_HALF_WIDTH / CURVE_SPAN),
            CURVE_HEIGHT * np.exp(-0.5 * offset ** 2),
            0.0,
        ])

    def _threshold_offset(self, weight: float) -> float:
        """Where the bar sits, in standard deviations: the effort's edge over the opponent.

        This is the one number the whole scene moves. At parity it is zero -- the bar stands on
        the mean and the category is a coin flip -- and every unit of effort pushes it a standard
        deviation to the right, which is exactly the win probability the score is counting.
        """
        return float(np.clip(weight - OPPONENT_EFFORT, -CURVE_SPAN, CURVE_SPAN))

    def _shaded_region(self, category_index: int, from_offset: float, to_offset: float, colour):
        """The area under the bell between two offsets, as a filled polygon."""
        if to_offset - from_offset < 1e-4:
            return None
        centre = self._panel_centre(category_index)
        top_edge = [self._point_on_curve(category_index, offset)
                    for offset in np.linspace(from_offset, to_offset, _SHADE_RESOLUTION)]
        base = [np.array([top_edge[-1][0], centre[1], 0.0]),
                np.array([top_edge[0][0], centre[1], 0.0])]
        return Polygon(*top_edge, *base, stroke_width=0, fill_color=colour, fill_opacity=0.6)

    # ── One category ──────────────────────────────────────────────────────────────────

    def current_weights(self) -> np.ndarray:
        """Every category's effort right now, at the current point of the drain and the trade."""
        return weights_at(self.progress.get_value(), self.reallocation.get_value())

    def current_weight(self, category_index: int) -> float:
        """The effort in one category right now.

        Read through a method rather than off `weights_at` directly so a scene that moves a
        single category on its own -- CategoryGradient does -- can override where one weight
        comes from without touching the drawing code or the drain path.
        """
        return float(self.current_weights()[category_index])

    def _build_shading(self, category_index: int) -> VGroup:
        """The three regions that say what this category is worth, and the bar dividing them.

        Yellow is the half of the distribution you win at parity -- what showing up is worth.
        Green is territory beyond the mean that extra effort has bought. Red is territory short
        of the mean that abandoning the category has given back. Only ever two of the three are
        present at once, because the bar is either right of the mean or left of it.
        """
        threshold = self._threshold_offset(self.current_weight(category_index))

        regions = VGroup()
        # The part of the win region that was there before any effort was spent.
        baseline_edge = min(threshold, 0.0)
        for region in (
            self._shaded_region(category_index, -CURVE_SPAN, baseline_edge, YELLOW),
            self._shaded_region(category_index, 0.0, threshold, GREEN_B) if threshold > 0 else None,
            self._shaded_region(category_index, threshold, 0.0, RED_C) if threshold < 0 else None,
        ):
            if region is not None:
                regions.add(region)

        centre = self._panel_centre(category_index)
        bar_x = centre[0] + threshold * (CURVE_HALF_WIDTH / CURVE_SPAN)
        regions.add(Line([bar_x, centre[1] - 0.08, 0.0],
                         [bar_x, centre[1] + CURVE_HEIGHT * 1.14, 0.0],
                         color=WHITE, stroke_width=3))
        return regions

    def _build_bell(self, category_index: int) -> VGroup:
        """The curve itself, which never moves -- drawn once, not redrawn per frame."""
        points = [self._point_on_curve(category_index, offset)
                  for offset in np.linspace(-CURVE_SPAN, CURVE_SPAN, 140)]
        return VGroup(*[
            Line(start, end, color=WHITE, stroke_width=2.5)
            for start, end in zip(points, points[1:])
        ])

    # ── Readout ───────────────────────────────────────────────────────────────────────

    def _build_score_readout(self) -> VGroup:
        current = categories_won(self.current_weights())
        # Ratcheted as the scene plays. Frames are rendered forwards in time, so a maximum kept
        # here is the maximum over everything actually shown -- which is what "best so far"
        # claims to be, and which a table indexed by progress alone can no longer give.
        self._best_seen = max(self._best_seen, current)
        best = self._best_seen

        # Laid out across rather than stacked: the grid of bells already reaches well down the
        # frame, and a stacked readout leaves no room beneath it.
        current_value = DecimalNumber(current, num_decimal_places=3, font_size=50,
                                      color=YELLOW if current >= best - 1e-9 else GREY_B)
        # What is actually on the table. It sits at the full budget while every category is
        # contested, falls as one is given up, and climbs back as what it freed is handed to the
        # others -- which is the whole of the first punt's argument, in one number. Without it
        # the drain and the payback look like the same kind of move, because both are curves
        # sliding sideways.
        invested = float(self.current_weights().sum())
        readout = VGroup(
            Text('categories won', font_size=19, color=GREY_B),
            current_value,
            Text(f'best so far  {best:.3f}', font_size=19, color=GREY_B),
            Text(f'total investment  {invested:.2f}', font_size=19,
                 color=GREY_B if invested >= EFFORT_BUDGET - 1e-6 else RED_C),
        )
        readout.arrange(RIGHT, buff=0.45)
        readout.move_to([0.0, -2.95, 0.0])
        return readout

    # ── The scene ─────────────────────────────────────────────────────────────────────

    def play_first_punt_in_two_halves(self, tracker) -> None:
        """Give the category up, and only then spend what giving it up freed.

        Every later punt does both at once, which is what a punt IS -- but done that way the
        first time, the two halves cancel on screen before either has been seen. The one curve
        slides right while the other eight slide left, the score barely moves, and the argument
        the line is making ("we lose expected value... meanwhile, we can reallocate") has no
        picture to point at.

        So the first punt waits. The abandoned category drains on its own, and the score falls,
        which is the cost with nothing yet paid back. On "Meanwhile" the other eight take up
        what it left, and the score passes where it started. The readout says the same thing in
        a second way, since it only shows yellow while standing at its best: it goes grey as the
        category is given up and comes back yellow once the trade has been made.
        """
        to_meanwhile = seconds_until_phrase(tracker, 'Meanwhile')
        self.play(self.progress.animate.set_value(1.0),
                  run_time=max(2.4, to_meanwhile - SETTLE_AFTER_A_PUNT))

        wait_until_phrase(self, tracker, 'Meanwhile')
        self.play(self.reallocation.animate.set_value(1.0),
                  run_time=max(2.4, tracker.get_remaining_duration() - SETTLE_AFTER_A_PUNT))

    def construct(self) -> None:
        self.set_speech_service(NarrationVoice())
        titles = VGroup(*[
            Text(name, font_size=16, color=GREY_B).move_to(
                self._panel_centre(index) + np.array([0.0, CURVE_HEIGHT * 1.36, 0.0]))
            for index, name in enumerate(CATEGORY_NAMES)
        ])
        bells = VGroup(*[self._build_bell(index) for index in range(CATEGORY_COUNT)])

        # The shading redraws itself off the weights, and an always_redraw mobject repaints over
        # a fade. So it arrives as a still copy -- which is exactly what the live one would be
        # drawing while nothing is moving yet -- and hands over once it is up.
        arriving_shading = VGroup(*[
            self._build_shading(index) for index in range(CATEGORY_COUNT)
        ])
        live_shading = VGroup(*[
            always_redraw(lambda index=index: self._build_shading(index))
            for index in range(CATEGORY_COUNT)
        ])

        # Drawn, not dropped in. Nine curves appearing in a single frame is a jump cut, and the
        # scene opens on it -- so the panels name themselves and the curves draw themselves in.
        #
        # All nine at once, though, rather than one after another. The panels are nine views of
        # the same fact, equivalent to each other in every way the scene is about; drawing them
        # in sequence puts an order on them that says the ninth follows from the first.
        #
        # All of it happens UNDER the opening line rather than before it. Drawing in silence and
        # then starting to speak cost four seconds at the top of the scene, which is a long time
        # to look at a picture nobody is talking about; and the first sentence is about punting
        # being a consequence of two things, not about the panels, so it does not need them
        # finished to make sense. By the second sentence, which is about the distributions, they
        # are there to be pointed at.

        # The yellow arrives on the clause that names it, finishing just as the words land. It
        # used to be added the instant the curves finished, which put it on screen through two
        # sentences that were about something else, and by the time it was called "the yellow
        # shaded region" it had been sitting there long enough to stop being new.
        with self.voiceover(text=NARRATION['opening']) as tracker:
            self.play(FadeIn(titles), run_time=TITLES_FADE_SECONDS)
            self.play(Create(bells, lag_ratio=0.0), run_time=CURVES_DRAW_SECONDS)
            wait_until_phrase(self, tracker, 'The yellow shaded region',
                              lead_seconds=SHADING_FADE_SECONDS)
            self.play(FadeIn(arriving_shading, lag_ratio=0.0), run_time=SHADING_FADE_SECONDS)
            self.remove(arriving_shading)
            self.add(live_shading)

        with self.voiceover(text=NARRATION['score']):
            self.add(always_redraw(self._build_score_readout))
            self.wait(1.6)

        # Abandon them one at a time. The first three each pay for themselves; the fourth gives
        # ground back, and watching it fail is what makes three the answer rather than a claim.
        # One line apiece, so that the fourth's reversal gets said as it happens.
        abandonment_lines = [NARRATION['abandon'], NARRATION['abandon_again'],
                             NARRATION['abandon_third'], NARRATION['abandon_too_far']]
        if len(abandonment_lines) != len(ABANDON_ORDER):
            raise ValueError(
                f'{len(abandonment_lines)} narration lines for {len(ABANDON_ORDER)} categories '
                f'abandoned: narration.py and ABANDON_ORDER have to agree.')
        for abandoned, line in enumerate(abandonment_lines):
            # The slider moves for as long as the line describing it lasts. These lines talk
            # about the investment being moved WHILE it moves ("as we move it far..."), so a
            # fixed two seconds of motion followed by twenty seconds of still frame would be
            # describing something that had already finished happening.
            with self.voiceover(text=line) as tracker:
                if abandoned == 0:
                    self.play_first_punt_in_two_halves(tracker)
                else:
                    self.play(self.progress.animate.set_value(abandoned + 1.0),
                              run_time=max(2.4, tracker.duration - SETTLE_AFTER_A_PUNT))
                self.wait(SETTLE_AFTER_A_PUNT)

        # Walk back to the best the search found and rest there.
        # Straight back to three as the closing line starts. The fourth punt has just been shown
        # to cost more than it pays, so leaving it standing while the conclusion is spoken would
        # keep the wrong answer on screen through the sentence explaining the right one.
        with self.voiceover(text=NARRATION['settle']):
            self.play(self.progress.animate.set_value(3.0), run_time=1.2)
            self.wait(3.0)
