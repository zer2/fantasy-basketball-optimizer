"""The whole future-pick model, in one plane.

Two real categories, so a player is a point and nothing has to be imagined. Everyone above the
value bar was drafted before this pick came round; the twenty-five below it are what you get to
choose from. A score line slides across until it touches the best survivor left, and that player
is the pick.

Two weightings run side by side throughout, and each keeps one colour for the whole scene:
YELLOW cares equally about both categories, BLUE cares three times as much about Free Throw %.
Handed the same pool they usually want different players -- in the six drafts shown here they
disagree in five.

Run the situation again and the pool is different, so the picks are different. Run it sixty
thousand times and they stop being anecdotes and become a distribution, and the mean of that
distribution is x(w) -- the quantity the model exists to compute. The gap between the two means
is the tilt: (+1.01, -1.28), a strategy buying Free Throw % and paying for it in Blocks.

The scene carries no explanatory text; see `narration_notes.md`. Run
`python visualizations/4_category_weights/prepare_plane_story_data.py` first.

    manim -ql visualizations/4_category_weights/plane_story.py PlaneStory
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from manim import (
    Scene, VGroup, ImageMobject, Circle, Dot, Line, DashedLine, Text, DecimalNumber,
    FadeIn, FadeOut, Create, Transform,
    DOWN, UP, LEFT, RIGHT,
    WHITE, GREY_A, GREY_B, GREY_D,
)
from manim_voiceover import VoiceoverScene

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.narration_timing import wait_until_phrase   # noqa: E402
from shared.narration_voice import NarrationVoice   # noqa: E402
from narration import NARRATION


PLANE_CENTRE = np.array([-2.55, 0.15, 0.0])
UNITS_PER_STANDARD_DEVIATION = 1.02
PLANE_HALF_EXTENT = 3.0          # matches the prep script's grid, so the images line up exactly

READOUT_X = 3.85
PLAYER_DOT_RADIUS = 0.055

# One colour per weighting, held for the whole scene -- lines, picks and density alike. These
# match the hues the prep script bakes into the density images, so a blob and the line that
# produced it are recognisably the same thing.
NEUTRAL_COLOUR = '#FFD650'
ALTERNATIVE_COLOUR = '#6EAAFF'

_DATA_PATH = Path(__file__).resolve().parent.parent / 'prepared_data' / 'plane_story.json'
_IMAGE_DIR = Path(__file__).resolve().parent.parent / 'prepared_assets' / 'plane_story'


def load_plane_story() -> dict:
    if not _DATA_PATH.exists():
        raise FileNotFoundError(
            f'{_DATA_PATH} is missing. Run `python visualizations/4_category_weights/prepare_plane_story_data.py` '
            f'first -- the scene deliberately does no sampling of its own.')
    return json.loads(_DATA_PATH.read_text(encoding='utf-8'))


class PlaneStory(VoiceoverScene):
    """Pool, bar, two score lines, experiments, then the distributions they were sampling."""

    def setup(self) -> None:
        self.prepared = load_plane_story()
        self.experiments = self.prepared['experiments']
        self.value_direction = np.array(self.prepared['value_direction'])
        self.settings = self.prepared['weight_settings']
        self.weightings = [
            ('neutral', np.array(self.settings['neutral']['weights']), NEUTRAL_COLOUR),
            ('alternative', np.array(self.settings['alternative']['weights']), ALTERNATIVE_COLOUR),
        ]

    # ── Plane geometry ────────────────────────────────────────────────────────────────

    def _at(self, coordinates) -> np.ndarray:
        """Frame position of a point given in standard deviations of the two categories."""
        return PLANE_CENTRE + np.array([
            coordinates[0] * UNITS_PER_STANDARD_DEVIATION,
            coordinates[1] * UNITS_PER_STANDARD_DEVIATION,
            0.0,
        ])

    def _line_endpoints(self, normal: np.ndarray, level: float):
        """Where {z : normal . z = level} enters and leaves the plotted square."""
        along = np.array([-normal[1], normal[0]])
        base = normal * level / (normal @ normal)
        inside = [
            base + step * along for step in np.linspace(-8.0, 8.0, 500)
            if max(abs(base + step * along)) <= PLANE_HALF_EXTENT + 1e-9
        ]
        if len(inside) < 2:
            return None
        return self._at(inside[0]), self._at(inside[-1])

    def _score_line(self, normal: np.ndarray, level: float, colour) -> Line | None:
        endpoints = self._line_endpoints(normal, level)
        if endpoints is None:
            return None
        return Line(*endpoints, color=colour, stroke_width=4)

    def _build_axes(self) -> VGroup:
        extent = PLANE_HALF_EXTENT
        axes = VGroup(
            Line(self._at([-extent, 0]), self._at([extent, 0]), color=GREY_D, stroke_width=1.5),
            Line(self._at([0, -extent]), self._at([0, extent]), color=GREY_D, stroke_width=1.5),
        )
        horizontal, vertical = self.prepared['categories']
        axes.add(
            Text(horizontal, font_size=19, color=GREY_B)
            .next_to(self._at([extent, 0]), DOWN, buff=0.18),
            Text(vertical, font_size=19, color=GREY_B)
            .next_to(self._at([0, extent]), UP, buff=0.14),
        )
        return axes

    # ── The right-hand column ─────────────────────────────────────────────────────────

    def _build_weight_readout(self) -> VGroup:
        """Both weightings as coefficients, each in its own colour.

        Shown as numbers rather than as arrows. An arrow would sit at right angles to the line it
        controls, which is true and unhelpful: what a viewer needs is that these two numbers fix
        the line's slope, and the line is what does the choosing.
        """
        columns = VGroup()
        for _, weights, colour in self.weightings:
            rows = VGroup()
            for category, weight in zip(self.prepared['categories'], weights):
                rows.add(VGroup(
                    Text(category, font_size=17, color=GREY_B),
                    DecimalNumber(weight, num_decimal_places=2, font_size=25, color=colour),
                ).arrange(RIGHT, buff=0.26))
            columns.add(rows.arrange(DOWN, buff=0.18, aligned_edge=LEFT))
        columns.arrange(DOWN, buff=0.62, aligned_edge=LEFT)
        columns.move_to([READOUT_X, 2.15, 0.0])
        return columns

    # ── Act one: the pool ─────────────────────────────────────────────────────────────

    def _build_pool(self, experiment_index: int) -> VGroup:
        experiment = self.experiments[experiment_index]
        survivors = VGroup(*[
            Dot(self._at(player), radius=PLAYER_DOT_RADIUS, color=WHITE)
            for player in experiment['players']
        ])
        # Drawn the same as everyone else. These are the players already off the board, but
        # nothing has said so yet when the pool first appears -- greying them from the start
        # answers the question the dividing line is about to ask, and the line then arrives to
        # explain a distinction the picture had already made.
        taken = VGroup(*[
            Dot(self._at(player), radius=PLAYER_DOT_RADIUS, color=WHITE)
            for player in experiment['taken']
        ])
        return VGroup(taken, survivors)

    def introduce_pool(self) -> None:
        """Axes and players, before a word is said about them.

        The opening clause is about what a dot IS, so the dots have to be standing there for it
        to be about anything. Drawn under the line, the first seven words played to an empty
        pair of axes.
        """
        self.axes = self._build_axes()
        self.play(Create(self.axes), run_time=1.0)

        self.pool = self._build_pool(0)
        self.play(FadeIn(self.pool), run_time=1.0)

    def play_act_one_pool(self) -> None:
        """The line that divides the pool, and the players it has already taken out of it.

        Runs on the clause that introduces it rather than at the top of the beat: the dividing
        line is the second thing the sentence says, and arriving early it dimmed half the pool
        while the voice was still describing the whole of it.
        """
        # The bar is dashed and colourless on purpose: the two solid coloured lines coming next
        # are score levels a strategy chose, and this one is a fact about the draft that no
        # strategy gets a say in.
        endpoints = self._line_endpoints(self.value_direction, 0.0)
        self.value_bar = DashedLine(*endpoints, color=GREY_A, stroke_width=4, dash_length=0.16)
        self.play(Create(self.value_bar), run_time=1.0)
        self.play(self.pool[0].animate.set_opacity(0.22), run_time=0.8)
        self.wait(1.1)

    # ── Act two: two ways of scoring the same players ─────────────────────────────────

    def _sweep_to_winners(self, experiment_index: int, steps: int) -> None:
        """Slide both score lines down until each one reaches its own best survivor.

        Sliding, not turning: a weighting's slope is what it MEANS and it never changes, while
        the level is only "how good a player would have to be", sweeping down until somebody
        qualifies. The two lines have different slopes, so they stop on different players.
        """
        experiment = self.experiments[experiment_index]
        targets = [max(experiment['neutral_scores']), max(experiment['alternative_scores'])]

        for fraction in np.linspace(0.0, 1.0, steps):
            replacements = []
            for line, (_, weights, colour), target in zip(self.score_lines, self.weightings, targets):
                level = 2.9 + fraction * (target - 2.9)
                replacement = self._score_line(weights, level, colour)
                if replacement is not None:
                    replacements.append(Transform(line, replacement))
            if replacements:
                self.play(*replacements, run_time=0.05)

    def _mark_picks(self, experiment_index: int) -> None:
        experiment = self.experiments[experiment_index]
        rings = []
        for key, (_, _, colour) in zip(('neutral_pick', 'alternative_pick'), self.weightings):
            winner = experiment['players'][experiment[key]]
            rings.append(Circle(radius=0.17, color=colour, stroke_width=4)
                         .move_to(self._at(winner)))
        self.play(*[Create(ring) for ring in rings], run_time=0.5)
        self.picked_marks.add(*rings)

    def play_act_two_score_lines(self) -> None:
        self.weight_readout = self._build_weight_readout()
        self.play(FadeIn(self.weight_readout), run_time=0.9)
        self.wait(0.9)

        self.score_lines = VGroup(*[
            self._score_line(weights, 2.9, colour) for _, weights, colour in self.weightings
        ])
        self.play(Create(self.score_lines), run_time=0.8)

        self.picked_marks = VGroup()
        self._sweep_to_winners(0, steps=26)
        self._mark_picks(0)
        self.wait(1.6)

    # ── Act three: the same situation, again ──────────────────────────────────────────

    def play_act_three_experiments(self) -> None:
        """Re-run the draft. Same two weightings, same bar -- a different set of players turns up.

        This is the act that turns two anecdotes into two distributions. Nothing about either
        strategy changes between these, so every difference in where a pick lands comes from the
        pool; and where the two rings land apart, the strategies genuinely wanted different men.
        """
        for experiment_index in range(1, len(self.experiments)):
            replacement_pool = self._build_pool(experiment_index)
            replacement_pool[0].set_opacity(0.22)
            self.play(FadeOut(self.pool, run_time=0.3), FadeIn(replacement_pool, run_time=0.4))
            self.pool = replacement_pool

            self._sweep_to_winners(experiment_index, steps=13)
            self._mark_picks(experiment_index)
            self.wait(0.6)
        self.wait(1.0)

    # ── Act four: where each strategy's picks land ────────────────────────────────────

    def _density_image(self, filename: str) -> ImageMobject:
        image = ImageMobject(str(_IMAGE_DIR / filename))
        image.height = 2 * PLANE_HALF_EXTENT * UNITS_PER_STANDARD_DEVIATION
        image.width = 2 * PLANE_HALF_EXTENT * UNITS_PER_STANDARD_DEVIATION
        image.move_to(PLANE_CENTRE)
        return image

    def play_act_four_pick_densities(self) -> None:
        """Sixty thousand of act three's experiments, at once, for each weighting.

        Only the two pick distributions are shown. The pool's own density could go underneath
        them, but it is the one field nobody is asking about -- where players are is the setup,
        and where each strategy ENDS UP is the answer, so the frame is given to the answer.

        Both clouds and both centres arrive in ONE cut. Shown one after the other, the first one
        up reads as the answer and the second as a correction to it; side by side they read as
        what they are, two strategies choosing differently out of the same pool.

        Nothing is animated once they are on screen. Added rather than faded in, because
        ImageMobject.set_opacity does not dim an image -- it overwrites every pixel's alpha with
        a single value, and alpha is what carries the density here. And no animation may play
        while they are up: an animation re-bakes the static frame, which composites these images
        without their alpha and washes the plane out for the length of the animation.
        """
        self.play(FadeOut(self.pool), FadeOut(self.score_lines),
                  FadeOut(self.picked_marks), run_time=0.9)

        # The mean of each cloud IS x(w) for that weighting. The neutral one sits just under the
        # bar rather than on it, which is the truncation bias the model subtracts off before it
        # reports anything.
        self.mean_dots = VGroup(*[
            Dot(self._at(self.settings[label]['pick_mean']), radius=0.10, color=colour)
            for (label, _, colour) in self.weightings
        ])
        self.add(self._density_image('pick_density_neutral.png'),
                 self._density_image('pick_density_alternative.png'),
                 self.mean_dots)
        self.wait(6.0)

    def construct(self) -> None:
        self.set_speech_service(NarrationVoice())
        self.introduce_pool()
        with self.voiceover(text=NARRATION['pool']) as tracker:
            wait_until_phrase(self, tracker, 'We assume that players with above average stats')
            self.play_act_one_pool()

        for line, act in (
            (NARRATION['score_lines'], self.play_act_two_score_lines)
            , (NARRATION['experiments'], self.play_act_three_experiments)
            , (NARRATION['densities'],   self.play_act_four_pick_densities)
        ):
            with self.voiceover(text=line):
                act()
