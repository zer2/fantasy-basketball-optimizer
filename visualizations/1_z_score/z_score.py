"""Why a category is worth winning: two random teams, and the gap between them.

Deals two thirteen-player rosters at random out of the pool a standard league drafts, totals
each side's per-game Points, and drops the difference into a histogram in the middle. Repeated
ten thousand times the histogram is a bell -- which is the fact every later scene leans on,
since it is what makes a category something you can be favoured or unfavoured to win rather
than a number you simply accumulate.

Everything shown is real: 2025-26 per-game scoring for the 156 players a twelve-team league
drafts, chosen by G-score. Run `python visualizations/prepare_season_data.py` to build the data
and headshots this reads; the draws are seeded there, so a re-render reproduces the same video.

Every player contributes the same number in every simulation here -- their season average. The
companion scene in weekly_differential.py deals each player a real week instead, which is what
that average is hiding.

Render one act while tuning it, the whole thing when it is right:

    manim -ql visualizations/scenes/team_differential.py ActOneSingleDraw
    manim -qh visualizations/scenes/team_differential.py TeamDifferentialFull
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import (
    Group, VGroup, Circle, Line, MathTex, FadeIn, FadeOut, Write,
    DOWN, UP, BLACK, YELLOW,
)
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.differential_base import DifferentialSceneBase   # noqa: E402


from narration import NARRATION   # noqa: E402


# ── The opening: the whole draftable pool, and two teams pulled out of it ─────────────

POOL_GRID_COLUMNS = 13        # 13 x 12 lays the 156 drafted players out exactly
POOL_CHIP_SPACING_X = 0.72
POOL_CHIP_SPACING_Y = 0.60
POOL_CHIP_DIAMETER = 0.48
POOL_GRID_CENTRE_Y = 0.15
POOL_INTRO_DRAWS = 3          # how many random rosters are pulled out before moving on
POOL_VEIL_OPACITY = 0.78      # how far an unchosen player is dimmed behind his veil


class SeasonAverageDifferential(DifferentialSceneBase):
    """The averages cut: a player is worth their weekly average, every single time."""

    data_filename      = 'pool_2025_26.json'
    # Three standard deviations either way, as in the weekly scene. Both scenes count the same
    # thing in the same unit -- points in a week -- so the two axes can be read against each
    # other: 270 here against 420 there is the whole difference real weeks make.
    differential_limit = 270
    bin_width          = 15
    axis_tick_step     = 90
    total_caption      = 'Points in an average week'
    spread_caption     = 'standard deviation: {spread:.0f} points in the week'

    def contribution_values(self, simulation_index: int) -> np.ndarray:
        """Each drafted player's weekly average, which is the same whenever they are drawn."""
        pool_averages = np.array(
            [player[self.prepared['value_key']] for player in self.prepared['pool']])
        return pool_averages[self.prepared['rosters'][simulation_index]]

    # ── Act zero: the pool everyone is drafting out of ────────────────────────────────

    def _pool_grid_position(self, pool_index: int) -> np.ndarray:
        row, column = divmod(pool_index, POOL_GRID_COLUMNS)
        rows = -(-len(self.prepared['pool']) // POOL_GRID_COLUMNS)
        return np.array([
            (column - (POOL_GRID_COLUMNS - 1) / 2) * POOL_CHIP_SPACING_X,
            POOL_GRID_CENTRE_Y + ((rows - 1) / 2 - row) * POOL_CHIP_SPACING_Y,
            0.0,
        ])

    def reveal_pool(self) -> None:
        """Put every drafted player on screen, before anything is said about them.

        Selection is shown by dimming everyone else, and the dimming is done with a dark disc laid
        OVER each portrait rather than by fading the portrait itself. The headshots are circular
        PNGs with transparent corners, and Manim's set_opacity replaces per-pixel alpha with a
        single value -- fading one directly would square it off into a grey block. A veil is a
        vector object and dims exactly as expected.
        """
        self.pool_chips, self.pool_veils = Group(), VGroup()
        for pool_index in range(len(self.prepared['pool'])):
            position = self._pool_grid_position(pool_index)
            portrait = self._headshot(pool_index)
            portrait.height = POOL_CHIP_DIAMETER
            portrait.move_to(position)
            self.pool_chips.add(portrait)
            self.pool_veils.add(Circle(
                radius=POOL_CHIP_DIAMETER / 2, stroke_width=0,
                fill_color=BLACK, fill_opacity=POOL_VEIL_OPACITY,
            ).move_to(position))

        self.add(self.pool_chips, self.pool_veils)
        self.play(FadeIn(self.pool_chips), FadeIn(self.pool_veils), run_time=1.2)
        self.wait(0.6)

    def play_pool_draws(self) -> None:
        """Pull a random thirteen out of the pool, a few times over.

        Thirteen, not twenty-six: this is one roster being chosen, and the second team is what
        act one introduces. The draws are the ones the rest of the scene goes on to use, so the
        team dealt in act one is one the viewer has already watched being picked.
        """
        for simulation_index in range(POOL_INTRO_DRAWS):
            drawn = self.prepared['rosters'][simulation_index][:self.team_size]
            self.play(*[self.pool_veils[int(pool_index)].animate.set_opacity(0.0)
                        for pool_index in drawn], run_time=0.7)
            self.wait(1.2)
            self.play(self.pool_veils.animate.set_opacity(POOL_VEIL_OPACITY), run_time=0.5)

        self.play(FadeOut(self.pool_chips), FadeOut(self.pool_veils), run_time=0.9)


class ActOneSingleDraw(SeasonAverageDifferential):
    def construct(self) -> None:
        self.build_static_frame()
        self.play_act_one_single_draw()


class ActTwoRepeatedDraws(SeasonAverageDifferential):
    def construct(self) -> None:
        self.build_static_frame()
        self.play_act_one_single_draw()
        self.play_act_two_repeated_draws()


class ActThreeMontage(SeasonAverageDifferential):
    def construct(self) -> None:
        self.build_static_frame()
        self.play_act_one_single_draw()
        self.play_act_two_repeated_draws()
        self.play_act_three_montage()


class ActZeroPool(SeasonAverageDifferential):
    """The opening alone, for tuning the pool grid without re-rendering the whole scene."""

    def construct(self) -> None:
        self.reveal_pool()
        self.play_pool_draws()


class TeamDifferentialFull(VoiceoverScene, SeasonAverageDifferential):
    """The Z-score scene, narrated.

    Each act plays inside the voiceover block that covers it, rather than every animation being
    timed to a clause. That keeps the mapping between line and act obvious, and means a rewritten
    line stretches or shortens only its own act -- if the audio outlasts the animation the last
    frame is held, and if the animation outlasts the audio it simply plays on.
    """

    def setup(self) -> None:
        super().setup()
        # gTTS: no key, no account. This is the only line to change to swap in a better voice.
        self.set_speech_service(GTTSService())

    def play_act_six_z_score(self) -> None:
        """From the height of the curve to the formula everyone already knows.

        The density at a dead heat is one over sigma root two pi. Root two pi is a constant, so
        the only thing in it that varies between categories is sigma -- which means a category's
        importance is inversely proportional to its spread. Writing that down with an empty
        numerator and then filling the numerator in is the whole derivation of a Z-score, and it
        arrives as a consequence of the simulation rather than as a definition handed down.
        """
        self.play(
            FadeOut(self.density_equation),
            FadeOut(self.normal_curve),
            FadeOut(self.spread_markers),
            FadeOut(self.axis),
            FadeOut(self.peak_marker),
            run_time=1.0,
        )

        # The fraction is assembled from separate pieces rather than typeset as one expression,
        # so the numerator can be WRITTEN INTO the empty space above a bar and a sigma that never
        # move. Transforming a whole fraction into another one would slide and morph the parts
        # that are meant to be standing still, and the point of the beat is that only the top
        # changes.
        vinculum = Line([-0.95, 0.0, 0.0], [0.95, 0.0, 0.0], color=YELLOW, stroke_width=5)
        denominator = MathTex(r'\sigma', font_size=96, color=YELLOW)
        denominator.next_to(vinculum, DOWN, buff=0.28)

        with self.voiceover(text=NARRATION['inverse_sigma']):
            self.play(Write(VGroup(vinculum, denominator)), run_time=1.2)
            self.wait(0.8)

        with self.voiceover(text=NARRATION['numerator']):
            numerator = MathTex(r'x - \mu', font_size=96, color=YELLOW)
            numerator.next_to(vinculum, UP, buff=0.28)
            self.play(Write(numerator), run_time=1.4)
            self.wait(0.6)

        with self.voiceover(text=NARRATION['z_score']):
            self.wait(1.2)
        self.wait(1.4)

    def construct(self) -> None:
        # The pool is on screen before a word is said about it. Starting the line over the fade-in
        # meant the first sentence played to an empty frame.
        self.reveal_pool()
        with self.voiceover(text=NARRATION['pool']):
            self.play_pool_draws()

        self.build_static_frame()
        # The central-limit line describes a process, so it runs over the whole of that process
        # rather than over the first draw alone: teams dealt, dealt again, then ten thousand
        # times. Under act one by itself it outlasted the animation by fourteen seconds and the
        # picture sat still through the half of the sentence that was about it moving.
        with self.voiceover(text=NARRATION['two_teams']):
            self.play_act_one_single_draw()
            self.play_act_two_repeated_draws()
            self.play_act_three_montage()

        with self.voiceover(text=NARRATION['bar_height']):
            self.play_act_four_normal_curve()
        with self.voiceover(text=NARRATION['simple_expression']):
            self.play_act_five_density_at_zero()
        self.play_act_six_z_score()
