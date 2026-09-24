"""Why the algorithm does not run gradient descent only once.

Run `python visualizations/shared/prepare_weight_surface_data.py` first: one bootstrap run
measures the slices both this scene and its sibling read.

    manim -ql visualizations/7_seed_menu/seed_menu.py SeedMenu
"""

from __future__ import annotations

import sys
from pathlib import Path

from manim import Create, FadeIn, FadeOut, WHITE, PURPLE_A
from manim_voiceover import VoiceoverScene

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.weight_surface_base import (                              # noqa: E402
    WeightSurfaceScene, CAMERA_DRIFT,
)

# The axes alone, before a word is said. Everything drawn before the first line is silence, so
# only the frame is built here -- about a second, enough that the line does not open on an empty
# screen. The surface itself is drawn under the words.
AXES_SECONDS = 0.9
# The surface cannot be slower than this however long the line is, and cannot be so quick that it
# snaps into place; it is drawn to land on the clause that first talks about its shape.
SURFACE_SECONDS = (1.0, 2.8)
POSTS_SECONDS = 0.5
CAPTION_SECONDS = 0.5
SETTLE_BEFORE_SEEDS = 0.6
from shared.narration_voice import NarrationVoice   # noqa: E402
from shared.narration_timing import seconds_remaining_until_phrase   # noqa: E402
from narration import NARRATION                                       # noqa: E402


class SeedMenu(VoiceoverScene, WeightSurfaceScene):
    """The menu of starting points, and the one the algorithm takes.

    Measured on the two-summit slice, which is the whole reason a menu exists: the seed a
    descent starts from decides which of the two peaks it ends on.

    The scene is the algorithm's own sequence. It scores every seed WHERE IT STANDS -- one
    objective call apiece, no descent -- takes the best of them and descends that one alone. The
    two it passes over are never stepped, not even once, so they are never walked here either:
    they stay where they are, which is exactly what happens to them.
    """

    data_filename = 'weight_surface.json'

    def construct(self) -> None:
        self.set_speech_service(NarrationVoice())
        # Only the axes are drawn in silence. The opening line opens by pointing -- "this is
        # another H-scoring surface" -- so the surface is drawn underneath that clause and is
        # finished by the time the line reaches its peaks, which is the first thing said that
        # needs a shape to point at. Building the whole picture first instead cost nearly three
        # seconds of silence before Alistair said anything.
        axes = self.introduce_axes(AXES_SECONDS)
        self.begin_ambient_camera_rotation(rate=CAMERA_DRIFT)
        with self.voiceover(text=NARRATION['surface']) as tracker:
            shortest, longest = SURFACE_SECONDS
            self.play(Create(self.build_surface(axes)), run_time=min(longest, max(
                shortest,
                seconds_remaining_until_phrase(self, tracker, 'It has multiple peaks'))))
            self.play(Create(self.build_corner_posts(axes)), run_time=POSTS_SECONDS)
            self.play(FadeIn(self.build_caption()), run_time=CAPTION_SECONDS)
            self.wait(SETTLE_BEFORE_SEEDS)

        # The three seeds this plane can show: a gentle punt of each category, and the balanced
        # build. The real menu has one seed per category plus the drafter's own earlier builds;
        # the six seeds that move a category the plane does not draw cannot be drawn here.
        climbs = [self.climb_named(label) for label in (
            f"punt {self.measured['category_a']}"
            , f"punt {self.measured['category_b']}"
            , 'balanced'
        )]
        starts = [climb['path'][0] for climb in climbs]

        balls = [self.mark_point(axes, start, WHITE) for start in starts]
        scores = [self.write_score(axes, start, climb['start_score'], WHITE)
                  for start, climb in zip(starts, climbs)]
        with self.voiceover(text=NARRATION['three_seeds']):
            self.play(*[FadeIn(ball, scale=2.0) for ball in balls],
                      *[FadeIn(score) for score in scores], run_time=1.2)
            self.wait(1.4)

        # The choice: the best seed where it stands, which is all the algorithm looks at.
        chosen = max(range(len(climbs)), key=lambda index: climbs[index]['start_score'])
        with self.voiceover(text=NARRATION['the_choice']):
            self.play(balls[chosen].animate.set_color(PURPLE_A),
                      scores[chosen].animate.set_color(PURPLE_A), run_time=0.8)
            self.wait(1.0)
            self.play(*[FadeOut(score) for score in scores], run_time=0.6)

        # The one descent that is actually run. The other two starts stay on screen, unmoved.
        with self.voiceover(text=NARRATION['the_descent']):
            self.play_climb(axes, climbs[chosen], PURPLE_A)
            self.play(FadeIn(self.write_score(axes, climbs[chosen]['path'][-1],
                                              climbs[chosen]['score'], PURPLE_A)), run_time=0.8)
            self.wait(2.0)
        self.stop_ambient_camera_rotation()
