"""Why the algorithm does not run gradient descent only once.

Run `python visualizations/shared/prepare_weight_surface_data.py` first: one bootstrap run
measures the slices both this scene and its sibling read.

    manim -ql visualizations/7_seed_menu/seed_menu.py SeedMenu
"""

from __future__ import annotations

import sys
from pathlib import Path

from manim import FadeIn, FadeOut, WHITE, YELLOW
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.weight_surface_base import (                              # noqa: E402
    WeightSurfaceScene, CAMERA_DRIFT,
)
from narration import NARRATION                                       # noqa: E402


class SeedMenu(VoiceoverScene, WeightSurfaceScene):
    """The menu of starting points, and the one the algorithm takes.

    Measured on the three-summit slice, which is the whole reason a menu exists.

    The scene is the algorithm's own sequence. It scores every seed WHERE IT STANDS -- one
    objective call apiece, no descent -- takes the best of them and descends that one alone. The
    two it passes over are never stepped, not even once, so they are never walked here either:
    they stay where they are, which is exactly what happens to them.
    """

    data_filename = 'weight_surface.json'

    def construct(self) -> None:
        self.set_speech_service(GTTSService())
        with self.voiceover(text=NARRATION['surface']) as tracker:
            axes = self.introduce_surface(tracker.duration)
        self.begin_ambient_camera_rotation(rate=CAMERA_DRIFT)

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
            self.play(balls[chosen].animate.set_color(YELLOW),
                      scores[chosen].animate.set_color(YELLOW), run_time=0.8)
            self.wait(1.0)
            self.play(*[FadeOut(score) for score in scores], run_time=0.6)

        # The one descent that is actually run. The other two starts stay on screen, unmoved.
        with self.voiceover(text=NARRATION['the_descent']):
            self.play_climb(axes, climbs[chosen], YELLOW)
            self.play(FadeIn(self.write_score(axes, climbs[chosen]['path'][-1],
                                              climbs[chosen]['score'], YELLOW)), run_time=0.8)
            self.wait(2.0)
        self.stop_ambient_camera_rotation()
