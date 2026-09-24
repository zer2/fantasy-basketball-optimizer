"""What gradient descent is: one start, walking uphill until it cannot.

Run `python visualizations/shared/prepare_weight_surface_data.py` first: one bootstrap run
measures the slices both this scene and its sibling read.

    manim -ql visualizations/8_gradient_descent/gradient_descent.py GradientDescent
"""

from __future__ import annotations

import sys
from pathlib import Path

from manim import FadeIn, WHITE, PURPLE_A
from manim_voiceover import VoiceoverScene

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.weight_surface_base import (                              # noqa: E402
    WeightSurfaceScene, CAMERA_DRIFT,
)
from shared.narration_voice import NarrationVoice   # noqa: E402
from narration import NARRATION                                       # noqa: E402


class GradientDescent(VoiceoverScene, WeightSurfaceScene):
    """One start, from the middle, walking uphill until it cannot: what descent IS.

    Measured on the simplest slice the search could find -- a single hill with its top inside the
    picture -- because this scene is about the mechanics of climbing, and nothing else should be
    competing for the viewer's attention while they are being explained.
    """

    data_filename = 'weight_surface_simple.json'

    def construct(self) -> None:
        self.set_speech_service(NarrationVoice())
        with self.voiceover(text=NARRATION['surface']) as tracker:
            axes = self.introduce_surface(tracker.duration)
        self.begin_ambient_camera_rotation(rate=CAMERA_DRIFT)

        climb = self.climb_named('the middle of the surface')
        start = climb['path'][0]
        with self.voiceover(text=NARRATION['the_start']):
            self.play(FadeIn(self.mark_point(axes, start, WHITE), scale=2.0), run_time=0.6)
            self.play(FadeIn(self.write_score(axes, start, climb['start_score'], WHITE)),
                      run_time=0.6)
            self.wait(1.0)

        with self.voiceover(text=NARRATION['the_climb']):
            self.play_climb(axes, climb, PURPLE_A)

        with self.voiceover(text=NARRATION['the_top']):
            self.play(FadeIn(self.write_score(axes, climb['path'][-1], climb['score'], PURPLE_A)),
                      run_time=0.8)
            self.wait(2.4)
        self.stop_ambient_camera_rotation()


