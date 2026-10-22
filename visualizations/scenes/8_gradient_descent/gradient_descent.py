"""What gradient descent is: one start, walking uphill until it cannot.

Run `python visualizations/shared/prepare_weight_surface_data.py` first: one bootstrap run
measures the slices both this scene and its sibling read.

    manim -ql visualizations/8_gradient_descent/gradient_descent.py GradientDescent
"""

from __future__ import annotations

import sys
from pathlib import Path

from manim import FadeIn, PURPLE_A
from manim_voiceover import VoiceoverScene

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.weight_surface_base import (                              # noqa: E402
    WeightSurfaceScene, CAMERA_DRIFT, CLIMB_ZOOM, DESCENT_RUN_TIME, READOUT_PURPLE,
)
from shared.narration_timing import wait_until_phrase                # noqa: E402

# How many of the thirty iterations are left off the end of the drawn climb. See
# stop_where_it_stops_climbing for what they do, and why not drawing them costs nothing.
SETTLED_TAIL = 14
from shared.narration_voice import NarrationVoice   # noqa: E402
from narration import NARRATION                                       # noqa: E402


class GradientDescent(VoiceoverScene, WeightSurfaceScene):
    """One start, from the middle, walking uphill until it cannot: what descent IS.

    Measured on the simplest slice the search could find -- a single hill with its top inside the
    picture -- because this scene is about the mechanics of climbing, and nothing else should be
    competing for the viewer's attention while they are being explained.
    """

    data_filename = 'weight_surface_simple.json'

    def stop_where_it_stops_climbing(self, climb: dict) -> dict:
        """The climb as it is drawn, with the settled tail left off the end.

        The last fourteen of the thirty iterations gain no height: the objective reads the same to
        every digit the readout shows. They gain no ground either -- the regulariser has pinned
        one weight exactly at its balanced value and the other oscillates by under one percent, so
        the ball walks back up the line it has already drawn and finishes sitting on it. A picture
        of a path should not end on top of itself.

        This is a drawing choice, so it is checked rather than asserted: if the tail ever turns
        out to carry height the readout would show, this raises instead of quietly writing a
        number the drawn ball never reached.
        """
        walked = dict(climb, path=climb['path'][:-SETTLED_TAIL])
        drawn = self.height_at(*walked['path'][-1])
        if f'{drawn * 100:.2f}' != f"{climb['score'] * 100:.2f}":
            raise RuntimeError(
                f'Leaving the last {SETTLED_TAIL} iterations off the climb drops it from '
                f"{climb['score'] * 100:.2f}% to {drawn * 100:.2f}%, so the readout would name a "
                f'height the ball on screen never reached. The tail is no longer dead weight -- '
                f'shorten SETTLED_TAIL, or draw the whole climb.')
        return walked

    def construct(self) -> None:
        self.set_speech_service(NarrationVoice())
        with self.voiceover(text=NARRATION['surface']) as tracker:
            axes = self.introduce_surface(tracker.duration)
        self.begin_ambient_camera_rotation(rate=CAMERA_DRIFT)

        climb = self.stop_where_it_stops_climbing(
            self.climb_named('the middle of the surface'))
        start = climb['path'][0]
        # The line spends most of itself on why a search is needed at all -- the grid, the many
        # dimensions -- and only names a starting point at the end of it. Putting the ball on
        # screen at the top of the line leaves it sitting there unexplained through all of that,
        # so it waits for the clause that introduces it.
        #
        # One colour throughout. There is nothing here to tell the start apart FROM: this scene
        # runs a single descent, and a second colour would imply a distinction the picture does
        # not make. The menu scene, which shows three starts and picks one, does need two.
        with self.voiceover(text=NARRATION['the_start']) as tracker:
            wait_until_phrase(self, tracker, 'That method is starting in one place')
            self.play(FadeIn(self.mark_point(axes, start, PURPLE_A, keep_apparent_size=True)),
                      run_time=0.6)
            # Under the ball, not over it. The climb leaves this point going straight up, so a
            # readout in the usual place has its own line drawn through the middle of it; below
            # the start there is nothing but hillside.
            start_readout = self.write_score(axes, start, climb['start_score'], READOUT_PURPLE,
                                             above=False)
            self.play(FadeIn(start_readout), run_time=0.6)

        # The camera pushes in on the climb as it is drawn. The start's readout goes out with it:
        # it is pinned to a point on the surface, and the push carries that point across the frame
        # while the text stays the same size, so it would drift away from what it labels. It is
        # written again at the arrival, where both heights can be read against each other.
        with self.voiceover(text=NARRATION['the_climb']) as tracker:
            self.play_climb(axes, climb, PURPLE_A, zoom=CLIMB_ZOOM,
                            fade_while_climbing=start_readout,
                            run_time=max(tracker.duration, DESCENT_RUN_TIME))

        with self.voiceover(text=NARRATION['the_top']):
            self.play(
                FadeIn(self.write_score(axes, start, climb['start_score'], READOUT_PURPLE,
                                        above=False))
                # Further out than the default: the climb curls over just above where it stops,
                # and the usual gap would put this inside the curl.
                , FadeIn(self.write_score(axes, climb['path'][-1], climb['score'],
                                          READOUT_PURPLE, clear_by=1.3))
                , run_time=0.8)
            self.wait(2.4)
        self.stop_ambient_camera_rotation()


