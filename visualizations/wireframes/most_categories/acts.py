"""One act of Most Categories at a time, for looking at a change without re-rendering the lot.

The full scene takes about ten minutes at low quality; a single act takes well under one. These
are the REAL scene class with a shorter construct, so nothing here can drift from what ships --
if an act looks right in here it looks right in the film.

    manim -ql acts.py Opening      the week, the rotation, the wide and narrow wins
    manim -ql acts.py Enumeration  the scenario table and its scroll
    manim -ql acts.py Walk         the two-category example, the lattice, the column sweep
    manim -ql acts.py Tipping      the slope question, the panel, the balance, the equation
    manim -ql acts.py Cutting      the backward walk, the cut, the glue, the slide, the board

Acts after the first need what the ones before them left on screen, so each runs the acts it
depends on -- Cutting has to build both walks before it can cut them. The listing above is in
scene order, and the cost grows down the list for that reason.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from most_categories import MostCategories, NarrationVoice   # noqa: E402


class Act(MostCategories):
    """Runs a prefix of the scene. `plays` names the methods this act needs, in order."""

    plays: tuple = ()

    def construct(self) -> None:
        self.set_speech_service(NarrationVoice())
        for name in type(self).plays:
            getattr(self, name)()


class Opening(Act):
    plays = ('play_a_week',)


class Enumeration(Act):
    plays = ('play_the_table',)


class Walk(Act):
    plays = ('play_walk',)


class Tipping(Act):
    # The panel clears the walk at its start and hands it back at its end, so it needs one.
    plays = ('play_walk', 'play_the_slope', 'play_what_tipping_means')


class Cutting(Act):
    # play_tipping_point carries the punt board and the tipping-point strip as well as the cut,
    # so this is the last act rather than the second to last. There was a Punting act beside it
    # naming the same four methods, which rendered the identical film under a second name.
    plays = ('play_walk', 'play_the_slope', 'play_what_tipping_means', 'play_tipping_point')
