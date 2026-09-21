"""Starting a beat partway through a spoken line.

A beat normally plays inside the line that covers it, which is enough while a line is about one
thing. Some lines are about two things in sequence -- a pool of players, and then those players
being drawn at random -- and the second beat has to wait for the clause that introduces it.
Waiting a hand-counted number of seconds would do it, and would be wrong the moment the line is
rewritten.

This finds a phrase by where it falls in the text and reads that back as a time. Alistair reads
at a very even rate: measured across this set, a phrase's position in the characters predicts
its position in the audio to within six tenths of a second, which is finer than the beats being
triggered. Rewriting the line moves the phrase and its trigger together.

    with self.voiceover(text=NARRATION['pool']) as tracker:
        wait_until_phrase(self, tracker, 'that players are chosen randomly')
        self.flash_pool_draws(tracker.get_remaining_duration())
"""

from __future__ import annotations

from manim_voiceover.helper import remove_bookmarks


def seconds_until_phrase(tracker, phrase: str) -> float:
    """How far into the line the voice reaches `phrase`, in seconds from the line's first word.

    Whitespace is normalised on both sides, so a phrase can be quoted from narration.py as it
    reads there even though the line is written across several source lines.
    """
    spoken = ' '.join(remove_bookmarks(tracker.data['input_text']).split())
    wanted = ' '.join(phrase.split())

    offset = spoken.find(wanted)
    if offset < 0:
        raise ValueError(
            f'"{wanted}" does not appear in the line it is meant to be timing:\n  {spoken}\n'
            f'The phrase is quoted from narration.py, so an edit there has to be made here too '
            f'-- silently starting the beat at the top of the line would hide that.')

    return tracker.duration * offset / len(spoken)


def seconds_remaining_until_phrase(scene, tracker, phrase: str) -> float:
    """How long from now until the voice reaches `phrase`, never negative.

    For a beat that should not merely START on a word but FINISH on one: an animation given this
    as its duration lands on the phrase however long the run-up to it happened to take.
    """
    elapsed = scene.renderer.time - tracker.start_t
    return max(0.0, seconds_until_phrase(tracker, phrase) - elapsed)


def wait_until_phrase(scene, tracker, phrase: str, lead_seconds: float = 0.0) -> None:
    """Hold the picture until the voice reaches `phrase` in the line already playing.

    `lead_seconds` starts the beat that far ahead of the phrase, for a beat that should have
    FINISHED arriving by the time the words land rather than be starting then -- something the
    voice is about to call by name wants to be there already, not appearing as it is named.

    Never runs backwards: if whatever came before this already overran the phrase, the next beat
    starts immediately rather than being waited into the past.
    """
    elapsed = scene.renderer.time - tracker.start_t
    remaining = seconds_until_phrase(tracker, phrase) - lead_seconds - elapsed
    if remaining > 0:
        scene.wait(remaining)
