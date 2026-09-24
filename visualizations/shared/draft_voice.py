"""The voice a wireframe is timed against: gTTS, free, and deliberately not the shipped one.

A wireframe exists to find out whether a scene works at all -- whether the beats land, whether
the picture carries the argument, whether the thing is two minutes or five. None of that needs
the good voice, and every line sent to the shipped voice costs money and lands in a cache that
is worth protecting. So drafts are read by gTTS and only promoted to Alistair once the script
has stopped changing.

The two services are interchangeable at the call site -- both are manim-voiceover SpeechServices
-- so promoting a scene is a one-line swap in its `construct`:

    self.set_speech_service(DraftVoice())        # while the script is still moving
    self.set_speech_service(NarrationVoice())    # once it has settled

gTTS reads faster and flatter than Alistair does, so a wireframe's timings are indicative and
not final. Anything anchored with narration_timing follows the words either way; anything given
a fixed run_time will need a look after the swap.
"""

from __future__ import annotations

from manim_voiceover.services.gtts import GTTSService


class DraftVoice(GTTSService):
    """gTTS at its default settings, named so a scene says which voice it is drafting against."""

    def __init__(self, lang: str = 'en', tld: str = 'com', **kwargs):
        super().__init__(lang=lang, tld=tld, **kwargs)
