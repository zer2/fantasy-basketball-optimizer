"""The voice the animations are narrated in: Alistair, from the ElevenLabs voice library.

manim-voiceover ships an ElevenLabs service, and it cannot be used here for two reasons.

It resolves a voice by fetching the account's own voices and filtering that list locally -- for
a voice id as much as for a name -- so a library voice, which lives in ElevenLabs' shared
library rather than in the account, is never found and the request quietly falls back to
whatever voice happens to be first. Adding the voice to the account would fix that, and needs a
`voices_write` key.

And its default model, `eleven_monolingual_v1`, has been retired: the API now rejects it and
names the replacements. So the model has to be chosen here regardless.

Both problems disappear by addressing the endpoint directly with the voice id, which is all the
API needs. Measured: a library voice id synthesises without being added to the account at all.

    from shared.narration_voice import NarrationVoice
    self.set_speech_service(NarrationVoice())

THE CACHE IS MONEY. This voice is billed per character against a monthly allowance, and the
cache under media/voiceovers is what stops a line being paid for twice. Deleting it re-buys
every line in the set; the cache is also gitignored, so nothing restores it. Before rendering,
`python visualizations/shared/narration_budget.py` says what a render would spend and what is
already paid for. Editing a line by one word makes it a new line, at full price.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import requests
from manim_voiceover.helper import append_to_json_file, remove_bookmarks
from manim_voiceover.services.base import SpeechService

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from backend.infra.secret_config import get_secret                   # noqa: E402

# "Alistair -- Clear, Neutral and Informative": British, middle-aged, the most used voice of the
# four Alistairs in the library. An explainer wants the one that sounds like it is explaining.
ALISTAIR_VOICE_ID = 'l30f87tf05uxyknGdDw6'

# eleven_monolingual_v1 and eleven_multilingual_v1 are retired. This is the current quality
# model; eleven_flash_v2_5 is the cheaper, faster one if the bill ever matters more than the
# reading.
DEFAULT_MODEL = 'eleven_multilingual_v2'

_ENDPOINT = 'https://api.elevenlabs.io/v1/text-to-speech'
_SECRET_NAME = 'ELEVENLABS_API_KEY'


class NarrationVoice(SpeechService):
    """ElevenLabs, addressed by voice id, with the cache contract manim-voiceover expects."""

    def __init__(self, voice_id: str = ALISTAIR_VOICE_ID, model: str = DEFAULT_MODEL, **kwargs):
        self.voice_id = voice_id
        self.model = model
        super().__init__(**kwargs)

    def generate_from_text(self, text: str, cache_dir: str = None, path: str = None,
                           **kwargs) -> dict:
        """One line of narration as an mp3, cached by its text the way every service caches.

        The cache is what keeps the bill down: a line is paid for once, however many times the
        scene is rendered, and only a rewritten line spends its characters again.
        """
        # The base's cache helpers build paths with `/`, so a string cache_dir would fail there
        # rather than here; coerced once, at the edge.
        cache_dir = Path(cache_dir if cache_dir is not None else self.cache_dir)

        input_text = remove_bookmarks(text)
        input_data = {'input_text': input_text, 'service': 'elevenlabs',
                      'voice_id': self.voice_id, 'model': self.model}

        cached = self.get_cached_result(input_data, cache_dir)
        if cached is not None:
            return cached

        audio_path = path if path is not None else self.get_audio_basename(input_data) + '.mp3'
        (Path(cache_dir) / audio_path).write_bytes(self._synthesise(input_text))

        json_dict = {'input_text': text, 'input_data': input_data, 'original_audio': audio_path}
        append_to_json_file(Path(cache_dir) / 'cache.json', json_dict)
        return json_dict

    def _synthesise(self, text: str) -> bytes:
        """The API call itself. A failure is raised rather than swallowed: silent narration is
        worse than a stopped render, because it only shows up on playback."""
        key = get_secret(_SECRET_NAME)
        if not key:
            raise RuntimeError(
                f'{_SECRET_NAME} is not set. Put it in .streamlit/secrets.toml or the '
                f'environment; the animations cannot be narrated without it.')
        response = requests.post(
            f'{_ENDPOINT}/{self.voice_id}',
            headers={'xi-api-key': key, 'Content-Type': 'application/json'},
            json={'text': text, 'model_id': self.model},
            timeout=120)
        if response.status_code != 200:
            detail = response.json().get('detail', response.text)
            message = detail.get('message', detail) if isinstance(detail, dict) else detail
            raise RuntimeError(f'ElevenLabs refused the line ({response.status_code}): {message}')
        return response.content
