"""How long every line takes to say, and what that does to the scenes that are paced off it.

Generating the audio is the expensive step and rendering is the slow one, so this does the first
without the second: every line is voiced (and cached, so a later render pays nothing), measured,
and compared against whatever the previous voice produced for the same words.

    python visualizations/shared/measure_narration.py

A scene whose beats are timed by hand does not much care -- a shorter line just holds its last
frame for less time. The one that does care is self-play, whose pass rate is the narration
divided by the number of passes: a faster reader makes the board move faster, and past about two
seconds a pass stops reading as a change.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

_VISUALIZATIONS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_VISUALIZATIONS))

from shared.narration_voice import NarrationVoice                        # noqa: E402

_CACHE = _VISUALIZATIONS / 'media' / 'voiceovers'
# Self-play divides these lines' total by the number of passes to get one steady rate.
_PACED_SCENE = '6_self_play'
_PACED_KEYS = ('early_passes', 're_estimation', 'late_passes')


def seconds_of(audio_path: Path) -> float:
    output = subprocess.run(
        ['ffprobe', '-loglevel', 'error', '-show_entries', 'format=duration', '-of',
         'csv=p=0', str(audio_path)], capture_output=True, text=True).stdout.strip()
    return float(output) if output else 0.0


def previous_seconds() -> dict[str, float]:
    """What the last voice produced for each line, read out of the cache it left behind."""
    cache_file = _CACHE / 'cache.json'
    if not cache_file.exists():
        return {}
    previous = {}
    for entry in json.loads(cache_file.read_text(encoding='utf-8')):
        if entry.get('input_data', {}).get('service') == 'elevenlabs':
            continue
        audio = _CACHE / entry['original_audio']
        if audio.exists():
            previous[' '.join(entry['input_text'].split())] = seconds_of(audio)
    return previous


def narration_of(folder: Path) -> dict[str, str]:
    spec = importlib.util.spec_from_file_location('narration', folder / 'narration.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NARRATION


def main() -> None:
    voice = NarrationVoice()
    before = previous_seconds()
    characters = 0
    for folder in sorted(_VISUALIZATIONS.glob('[1-8]_*')):
        print(f'\n{folder.name}')
        total_now = total_before = 0.0
        paced_now = paced_before = 0.0
        for key, line in narration_of(folder).items():
            spoken = ' '.join(line.split())
            characters += len(spoken)
            result = voice.generate_from_text(spoken)
            now = seconds_of(_CACHE / result['original_audio'])
            was = before.get(spoken, 0.0)
            total_now += now
            total_before += was
            if folder.name == _PACED_SCENE and key in _PACED_KEYS:
                paced_now += now
                paced_before += was
            change = f'{now - was:+5.1f}s' if was else '   new'
            print(f'   {key:20} {was:5.1f}s -> {now:5.1f}s  {change}')
        shrink = (1 - total_now / total_before) * 100 if total_before else 0.0
        print(f'   {"TOTAL":20} {total_before:5.1f}s -> {total_now:5.1f}s  '
              f'({shrink:+.0f}% shorter)')
        if paced_before:
            print(f'   pass rate: {paced_before / 32:.2f}s -> {paced_now / 32:.2f}s per pass')
    print(f'\n{characters} characters voiced (cached, so a render spends nothing more)')


if __name__ == '__main__':
    main()
