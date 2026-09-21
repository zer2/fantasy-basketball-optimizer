"""Check every narration file for the faults that reach the rendered audio, or the bill.

Run before rendering; it costs nothing and catches what listening for would cost a render:

    python visualizations/shared/check_narration.py

**Joined words.** A line is written as adjacent string literals, one per source line. Python
concatenates them with nothing in between, so a literal that ends without a space followed by one
that starts without a space produces "theirH-score" -- which the voice reads as a single nonsense
word and a listener hears as a glitch.

**Lines nothing says.** A key in narration.py that no scene reads is a line that will be voiced,
paid for, and never heard. A key a scene reads that narration.py does not have is a KeyError part
way through a render, after the earlier lines have already been bought.

**Renders left behind.** Editing a line does not re-render the scene, and a finished .mp4 looks
exactly as finished whether or not it still says what narration.py says. So each scene's most
recent subtitle track is read back and matched against its lines: anything a render does not
speak is a render that needs making again before the scene is judged or shipped.

This used to also report the pauses gTTS forced mid-phrase, by asking gTTS where it would split
a line into hundred-character requests. Alistair takes a whole line in one request and puts no
seams in it, so there is nothing left for that check to find; it went when the voice did.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

_VISUALIZATIONS = Path(__file__).resolve().parent.parent
_RENDERS = _VISUALIZATIONS / 'media' / 'videos'


def find_joined_words(source: str) -> list[tuple[int, str]]:
    """Places where two string literals meet with no space between them."""
    faults = []
    lines = source.splitlines()
    for index, line in enumerate(lines[:-1]):
        stripped, following = line.strip(), lines[index + 1].strip()
        if not (stripped.endswith("'") and following.startswith("'")):
            continue
        # The text inside each literal, without its quotes.
        ending = stripped.rstrip(',').rstrip("'")
        starting = following[1:]
        # A literal that opens with punctuation is right to have no space before it: a comma
        # or a full stop belongs against the word it follows.
        if (ending and not ending.endswith(' ')
                and starting and starting[0] not in ' ,.;:!?'):
            faults.append((index + 1, f'{ending[-30:]}" + "{starting[:30]}'))
    return faults


def narration_keys(folder: Path) -> set[str]:
    """Every key narration.py defines, read by importing it rather than by parsing it."""
    spec = importlib.util.spec_from_file_location('narration', folder / 'narration.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return set(module.NARRATION)


def keys_the_scene_reads(folder: Path) -> tuple[set[str], set[str]]:
    """What the scene files ask NARRATION for, and every name they mention at all.

    Two sets, because the two questions want different strictness. A key subscripted by name is
    definitely wanted, and one that is not defined will stop the render -- so that set is matched
    exactly. But a scene may reach its lines indirectly, as self-play does, listing the keys of
    its repeating beats in a tuple and looping over them; those never appear as a subscript. So
    "nothing says this line" is answered against every quoted name in the file, which is loose in
    the harmless direction: it can miss a line gone unused, and it will not invent one.

    The prep scripts are skipped: they build the data a scene draws and never speak.
    """
    subscripted, mentioned = set(), set()
    for path in sorted(folder.glob('*.py')):
        if path.name == 'narration.py' or path.name.startswith('prepare_'):
            continue
        source = path.read_text(encoding='utf-8')
        subscripted.update(re.findall(r"NARRATION\['([a-z_0-9]+)'\]", source))
        mentioned.update(re.findall(r"'([a-z_0-9]+)'", source))
    return subscripted, mentioned


def lines_the_render_does_not_say(folder: Path) -> tuple[list[str], str]:
    """Which of this scene's lines its newest render does not speak, and which render that was.

    Manim files a render under its module name, so the subtitle track is found by looking for a
    directory named after one of the scene files rather than by keeping a table of class names
    in step by hand. The newest .srt under it is the render being judged.

    Whitespace is collapsed on both sides before comparing: narration.py wraps its lines across
    source lines and the subtitle track wraps them across cues, and neither break is spoken.
    """
    candidates = [path.stem for path in sorted(folder.glob('*.py'))
                  if path.name != 'narration.py' and not path.name.startswith('prepare_')]
    subtitles = [found
                 for name in candidates
                 for found in (_RENDERS / name).glob('*/*.srt')]
    if not subtitles:
        return [], ''

    newest = max(subtitles, key=lambda path: path.stat().st_mtime)
    heard = ' '.join(' '.join(
        line for line in newest.read_text(encoding='utf-8').splitlines()
        if line.strip() and '-->' not in line and not line.strip().isdigit()).split())

    spec = importlib.util.spec_from_file_location('narration', folder / 'narration.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    unsaid = [key for key, value in module.NARRATION.items()
              if ' '.join(str(value).split()) not in heard]
    return unsaid, newest.stem


def main() -> int:
    faults = 0
    # The numbered scenes, then any wireframe drafts. Wireframes are checked for exactly
    # the same faults -- a joined word is a joined word whichever voice reads it -- but
    # they have no render to fall behind, so that part simply finds nothing.
    folders = (sorted(_VISUALIZATIONS.glob('[1-8]_*'))
               + sorted(path for path in _VISUALIZATIONS.glob('wireframes/*')
                        if (path / 'narration.py').exists()))
    for folder in folders:
        source = (folder / 'narration.py').read_text(encoding='utf-8')
        joined = find_joined_words(source)
        written = narration_keys(folder)
        subscripted, mentioned = keys_the_scene_reads(folder)
        unread = sorted(written - subscripted - mentioned)
        missing = sorted(subscripted - written)

        print(folder.name if folder.parent == _VISUALIZATIONS
              else f'{folder.parent.name}/{folder.name}')
        for line_number, sample in joined:
            print(f'   JOINED at line {line_number}: "{sample}"')
        for key in missing:
            print(f'   MISSING: the scene reads {key!r}, narration.py does not define it '
                  f'-- the render will stop there')
        for key in unread:
            print(f'   UNREAD: narration.py defines {key!r} and no scene says it '
                  f'-- it would be voiced and paid for unheard')

        # Reported but NOT counted as a fault: a scene mid-edit is expected to be ahead of its
        # render, and failing the check for that would make it useless during the edit it is
        # most wanted for. It is the last thing to settle before a scene is judged, not a
        # reason to refuse to look at it.
        unsaid, render = lines_the_render_does_not_say(folder)
        if unsaid:
            print(f'   BEHIND: the {render} render does not say {", ".join(map(repr, unsaid))} '
                  f'-- re-render before judging this scene')

        faults += len(joined) + len(missing) + len(unread)
        if not (joined or missing or unread or unsaid):
            print('   clean')
    return 1 if faults else 0


if __name__ == '__main__':
    sys.exit(main())
