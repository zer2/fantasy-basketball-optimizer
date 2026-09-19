"""Check every narration file for the two faults that keep reaching the rendered audio.

Run before rendering; it costs nothing and catches what listening for would cost a render:

    python visualizations/shared/check_narration.py

**Joined words.** A line is written as adjacent string literals, one per source line. Python
concatenates them with nothing in between, so a literal that ends without a space followed by one
that starts without a space produces "theirH-score" -- which the voice reads as a single nonsense
word and a listener hears as a glitch.

**Forced pauses.** gTTS sends at most a hundred characters per request, and a pause that falls
BETWEEN two requests runs about three quarters of a second -- against roughly a quarter for one
inside a request. That is exactly what a comma is for, so it sounds right wherever the writer put
one, and wrong wherever the character limit put one instead.

This reports the second kind, and the words each will fall after, asked of gTTS itself rather
than guessed. To clear one: put a comma or a full stop where the break wants to be, close enough
to the start of the sentence that the text before it fits in a hundred characters. Then the pause
lands on punctuation and reads as a pause rather than as a fault.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from gtts import gTTS

GTTS_CHUNK_LIMIT = gTTS.GOOGLE_TTS_MAX_CHARS

# Words that begin a phrase rather than end one. A break landing just before one of these reads
# as a phrase boundary; one landing after leaves a preposition stranded, which is what makes a
# break audible as a fault rather than as a pause.
PHRASE_OPENERS = frozenset('''
    a an the and or but nor so yet for of to in on at by with from into onto over under
    that which who whom whose when where while because since although though if unless
    as than about after before between during against across through toward towards
'''.split())


def forced_seam_words(text: str, language: str) -> list[str]:
    """The words gTTS will pause after despite the writer not asking it to.

    gTTS tokenises at punctuation, then merges adjacent tokens into one request of at most a
    hundred characters. A pause inside a request comes out natural; a pause BETWEEN requests is
    two files' padding back to back and runs about three quarters of a second. That is fine where
    the writer put a comma and wrong where the character limit put one, so this reports the
    second kind -- asked of gTTS itself rather than guessed at.
    """
    forced = []
    position = 0
    for piece in gTTS(text, lang=language)._tokenize(text)[:-1]:
        stripped = piece.strip().rstrip('.,;:!?')
        if not stripped:
            continue
        found = text.find(stripped, position)
        if found == -1:
            continue
        position = found + len(stripped)
        following = text[position:].lstrip()[:1]
        if following not in '.,;:!?()-':
            forced.append(' '.join(stripped.split()[-3:]))
    return forced


_NARRATION_DIRECTORY = Path(__file__).resolve().parent.parent


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


def find_forced_pauses(source: str) -> list[tuple[str, str]]:
    """Every pause gTTS will impose mid-phrase, and the words it will fall after."""
    found = []
    for key, line in read_lines(source):
        for words in forced_seam_words(line, 'en'):
            found.append((key, words))
    return found


def read_lines(source: str) -> list[tuple[str, str]]:
    """Each narration key and the whole line it will be spoken as, literals already joined."""
    lines = []
    key = None
    literals: list[str] = []
    for raw in source.splitlines():
        stripped = raw.strip()
        named = re.match(r"^'([a-z_]+)':$", stripped)
        if named:
            if key is not None and literals:
                lines.append((key, ' '.join(''.join(literals).split())))
            key, literals = named.group(1), []
            continue
        if key is not None and stripped.startswith("'"):
            literals.extend(re.findall(r"'([^']*)'", stripped))
    if key is not None and literals:
        lines.append((key, ' '.join(''.join(literals).split())))
    return lines


def main() -> int:
    joined_total = 0
    for path in sorted(_NARRATION_DIRECTORY.glob('*/narration.py')):
        source = path.read_text(encoding='utf-8')
        joined = find_joined_words(source)
        forced = find_forced_pauses(source)
        joined_total += len(joined)
        print(f'{path.parent.name}')
        for line_number, sample in joined:
            print(f'   JOINED at line {line_number}: "{sample}"')
        for key, words in forced:
            print(f'   forced pause in {key}, after "...{words}" '
                  f'-- put a comma or full stop at that break')
        if not joined and not forced:
            print('   clean')
    return 1 if joined_total else 0


if __name__ == '__main__':
    sys.exit(main())
