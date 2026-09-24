"""What a render would cost, before it costs it.

The voice is billed per character against a monthly allowance, so the question worth answering
before rendering is not "how long will this take" but "how much of the month does this spend".
This answers it without sending anything: it compares the lines in every narration.py against
what is already cached, and reports only the difference.

    python visualizations/shared/narration_budget.py

A line already in the cache is free however many times it is rendered. A line that has been
edited -- even by one word -- is a new line, and costs its whole length again.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_VISUALIZATIONS = Path(__file__).resolve().parent.parent
# Where manim-voiceover keeps what has been bought. It derives this as media_dir/voiceovers, and
# manim.cfg points media_dir at render_cache, so the two have to agree -- this path was left
# behind at the old media/ name once and reported every line as unpaid, which is the most
# alarming thing this script can say and it was not true.
_CACHE_FILE = _VISUALIZATIONS / 'render_cache' / 'voiceovers' / 'cache.json'

# What the plan allows per month. Not enforced anywhere -- it is here so the report can say how
# much of the month a render would take, which is the number a decision is actually made on.
MONTHLY_CHARACTER_ALLOWANCE = 40_000


def cached_lines() -> set[str]:
    """Every line the paid voice has already spoken, as normalised text.

    A missing cache file raises rather than reading as an empty cache. The two are indistinguishable
    in the report -- both say every line must be bought -- but they mean opposite things: one is a
    set that has genuinely never been rendered, the other is this script looking in the wrong place
    while the narration sits paid for somewhere else.
    """
    if not _CACHE_FILE.exists():
        raise FileNotFoundError(
            f'No narration cache at {_CACHE_FILE}. If narration really has never been bought, '
            f'this file appears on the first render; if it has, this path is wrong and the report '
            f'below would claim a full-price render for lines that are already paid for.')
    return {
        ' '.join(entry['input_text'].split())
        for entry in json.loads(_CACHE_FILE.read_text(encoding='utf-8'))
        if entry.get('input_data', {}).get('service') == 'elevenlabs'
    }


def narration_of(folder: Path) -> dict[str, str]:
    spec = importlib.util.spec_from_file_location('narration', folder / 'narration.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NARRATION


def main() -> int:
    cached = cached_lines()
    owed = 0
    for folder in sorted(_VISUALIZATIONS.glob('[1-8]_*')):
        pending = [(key, ' '.join(line.split()))
                   for key, line in narration_of(folder).items()
                   if ' '.join(line.split()) not in cached]
        if not pending:
            print(f'{folder.name:22} free -- every line is cached')
            continue
        cost = sum(len(line) for _, line in pending)
        owed += cost
        print(f'{folder.name:22} {cost:5} characters for {len(pending)} line(s): '
              f'{", ".join(key for key, _ in pending)}')

    print()
    if owed:
        print(f'A render now would send {owed} characters, '
              f'{owed / MONTHLY_CHARACTER_ALLOWANCE:.1%} of the monthly allowance.')
    else:
        print('A render now would send nothing: every line is already paid for.')
    print(f'{len(cached)} lines cached, {sum(len(line) for line in cached)} characters bought '
          f'so far.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
