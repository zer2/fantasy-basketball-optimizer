"""What the Most Categories animation says out loud.

WIREFRAME. Read by gTTS, not Alistair, while the script is still moving -- see
shared/draft_voice.py. Lines here are drafts and are expected to change.

This is the only file to edit for narration. Each key is one beat of the scene, and that beat's
animation plays INSIDE its line: rewriting a line retimes its own beat rather than
desynchronising everything after it.

The voice reads what is written here literally, so spell out anything a reader would say
differently from how it is written ('sigma', 'two to the ninth').
"""

NARRATION = {
    'setup':
        'In Most Categories scoring, winning eight categories is worth exactly the same as '
        'winning five. All that matters is whether you take the majority.',
    'tree':
        'So to know your chances, the algorithm has to consider every way the week could go. '
        'With nine categories, each of which you either win or lose, that is two to the ninth '
        'possible outcomes. Five hundred and twelve of them.',
    'majority':
        'Every one of those outcomes is either a win or a loss, depending on whether you took '
        'at least five of the nine. Your chance of winning the matchup is the total probability '
        'of all the branches that end in a majority.',
    'explosion':
        'Adding one more category doubles the tree. This is far too slow to run for every '
        'candidate player, thousands of times a draft.',
    'dynamic':
        'But the whole tree is not necessary. All that matters about a branch is how many '
        'categories it has won so far, not which ones.',
    'collapse':
        'So the algorithm keeps a running tally instead: the probability of having won zero '
        'categories, one category, two, and so on. Each new category updates that tally in a '
        'single pass.',
    'tipping':
        'This also answers a more useful question. A category only matters when it is the one '
        'that decides the matchup: when the other eight leave you exactly on the boundary. '
        'The chance of that is called the tipping point probability, and it is what tells the '
        'algorithm how much a category is worth.',
    'punting':
        'Categories you are already certain to win, or certain to lose, almost never tip the '
        'result. That is why Most Categories scoring punts hardest of the three formats.',
}
