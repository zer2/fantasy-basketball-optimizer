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
        'But the whole tree is not necessary. Think of the week as a walk. Every category you '
        'win is a step up, and every category you lose is a step down. Nine categories, nine '
        'steps. You take the majority exactly when you finish above where you started.',
    'collapse':
        'And nothing about a path matters except where it has reached. Two paths that arrive at '
        'the same height are worth the same from there on. So the algorithm never follows the '
        'paths. It keeps a single column of probabilities, one for each height, and every '
        'category translates that column into the next one in a single pass.',
    'tipping':
        'This also answers a more useful question. A category only decides the matchup when '
        'everything else leaves you exactly level. How often that happens is called the tipping '
        'point probability, and it is what tells the algorithm how much a category is worth.',
    'backward':
        'To get it, the algorithm runs the walk a second time, from the other end. The same nine '
        'categories, taken in the opposite order.',
    'the_cut':
        'Now pick a category, and cut the walk open there. On the left is everything before it, '
        'carried forward. On the right is everything after it, carried back.',
    'convolution':
        'For this category to be the one that decides the matchup, the two halves have to meet '
        'level. So pair every height on the left with the opposite height on the right, multiply, '
        'and add them up. That sum is the tipping point.',
    'slide':
        'And now the reason for running the walk twice. Moving the cut to a different category '
        'does not need either walk to be run again. The two sweeps are already done, and every '
        'category is just a different place to cut them open.',
    'punting':
        'Categories you are already certain to win, or certain to lose, almost never tip the '
        'result. That is why Most Categories scoring punts hardest of the three formats.',
}
