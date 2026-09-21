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
    # -- A week, and who took it -------------------------------------------------------
    'a_week':
        'Here is one week of a head to head matchup. Nine categories, and each one goes to '
        'whichever team put up the better number that week.',
    'count_them':
        'Team one took six of the nine, so team one wins the matchup.',
    'another_week':
        'A different week, between the same two teams. This time it goes the other way, and '
        'team two takes it.',
    'all_or_nothing':
        'And here is the thing that shapes everything the algorithm does. Winning six categories '
        'and winning all nine are worth exactly the same: one win. Only the majority counts.',

    # -- What has to be added up -------------------------------------------------------
    'the_table':
        'So the question is not how many categories you win on average. It is how often you take '
        'at least five of them. Every combination of nine wins and losses is one row, and each '
        'row is either a win for you or a loss.',
    'how_many_rows':
        'There are two to the ninth of those rows. Five hundred and twelve. Adding one more '
        'category doubles it, and this has to run for every candidate player, thousands of times '
        'in a draft.',

    # -- The walk ----------------------------------------------------------------------
    'dynamic':
        'But the rows do not all need visiting. Think of the week as a walk. Every category you '
        'win is a step up, and every category you lose is a step down. Nine categories, nine '
        'steps. You take the majority exactly when you finish above where you started.',
    'collapse':
        'And nothing about a path matters except where it has reached. Two paths that arrive at '
        'the same height are worth the same from there on. So the algorithm never follows the '
        'paths. It keeps a single column of probabilities, one for each height, and every '
        'category translates that column into the next one in a single pass.',

    # -- What a category is worth ------------------------------------------------------
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
