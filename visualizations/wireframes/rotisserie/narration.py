"""What the Rotisserie animation says out loud.

WIREFRAME. Read by gTTS, not Alistair, while the script is still moving -- see
shared/draft_voice.py. Lines here are drafts and are expected to change.

This is the only file to edit for narration. Each key is one beat of the scene, and that beat's
animation plays INSIDE its line.

The voice reads what is written here literally, so spell out anything a reader would say
differently from how it is written ('sigma', 'fifty-fifty').
"""

NARRATION = {
    # -- What a Rotisserie score even is -----------------------------------------------
    'the_points':
        'Rotisserie is not a series of matchups. You play every team at once, all season. At the '
        'end, each category is ranked, and you score points for where you finished.',
    'the_scale':
        'In a twelve team league you get twelve points for winning a category, down to one point '
        'for finishing last. Nine categories, so a perfect season is a hundred and eight points, '
        'and an average one is around fifty eight.',

    # -- The bar you actually have to clear --------------------------------------------
    'the_bar':
        'But your score is not what decides it. What decides it is whether you beat the best of '
        'the other eleven teams, and that best score is itself uncertain. It usually lands around '
        'seventy five.',
    'simulate':
        'So here are a thousand simulated seasons. Each one is a dot: where you finished, against '
        'what it took to win that year. Every dot above the line is a season you won the league.',
    'its_hard':
        'An average team wins about eight percent of the time. That is what winning a league '
        'looks like. You are not trying to be a bit above average. You are trying to reach up '
        'to a bar that sits well above you.',

    # -- Why spread is worth having ----------------------------------------------------
    'widen':
        'Which changes what is worth buying. If the algorithm could make your season more '
        'uncertain, without improving it at all, your chances would go up.',
    'why_wide':
        'Because finishing a little below the bar and finishing far below it are the same '
        'result. There is nothing to lose on the left. But the extra spread on the right reaches '
        'into the seasons you win.',

    # -- Where spread comes from -------------------------------------------------------
    'two_builds':
        'So compare two ways of building a team. For each one, here is the chance of beating '
        'every opponent in every category.',
    'coin_flips':
        'The first is competitive everywhere. Every category is close to a coin flip against '
        'every rival. Nothing is settled, so the season total swings widely.',
    'certainties':
        'The second commits. It locks in five categories and abandons four. Those are near '
        'certainties, and certainties do not vary, so the total barely moves from year to year.',
    'the_result':
        'Now run both. And here is the part worth pausing on. The committed team actually scores '
        'more points on average, sixty four against fifty eight. It is the better team by the '
        'obvious measure.',
    'conclusion':
        'And it wins the league almost never. Two tenths of one percent, against nearly eight '
        'percent for the balanced team. It is pinned too tightly to its own average ever to '
        'reach the bar. That is why the Rotisserie algorithm holds categories near fifty-fifty, '
        'and why it punts far less than the head to head formats do.',
}
