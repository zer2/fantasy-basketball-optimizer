"""What the Rotisserie animation says out loud.

WIREFRAME. Read by gTTS, not Alistair, while the script is still moving -- see
shared/draft_voice.py. Lines here are drafts and are expected to change.

This is the only file to edit for narration. Each key is one beat of the scene, and that beat's
animation plays INSIDE its line.

The voice reads what is written here literally, so spell out anything a reader would say
differently from how it is written ('sigma', 'fifty-fifty').
"""

NARRATION = {
    # -- The scale, assumed rather than taught -----------------------------------------
    'the_scale':
        'A Rotisserie season comes down to one number. Nine categories, twelve points for '
        'winning one and one point for finishing last, so a season lands somewhere between nine '
        'and a hundred and eight. An average team scores about fifty eight.',

    # -- The bar you actually have to clear --------------------------------------------
    'the_bar':
        'But your score is not what decides it. What decides it is whether you beat the best of '
        'the other eleven teams, and that best score is itself uncertain. It usually lands around '
        'seventy five.',
    'simulate':
        'The blue here is how often you finish on each total. And laid over it in yellow is how '
        'often that total was actually enough to win the league. Low scores are never enough, so '
        'they stay blue. High ones almost always are.',
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
        'The second commits. It goes after five categories hard enough to win them nine times in '
        'ten, and gives up the other four just as hard. Very little is left in doubt, so the '
        'season total barely moves from one year to the next.',
    'the_result':
        'Now run both. And here is the part worth pausing on. The committed team actually scores '
        'more points on average, sixty three against fifty eight. It is the better team by the '
        'obvious measure.',
    'conclusion':
        'And it wins the league less often. Five percent against nearly eight. It is packed too '
        'tightly around its own average to reach up to the bar as often. That is why the '
        'Rotisserie algorithm keeps categories closer to fifty-fifty, and why it punts far less '
        'than the head to head formats do.',
}
