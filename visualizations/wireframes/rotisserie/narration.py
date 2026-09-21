"""What the Rotisserie animation says out loud.

WIREFRAME. Read by gTTS, not Alistair, while the script is still moving -- see
shared/draft_voice.py. Lines here are drafts and are expected to change.

This is the only file to edit for narration. Each key is one beat of the scene, and that beat's
animation plays INSIDE its line.

The voice reads what is written here literally, so spell out anything a reader would say
differently from how it is written ('sigma', 'fifty-fifty').
"""

NARRATION = {
    'the_bar':
        'Rotisserie is not a series of matchups. You are not trying to beat one opponent this '
        'week; you are trying to finish ahead of everyone, over a whole season.',
    'winning_total':
        'So what matters is the total you would need to win the league. That total is itself '
        'uncertain, because it depends on how the luckiest manager happens to do. Call it a '
        'distribution.',
    'your_total':
        'Your own season total is a distribution too, centred on what the algorithm expects '
        'you to score.',
    'the_overlap':
        'You win the league when your total lands above the bar. That is the overlap between '
        'the two curves, and it is small. Winning a league is hard.',
    'widen':
        'Now here is the part that is not obvious. If the algorithm could make your curve '
        'wider, without moving its centre at all, your chance of winning would go up.',
    'why_wide':
        'A wider curve loses you nothing that matters. Finishing a little below average and '
        'finishing far below average are both simply losing the league. But the extra width on '
        'the right reaches into the region where you win.',
    'two_builds':
        'So consider two ways of building a team, with the same expected total.',
    'coin_flips':
        'The first is competitive everywhere. Nine categories, each close to a coin flip. Those '
        'are the most uncertain outcomes there are, so the totals they produce are spread wide.',
    'certainties':
        'The second locks in some categories and abandons others. Those are near certainties, '
        'and certainties do not vary. The total is pinned close to its average.',
    'conclusion':
        'The first team wins the league more often, despite the identical expectation. That is '
        'why the Rotisserie algorithm keeps categories near fifty-fifty, and why it punts far '
        'less than the head to head formats do.',
}
