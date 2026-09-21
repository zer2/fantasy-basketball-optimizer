"""What the SAVOR animation says out loud.

WIREFRAME. Read by gTTS, not Alistair, while the script is still moving -- see
shared/draft_voice.py. Lines here are drafts and are expected to change.

This is the only file to edit for narration. Each key is one beat of the scene, and that beat's
animation plays INSIDE its line.

The voice reads what is written here literally, so spell out anything a reader would say
differently from how it is written ('sigma', 'mu').
"""

NARRATION = {
    'three_players':
        'Here are three players you could buy in an auction. A star, a decent starter, and a '
        'one dollar flyer at the very end. Each one is a projection, and a projection is a '
        'guess, so each is really a spread of what they might actually deliver.',
    'the_floor':
        'They all share a floor. If any of them turns out worse than the free agents available '
        'all season, you simply drop them and pick somebody else up. So nothing below this line '
        'is ever really yours. You keep the upside and you hand back the downside.',
    'the_question':
        'Now ask the question an auction actually asks. If you could improve one of these '
        'players slightly, how much would it be worth?',
    'the_flyer_half':
        'For the flyer, sitting right at the line, only half of it counts. Half of their '
        'distribution is already below replacement, and improving an outcome you were going to '
        'throw away anyway buys you nothing. A dollar of projection is worth about fifty cents.',
    'the_star_whole':
        'For the star it is almost the whole distribution. Nearly every outcome is one you '
        'actually keep, so nearly every outcome carries the improvement through. A dollar of '
        'projection is worth about ninety nine cents.',
    'the_general_rule':
        'And that is the whole adjustment. An extra unit of projection is worth exactly the '
        'share of a player that stays above replacement.',
    'concentration':
        'Which is why money concentrates at the top of an auction. The same improvement is worth '
        'twice as much on a star as it is on a replacement level flyer, so the stars are worth '
        'more than their projections suggest, and the players at the bottom are worth close to '
        'the dollar it costs to replace them.',
}
