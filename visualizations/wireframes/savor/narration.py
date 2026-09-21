"""What the SAVOR animation says out loud.

WIREFRAME. Read by gTTS, not Alistair, while the script is still moving -- see
shared/draft_voice.py. Lines here are drafts and are expected to change.

This is the only file to edit for narration. Each key is one beat of the scene, and that beat's
animation plays INSIDE its line.

The voice reads what is written here literally, so spell out anything a reader would say
differently from how it is written ('sigma', 'mu').
"""

NARRATION = {
    'projection_is_a_guess':
        'A projection is not a fact. It is a guess, and the season is what settles it. So the '
        'value a player actually delivers is a distribution around what you projected.',
    'the_floor':
        'But there is a floor under that distribution. If a player turns out to be worse than '
        'the free agents available all season, you simply drop them and pick up a free agent. '
        'You never actually receive the bad outcomes.',
    'truncation':
        'So the value you get is the projection, cut off on the left. Everything below the '
        'replacement line collapses onto it.',
    'two_players':
        'Now compare two players. A star, projected far above replacement, and a marginal '
        'player, projected only a little above it.',
    'star_unaffected':
        'For the star, the floor is almost irrelevant. Nearly the whole distribution sits above '
        'it, so the cut removes almost nothing, and what you expect to get is very close to '
        'what you projected.',
    'marginal_shrinks':
        'For the marginal player it is completely different. A large part of their distribution '
        'is below the line, and all of that is worth exactly replacement level, not what you '
        'projected. Their edge over a free agent is much smaller than it looks.',
    'the_flyer':
        'And this is the comparison that matters, because the alternative is never nothing. At '
        'the end of an auction you can take a flyer for a dollar: a player projected at exactly '
        'replacement, whose upside you keep and whose downside you can drop.',
    'the_subtraction':
        'That flyer is worth something on its own. So a player is only worth paying for to the '
        'extent that they beat it. Subtract the flyer, and what is left is the SAVOR value.',
    'conclusion':
        'The effect is that money concentrates at the top. Stars keep nearly their full '
        'projected value, while the players at the bottom of the auction are worth close to the '
        'dollar it costs to replace them.',
}
