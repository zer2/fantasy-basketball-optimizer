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
        'When we take players in an auction before a season, we do not know exactly how they will perform during the season. We might expect them '
        'to have some level of value, but they could end up well above or below that mark. If we think of players as having a Normally distributed '
        'error in their value projection, then their values are Normal distributions. Here are some potential distributions, for a star, a decent '
        'starter, and a flyer.',
    'the_floor':
        'Here\'s a complication: if a player ends up below the replacement level, we are not going to keep them. So if we draw a line on here for the '
        'replacement level, the parts of the distribution that are to the left of the line do not actually help us. With a player who is a flyer or just a bit'
        ' better than a flyer, we are banking on their positive outcome; if they dip a little they will not be worth anything to us. ',
    'the_flyer_half':
        'This affects how much value we can actually realize from investing into players. Imagine starting from a flyer player and adding '
        'one dollar of expected value. The distribution moves, but only about half of it is above replacement, so '
        'we only get to capture that extra dollar of value about half the time. That one dollar is worth an expected fifty cents of real value.',
    'the_star_whole':
        'On the other hand, increasing the value of a star is almost definitely going to pay off, since they will almost definitely stay on the team. '
        'A dollar of projection for a star is worth about ninety nine cents.',
    'the_general_rule':
        'SAVOR adjusts for this asymmetry by calculating the expected realized values of all players relative to flyers. It then scales the values back up '
        'so the total amount of value still equals the total in the pot.',
    'concentration':
        'The SAVOR calculation concentrates value at the top of the auction, because marginal increases at the top end are worth '
        'almost twice as much as marginal increases on the low end. This matches the intuition that you ought to invest more in the '
        'players you expect to stay on your team for the long term. ',
}
