"""What the Z-score versus G-score animation says out loud.

This is the only file to edit for narration. Each key is one beat of the scene, and that beat's
animation plays INSIDE its line: rewriting a line retimes its own beat rather than
desynchronising everything after it. A beat whose line is short waits for the animation, and one
whose line is long holds the last frame until the sentence finishes.

The voice is Alistair, from ElevenLabs, so what is written here is what is spoken -- spell out
anything a reader would say differently from how it is written ('sigma', 'twenty-six'). The
teams are numbered rather than lettered because a lone capital letter cannot be made to read as
one: every spelling of it comes out as a word instead.

Every line is billed by the character, and a line edited by one word is a new line at full
price. `python visualizations/shared/narration_budget.py` says what a render would spend before
it spends it.

Players named here have to be players this scene actually deals -- the rosters are real and the
faces are on screen while the line is spoken. Week numbers are the season's own weeks, and
nothing checks them: the prepared data keeps each player's weekly totals as a bare list of
numbers with no record of which week each came from. A week number is illustration, and only
has to be a week the season had.
"""

# ── The spoken track ─────────────────────────────────────────────────────────────────
# Edit only this block. Each beat plays inside the line that covers it, so a rewritten line
# retimes its own beat rather than desynchronising everything after it.
#
# PLACEHOLDER lines are mine, holding the timing until the real ones are written.

NARRATION = {
    'fixed_matchup':
        'The logic of Z-scores assumed that the outcome of a matchup was entirely determined '
        'by one thing: which players were on which team. But that is not true, because players '
        'do not have the same stats every week.',
    'fixed_matchup_result':
        'This simulation takes two teams, and randomizes the performances of each player '
        'on each team by independently sampling from weeks of a real season. The idea is to roughly simulate'
        ' the spectrum of possibilities for how players could perform on any given week. The result is another bell curve, '
        'this time centered at how much more team 1 tends to score on average.',
    'fixed_matchup_win_rate':
        'Instead of winning every time, as their roster total might suggest, team 1 wins just a '
        'majority of the time. Their margin is determined by how spread out the distribution is; '
        'if the distribution was tighter, less of it would be below zero.',
    'both_vary':
        'Now let\'s add this mechanism back to the original Z-score simulation. We know neither which '
        'players will be on which team nor how players will perform in any given week. Players are chosen '
        'randomly and then their weeks are chosen randomly as well. ',
    'both_vary_result':
        'The result is again a bell curve, this time, quite wide. It is wide because it incorporates both '
        'sources of variance.',
    'opening':
        'The weight of a category is determined by the height in the middle, which is a function of the '
        'standard deviation.',
    'cross_player':
        'For Z-scores, that was determined by the standard deviation between player averages.',
    'week_to_week':
        'The additional variance that we need to add in comes from week-to-week variation. This is '
        'across all players, so it is not exactly the same as what we got from two specific teams',
    'both':
        'The result is a curve that is wider than both of the originals, because it has both of their variances',
    'the_question':
        'When distributions are added together, their standard deviations squared are added together. So the new '
        'standard deviation is the square root of the two original standard deviations squared, like the length of a hypotenuse.',
    'g_scores' :
        'Subbing in this new standard deviation to Z-scores, we get G-scores. They work a bit better than Z-scores in simple simulations, '
        'though they are still massive simplifications; there is no perfect way to make static scores for a dynamic game.'

}
