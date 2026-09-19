"""What the Z-score versus G-score animation says out loud.

This is the only file to edit for narration. Each key is one beat of the scene, and that beat's
animation plays INSIDE its line: rewriting a line retimes its own beat rather than
desynchronising everything after it. A beat whose line is short waits for the animation, and one
whose line is long holds the last frame until the sentence finishes.

The voice is gTTS, so what is written here is what is spoken -- spell out anything a reader would
say differently from how it is written ('sigma', 'twenty-six'). A lone capital letter is read
as a word rather than as a letter, so the team on the left is written 'team Ay': measured, 'A'
adds a tenth of a second to the line and 'Ay' adds two tenths, which is the difference between a
swallowed schwa and the letter being said.
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
        'This simulation randomizes each player\'s performance by sampling from weeks of '
        'a real season. The result is another bell curve. Instead of winning every time, as '
        'their roster might suggest, team Ay wins just a majority of the time. Their margin '
        'is determined by how spread out the distribution is.',
    'both_vary':
        'Now let\'s add this mechanism back to the original Z-score simulation. We know neither which '
        'players will be on which team nor how players will perform in any given week.',
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

}
