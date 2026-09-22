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
        'There are no matchups in Rotisserie; it\'s you against everyone else simultaneously. That means you '
        'need a strong performance to even have a chance of winning. The average team in a typical 9-cat 12-team '
        'league scores 58.5 points. The bar needed to be the top team is always going to be far above that.',
    # -- The bar you actually have to clear --------------------------------------------
    'the_bar':
        'You cannot know exactly what the bar is going to be before the season happens. We can think of it as a  '
        'distribution, centered somewhere far above the mean.',
    'simulate':
        'The blue here represents the score distribution of an average manager. And laid over it in yellow is how '
        'often the scores would actually end up being enough to win the league. Low scores are never enough, so '
        'they stay blue. It takes a high score to have any shot.',
    'its_hard':
        'This is only a small fraction of the distribution. The left-hand side of the distribution is irrelevant; what '
        'matters is maximizing the area on the right that has some yellow. That means you should be willing to give up '
        'quite a bit of expected value on the left side if it leads to an increased possibility of getting to the far right '
        'side.',
    # -- Why spread is worth having ----------------------------------------------------
    'widen':
        'So we don\'t just care about moving the mean of our distribution up. We want a high variance too, and sometimes '
        'it will make sense to trade off a little bit of mean for a lot of variance.',
    # -- Where spread comes from -------------------------------------------------------
    'two_builds':
        'The way that we control variance is through the kinds of builds that we design. We can increase variance by leaving the '
        'possibility of winning many points in many categories open. Consider two ways to build a team- one where every individual '
        'fantasy point is a coinflip, and one in which most are nearly guaranteed wins, and the rest are nearly guaranteed losses.',
    'coin_flips':
        'The first has an average expected value and strong variance.',
    'certainties':
        'The second has a higher expected value, with much lower variance. This would be an amazing Most Categories team, '
        'almost guaranteed to win every matchup.',
    'conclusion':
        'But in Rotisserie, it is the first team that wins the league more often. The team that punts heavily is too concentrated '
        'around its mean; it did not leave open the possibility of doing broadly well across all categories, which made its only '
        'winning scenario nearly impossible.',
}
