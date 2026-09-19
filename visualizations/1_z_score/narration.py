"""What the Z-score animation says out loud.

This is the only file to edit for narration. Each key is one beat of the scene, and that beat's
animation plays INSIDE its line: rewriting a line retimes its own beat rather than
desynchronising everything after it. A beat whose line is short waits for the animation, and one
whose line is long holds the last frame until the sentence finishes.

The voice is gTTS, so what is written here is what is spoken -- spell out anything a reader would
say differently from how it is written ('sigma', 'twenty-six').
"""

NARRATION = {
    'pool':
        'Real drafting behavior is complicated. To make a justification for Z-scores, we need to imagine a simpler '
        'version of fantasy basketball, which is that players are chosen randomly from a pool of fantasy-relevent players. '
        'Obviously this is not perfectly accurate to real fantasy basketball, but it is not too crazy either, and it has '
        'convenient properties that will make Z-scores simple.',
    'two_teams':
        'For a weekly matchup, we have two teams of thirteen players chosen randomly, for a '
        'total of twenty-six players. That is quite a few, and it means that the central limit '
        'theorem comes into play, which says that when you add or subtract a bunch of random '
        'numbers together, the end result looks like a random bell curve.',
    # Split in two so each half has its own act to run under. As one line it was twenty-four
    # seconds of audio over four seconds of animation.
    'bar_height':
        'Notice how the height in the middle tells us how important a single additional point is. '
        'If we move over to the right through a high bar, that is a lot of scenarios where the '
        'extra point helped us win.',
    'simple_expression':
        'So we can say, the height in the middle is how important the stat is. Fortunately, there '
        'is a simple expression for the height of the middle, based on the fact that this is '
        'roughly a bell curve. ',
    'inverse_sigma':
        'We can apply another useful fact: the standard deviation of the total is proportional to '
        'that of the components. Therefore, category importance is inversely proportional to the '
        'category\'s standard deviation across players.',
    'numerator':
        'To make the average zero, we can have the top be the difference from the average.',
    'z_score':
        'This is the formula for a Z-score. It is not perfect, but it has stood the test of time as a '
        'reasonable way to evaluate players',
}
