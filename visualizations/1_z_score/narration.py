"""What the Z-score animation says out loud.

This is the only file to edit for narration. Each key is one beat of the scene, and that beat's
animation plays INSIDE its line: rewriting a line retimes its own beat rather than
desynchronising everything after it. A beat whose line is short waits for the animation, and one
whose line is long holds the last frame until the sentence finishes.

The voice is Alistair, from ElevenLabs, so what is written here is what is spoken -- spell out
anything a reader would say differently from how it is written ('sigma', 'twenty-six').

Every line is billed by the character, and a line edited by one word is a new line at full
price. `python visualizations/shared/narration_budget.py` says what a render would spend before
it spends it.

Two beats inside 'two_teams' are triggered by phrases rather than by the top of the line: the
simulation starts on "What you will notice". Rewording that phrase moves the trigger with it;
deleting it stops the render with a message saying so.
"""

NARRATION = {
    'pool':
        'Real drafting behavior is complicated. To make a justification for Z-scores, we need to imagine a simpler '
        'version of fantasy basketball, which is that players are chosen randomly from a pool of fantasy-relevent players. '
        'Obviously this is not perfectly accurate to real fantasy basketball, but it is not too crazy either, and it has '
        'convenient properties that will make Z-scores simple.',
    'two_teams':
        'Let\'s simulate this version of fantasy basketball and see what happens to the margin in the points category '
        'between two random teams. What you will notice is that after sampling many random teams, the distribution '
        'of the margin begins to look like a bell curve. This happens because of the central limit theorem, which says '
        'that when you add or subtract a bunch of random numbers together, as we are doing with point averages, '
        'the resulting distribution will be close to a bell curve.',
    # Split in two so each half has its own act to run under. As one line it was twenty-four
    # seconds of audio over four seconds of animation.
    'bar_height':
        'Notice how the height in the middle tells us how important a few additional points are. '
        'If we move over to the right through a high bar, that is a lot of scenarios where the '
        'extra points helped team 1 win.',
    'simple_expression':
        'So we can say, the height in the middle is how important the stat is. Fortunately, there '
        'is a simple expression for the height of the middle, based on the fact that this is '
        'roughly a bell curve. We can ignore the square root of two pi because it is a constant across categories. ',
    'inverse_sigma':
        'We can also apply another useful fact: the standard deviation of the total is proportional to '
        'that of the components. Therefore, category importance is inversely proportional to the '
        'category\'s standard deviation across players.',
    'numerator':
        'To make the average zero, we can have the top be the difference from the average.',
    'z_score':
        'This is the formula for a Z-score, for a counting statistic like points.',
    'ratio_statistics':
        'If you are wondering about the percentages like field goal percent, we can justify their Z-score formulas with a bit more work. '
        'Approximating that the total volume stays constant, an additional player changes the overall percentage '
        'by a factor of their volume times how far apart from average they are. That means we can use the same math as before, just '
        'factoring in volume. This lines up with the Z-score formula, which multiplies percentages by volume over average volume.',
    'conclusion': 
        'So Z-scoring does have a justification. It is not perfect, but it has stood the test of time as a '
        'reasonable way to evaluate players'
}
