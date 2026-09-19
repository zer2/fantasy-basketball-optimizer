"""What the category weights animation says out loud.

This is the only file to edit for narration. Each key is one beat of the scene, and that beat's
animation plays INSIDE its line: rewriting a line retimes its own beat rather than
desynchronising everything after it. A beat whose line is short waits for the animation, and one
whose line is long holds the last frame until the sentence finishes.

The voice is gTTS, so what is written here is what is spoken -- spell out anything a reader would
say differently from how it is written ('sigma', 'twenty-six').
"""

NARRATION = {
    'pool':
        'Every dot represents a hypothetical player and their expected stats across two categories, ignoring other '
        'categories for this example. We assume that players with above average stats will have already been taken.',
    'score_lines':
        'We will choose players based on weights, represented by lines. We push the lines down until we find a '
        'player on the line; that means they have the highest score with those weights.',
    'experiments':
        'Using different choices for weights leads to different players getting picked. The blue weights, which '
        'value free throws, tend to hit players who have good free throw rates.',
    'densities':
        'If we do many of these simulations, we get a cloud of outcomes for each weighting, and a center of mass '
        'for both of them. H-scoring estimates what this center of mass will be based on weights.',
}
