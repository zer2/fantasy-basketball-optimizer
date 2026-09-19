"""What the seed menu animation says out loud.

This is the only file to edit for narration. Each key is one beat of the scene, and that beat's
animation plays INSIDE its line: rewriting a line retimes its own beat rather than
desynchronising everything after it. A beat whose line is short waits for the animation, and one
whose line is long holds the last frame until the sentence finishes.

The voice is gTTS, so what is written here is what is spoken -- spell out anything a reader would
say differently from how it is written ('sigma', 'twenty-six').
"""

NARRATION = {
    'surface':
        'This is another H-scoring surface. It has multiple peaks corresponding to different '
        'kinds of builds',
    'three_seeds':
        'The algorithm does not start in one pre-determined place. Instead, it tries a few '
        'different seeds, in the directions of various punts.',
    'the_choice':
        'It scores them where they stand, takes the best one, and climbs from there.',
    'the_descent':
        'This way, gradient descent is likely to find the best peak, instead of going up '
        'a less promising one and getting stuck.',
}
