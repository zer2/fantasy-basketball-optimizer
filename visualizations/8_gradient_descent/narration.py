"""What the gradient descent animation says out loud.

This is the only file to edit for narration. Each key is one beat of the scene, and that beat's
animation plays INSIDE its line: rewriting a line retimes its own beat rather than
desynchronising everything after it. A beat whose line is short waits for the animation, and one
whose line is long holds the last frame until the sentence finishes.

The voice is gTTS, so what is written here is what is spoken -- spell out anything a reader would
say differently from how it is written ('sigma', 'twenty-six').
"""

NARRATION = {
    'surface':
        'H-scoring provides a function for overall H-score based on the input parameters, including category weights. '
        'This is a slice of what the function looks like. The height of the surface is the H-score based on the weights.',
    'the_start':
        'We want to find the peak. In this simple example, it would be easy enough to just traverse the grid looking for it. '
        'In practice, we are optimizing over many more dimensions, so we need a more efficient method. That method is '
        'starting in one place and climbing up the hill.',
    'the_climb':
        'Gradient descent (or in this case, ascent) takes repeated steps, each time in the direction that is steepest upwards',
    'the_top':
        'We stop when we cannot move up much more. After thirty steps, we usually are quite close to the top. This works '
        'well in general, even across many dimensions.',
}
