"""What the punting animation says out loud.

This is the only file to edit for narration. Each key is one beat of the scene, and that beat's
animation plays INSIDE its line: rewriting a line retimes its own beat rather than
desynchronising everything after it. A beat whose line is short waits for the animation, and one
whose line is long holds the last frame until the sentence finishes.

The voice is gTTS, so what is written here is what is spoken -- spell out anything a reader would
say differently from how it is written ('sigma', 'twenty-six').
"""

NARRATION = {
    'opening':
        'Punting is a natural consequence of two things: the structure of fantasy basketball, and the central limit theorem. '
        'These distributions are expected win differentials between two teams, approximated as bell curves through the CLT. '
        'The yellow shaded region is where the differential favors you, and you win.',
    'score':
        'Add up the area of all the yellow regions to get the expected number of categories you win. With parity across '
        'categories, the total is four and a half.',
    'abandon':
        'Let\'s shift our category investment around. If we take one category, and keep reducing our investment in it'
        ', we lose expected value depending on the thickness of the distribution. As we move it far, the thickness goes down, '
        'so we lose less and less. Meanwhile, we can reallocate to other categories which are still in their thick middles, '
        'gaining back more expected value than we lost.',
    'abandon_again':
        'Give up a second category, and the score goes up again.',
    'abandon_third':
        'And a third, which is still an improvement.',
    'abandon_too_far':
        'This does not continue forever. At the fourth category, there is not enough to gain from the other categories to make '
        'the punt worth it. They are already far to the right, ',
    'settle':
        'So punting is not just a practical decision. There is mathematical rigor behind it.',
}
