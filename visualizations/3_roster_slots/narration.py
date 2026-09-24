"""What the roster-slot assignment animation says out loud.

This is the only file to edit for narration. Each key is one beat of the scene, and that beat's
animation plays INSIDE its line: rewriting a line retimes its own beat rather than
desynchronising everything after it. A beat whose line is short waits for the animation, and one
whose line is long holds the last frame until the sentence finishes.

The voice is Alistair, from ElevenLabs, so what is written here is what is spoken -- spell out
anything a reader would say differently from how it is written ('sigma', 'twenty-six').

Punctuation is how long a pause is asked for. Measured across every line he has read for this
set -- 75 pauses that could be attributed to one mark -- a comma runs 0.18s and never went past
0.24s, while a full stop has a median of 0.42s but ranges up to 1.41s depending on where in the
paragraph it falls. A hyphen-dash has too few examples to characterise, and one of them ran to
1.28s.

A semicolon, measured on this file's last line, came out at 0.36s where the full stop it
replaced had run to 1.26s -- and shortened the whole line by a second.

So the ranking is comma, semicolon, full stop, and only the comma is reliably short. A full stop
asks for a pause and lets him choose how long, which is right at the end of a thought and wrong
in the middle of one. Where a small breath is wanted between clauses, a semicolon buys it.

Every line is billed by the character, and a line edited by one word is a new line at full
price. `python visualizations/shared/narration_budget.py` says what a render would spend before
it spends it.
"""

NARRATION = {
    'slots':
        'Under H-scoring, teams need to fit into a pre-determined position structure of basic positions and '
        'flex spots. It figures out how to do this through a sub-algorithm, matching players to positions.',
    'the_roster':
        'Players that are already drafted must each take up a position slot that they are eligible for; '
        'Their ineligible positions are marked with negative infinities so that they are never chosen.',
    'future_picks':
        'Below the line are picks that will be made in the future.',
    'future_rows':
        'Future picks can be anything, but some position slots are worth more than others.',
    'best_slot':
        'The value of a position slot is calculated by applying category weights to position averages. In this case'
        ', guard slots are worth more, perhaps the team is in need of more assists. Flex spots are always worth '
        'the highest of their constituent parts, plus a small bonus for flexibility.',
    'one_slot_each':
        'Every player, including the future picks, must be assigned to one slot, and every slot needs one player filling it. '
        'This kind of problem is called an assignment problem',
    'the_clash':
        'The rules make it impossible for two players to both occupy the same slot, ',
    'clash_two':
        'or for the same player to occupy two slots.',
    'the_greedy_guess':
        'Algorithms can figure out how to do the matching. One simple approach is to go down the line, assigning each player '
        'to the slot they have the highest reward for, or choosing randomly when rewards are equal. ',
    'the_total':
        'And here is the total reward based on that strategy.',
    'solve_it_whole':
        'It is fine, but we can do better. Solving the whole board at once, using a more sophisticated algorithm,'
        ' yields a higher total.',
    'the_swap':
        'The difference is a single trade: Jabari moves to the SF slot, opening up another utility slot.',
    'why_important':
        'This process is important to the algorithm because it maps out the expected positions of future picks, '
        'impacting category totals; in this example, there will likely be a tilt towards guard metrics '
        'like assists, and away from big man metrics like blocks.'

}