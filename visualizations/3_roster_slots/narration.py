"""What the roster-slot assignment animation says out loud.

This is the only file to edit for narration. Each key is one beat of the scene, and that beat's
animation plays INSIDE its line: rewriting a line retimes its own beat rather than
desynchronising everything after it. A beat whose line is short waits for the animation, and one
whose line is long holds the last frame until the sentence finishes.

The voice is gTTS, so what is written here is what is spoken -- spell out anything a reader would
say differently from how it is written ('sigma', 'twenty-six').
"""

NARRATION = {
    'slots':
        'Under H-scoring, entire teams need to fit into a pre-determined position structure. That can '
        'include the basic positions and flex spots. ',
    'the_roster':
        'Players that are already drafted must take up one of those slots, each. They can only be assigned to '
        'slots they are eligible for. Their ineligible positions are marked with negative infinities, ensuring '
        'the algorithm always chooses eligible positions for them.',
    'future_picks':
        'Below the line are picks that will be made in the future.',
    'future_rows':
        'The goal is to leave spots open for future picks for positions that are desirable, based on the team\'s build. ',
    'best_slot':
        'Desirability is calculated by applying category weights to position averages. In this case'
        ', guard slots are worth more, perhaps because the team is punting field goal percentage. Flex spots are worth '
        'more than their constituent parts, because they allow for flexibility.',
    'one_slot_each':
        'Every player, including the future picks, must be assigned to one slot. And every slot needs one player filling it. '
        'This kind of problem is called an assignment problem',
    'the_clash':
        'The rules makes it impossible for two players to both occupy the same position slot. So this solution is not permissible.',
    'the_greedy_guess':
        'One simple approach is to go down the line, assigning each player to the slot they have the highest reward for, or choosing '
        'randomly when rewards are equal. ',
    'the_total':
        'And here is the total reward based on that strategy.',
    'solve_it_whole':
        'It is fine, but we can do better. Solving the whole board at once yields a higher total',
    'the_swap':
        'The difference is a single trade: Jabari moves to the SF slot, opening up another utility slot.',
    'why_important':
        'This is important to the algorithm because it maps out the potential positions of future picks, '
        'impacting category totals. In this example, there will likely be a tilt towards guard metrics '
        'like assists, and away from big man metrics like blocks.'

}