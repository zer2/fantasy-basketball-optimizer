"""What the cash packing animation says out loud.

Read by the shipped voice (shared/narration_voice.py); a changed line is bought again on the next render.

This is the only file to edit for narration. Each key is one beat of the scene, and that beat's animation plays INSIDE
its line. Once a line is written, the scene can cue its pictures to phrases in it (shared/narration_timing).

The voice reads what is written here literally, so spell out anything a reader would say differently from how it is
written ('r one', 'two hundred dollars').
"""

NARRATION = {
    # -- Packing a budget is not trivial ---------------------------------------------------------
    'packing':
        'Spending money efficiently in an auction can be difficult. Assuming that players have agreed-upon dollar '
        'values adding up to 200 per drafter, to get your full 200 dollars worth, you need to find thirteen '
        'players that are worth 200 dollars combined. You also want your set of players to be positionally '
        'balanced and to fit a coherent punt build. In a real auction you need to balance all this on the fly, '
        'because your valuations of players and those of other drafters change with context, and that makes it '
        'impossible to predict what every player will cost. With all these complications, it is not obvious how '
        'to get the full value out of your money.',
    # -- It gets harder as the options run out ---------------------------------------------------
    'fewer_options':
        'It gets harder as the number of players left in the pool decreases. There are fewer ways to make things '
        'work, and therefore fewer avenues for spending your money efficiently.',
    # -- The ratio -------------------------------------------------------------------------------
    'ratio':
        'Eventually, the problem becomes acute when the best available options are worth less to you than the '
        'amount of money you have left. At that point, it becomes impossible to spend your money efficiently.',
    # -- Simple scaling --------------------------------------------------------------------------
    'simple_scaling':
        'When that happens, you might as well plan to overbid on the remaining players, so that you spend your '
        'money on the best options, and don\'t end up with even worse options. Scaling up your valuations of the '
        'remaining players by the ratio of your remaining cash to the value of the best players for your open '
        'slots gives you a way to spend the rest of your money. When this "cash ratio" is below 1, there is no '
        'scaling, so effectively this is a multiplier that gets applied as the maximum of one and the cash ratio. '
        'Applying it makes it less likely that you leave money on the table. But this is far from a complete fix, '
        'because you will likely pay far more for the last few players than they are worth to you. It would have '
        'been better to pay a bit more earlier, filling up your team before you ran out of options, and avoiding '
        'this situation altogether.',
    # -- The smooth version ----------------------------------------------------------------------
    'smooth_scaling':
        'To mitigate this issue more fully, we can extend the scaling fix to kick in before there is literally no '
        'way to spend the remaining money efficiently. That means smoothing out the sharp angle in how this '
        'multiplier is applied, and letting it go above one before the cash ratio goes above one. This expression '
        'is a helpful construction, since it is approximately 1 when x is small, and approximately x when x is '
        'large. Calibrating that, we get a smooth approximation of the max curve, which starts increasing before '
        'the cash ratio goes above one. We can scale player values by it during an auction, making their values '
        'drift up as the packing situation starts to get tougher.',
    # -- The feedback process --------------------------------------------------------------------
    'feedback_process':
        'This creates a feedback process, wherein the more cash you have on hand relative to the available '
        'options, the more likely it is that you slightly overpay and win a bid for a player. That will decrease '
        'your cash, pushing the cash ratio down, and keeping you from ever getting to the far right zone, where '
        'you are forced to overpay to a much more significant degree.',
    # -- Weighting the boost toward the best players ---------------------------------------------
    'other_modification':
        'There is also one other modification to this process. The raw multiplier inflates all values, which is a '
        'bit awkward, because we do not want to accidentally spend too much money on a marginal player and lose '
        'out on one of the few high-value players remaining. That would be quite bad. For example, in this case, '
        'if we bid up the ten dollar player and lose the twenty dollar player, we lose a full eleven dollars of '
        'potential value. Losing the ten dollar player would not be nearly as bad because it could be replaced '
        'with the nine dollar player, losing only one dollar of value. It is hard to say what the exact '
        'possibilities will be in any given situation, but we can say that with excess cash, the most important '
        'thing to do is secure the highest-value players available. That motivates allocating most of the excess '
        'bidding to the highest-value players. To do that, the multiplier is modified to act proportional to the '
        'square of player value. E.g. a twenty dollar player will get four times the boost of a ten dollar '
        'player, instead of twice as much of a boost.',
}
