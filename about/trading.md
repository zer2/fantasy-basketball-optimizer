# Mid-season moves via waivers and trading

Fantasy basketball does not end at the draft stage. During a season, drafters can still improve their rosters via two mechanisms: the waiver wire and trading. Algorithmic approaches to both are included under the Draft tab 

## The waiver wire

Players that are not on any team are considered "on the waiver wire", and can be picked up as free agents. If there is a player on your team who is not contributing much or does not fit your strategy well, it might be worth exchanging them for a player on the waiver wire. Evaluating whether or not this is a good decision is accomplished simply by analyzing the change in H-score for the team when the substitution is made. If the H-score improves after the change is made, then the exchange is promising. It is worth keeping in mind that the calculation does not explicitly consider position. A waiver move might be risky if it improves your team's H-score slightly but also makes your team less positionally balanced, and might be worth it even if it decreases H-score if it greatly improves team balance. 

## Trading

Trading provides an alternative mechanism for improving your team. It is useful when one of your players has strong general value, but does not fit your team well. If you can find a corresponding player or set of players on another team, it is possible to construct a trade that benefits both parties. 

Evaluating whether a trade is mutually beneficial is complicated, especially when trades are numerically asymmetrical. The trading module provides one perspective, leveraging H-score to evaluate the effect of the trade on both teams

### Symmetric trade

Symmetric trades are simple to evaluate. There are no choices to be made; the post-trade teams are entirely deterministic. The victory probabilities for each team can be brute-forced based on the means and variances assumed by the H-scoring algorithm

### Asymmetric trade- reducing players

When a team trades down, e.g. trades two players for one, one or more positions on the team open up. This makes it more difficult to evaluate whether the trade makes sense or not. 

One approach is to manually check all waiver wire candidates that could fill up the open opositions and find the set that optimizes for H-score, taking that H-score to be the post-trade value of the team. The issue with this approach is that explicitly checking every combination of players is exponentially complicated and quickly becomes intractable. With $400$ waiver candidates and four open positions, there would be 26 billion combinations to evaluate. 

A more tractable approach is to simply run the H-scoring algorithm with the remaining team. One candidate is selected for explicitly, and the rest are modeled via approximation. This is how the trading module works in this case.

It should be noted that a single incorrectly listed waiver wire player can render the results of this approach meaningless. If a very strong player is considered available in the waiver wire pool, when they actually should not be, that will make the H-scoring algorithm erroneously believe that it can fill holes in its roster with that very strong player. Then many trade-down scenarios will appear greatly beneficial, even when they are not. So it is important for traders to make sure that the list of players to exclude from the analysis is correct before evaluating a trade

### Asymmetric trade- increasing players

When a team trades up, e.g. trades one players for two, they need to drop a player from their resulting team. 

The complexity of this kind of trade also increases drastically with the number of players needed to drop, but not to the same degree as the trade-down case. The number of combinations available if e.g. four players are added is just seventeen choose four, which is slightly over two thousand. That is trivial for a computer. Trading down more than six players does run into memory issues, so that is disabled, but a trade that lopsided is not common

## Heuristic trade guide

In addition to evaluating trade suggestions, the app also provides some guidance on the most promising avenues for trades. This guidance can be used as inspitation for concrete trade ideas to evaluate

### Evaluating trade candidates

Good trade candidates would have more value on other teams than on your team. The trade candidates tab quantifies this value disparity for all combinations of your own players and possible receiving teams.

For a particular player/receiving team pair it does this by:
- Evaluating the impact of the player to your team. For this it uses an equivalent methodology to the asymmetric trade with decreasing players, meaning that the algorithm finds the best free agent/waiver wire replacement and observes how much worse the team gets with that replacement. If the team actually gets better when the player gets dropped, the value is set to zero
- Evaluating the impact of adding the player to the receiving team, with equivalent methodology to the asymmetric trade with increasing players. That is, the algorithm runs through all drop options and finds the one which works the best to determine the post-addition score, then observes how much higher it is than the pre-addition score 
- Subtracting the benefit the player offers to you from the benefit they would offer to the receiving team 

After all values are calculated, there is also a "regularization" step for each receiving team. Some constant is added or subtracted from all the differential values to get a set of values that average to zero. The reason for this step is that some teams may have very poor-fitting players, leading to high values given to all potential trade acquisitions, since jettisoning the poor-fitting player would be a boon to the team. The regularization step makes it so that such teams do not stand out as attractive trade destinations for all players across the board. 

### Evaluating trade targets

Good trade targets would have more value on your team than on their current team. The trade targets tab quantifies this value disparity for all of another team's players

For a particular player it does this by:
- Evaluating the impact of the player to its current team. For this it uses an equivalent methodology to the asymmetric trade with decreasing players, meaning that the algorithm finds the best free agent/waiver wire replacement and observes how much worse the team gets with that replacement. If the team actually gets better when the player gets dropped, the value is set to zero
- Evaluating the impact of adding the player to your team, with equivalent methodology to the asymmetric trade with increasing players. That is, the algorithm runs through all drop options and finds the one which works the best to determine the post-addition score, then observes how much higher it is than the pre-addition score 
- Subtracting the benefit the player offers to their current team from the benefit they would offer to you

After all values are calculated, there is also a "regularization" step for each sending team. This is to keep the averages around zero and in line with the candidates tab