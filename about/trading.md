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

The complexity of this kind of trade also increases drastically with the number of players needed to drop, but not to the same degree as the trade-down case. The number of combinations available if e.g. four players are added is just seventeen choose four, which is slightly over two thousand. That is a large number for a human to evaluate, but trivial for a computer. The absolute maximum possible number of combinations, resulting from the absurd scenario of a team trading its entire roster for a single player, is only twenty-five choose thirteen or around five million. Even that is doable for the computer. For this reason, trade-ups are evaluated by explicitly checking every combination
