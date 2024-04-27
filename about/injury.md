# Injuries

As unfortunate as they are, injuries are an important part of fantasy basketball. The best-laid plans can easily be ruined by an injury to a key player during a key matchup. 

How to handle injury risk is something of an open question in the fantasy basketball community. There is no consensus on whether rankings should use yearly totals, which account for the volume of games not played, or per-game totals which assume that players go uninjured. On the one hand, injury risk is a real factor, which should ideally be accounted for. On the other, injury risk is hard to predict, and arguably managers need to assume their players will be healthy if they want any chance of winning a championship.

There are many complicating factors which make it difficult to answer this question in a universal way. For example, different leagues have different numbers of "IL" spots which ameliorate the negative impact of injury. 

Because of this heterogeneity between leagues, the approach of this website is to handle injuries in a flexible way. It uses two parameters: $\upsilon$, which encodes a degree of luck bias, and $\psi$, which encodes the degree to which injured players can be replaced

## Biasing towards optimism 

Say a manager takes Nikola Jokic with their first pick. If Jokic then is injured and end up not playing much, that manager is sorely out of luck and has almost no chance of winning the championship. If the manager is ineveitably doomed by a Jokic injury, then does it really matter if Jokic misses half of the season versus all of it? 

One way to frame this mathematically is that managers are optimizing for upside potential rather than expected value. They want the highest chance of winning a season; how badly they lose if they do lose is irrelevant. 

The math for this could theoretically get quite complicated and take into account probabilities of all players being injured for different numbers of games to optimize for the likelihood of overall victory. In practice, this is a difficult calculation to make and is not worth the computational intensity. Instead, a heuristic should be applied.

The heuristic used by this website is to simply scale-down injury risk by some constant factor $\upsilon$. For example, if a player is expected to miss $20\%$ of games and $\upsilon$ is $75\%$, then their injury risk is adjusted to $15\%$ instead. Then all of their counting statistics are multipled by $85\%$. In numbers, 

$$
I' = I *\upsilon
$$

$$
C' = C * ( 1 - I')
$$

Where $I'$ is the effective injury rate and $C'$ is effective covalue for a counting statistic

## Subbing in a replacement player 

When IL spots are available or dropping a player is an option, injured players can be replaced by replacement-level alternatives. This defrays some of the pain of a player being injured. 

The overall replacement level value is easy to estimate, since it is the total score of the $N$th best player, where score is some appropriate scoring metric like G-score or Z-score. Distributing the overall value to categories is more complicated since replacement-level value is not necessarily distributed evenly. My rule of thumb is to distribute it evenly to all categories, except with an opposite sign for turnovers, since  a disadvantage in other stats roughly translates to an equivalent advantage in the turnover stat. So for example if the replacement value is $-3$, by category it is broken down by $R = - \frac{3}{7}$ for non-turnover categories and $R = \frac{3}{7}$ for turnovers. The sum is $-3$ as needed.

It is worth keeping in mind that replacement is not always possible. This is parameterised by $\psi$, representing the fraction of missed games for which a player can be replaced. 

The total fraction of replacement games is then the injury risk multiplied by $psi$. Mathematically, this means 

$$
X' = X + I' * \psi * R
$$

Where $X'$ is the effective score for a category 
