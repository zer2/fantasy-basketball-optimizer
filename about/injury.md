# Injuries

As unfortunate as they are, injuries are an important part of fantasy basketball. The best-laid plans can easily be ruined by an injury to a key player during a key matchup. 

Injuries are difficult to predict in advance. Still, some level of careful analysis can produce better forecasts than guessing blindly. [This meta-analysis](https://jeo-esska.springeropen.com/articles/10.1186/s40634-021-00346-x) found many forecasting methods which somewhat reliably predicted which players would be at higher risk of injury in the future. 

Given that we can expect different players to get injured at different rates, the question becomes how to incorporate that into rankings. This is something of an open question in the fantasy basketball community. There is no consensus on whether rankings should use per-season totals, which account for all games, or per-game totals which filter out games when players are injured. On the one hand, injury risk is a real factor that should ideally be accounted for. On the other hand, teams need some degree of injury luck for a chance to compete, suggesting that managers should ignore injury risk and just hope for the best. 

There are many complicating factors which make it difficult to resolve this conundrum in a universal way. For example, different leagues have different numbers of "Injury List" roster spots and different rules about which players are eligible. So coming up with one single answer, that will always apply, is impossible. 

What is certain is that the appropriate solution is somewhere between the two extremes of per-season and per-game. Because of the heterogeneity between leagues, the approach of this website is to handle injuries flexibly within that space between the extremes. It uses two parameters: $\upsilon$, which accounts for necessary upside bias, and $\psi$, which encodes the degree to which injured players can be replaced by waiver targets/free agents. Setting $\upsilon$ to one and $\psi$ to zero results in yearly total scores. Setting $\upsilon$ to zero results in the other extreme, per-game averages. Setting the parameters to other values results in scores somewhere between the two.

More detail on the two parameters is included below

## Biasing towards optimism with $\upsilon$

When a team has below-average injury luck, it is unlikely to have any shot at competing for a championship. So if a team is aiming for a championship, it makes sense for them to strategize with the assumption that their injury luck is reasonably good. 

The math for this could theoretically get quite complicated and take into account probabilities of all players being injured for different numbers of games to optimize for the likelihood of overall victory. In practice, this is a difficult calculation to make and is not worth the computational intensity.

The site's simple heuristic is to scale-down injury risk by a constant factor $\upsilon$. For example, if a player is expected to miss $20\%$ of games on average and $\upsilon$ is $75\%$, then their injury risk is adjusted to $15\%$ instead. Then all of their counting statistics (including number of free throw attempts and field goal attempts) are multipled by $85\%$, the adjusted percentage of games they are expected to play. 

In numbers, 

$$
I' = I *\upsilon
$$

Where $I'$ is the adjusted version of the injury rate $I$, and

$$
\mu_c' = \mu_c * ( 1 - I')
$$

Where $\mu_c'$ is the adjusted version of the mean for a counting statistic $\mu_c$ 

The default value of $\upsilon$ is $100\%$. This is appropriate for managers who want to do well in general, and don't want to gamble on risky players in the hopes of winning a championship if they get lucky. For managers who want to win championships and need some level of injury luck to get there, setting the value lower is appropriate

## Quantifying replacability with $\psi$ 

When IL spots are available or dropping a player is an option, injured players can be replaced by replacement-level alternatives. This defrays some of the pain of a player being injured. 

The overall replacement level value is easy to estimate, since it is the total score of the $N$th best player, where score is some appropriate scoring metric like G-score or Z-score. Distributing the overall value to categories is more complicated since replacement-level value is not necessarily distributed evenly. My rule of thumb is to distribute it evenly to all categories, except with an opposite sign for turnovers, since  a disadvantage in other stats roughly translates to an equivalent advantage in the turnover stat. So for example if the replacement value is $-3$, by category it is broken down by $R = - \frac{3}{7}$ for non-turnover categories and $R = \frac{3}{7}$ for turnovers. The sum is $-3$ as needed.

Of course, replacement is not always possible. That is where the $\psi$ parameter comes in, representing the fraction of missed games for which a player can be replaced. 

The total fraction of replacement games is then the injury risk multiplied by $\psi$. Mathematically, this means 

$$
X' = X + I' * \psi * R
$$

Where $X'$ is the adjusted version of score $X$.

The default value of $\psi$ is $80\%$. This can easily be adjusted for leagues where replacing injured players is easier or harder