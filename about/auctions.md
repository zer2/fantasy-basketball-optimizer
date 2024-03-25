# Auctions

Auctions are more complicated and strategically interesting than snake drafts. Where snake drafts implicitly ask "which player is the most valuable?" auction drafts ask the more difficult question, "how much is each player worth"? 

Most fantasy analysts approach this question from the static perspective, aiming to quantify how much each player is worth in a given auction environment. There is an established method for this. However I believe that it is flawed and can be improved with a modification that I call SAVOR: streaming-adjusted value over replacement. 

Of course, just like for drafting, any static system will be suboptimal, no matter how sophisticated. A more optimal way of valuing players can be designed with a variant of H-scores designed for auctions. 

## Static value quantification

### The basic auction model

A well-known heuristic for quantifying auction value is described in many places including [this article from rotowire](https://www.rotowire.com/basketball/article/nba-auction-strategy-part-2-21393). For reference, the procedure is
1) Calculate the replacement-level Z-score. That is, if 156 players will be chosen, the 157th-highest Z-score is the replacement value
2) Adjust all Z-scores by subtracting out the replacement-level value. If this would make a score go below zero, set it to zero instead
3) Calculate the sum of Z-scores above replacement. This is the total amount of real value available in the auction
4) Divide the total number of dollars available by the total amount of real value available. This yields a conversion rate from Z-score above replacement to dollars
5) Multiply each players' Z-score above replacement with the conversion rate calculated in the previous step. The result is each players' auction value

This strategy is entirely logical if Z-score truly represents value. Players who are not worth including on a team have no value, and players who are worth including are worth an amount of money proportional to how much more valuable they are than a replacement player

### Value deterioration

Every player has some probability of losing value over the course of the season, perhaps to the degree that they are no longer worth having on a team. In that case, they would be replaced by a player with exactly replacement value. 

Intuitively, it makes sense that this possibility should be reflected in a player's auction value. For drafting we did not have to worry about it, because so long as players had roughly similar variance over time to each other, the higher-value player was always worth taking over the lower-value player. However, in the auction context, absolute value matters too, not just the order of players.

To get a sense of how this effects value, we can model the situation mathematically. This thought experiment leads to the SAVOR procedure

### Streaming-adjusted value over replacement 

Let's assume that the replacement value remains the same over the course of a season. This is intuitively reasonable; individual players may go up or down but the value of the player at rank e.g. 157 should be relatively consistent. 

Further, let's assume that each player's value is perturbed by noise $\epsilon$ with mean zero by some point during the season. If $\epsilon$ drops the player's value below the replacement value, then they are not worth having on a team and will be dropped in favor of a player with exactly the replacement value. 

Consider the expected performance value of a player above replacement to be $\mu$, with perturbation $\epsilon_f$. Then, their perturbed value $F$ is 

$$
F= 
\begin{cases}
    \mu + \epsilon_f,& \text{if } \mu + \epsilon_f \geq 0\\
    0,              & \text{otherwise}
\end{cases}
$$

Players with exactly replacement value before perturbation are essentially costless to pick up. There are more players of this value available than draft spots, so you could always wait until after the auction and pick one of them up for free. They have no value when unperturbed, but could be valuable in the future. Their perturbed value $S$ is 

$$
S= 
\begin{cases}
    \epsilon_s ,& \text{if } \epsilon_s \geq 0\\
    0,              & \text{otherwise}
\end{cases}
$$

Since picking a player removes a spot that could be used for one of these "streamers", the real value of picking a player is $E[F-S]$ or $E[F] - E[S]$

Using the definition of the expected value and the probability density of the normal distribution,

$$
E[F] = \frac{1}{\sqrt{2\pi}} \int_0^{\infty} x e^{\frac{-\left( x - \mu \right)^2}{2\sigma}} dx
$$

$$
E[S] = \int_0^{\infty} \frac{x}{\sqrt{2\pi}} e^{\frac{-x^2}{2\sigma}} dx
$$


Changing the variable in the expression for $E[F]$ from $x$ to $y + \mu$ yields 

$$
E[F] = \frac{1}{\sqrt{2\pi}} \int_{-\mu}^{\infty} (y + \mu) e^{\frac{-y^2}{2\sigma}} dy
= \frac{1}{\sqrt{2\pi}} \left(  \int_{-\mu}^{\infty} y e^{\frac{-y^2}{2\sigma}} dy + \int_{-\mu}^{\infty} \mu e^{\frac{-y^2}{2\sigma}} dy \right)
= \frac{1}{\sqrt{2\pi}} \left( \int_{-\mu}^{\infty} y e^{\frac{-y^2}{2\sigma}} dy + \mu \int_{-\mu}^{\infty}  e^{\frac{-y^2}{2\sigma}} dy \right)
$$

Therefore

$$
E[F] - E[S] = \frac{1}{\sqrt{2\pi}} \left( \int_{-\mu}^{\infty} y e^{\frac{-y^2}{2\sigma}} dy + \mu \int_{-\mu}^{\infty}  e^{\frac{-y^2}{2\sigma}} dy - \int_0^{\infty} x e^{\frac{-x^2}{2\sigma}} dx \right)
= \frac{1}{\sqrt{2\pi}} \left( \int_{-\mu}^{0} y e^{\frac{-y^2}{2\sigma}} dy + \mu \int_{-\mu}^{\infty} e^{\frac{-y^2}{2\sigma}} dy \right)
$$

The second part of the expression can be identified as $\mu * CDF(\mu)$. This simplifies to 

$$
E[F] - E[S] =\mu * CDF(\mu) + \int_{-\mu}^{0} \frac{y}{\sqrt{2\pi}} e^{\frac{-y^2}{2\sigma}} dy
$$

Substituting $- \sigma * d(- \frac{y^2}{2 \sigma })$ for $ydy$ yields 
$$
E[F] - E[S] = \mu * CDF(\mu) - \int_{-\mu}^{0} \frac{\sigma}{\sqrt{2\pi}} e^{\frac{-y^2}{2\sigma}} d(- \frac{y^2}{2 \sigma})
$$

Upon integration, this becomes

$$
E[F] - E[S] = \mu * CDF(\mu) - \frac{\sigma}{\sqrt{2\pi}} \left( e^{\frac{-y^2}{2\sigma}} |_{-\mu}^{0} \right) 
= \mu * CDF(\mu) - \frac{\sigma}{\sqrt{2\pi}} \left( 1 - e^{\frac{- \mu^2}{2\sigma}} \right)
$$

It is easy enough to test this equation by simulating post-perturbation value a large number of times. I've done that and made sure that it works, given the assumptions of the problem setup. 

One way of grasping the meaning of the expression conceptually is that the value of a given player is their expected value multiplied by the probability that they remain above replacement level, with a small adjustment factor to account for the imprecision of that calculation. 

This step can be inserted into the previously outlined procedure. With the SAVOR methodology, it becomes

1) Calculate the replacement-level Z-score. That is, if 156 players will be chosen, the 157th-highest Z-score is the replacement value
2) Adjust all Z-scores by subtracting out the replacement-level value. If this would make a score go below zero, set it to zero instead
3) Further adjust all scores by calculating $E[F-S]$ for all players. Call the result real value
4) Calculate the sum of real value. This is the total amount of real value available in the auction
5) Divide the total number of dollars available by the total amount of real value available. This yields a conversion rate from real value to dollars
6) Multiply each players' real value with the conversion rate calculated in the previous step. The result is each players' auction value

SAVOR has one parameter- the noise level $\sigma$. I have set that to one because that seems reasonable to me and the results look reasonable too, but the parameter could perhaps be refined. 

## Dynamic value quantification 

### Implementing H-scoring for auctions

The H-scoring framework for drafting can also be applied to auctions. With a certain number of players remaining, we can assume some level of control over the weighting applied to those players to account for a punting strategy. All that differs is that we need a different way to account for the mean of the Bell curve of the category differential.

It is helpful to start by breaking down overall metrics in the following way
- $X = X_s + X_p + X_r + X_m + X_{\delta}$ where
  - $X_s$ is the aggregate statistics of team $A$'s already selected players
  - $X_p$ is the statistics of the candidate player
  - $X_r$ is the statistics of aggregate statistics replacement-level players, filling all empty slots 
  - $X_m$ is the general benefit of leveraging extra money to get above-replacement players
  - $X_{\delta}$ is the differential effect of punting strategy on the above-replacement players that will be selected instead of the replacement-level players 
- $X_{o_\mu} = X_{o_s} + X_{o_r} + X_{o_m}$ where
  - $X_{o_s}$ is the aggregate statistics of team $B$'s already selected players
  - $X_{o_r}$ is the statistics of aggregate statistics replacement-level players, filling all empty slots 
  - $X_{o_m}$ is the general benefit of leveraging extra money to get above-replacement players

Then 

$$
X - X_{o_\mu} =  X_s + X_p - X_{o_s} + X_r - X_{o_r} + X_m - X_{o_m} + X_{\delta} 
$$

This equation can be grouped into four parts 
- $X_s + X_p - X_{o_s}$: difference of known player statistics
- $X_r - X_{o_r}$: difference of replacement-level values. E.g. if after adding the chosen player team $A$ has one more player selected already, then team $B$ has an additional replacement-level player which is subtracted out 
    - I define a replacement-level player as having overall replacement-level value spread out equally across categories
- $X_m - X_{o_m}$: the differential effect of team $A$ having more money remaining than team $B$.
    - The overall monetary differential can be estimated as the sum of available above-replacement value over the sum of remaining money in the pool, multiplied by the difference in money between team $A$ and team $B$. To get per-category values, spread the overall value by per-category generic weight (as in, multiply by the $v$ vector)
- $X_{\delta}$: differential from punting strategy
    - This value can be calculated in the same way as it was for the drafting context 

### Estimating auction value from H-score

The H-score calculation yields a win probability for each candidate player if they could be selected without costing any money. Of course, this is unrealistic. But we can use H-score as a proxy for general value; a player that increases win probability over a replacement level player by twice as much as a different player is probably twice as valuable and worth twice as much money. 

With H-scores proxying general value, the SAVOR algorithm can be applied to them, just like in the static context. 

### Using auction values 

The full procedure outlined above yields rough auction values for a particular auction drafter.

It should be noted that just having this information available does not guarantee a good auction: if a bidder always pays full money for each of their picks, they won't have an above-average team even in theory. 

One way to make a good team is to target players that have high auction values according to H-score versus what others are willing to pay, which could perhaps be inferred from typical auction values or the static Z-score procedure. Even then, there are many additional dimensions to auction drafting: the very best auction drafters know how to use psychology to get good deals on the players they want or bump up the prices of players that they are not interested in. Perfecting auction drafting is difficult, and goes far beyond the mathematics of player value. 







