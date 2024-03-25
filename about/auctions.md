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
\mu * CDF(\mu) + \int_{-\mu}^{0} \frac{y}{\sqrt{2\pi}} e^{\frac{-y^2}{2\sigma}} dy
$$

Substituting $- \sigma * d(- \frac{y^2}{2 \sigma })$ for $ydy$ yields 
$$
\mu * CDF(\mu) - \int_{-\mu}^{0} \frac{\sigma}{\sqrt{2\pi}} e^{\frac{-y^2}{2\sigma}} d(- \frac{y^2}{2 \sigma})
$$

This is equivalent to 

$$
\mu * CDF(\mu) - \frac{\sigma}{\sqrt{2\pi}} \left( e^{\frac{-y^2}{2\sigma}} |_{-\mu}^{0} \right) 
= \mu * CDF(\mu) - \frac{\sigma}{\sqrt{2\pi}} \left( 1 - e^{\frac{- \mu^2}{2\sigma}} \right)
$$

One way of grasping the meaning of this expression conceptually is that the value of a given player is their expected value multiplied by the probability that they remain above replacement level, with a small adjustment factor to account for the imprecision of that calculation

This step can be inserted into the previously outlined procedure. With the SAVOR methodology, it becomes

1) Calculate the replacement-level Z-score. That is, if 156 players will be chosen, the 157th-highest Z-score is the replacement value
2) Adjust all Z-scores by subtracting out the replacement-level value. If this would make a score go below zero, set it to zero instead
3) Further adjust all scores by calculating $E[F-S]$ for all players. Call the result real value
4) Calculate the sum of real value. This is the total amount of real value available in the auction
5) Divide the total number of dollars available by the total amount of real value available. This yields a conversion rate from real value to dollars
6) Multiply each players' real value with the conversion rate calculated in the previous step. The result is each players' auction value

## Dynamic value quantification 

