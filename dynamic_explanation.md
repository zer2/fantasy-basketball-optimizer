Static ranking lists are convenient but obviously suboptimal. An algorithm that can adapt to draft circumstances is preferable

## 1. Draft decisions as an optimization problem 

While no information about any other players was available in the static context, a dynamic algorithm knows previously drafted players and is aware that future picks will be made with the same knowledge. This allows the algorithm to treat player choice as an optimization problem; attempting to maximize probability of victory as a function of the statistics of the team it chooses. 

Define $V(X)$ as the objective function relative to the team $A$'s stat distribution $X$. With $w_c(X)$ as the probability of winning a category based on $X$, the objective function for the Each Category format is simply 

$$
V(X) = \sum_c w_c(X)
$$

For Most Categories, $V(x)$ is slighly more complicated, since it is the probability of winning the majority of categories. It can be written as

$$
V(j)  = w_1(X) * w_2(X) * w_3(X) * w_4(X) * w_5(X) * (1-w_6(X)) * (1-w_7(X)) * (1-w_8(X)) * (1- w_9(X)) + \cdots
$$

Where there is a term for each scenario including five or more scenario wins

## 2. A formula for $w_c(X)$

The discussion of static ranking lists established that point differentials between teams can be modeled as Normal distributions. This can be applied in the dynamic context as well. One necessary modification is that players on team $A$ do not contribute to player-to-player variance, since they are under control of the drafter. The resulting normal distribution can then be defined
- The mean is $X - X_{\mu}$, where $X_{\mu}$ is the expected value of $X$ for other teams
- The variance is $N * m_{\sigma}^2 + 2 * N * m_{\tau}^2$

$X$ and $X_\mu$ are not particularly helpful in and of themselves, because it is not obvious how to estimate them. One useful way to decompose them is into
- $X = X_s + X_p + X_{\mu_u} + X_u$ where $X_s$ is the aggregate statistics of already selected players, $X_p$ is the statistics of the candidate player, $X_{\mu_u}$ is the expected statistics of unchosen players, and $X_u$ is a difference factor to adjust for future picks being different from expected
- $X_\mu = X_{\mu_s} + X_{\mu_u}$ where $X_{\mu_s}$ is the expected aggregate statistics of players drafted up to the round player $p$ is being drafted

This allows the distribution mean to be rewritten to 

$$
X_s + X_p - X_{\mu_s} + X_u
$$

The first two quantities are known. $X_{\mu_s}$ can be estimated by finding the averages of all players drafted up to a certain round, based on a heuristic metric like G-score or Z-score. The tricky quantity to compute is $X_u$

## 3. Approximating X_u

## 4. Optimizing for j

## 3. Limitations
The most important weaknesses to keep in mind for H-scoring are 
* The algorithm does not adjust for the choices of other drafters. If you notice another drafter pursuing a 
particular punting strategy, you might want to avoid that strategy for yourself so that you do not compete
for the same players
* The algorithm understands that it cannot pick all players of the same position with future picks through the $\nu$ parameter, but it does not adjust H-scores by
position, even if the top scorers are heavily tilted towards some positions over others. It works this way because in my simulations, the greedy heuristic of simply taking the highest-scoring available player
that can fit on the team at every stage of the draft does fine, and I have not found a value-above-replacement system which improves
algorithm performance. However, this may be impractical for real drafting. Real fantasy basketball has no concrete rules around team construction
and most drafters want to avoid accidentally constructing unbalanced teams. So you might want to pick players of new positions even if they have slightly lower H-scores
e.g. If the algorithm is leaning towards centers to align with its punting strategy, and it finds a point guard that is only slightly below the top pick in terms of overall
H-score, you might want to pick it. 
* The extension of H-scoring to Rotisserie implemented in this tool is not described in the paper.
It is similar to the Each Category algorithm, except that week-to-week variance is set to zero and it is assumed
that other drafters will be drafting based on Z-scores. It has not been verified in the Rotisserie context, so there
is even more reason for skepticism when interpreting its results than for the other formats
