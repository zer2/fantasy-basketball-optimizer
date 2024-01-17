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

Future draft picks are tricky to model because they are neither completely under the drafter's control (since they don't know which players will be available later) nor completely random (since the drafter will use the dynamic algorithm to draft them). Instead, they fall somewhere between the two extremes. 

One way of approaching this dilemma is allowing the drafter to choose per-category weights for future picks, then approximating the aggregate statistics of future picks based on those weights. This allows the drafter to have some measure of control over future picks, albeit a noisy one that does not anchor on specific players. 

A natural choice for modeling the statistics of draft picks is the [multivariate normal distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution). It has two useful properties
- It can incorporate correlations between different categories. This is essential because it allows the algorithm to understand that some combinations of categories are easier to jointly optimize than others, e.g. prioriting both rebounds and blocks is easier than prioritizing assists and turnovers
- It is relatively easy to work with. A more complicated function, while perhaps more suited to real data, would make the math much more complicated

It is simple to derive a parameterization for $X_u$ when it is not conditional on any weight. One could simply compute the mean, variance, and correlations of real player data. The key to modeling $x_u$ is understanding how it changes when two conditions are applied
- All players above a certain threshold of general value have been picked
- The chosen player is the highest-scoring of those remaining based on some custom weight vector, which we will call $j$

The details of this calculation are mathy. You can find them in the paper if you are interested, or take it for granted that the resulting equation is 

$$
X_u(j) = \Sigma * \left( v j^T - j v^T \right) * \Sigma * \left( - \gamma j - \omega v \right) * \frac{
   \sqrt{\left(j -  \frac{v v^T \Sigma j}{v^T \Sigma v} \right) ^T \Sigma \left( j -  \frac{v v^T \Sigma j}{v^T \Sigma v}  \right) }
  }{j^T \Sigma j * v^T \Sigma v - \left( v^T \Sigma j \right) ^2}
$$

Where $\Sigma$ is the [covariance matrix](https://en.wikipedia.org/wiki/Covariance_matrix) across players 

## 4. Optimizing for j

We have all the ingredients for calculating H-score based on the choice of $j$. However, that does not imply that that we know the choice of $j$ that optimizes H-score. In fact, this question is quite difficult to solve: there are infinite choices for $j$ and even if we were to simplify it to say $10$ choices of weight per category, there would still be $\approx 10^9$ options to look through, which is a lot!

Instead of looking through all the options at random, we can use a method called [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent). Essentially, gradient descent conceives of the solution space as a multi-dimensional mountain, and repeatedly moves in the direction of the highest slope to eventually reach a peak. 

## 5. Results

## 6. Limitations
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
