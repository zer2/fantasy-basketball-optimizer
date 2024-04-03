# Adapting to Rotisserie

The Rotisserie format has considerably different dynamics from its head-to-head counterparts. For one, scores are aggregated across an entire season rather than across individual weeks. Additionally, winning in Rotisserie requires winning against every single opponent at the same time, rather than individual opponents across multiple matchups. These two distinctions pose unique problems when trying to game-plan for Rotisserie, and warrant significant adjustments. 

I've done my best to adjust appropriately. My solutions are a bit duck-tape-and-glue, but I think they broadly make sense. 

## Season-long aggregation

In theory, if exact probability distributions were known for player performances, overall variance for the entire season would be easy to calculate. From the perspective of weekly statistics, the variance would be the same as for head-to-head except divided by the total number of weeks, because variance scales down by sample size. 

However, this assumption is unrealistic because there is some uncertainty in the underlying distribution for each player, especially before the season begins. This inconvenient fact can largely be ignored for head-to-head formats because week-to-week variance is generally more significant than uncertainty in the underlying distributions. But when week-to-week variance is minimized due to a larger sample size, it becomes untenable to ignore it. 

The best way to account for this would be with a measure of pre-season projection reliability. I don't have anything like that, so I use a duck-tape-and-glue fudge factor that I call $\chi$. Only relevant for Rotisserie, $\chi$ is a manual adjustment to week-to-week variance for Rotisserie. If performance means were known exactly beforehand, $\chi$ would be the same as for head-to-head except divided by number of weeks as discussed before. In practice, larger values of $\chi$ make more sense for Rotisserie: $0.6$ is the default. 

## Going for gold

The goal for Rotisserie is not increasing the probability of winning an arbitrary matchup. Instead, it is the probability of getting the most category points overall. The objective function is 

$$
\sum_{s \in S_w} P(s) 
$$

Where $S_w$ is a particular scenario in terms of category ordering, for which you are the winner, and $P(S)$ is the probability of that scenario happening. This very different objective necessitates approaching the format differently from how we approached head-to-head

### Explicit solution

If computation was not a limiting factor, it would be simple to brute force the probability of winning in Rotisserie, given approximate normal distributions for each teams' performances for each category. Unfortunately, the calculation is too intensive to be practical.

For a given category, there are $T!$ ([factorial](https://en.wikipedia.org/wiki/Factorial)) possible orderings, where $T$ is the number of teams. If $T$ is $12$ that translates to more than $479$ million. Further, to get total orderings for each category, that number must be raised to the power of number of categories $C$. With $9$ categories, the result is $1.32 * 10^{78}$. which is roughly equal to the number of atoms in the universe. No cloud service is performing that calculation! 

### Simulation

When computation is impractical, the natural next course of action is to try simulating the problem. 

Unfortunately, simulating Rotisserie is still more computationally intensive than we would like. A single simulation for a single player requires drawing $T*C$ random numbers, then performing a number of computations to determine the winner. With ten thousand simulations (often necessary to get a reliable result) across five hundred players thirty times (a typical number of iterations for H-scoring), simulation requires careful analysis of more than sixteen billion random numbers. That is less outrageous, but still stretches the limitations of modern hardware. The best random number generators can generate a random normal in [about two nanoseconds](https://github.com/miloyip/normaldist-benchmark). It would take more than thirty seconds just to generate the sixteen billion random normals. And that is before all of the necessary subsequent calculations, which are not simple. 

### Invoking a heuristic 

When all else fails, the problem must be simplified.

There are many ways to create a heuristic for Rotisserie. The method I have implemented assumes 
- The overall stats of other teams are (roughly) known 
    - If they are not known, such as at the beginning of a draft, some heuristic must be used to fill in these values
- The maximum number of category points scored by any opponent can be approximated by a Normal distribution, independent from the drafter's own category points 
    - There are many ways to estimate this distribution. I've used some very rough math to estimate it, but it could also be determined empirically
    - In reality, the number of points needed to win is definitely not independent from the drafter's own success because there are only so many points available. So this method probably slightly overestimates how difficult it is to win in Rotisserie
- If a drafter has an $X\%$ chance of winning against team $A$ and a $>X\%$ chance of beating team $B$, if they win against team $A$ then they must also win against team $B$
- The drafter's number of category points can be approximated by a Normal distribution 

Below is a slightly mathy summary of how the model works

#### The drafter's distribution 

Based on the assumptions, a mean and variance can be calculated for the drafter's total number of points earned in a category. A detailed derivation is in the paper, the result of which is
- $\mu = \sum_n P_n$
- $\sigma^2 =  \mu^2 + \sum_n P_n (2N - 2n - 2 \mu + 1)$

Where $P_1$, $P_2$ etc. are the probabilities of winning against each opponent, in ascending order. $N$ is the total number of opponents.

To get the overall distribution for the drafter, these values for all categories can be added together. Call the sums $\mu_D$ and $\sigma_D^2$

#### The objective function

The difference between two Normal distributions is also a Normal distribution. Therefore, the differential between the drafter's total points and the number of points needed to win the league can be approximated as a Normal distribution. 

$$
N(\mu_M - \mu_D, \sigma_D^2 + \sigma_M^2)
$$

The CDF of this distribution at zero is the probability that the drafter scores at least as many category points as necessary to win, making it an appropriate objective function. It can be optimized via gradient descent to find the optimal player to pick 

## Results and observations 

In general, $\mu_M$ is larger than $\mu_D$, so the center of the distribution is to the right of $0$ and the value of the CDF is below $50\%$. This implies that increasing the total variance through $\sigma_D^2$ increases the CDF and therefore the objective function. While uncertainty is undesirable in most cases, in this context, it turns out to be a good thing! Intuitively, the reason that this happens is that scoring above every other drafter takes luck, and luck comes more easily with higher volatility. 

The natural follow-up question is how volatility for the drafter's score can be increased. Mathematically the answer is that it can be increased by improving the likelihood of winning low-probability points at the expense of high-probability points. Common sense bears this out; it stands to reason that the sum of two $0/1$ coin flips is more volatile than the sum of one guaranteed $0$ and one guaranteed $1$.

The upshot is that the most important categories and matchups to try to win in Rotisserie are those for which the drafter is at a disadvantage. So if it seems like a drafter is falling behind in a category, they likely would benefit by shoring it up with their future picks. This aligns with the conventional wisdom that balanced teams are generally stronger for Rotisserie than they are for head-to-head. 

That is not to say that punting is an inherently terrible idea in Rotisserie; rather, the argument for punting is just weaker because of a countervailing effect. In practice, the algorithm tends to advocate for some degree of very soft punting in Rotisserie. In particular it likes the idea of deprioritizing steals slightly, using it to eke out a significant advantage in other categories, and hoping that the volatile nature of steals allows the drafter to pick up some wins in that category despite their slight disadvantage. 

Finally, I would be remiss not to emphasize that the derived method for Rotisserie is *extremely* rough. Even using the explicit solution or simulation method would be an approximation to some degree; the heuristic method adds another layer of simplification on top of that. It is far from a guaranteed correct answer and should be taken with a grain of salt!