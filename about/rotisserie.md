# Adapting to Rotisserie

I did not include Rotisserie in the H-scoring paper because its objective function is hard to calculate. Still, I wanted to have some sort of implementation for this site, so I designed a duck-tape-and-glue way of handling Rotisserie. 

The Rotisserie format has considerably different dynamics from its head-to-head counterparts. For one, scores are aggregated across an entire season rather than across individual weeks. Additionally, winning in Rotisserie requires winning against every single opponent at the same time, rather than individual opponents across multiple matchups. These two distinctions pose unique problems when trying to game-plan for Rotisserie, and warrant significant adjustments. 

## Season-long aggregation

In theory, if exact probability distributions were known for player performances, overall variance for the entire season would be easy to calculate. From the perspective of weekly statistics, the variance would be the same as for head-to-head except divided by the total number of weeks, because variance scales down by sample size. 

However, this assumption is unrealistic because there is some uncertainty in the underlying distribution for each player, especially before the season begins. This inconvenient fact can largely be ignored for head-to-head formats because week-to-week variance is generally more significant than uncertainty in the underlying distributions. But when week-to-week variance is minimized due to a larger sample size, it becomes untenable to ignore it. 

The best way to account for this would be with a measure of pre-season projection reliability. I don't have anything like that, so I use a duck-tape-and-glue fudge factor that I call $\chi$. Only relevant for Rotisserie, $\chi$ is a manual adjustment to week-to-week variance for Rotisserie. If performance means were known exactly beforehand, $\chi$ would be the same as for head-to-head except divided by number of weeks as discussed before. In practice, larger values of $\chi$ make more sense for Rotisserie: $0.6$ is the default. 

## Going for gold

The other difference is in the objective function. Rather than winning an arbitrary matchup, a Rotisserie manager wants to win against every opponent simultaneously. One way of writing the objective function is 

$$
\sum_{s \in S_w} P(s) 
$$

Where $S_w$ is a particular scenario in terms of category ordering, for which the manager is the winner, and $P(S)$ is the probability of that scenario happening. This very different objective necessitates approaching the format differently from ho head-to-head

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
- The maximum number of category points scored by any opponent can be approximated by a Normal distribution, independent from the team's own category points 
    - There are many ways to estimate this distribution. I've used some very rough math to estimate it, but it could also be determined empirically
    - In reality, the number of points needed to win is definitely not independent from the team's own success because there are only so many points available. So this method probably slightly overestimates how difficult it is to win in Rotisserie
- If a team has an $X\%$ chance of winning against opposing team $A$ and a $>X\%$ chance of beating opposing team $B$, if they win against team $A$ then they must also win against team $B$
- The team's number of category points can be approximated by a Normal distribution 

Since this wasn't in the paper, I will include a mathy explanation of what follows from these assumptions 

#### Math

Consider a simple example of three opponents for a specific category: $t_0$, $t_1$, and $t_2$. The team has $10\%$, $50\%$, and $60\%$ chances to defeat them respectively. The team's average outcome is defeating $\sum p = 1.2$ of the other teams, earning $1.2$ points. 

Based on how the team scores, there are four possible outcomes 
- With probability $10\%$, the team scores better than all three other teams. The variance of this outcome from expected is $(+1.8)^2 = 3.24$
- With probability $40\%$, the team scores better than both $t_1$ and $t_2$ but not $t_3$. The variance of this outcome from expected is $(+0.8)^2 = 0.64$
- With probability $10\%$, the team scores better than $t_1$ and worse than the other two. The variance of this outcome from expected is $(-0.2)^2 = 0.04$
- With probability $40\%$, the team scores worse than all three other teams. The variance of this outcome from expected is $(-1.2)^2 = 1.44$
\end{itemize}

The total variance is 

$$
0.1 * 3.24 + 0.4 * 0.64 + 0.1 * 0.04 + 0.4 * 1.44 = 1.16
$$

More generally, this can be framed as 

$$
p_1 \left( N  - \mu_c \right)^2  + \sum_{n=2}^{N} \left[ \left( p_n - p_{n-1} \right) \left(N - n + 1 - \mu_c \right)^2 \right]+ \left( 1 - p_{n} \right) \mu^2
$$

Where $p_1$, $p_2$ et cetera are the probabilities of beating the other teams in in ascending order and $\mu_c$ is their sum. This can be rewritten to 

$$
\mu_c^2 + \sum_{n=1}^{N} \left[ p_{n,c} \left( (N - n + 1 - \mu_c)^2 - (N - n -\mu_c)^2 \right) \right]
$$

$$
=\mu_c^2 + \sum_{n=1}^{N} \left[ p_{n,c} \left( 2N - 2n - 2\mu_c + 1 \right) \right]
$$

For a sanity check, it is easy to verify that this equation works with the example. 

The total variance is 

$$
\sigma_D^2 = \sum_c \left[ \mu_c^2 + \sum_{n=1}^{N} \left[ p_{n,c} \left( 2N - 2n - 2\mu_c + 1 \right) \right] \right]
$$

The difference between the necessary number of points to win and the team's score is then
$$
N\left( \mu_M - \sum_{n,c} p_{n,c} , \sigma_D^2 + \sigma^2_M \right)
$$

The objective function is the cdf of that Normal distribution at zero. 

$$
V = CDF \left( N\left( \mu_M - \sum_{n,c} p_{n,c}, \sigma_D^2 + \sigma^2_M \right), 0 \right) 
$$

The gradient is somewhat complicated to calculate.The objective function has already been determined to be

$$
V = CDF \left( N\left( \mu_M - \sum_{n,c} p_{n,c}, \sigma_D^2 + \sigma^2_M \right), 0 \right) 
$$

First, the Normal distribution can be converted into the basis of a standard normal

$$
V = \phi  \left( \frac{ \sum_{n,c} p_{n,c} - \mu_M }{\sqrt{ \sigma_D^2 + \sigma^2_M}} \right)
$$

By chain rule, the gradient is the PDF multiplied by the gradient of the inside. Since the PDF is essentially a constant (it is the same for all categories) and only the direction of the gradient is needed, the pdf term can be dropped. All that is required is the gradient of the inside. 

$$
\nabla V \sim \nabla \left( \frac{ \sum_{n,c} p_{n,c} - \mu_M  }{\sqrt{ \sigma_D^2 + \sigma^2_M}} \right)
$$

Invoking the quotient rule yields 

$$
\nabla V \sim \frac{\sum_{n,c} \frac{dp_{n,c}}{dX} \sqrt{ \sigma_D^2 + \sigma^2_M} - \left( \sum_{n,c} p_{n,c} - \mu_M   \right) \frac{1}{2} \left(\sigma_D^2 + \sigma^2_M \right)^{-0.5}   \frac{d \sigma_D^2}{dX}  }{ \sigma_D^2 + \sigma^2_M}
$$

Thanks to similarity this becomes 

$$
\nabla V \sim \sum_{n,c} \frac{dp_{n,c}}{dX} \left( \sigma_D^2 + \sigma^2_M \right) + \left( \sum_{n,c} p_{n,c} - \mu_M   \right) \frac{1}{2}  \frac{d \sigma_D^2}{dX}  
$$

Now we need to use that 

$$
\frac{d \sigma_D^2}{dX}  = \sum_{n,c} \frac{dp_{n,c}}{dX} \left( 2N - 2n + 1 - 2\mu_c \right)
$$

To see this, consider that 

$$
\frac{d \sigma_D^2}{dX}  = \frac{d}{dX} \sum_c \left[ \mu_c^2 + \sum_{n=1}^{N} \left[ p_{n,c} \left( 2N - 2n - 2\mu_c + 1 \right) \right] \right]
$$

$$
= \sum_{c} \frac{d\mu_c^2}{dX} + \sum_{n.c} \frac{dp_{n,c}}{dX} \left[ 2N - 2n + 1 \right] - \sum_{n.c} \left[ \frac{d2\mu_c}{dX} \right]
$$

$$
= \sum_{n,c} \left[ \frac{dp_{n,c}}{dX} 2 \mu_c \right] + \sum_{n,c} \left[ \frac{dp_{n,c}}{dX} \left( 2N - 2n + 1 \right) \right] - \sum_{n.c} \left[ \frac{d2 \sum_{n} p_{n,c}}{dX} \right]
$$

$$
= \sum_{n,c} \left[ \frac{dp_{n,c}}{dX} 2 \mu_c \right] + \sum_{n,c} \left[ \frac{dp_{n,c}}{dX} \left( 2N - 2n + 1 \right) \right] - \sum_{n.c} \left[ \frac{dp_{n,c}}{dX} 4 \mu_c \right]
$$

And factoring out $\sum_{n,c} \frac{dp_{n,c}}{dX}$, results in 

$$
\nabla V \sim \sum_{n,c} \frac{dp_{n,c}}{dX} \left[ \left( \sigma_D^2 + \sigma^2_M \right) - \frac{ \sum_{n,c} p_{n,c} - \mu_M  }{2} \left( 2N - 2n + 1 - 2\mu_c \right) \right]
$$

Or 

$$
\nabla V \sim \sum_{o,c} PDF(X_o(j)) * \nabla X_o(j)\left[ \left( \sigma_D^2 + \sigma^2_M \right) + \left( \mu_M - \sum_{c} \mu_c \right) \left( N - n - \mu_c + \frac{1}{2}\right) \right]
$$

Where $X_o$ is the mean differential against a specific opponent

It is apparent from the gradient that when $\mu_M > \sum_{n,c} p_{n,c}$ as it generally will be, the matchups with low $n$ and low $\mu_c$ yield the greatest returns on investment. This aligns with the intuition that Rotisserie managers want to increase the variance and therefore upside of their own strategies, since increasing the probabilities of winning low-probability categories increases variance

## Results and observations 

Interestingly, the derived equation implies that teams are better off with more volatile results. Explaining that with some brief math:
-In general, the expected top score $\mu_M$ is larger than the manager's own expected score $\mu_D$. Therefore, center of the distribution is to the right of $0$ and the value of the CDF is below $50\%$. 
-This implies that increasing the total variance through $\sigma_D^2$ and spreading out the distribution of $D$ increases the likelihood that it will go above $\mu_M$

While uncertainty is undesirable in most cases, in this context, it turns out to be a good thing! Intuitively, the reason that this happens is that scoring above every other team takes luck, and luck comes more easily with higher volatility. 

The natural follow-up question is how volatility for the team's score can be increased. Mathematically the answer is that it can be increased by improving the likelihood of winning low-probability points at the expense of high-probability points. Common sense bears this out; it stands to reason that the sum of two $0/1$ coin flips is more volatile than the sum of one guaranteed $0$ and one guaranteed $1$. Similarly, being decent at two categories leads to more volatility than being amazing at one and terrible at another. 

The upshot is that the most important categories and matchups to try to win in Rotisserie are those for which the team is at a disadvantage. So if it seems like a manager's team is falling behind in a category, they likely would benefit by shoring it up with their future picks. This aligns with the conventional wisdom that balanced teams are generally stronger for Rotisserie than they are for head-to-head. 

That is not to say that punting is an inherently terrible idea in Rotisserie; rather, the argument for punting is just weaker because of a countervailing effect. The algorithm tends to advocate for some degree of very soft punting. In particular it likes the idea of deprioritizing steals slightly, using it to eke out a significant advantage in other categories, and hoping that the volatile nature of steals allows the team to pick up some wins in that category despite their slight disadvantage. 

Finally, I would be remiss not to emphasize that the derived method for Rotisserie is *extremely* rough. Even using the explicit solution or simulation method would be an approximation to some degree; the heuristic method adds another layer of simplification on top of that. It is far from a guaranteed correct answer and should be taken with a grain of salt!