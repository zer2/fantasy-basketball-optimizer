# Adapting to Rotisserie

The Rotisserie format has considerably different dynamics from its head-to-head counterparts. For one, scores are aggregated across an entire season rather than across individual weeks. Additionally, winning in Rotisserie requires winning against every single opponent at the same time, rather than individual opponents across multiple matchups. These two distinctions pose unique problems when trying to game-plan for Rotisserie, and warrant significant adjustments 

## Season-long aggregation

In theory, if exact probability distributions were known for player performances, overall variance for the entire season would be easy to calculate. From the perspective of weekly statistics, the variance would be the same as for head-to-head except divided by the total number of weeks, because variance scales down by sample size. 

However, this assumption is unrealistic because there is some uncertainty in the underlying distribution for each player, especially before the season begins. We could largely ignore this inconvenient fact for head-to-head formats because week-to-week variance ig generally more significant than uncertainty in the underlying distributions. But when week-to-week variance is minimized due to a larger sample size, it becomes untenable to ignore it. 

The best way to account for this would be with a measure of pre-season projection reliability. I don't have anything like that, so I use a duck-tape-and-glue fudge factor that I call $\chi$. Only relevant for Rotisserie, $\chi$ is a manual adjustment to week-to-week variance for Rotisserie. If performance means were known exactly beforehand, $\chi$ would be the same as for head-to-head except divided by number of weeks as discussed before. In practice, larger values of $\chi$ make more sense for Rotisserie: $0.4$ is the default. 

## Going for gold

The goal for Rotisserie is not increasing the probability of winning an arbitrary matchup. Instead, it is the probability of getting the most category points overall. The objective function is 

$$
\sum_{s \in S_w} P(s) 
$$

Where $S_w$ is a particular scenario in terms of category ordering, for which you are the winner, and $P(S)$ is the probability of that scenario happening. This very different objective necessitates approaching the format differently from how we approached head-to-head

### Explicit solution

If computation was not a limiting factor, it would be simple to brute force the probability of winning in Rotisserie, given approximate normal distributions for each teams' performances. Unfortunately, the calculation is too intensive to be practical.

For a given category, there are $T!$ possible orderings, where $T$ is the number of teams. If $T$ is $12$ that translates to more than $479$ million. Further, to get total orderings for each category, that number must be raised to the power of number of categories $C$. With $9$ categories, the result is $1.32 * 10^{78}$. which is roughly equal to the number of atoms in the universe. No cloud service is performing that calculation! 

### Simulation

When computation is impractical, the next logical course of action is to try simulating the problem. 

Unfortunately, simulating Rotisserie is still more computationally intensive than we would like. A single simulation for a single player requires drawing $T*C$ random numbers, then performing a number of computations to determine the winner. With ten thousand simulations (often necessary to get a reliable result) across five hundred players thirty times (a typical number of iterations for H-scoring), simulation requires careful analysis of more than sixteen billion random numbers. That is less outrageous, but still stretches the limitations of modern hardware. The absolute best random number generators can generate a random normal in [about two nanoseconds](https://github.com/miloyip/normaldist-benchmark). It would take more than thirty seconds just to generate the sixteen billion random normals. And that is before all of the necessary subsequent calculations, which are not simple. 

### A heuristic 

When all else fails, the problem must be simplified. 

There are many ways to create a heuristic adjustment for Rotisserie. The method I have implemented is based off the simple observation that in order to win Rotisserie, you probably need a little bit of luck. It stands to reason that given that you have some luck, you want to win as many categories as possible. 

Luck is a matter of variance. In terms of week-to-week variance, the overall variance for a category differential is the number of players per matchup (since opposing players doing poorly also helps) $P*2$ times $\chi$. The aggregate variance in terms of the sum across all categories is that same number times the number of categories, $C$. The expected best luck of any team is the expected maximum of luck across the number of teams. It is known that the maximum of $N$ random variables is [approximately](https://math.stackexchange.com/questions/89030/expectation-of-the-maximum-of-gaussian-random-variables) $\sigma \sqrt{2 log(N)}$. So the expected best luck is 

$$
\sqrt{P * 2 * \chi * C * 2 * log(T)}
= 2 \sqrt{P * \chi * C * log(T)}
$$

Distributing this across categories, the luck per category is 

$$
\frac{2 * \sqrt{P * \chi * C * log(T)}}{C} = 2 *  \sqrt{\frac{P * \chi * log(T)}{C}}
$$

One strategy is to assume that your team will get the expected best luck of any competitor. However, you likely want to win even if your luck isn't quite that obscene. For that purpose I define a parameter $\Upsilon$ and then add the following advantage to each category 

$$
\Upsilon * 2 *  \sqrt{\frac{P * \chi * log(T)}{C}}
$$

With $\Upsilon$ set to the default of $0.7$ and typical league settings, this comes out to approximately 

$$
1.7 \sqrt{\chi}
$$

In words, this means that the assumed advantage is about equivalent to one or two players performing a standard deviation above expected