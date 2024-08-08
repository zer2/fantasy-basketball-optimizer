# Static rankings from Z-score to G-score 

Ideally, player rankings should have some mathematical backbone, which requires a metric quantifying player value. The established metric for category leagues is called 'Z-scoring', and it drives most numerical ranking lists. However, when I looked for a mathematical argument explaining why Z-scores are a good choice for fantasy basketball rankings, I could not find one.

This lack of justification inspired me to lay out a mathematical argument for Z-scores myself, which I put forward in the [paper](https://arxiv.org/abs/2307.02188). One surprising wrinkle is that the justification only works given that player performances are known exactly; otherwise, Z-scores are missing a crucial component. To 'fix' it I formulated a new version called G-score. 

I realize that the paper's explanation is a little dense, so I will also provide a simplified version here

## 1.	What are Z-scores?

In the field of statistics, Z-scores are what happens to a set of numbers after subtracting the mean (average) signified by $\mu$ and dividing by the standard deviation (how "spread out" the distribution is) signified by $\sigma$. Mathematically, $Z(x) = \frac{x - \mu}{\sigma}$

For use in fantasy basketball, a few modifications are made to basic Z-scores 
-	The percentage categories are adjusted by volume. This is necessary because players who shoot more matter more; if a team has one player who goes $9$ for $9$ ($100\%$) and another who goes $0$ for $1$ ($0\%$) their aggregate average is $90\%$ rather than $50\%$. The fix is to multiply scores by the player's volume, relative to average volume
-	$\mu$ and $\sigma$ are calculated based on the $\approx 156$ players expected to be on fantasy rosters, rather than the entire NBA
  
Denoting
- Player $p$'s weekly average as $m_p$ 
- $\mu$ of $m_p$ across players expected to be on fantasy rosters as $m_\mu$
- $\sigma$ of $m_p$ across players expected to be on fantasy rosters as $m_\sigma$ 

Z-scores for standard categories (points, rebounds, assists, steals, blocks, three-pointers, and sometimes turnovers) are  

$$
\frac{m_p - m_\mu}{m_\sigma}
$$ 

The same definition can be extended to the percentage categories (field goal % and free throw %). With $a$ signifying attempts and $r$ signifying success rate, their Z-scores are

$$
\frac{\frac{a_p}{a_\mu} \left(r_p - r_\mu \right)}{r_\sigma}
$$

See below for an animation of weekly blocking numbers going through the Z-score transformation step by step. First the mean is subtracted out, centering the distribution around zero, then the standard deviation is divided through to make the distribution more narrow. Note that a set of $156$ players expected to be on fantasy rosters is pre-defined

<iframe width = "896" height = "504" src="https://github.com/zer2/Fantasy-Basketball--in-progress-/assets/17816840/5996da7a-a877-4db1-bb63-c25bed81415f"> </iframe>

Adding up the results for all categories yields an aggregate Z-score

## 2. Justifying Z-scores 

### A. Assumptions and setup

Consider this problem: **Team one has $N-1$ players randomly selected from a pool of players, and team two has $N$ players chosen randomly from the same pool. Which final player should team one choose to optimize the expected value of categories won against team two, assuming all players perform exactly as expected?**

The simplified problem can be approached by calculating the probability for team one to win each category, then optimizing for their sum

### B.	Category differences

The difference in category score between two teams tells us which team is winning the category and by how much. By randomly selecting the $2N -1$ random players many times, we can get a sense of what team two's score minus team one's score will be before the last player is added. See this simulation being carried out for blocks below with $N=12$

<iframe  width = "896" height = "504" src="https://github.com/zer2/Fantasy-Basketball--in-progress-/assets/17816840/73c3acaa-20c9-4a61-907a-ee0de2ff7e3b"> </iframe>

You may notice that the result looks a lot like a Bell curve even though the raw block numbers look nothing like a Bell curve. This happens because of the surprising "Central Limit Theorem", which essentially says that when you add up a bunch of random numbers, the result looks like a Bell curve even if the original random numbers did not. Bell curves are also called Normal or Gaussian distributions. If you're interested in learning more about the CLT, I recommend 3Blue1Brown's video about it, embedded below

<iframe width = "800" height = "450" padding = "none" scrolling="no" src="https://www.youtube.com/embed/zeJD6dqJ5lo
"> </iframe>


### C.	Properties of the category difference

The mean and standard deviation of the Bell curves for category differences can be calculated via probability theory. Including the unchosen player with category average $m_p$
- The mean is $m_\mu - m_p$
- The standard deviation is $\sqrt{2N-1} * m_\sigma$ 

### D.	Calculating probability of victory

When the category difference is below zero, team one will win the category

The probability of this happening can be calculated using something called a cumulative distribution function. $CDF(x) =$ the probability that a particular distribution will be less than $x$. We can use $CDF(0)$, then, to calculate the probability that the category difference is below zero and team one wins. 

The $CDF$ of the Bell curve is well known. Approximately, it can be said that

$$
CDF(0) = \frac{1}{2}\left[ 1 + \frac{2}{\sqrt{\pi}}* \frac{- \mu }{ \sigma} \right]
$$

We already know $\mu$ and $\sigma$ for the standard statistics. Substituting them in yields

$$
CDF(0) = \frac{1}{2}\left[ 1 + \frac{2}{\sqrt{(2N-1) \pi}}* \frac{m_p - m_\mu}{m_\sigma} \right]
$$

And analagously for the percentage statistics 

$$
CDF(0) = \frac{1}{2} \left[ 1 + \frac{2}{\sqrt{(2N-1) \pi}} * \frac{ \frac{a_p}{a_\mu} \left( r_p - r_\mu \right) }{r_\sigma}\right]
$$

### D. Implications for Z-scores

The last two equations included Z-scores. Adding up all the probabilities to get the expected number of categories won by team one, with $Z_{p,c}$ as player $p$'s Z-score for category $c$, the formula is

$$
\frac{1}{2}\left[9 + \frac{2}{\sqrt{(2N-1) \pi}} * \sum_c Z_{p,c} \right]
$$

It is clear that the expected number of category victories is directly proportional to the sum of the unchosen player's Z-scores. This tells us that under the aforementioned assumptions, the higher a player's total Z-score is, the better they are.

One common misconception is that Z-scores are only applicable to Normally distributed data, so using Z-scores for categories that are not distributed Normally across players is inappropriate. This is not the case; the justification for Z-scores laid out here does not require the underlying data to be Normally distributed. The misconception that Normality is required is probably rooted in the concept of the [standard Z-table](https://en.wikipedia.org/wiki/Standard_normal_table), which does require Normally distributed data. But we aren't using a Z-table anywhere so it doesn't matter in this case

## 3. Accounting for uncertainty

In reality, player performances are not known exactly beforehand. To account for that we can imagine randomly choosing a weekly performance for each player, instead of assuming they will perform at a pre-determined level. Essentially this means sampling from player/week pairs instead of sampling just from players. Below, see how metrics for blocks change when we look at every weekly performance of the top $156$ players, instead of just their averages 

<iframe  width = "896" height = "504" src="https://github.com/zer2/Fantasy-Basketball--in-progress-/assets/17816840/ab41db2a-99f2-45b1-8c05-d755c014b30f"> </iframe>

Although the mean remains the same, the standard deviation gets larger. This makes sense, because week-to-week "noise" adds more volatility, which is reflected in the additional $m_\tau$ term. Note that the new standard deviation is $\sqrt{m_\sigma^2 + m_\tau^2}$ rather than $m_\sigma + m_\tau$ because of how standard deviation aggregates across multiple variables, as discussed in section 2B

## 4.	Formulating G-scores 

Most of the logic from section 2 can also be applied to the new model. The only difference is that we need to use metrics from the pool of players and performances, as laid out in section 3, rather than just players as we did in section 2. The mean is still $m_\mu$ as shown in the example of blocks above. Therefore all we need to do is replace $m_\sigma$ with $\sqrt{m_\sigma^2 + m_\tau^2}$, which yields

$$
\frac{m_p - m_\mu}{\sqrt{m_\sigma^2 + m_\tau^2}} 
$$

And analagously for the percentage statistics, 

$$
\frac{\frac{a_p}{a_\mu} \left( r_p - r_\mu \right) }{\sqrt{r_\sigma^2 + r_\tau^2}} 
$$

I call these G-scores, and it turns out that these are quite different from Z-scores. With 2022-23 data, they are as follows

Category | G-score as fraction of Z-score
| -------- | ------- | 
Assists | 75\% 
Blocks | 68\% 
Field Goal % |  56\% 
Free Throw % | 58\% 
Points |  65\% 
Rebounds | 69\% 
Steals | 44\% 
Threes | 72\% 
Turnovers | 62\% 

G-scores are different from Z-scores because different categories have different levels of week-to-week volatility relative to player-to-player volatility. Steals for example are relatively volatile on a week to week basis, and adding the $m_\tau^2$ term makes their denominator quite large, decreasing their overall weight. 

Intuitively, why does this happen? The way I think about it is that investing heavily into a volatile category will lead to only a flimsy advantage, and so is likely less worthwhile than investing into a robust category. Many managers have this intuition already, de-prioritizing unpredictable categories like steals relative to what Z-scores would suggest. The G-score idea just converts that intuition into mathematical rigor
  
## 5.	Head-to-head simulation results

Our logic relies on many assumptions, so we can't be sure that G-scores work in practice. What we can do is simulate actual head-to-head drafts and see how G-score does against Z-score. 

The expected win rate if all strategies are equally good is $\frac{1}{12} = 8.33\%$. Empirical win rates are shown below for *Head-to-Head: Each Category* 9-Cat, which includes all categories, and 8-Cat, a variant which excludes turnovers 

|     | G-score vs 11 Z-score | Z-score vs. 11 G-score|
| -------- | ------- |------- |
| __9-Cat__    |  | |
| 2021  | $15.9\%$   | $1.5\%$  |
| 2022 | $14.3\%$   | $1.3\%$  |
| 2023    | $21.7\%$    | $0.4\%$  |
| Overall    | $17.3\%$    | $1.4\%$ |  
| __8-Cat__    |  | |
| 2021  | $10.7\%$   | $2.9\%$  |
| 2022 | $12.0\%$   | $1.5\%$  |
| 2023    | $15.4\%$    | $0.9\%$  |
| Overall    | $12.7\%$    | $1.8\%$ |  

When interpreting these results, it is important to remember that they are for an idealized version of fantasy basketball. Still, the dominance displayed by G-scores in the simulations suggests that the G-score modification really is appropriate.

To confirm the intuition about why the G-score works, take a look at its win rates by category against $11$ managers using Z-score in 9-Cat

|     | G-score win rate | 
| -------- | ------- |
| Points  | $77.7\%$   | 
| Rebounds | $70.8\%$   | 
| Assists    | $81.6\%$    | 
| Steals    | $25.7\%$    |  
| Blocks  | $44.9\%$   | 
| Three-pointers | $77.3\%$   | 
| Turnovers    | $16.2\%$    | 
| Field goal %    | $34.9\%$    | 
| Free throw %    | $40.6\%$    | 
| Overall   | $52.2\%$    | 

The G-score manager performs well in stable/high-volume categories like assists and poorly in volatile categories like turnovers, netting to an average win rate of slightly above $50\%$. As expected, the marginal investment in stable categories is worth more than the corresponding underinvestment in volatile categories, since investment in stable categories leads to reliable wins and the volatile categories can be won despite underinvestment with sheer luck. 

Simulations also suggest that G-scores work better than Z-scores in the *Head-to-Head: Most Categories* format. I chose not to include the results here because it is a very strategic format, and expecting other managers to go straight off ranking lists is probably unrealistic for it