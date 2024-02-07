# Diet soda, the electoral college, and turnovers

Turnovers are unique in fantasy basketball as the only "negative" category. This unique property makes turnovers a weird category to strategize around and leads many fantasy basketball analysts to recommend manually down-weighting the turnovers category by $75\%$ or even $100\%$.

There are real arguments to be made for this approach. However, my own analysis has led me to believe that none of the arguments are strong enough to warrant such an extreme treatment. As such, the website's default is to weigh turnovers as normal. 

This article is my attempt to explain why none of the arguments are convincing to me. As it turns out, the underlying logic gets *very interesting*. It touches on concepts that are important to seemingly unrelated topics, including the effect of diet soda on health and how voter power is distributed by the electoral college. My exploration here is wide-reaching and not enomously rigorous. So think of this more as a description of my own thought process, rather than a rigporous argument

## 1. The first argument: low-turnover teams tend to lose 

The most common argument for ignoring turnovers, especially for the Most Categories format, is that teams which perform poorly in turnovers tend to do better overall and vice versa. 

The statement of fact is incontrovertibly true. However, concluding that doing poorly in turnovers causally leads to success is a fallacy of [reverse causality](https://en.wikipedia.org/wiki/Correlation_does_not_imply_causation). __Just because losses in turnovers are associated with overall wins does not mean that trying to win turnovers is any less helpful for winning overall than trying to win other categories__

### Unpacking the fallacy 

If this concept is difficult to grasp intuitively, you shouldn't feel bad, because it is difficult for professional researchers and science reporters too. My favorite example of this is the oft-reported link between diet soda and obesity. Many studies have found a strong association between drinking diet soda and becoming obese, leading many researchers to suggest that some hidden mechanism makes drinking diet soda unhealthy. However, a [meta-analysis](https://academic.oup.com/nutritionreviews/article/71/7/433/1807065) summarizing all research suggests that this link is most likely ephemeral, and the result of a reverse-causality fallacy. It is true that people who drink diet soda are more likely to gain weight. But those are also the same people most likely to be worried about their weight because of individual risk factors such as activity level or genetics. When controlling for those factors, the link disappears or even reverses. In other words- it is not drinking diet soda that makes people obese, but the risk factors for becoming obese that make someone drink diet soda. If anything, drinking diet soda likely decreases obesity risk. It only appears to be the reverse in the data because of a failure to control for countervailing factors. 

The fantasy basketball equivalent of this statement is that it is not de-prioritizing turnovers that leads to success. Instead, it is the conditions that lead to success which also lead to poor scores for turnovers. 

To see how this fallacy can manifest itself in real data, consider the following simplified results table 

| | Drafter A | Drafter B | Drafter C | Drafter D| 
|:-------|:-------|:-------|:-------|:-------|
|Turnover weight | $100\%$ | $0\%$ | $100\%$ | $0\%$ |
|Player m/g | $40$ | $40$ | $20$ | $20$ |
|Result- turnovers | Middling | Bad | Good | Middling | 
|Result- placement | First | Second | Third | Fourth |

Ignoring the player minutes per game numbers, one might naively infer two effects
- 1: Investing in turnovers leads to a better performance in turnovers. The evidence is
  - Drafters A and C invested in turnovers, and were middling/good in them
  - Drafters B and D did not invest in turnovers, and were middling/bad in them 
- 2: Doing better in turnovers leads to doing worse in the league. The evidence is
  - Performing well in turnovers leads to an average placement of third place (drafter C finished third)
  - Performing average in turnovers leads to an average placement of between second and third place (drafter A finished first, drafter D finished fourth)
  - Performing badly in turnovers leads to an average of second place (drafter B finished second) 
 
Putting these two inferences together makes a seemingly solid case that investing in turnovers leads to bad performance. 

However, this backwards-causality approach is not logical because the second inference is an illusion of reverse causality. Performing badly in turnovers didn't help the drafters succeed- their players getting more minutes per game did, and that harmed their performance in turnovers.

In fact, when accounting for minutes per game, the opposite effect is uncovered. Of the two drafters whose players played $40$ minutes per game, drafter A, who cared about turnovers, did better. And of the two drafters whose players played $20$ minutes per game, drafter C, who cared about turnovers, also did better. So the simple way of looking at the data was totally misleading and directionally incorrect. 

This will not necessarily be the case in real fantasy basketball. The point here is that raw results may be misleading and should not be over-interpreted

### Introducing math  

The intuitive version of the argument is not entirely clear. Can a mathematical approach bring clarity? 

First, let's establish a theoretical framework which captures the idea that turnovers are generally lost by winning teams. This can be done by finding the correlation matrix of category scores then modeling the overall distribution as a corresponding multivariate normal

Correlation is a measure of how related two metrics are. When two metrics tend to be either both high or both low, they are highly correlated. When they tend to be either high/low or low/high, they are negatively correlated. when they are totally unrelated, they are uncorrelated, or have a correlation of zero.  A correlation matrix contains the pairwise correlations between many metrics. For the fantasy basketball setting, with scores normalized by week-to-week variance (and turnovers muliplied by -1), the correlation matrix is 
                                                        
 |        | Points    | Rebounds    | Assists    | Steals    | Blocks    | Threes    | Turnovers    | Free Throw \%   | Field Goal \%   |
 |:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:---------|:---------|
 | Points    | 100.0\% | 47.2\%  | 57.7\%  | 40.1\%  | 18.6\%  | 63.0\%  | -66.5\% | 17.5\%    | 19.0\%    |
 | Rebounds    | 47.2\%  | 100.0\% | 24.1\%  | 21.6\%  | 46.9\%  | 2.5\%   | -41.4\% | -20.9\%   | 27.5\%    |
 | Assists    | 57.7\%  | 24.1\%  | 100.0\% | 41.1\%  | -4.1\%  | 35.5\%  | -63.0\% | 11.0\%    | -9.2\%    |
 | Steals    | 40.1\%  | 21.6\%  | 41.1\%  | 100.0\% | 8.9\%   | 28.6\%  | -36.0\% | 6.5\%     | -6.8\%    |
 | Blocks    | 18.6\%  | 46.9\%  | -4.1\%  | 8.9\%   | 100.0\% | -7.4\%  | -12.5\% | -14.8\%   | 24.0\%    |
 | Threes    | 63.0\%  | 2.5\%   | 35.5\%  | 28.6\%  | -7.4\%  | 100.0\% | -34.2\% | 21.0\%    | -11.6\%   |
 | Turnovers    | -66.5\% | -41.4\% | -63.0\% | -36.0\% | -12.5\% | -34.2\% | 100.0\% | -4.7\%    | 1.2\%     |
 | Free Throw \% | 17.5\%  | -20.9\% | 11.0\%  | 6.5\%   | -14.8\% | 21.0\%  | -4.7\%  | 100.0\%   | -13.8\%   |
 | Field Goal \% | 19.0\%  | 27.5\%  | -9.2\%  | -6.8\%  | 24.0\%  | -11.6\% | 1.2\%   | -13.8\%   | 100.0\%   |

(You might note that this is the correlation matrix for individual players and not for teams. Fortunately the two are equivalent, since correlation, variance, and covariance are all bilinear)

Simulating the multivariate distribution with that correlation matrix, it is easy to confirm the expected result, that turnovers are usually lost in overall wins and won in overall losses

 |        | Points    | Rebounds    | Assists    | Steals    | Blocks    | Threes    | Turnovers    | Free Throw \%   | Field Goal \%   |
 |:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:---------|:---------|
 | Loss | 18.3\% | 28.7\% | 29.2\% | 28.8\% | 34.3\% | 29.2\% | 61.6\% | 39\%   | 36.9\% |
 | Win | 81.6\% | 71.2\% | 70.5\% | 71.1\% | 65.4\% | 71\%   | 38.6\% | 61.1\% | 62.8\% |

The same effect would be observed in Each Categories, with a higher number of categories won being associated with losng turnovers more often. 
 
However, as already established, just because winning turnovers is associated with losses does not mean that the effect is causal. 

Let's take a step back and be clear about what "optimal category weights" really are. The discussion of static ranking lists on the G-score page gives a framework for thinking about proper weighting. It models a situation wherein all players except one have been selected from a pool with arbitrary statistics. The proper weighting is designed so that a player's overall score is proportional to the benefit they incur to the reward function. On an individual category level, the weights then reflect the marginal improvement in the reward function earned by each increment of investment into the categories. This is equivalent to the definition of a partial derivative. So another way to frame the proper weight of a category is the partial derivative of the reward function (in this case, the probability of winning a matchup) with respect to investment in that category. 

 Partial derivatives can be calculated based on the model above. I did the math on them in the paper. 
 
 For Most Categories, the partial derivatives and boil down to two factors multiplied together: 
- How likely is it that the other eight categories are tied? I call this situation a "tipping point" for the category
- How likely is it that an incremental improvement in turnovers flips the category from a loss to a win? Or in a more technical sense, what is the probability density of turnovers around zero, conditional on the first criteria?

The intuition here is that the two conditions together specify a situation under which a tiny investment in a category can flip the overall result from a loss to a win. This is analagous to 538's [voter power index](https://projects.fivethirtyeight.com/2016-election-forecast/#tipping-point), a way of quantifying the importance of voters under the electoral college. The importance of a particular voter is equal to the probability that they can flip the result of their state and flipping that state flips the result of the electoral college. 

The probability of both criteria occuring can be estimated by approximating the values of all categories as multivariate normals with mean zero and sampling from the distribution many times. For each scenario with five category wins, all of the winning categories are considered tipping points. For each scenario with four category wins, the losing categories are considered tipping points. Then after the tipping points are identified, the probability of the tipping point category being around zero is estimated. I tried this and got

 | Points    | Rebounds    | Assists    | Steals    | Blocks    | Threes    | Turnovers    | Free Throw \%   | Field Goal \%   |
|:------|:------|:------|:------|:------|:------|:------|:---------|:---------|
| 10.3\% | 6.8\% | 6.2\% | 9.0\% | 7.1\% | 6.6\% | 7.2\% | 7.1\%    | 7.4\%    |

These can be considered proper weights, for this simplified model. Note that despite turnovers being associated with losses in aggregate, the reward for investing in turnovers is comparable to the reward for investing in other categories! 

For each Category, the analysis is even simpler. The partial derivative is just the probability density of each category around zero. Re-using the experimental apparatus, the result is 

 | Points    | Rebounds    | Assists    | Steals    | Blocks    | Threes    | Turnovers    | Free Throw \%   | Field Goal \%   |
|:------|:------|:------|:------|:------|:------|:------|:---------|:---------|
| 33.0\% |34.0\% | 30.9\% | 27.7\% | 36.4\% | 30.6\% | 31.0\% | 33.4\% | 33.4\%   | 34.4\%   |

Unsurprisingly, the importances are almost all the same 

## 2. The second argument: banking on overperformance

One might note that the math in the last section was predicated on neither drafter having an advantage in any category coming into the week. That assumption is arguably problematic, because in many contexts, drafters need to have some advantage to have any shot at winning. For example, say a league has no playoffs at all, and the top team after the regular season wins. Presumably the top drafter will have to come into each matchup with an advantage because they chose better players. 

The most reliable way for a drafter to obtain a consistent advantage is by choosing players who get more playing time than expected. This gives them some advantage in all of the counting statistics. 

The math from the previous section can be expanded to the situation where one drafter has an advantage in the counting stats coming into the week. I added a small constant to each of the counting stats (including turnovers), then re-ran the experiment, noting what percent of matchups were overall victories for each advantage state. The results are as follows

For Most Categories 

 | Likelihood of winning the matchup  | Points    | Rebounds    | Assists    | Steals    | Blocks    | Threes    | Turnovers    | Free Throw \%   | Field Goal \%   |
|-----:|:-------|:------|:------|:------|:------|:------|:------|:---------|:---------|
| 50\%   | 10.3\% | 6.8\% | 6.2\% | 9.0\% | 7.1\% | 6.6\% | 7.2\% | 7.1\%    | 7.4\%    |
| 59.7\% | 10.0\% | 7.4\% | 6.7\% | 8.6\% | 5.9\% | 6.8\% | 7.0\% | 6.9\%    | 7.1\%    |
| 68.9\% | 9.1\%  | 6.4\% | 6.1\% | 8.0\% | 5.6\% | 6.0\% | 6.5\% | 6.4\%    | 6.3\%    |
| 77.1\% | 8.4\%  | 5.5\% | 5.1\% | 6.4\% | 4.6\% | 5.3\% | 5.2\% | 5.0\%    | 5.5\%    |
| 83.9\% | 6.5\%  | 4.6\% | 4.3\% | 5.3\% | 3.8\% | 4.1\% | 4.0\% | 4.3\%    | 4.4\%    |

It does appear to be the case that with an increasing advantage, turnovers become less likely to be a tipping point relative to other categories. However this effect is small even when the advantage is extreme. Intuitively this makes sense because no matter how large the advantage state is, tipping points for all categories always require there to be a split among the counting statistics. Given that condition, there is no particular reason to expect that turnovers would be tipping points less often, or that tipping points would have low.

For Each Categories 

 | Averagte category winning %  | Points    | Rebounds    | Assists    | Steals    | Blocks    | Threes    | Turnovers    | Free Throw \%   | Field Goal \%   |
|-----:|:-------|:------|:------|:------|:------|:------|:------|:---------|:---------|
| 50\%   | 34.0\% | 30.9\% | 27.7\% | 36.4\% | 30.6\% | 31.0\% | 33.4\% | 33.4\%   | 34.4\%   |
| 54.3\% | 32.3\% | 30.9\% | 29.1\% | 36.4\% | 30.8\% | 30.3\% | 33.7\% | 33.4\%   | 34.4\%   |
| 58.5\% | 29.6\% | 29.1\% | 27.3\% | 33.4\% | 28.2\% | 28.9\% | 30.9\% | 33.4\%   | 34.4\%   |
| 62.3\% | 27.2\% | 27.0\% | 24.8\% | 29.2\% | 25.8\% | 26.0\% | 27.0\% | 33.4\%   | 34.4\%   |
| 65.8\% | 22.8\% | 23.2\% | 21.7\% | 24.5\% | 23.0\% | 22.1\% | 24.2\% | 33.4\%   | 34.4\%   |

There is an interesting takeaway from this analysis, but it has nothing to do with turnovers! It turns out that the percentage statistics have outsize importance in situations where one drafter has a playing time advantage. In retrospect this is obvious: with the counting statistics largely shored up, the percentage statistics, which are unbiased by playing time, are still just as difficult to win and become relatively more important. 

## 3. The third argument: turnovers are volatile

One final argument is that turnovers are hard to predict on a week-to-week basis, and therefore are not worth investing in. 

It is true that turnovers are relatively volatile from week to week. However, this is not unique; all categories have some level of week to week volatility. Turnovers are not even most volatile category. Steals are, by a wide margin. 

G-scores deal with this by incorporating week-to-week variance. They do downweight turnovers relative Z-scores, but not in an extreme way

## 4. Testing 

To some degree, the hypothesis that down-weighting turnovers can be tested. I ran a test with the following setup
- For each category
  - Divide all drafters into two groups of six
  - One group downweights the category by a certain factor. The other does not
  - For each sequential arrangement of seats (6 down-weighters/6 normals, or 1 normal/6 down-weighters/5 normals, etc. )
    - One thousand seasons are simulated by sampling from the actual season of data
    - The team with the highest regular season record wins

Down-weighting a single category can be a good strategy in general- it is essentially punting. If down-weighting turnovers is a uniquely important measure to take, then the benefit of down-weighting turnovers would be greater than the benefits of down-weighting other categories

The results are as follows for the turnover down-weighter's win rate:

| Weight |  Most Categories | Each Category | 
|----:|:------|:------|
| 0\% | 10.6\% | 12.8\% |
| 25\% | 9.2\% | 11.3\% |
| 50\% | 7.9\% | 9.9\% |
| 75\% | 6.1\% | 8.3\% |

Versus for points 

| Weight |  Most Categories | Each Category | 
|----:|:------|:------|
| 0\% | 12.0\% | 12.1\% |
| 25\% | 10.6\% | 10.5\% |
| 50\% | 10.3\% | 9.7\% |
| 75\% | 11.1\% | 9.8\% |

And for rebounds 

| Weight |  Most Categories | Each Category | 
|----:|:------|:------|
| 0\% | 11.0\% | 11.4\% |
| 25\% | 9.5\% | 10.4\% |
| 50\% | 8.6\% | 9.5\% |
| 75\% | 8.8\% | 9.1\% |

Punting or soft-punting turnovers is about as beneficial as punting other categories. With this evidence, there is no reason to treat it differently. 

It should be noted that this test does not cover the third argument because all player statistics are known beforehand. There remains the possibility that in a real league, with some uncertainty about how players are going to perform, the best-positioned drafters will have a significant advantage in general. If the heuristic of section three is wrong or misleading, perhaps turnovers become significantly less important in that case

## 5. Conclusion

I have not seen a convincing mathematical argument that turnovers should be down-weighted to an extreme degree. That's why I've set the default to treating turnovers like every other category. 

Still, absence of evidence is not evidence of absence, and there might be some more nuanced reason that turnovers should be downweighted not captured here. If you want to use this website and want to downweight turnovers, feel free to manually set the turnover multiplier on the parameters page

