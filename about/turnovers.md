# The curious case of turnovers 

The conventional wisdom is that turnovers are relatively unimportant, and as such should be downweighted by $~75\%$. I am not convinced. 

It is natural to look at results and conclude that turnovers are not important to invest in. So many winning teams lose turnovers, and so many losing teams win it. Of course, seeing that, anyone would naturally start to believe that investing in turnovers is not very helpful or important. But that is not exactly logical. What matters is the causal relationship between investing in turnovers and doing well: on the margins, does investing in turnovers help you do better as much as investing in other categories? My analysis suggests that it does, which is why this website's default is to treat turnovers like every other category (multiplied by -1 of course). 

My thinking on this topic is not entirely rigorous, which is why I am not including it in the paper or incorporating it into H-scoring. Still, I realize that deciding how to treat turnovers is important, and that my default goes against the grain of community wisdom. So I will walk through my thought process here 

## 1. Correlation does not imply causation

The simplest argument for ignoring turnovers is that teams which perform poorly in turnovers tend to do better overall and vice versa. The implication is that investing more in turnovers must make you more likely to lose, or at least not be enormously helpful. 

The statement of fact is incontrovertibly true. However, concluding from it that investing in turnovers does not increase overall win probability is a fallacy because [correlation does not imply causation](https://en.wikipedia.org/wiki/Correlation_does_not_imply_causation). __It may be true that doing well in turnovers is associated with losing overall. However, that does not necessarily mean anything about the causal relationship between the two. Investing in turnovers could still increase the probability of winning as much as investing in any of the other categories does or more.__ 

To see how this fallacy can manifest itself in real data, consider the following simplified results table 

| | Drafter A | Drafter B | Drafter C | Drafter D| 
|:-------|:-------|:-------|:-------|:-------|
|Turnover weight | $100\%$ | $0\%$ | $100\%$ | $0\%$ |
|Player m/g | $40$ | $40$ | $20$ | $20$ |
|Result- turnovers | Middling | Bad | Good | Middling | 
|Result- placement | First | Second | Third | Fourth |

Ignoring the player minutes per game numbers, one might naively infer that doing better in turnovers leads to doing worse overall. The evidence is: 
  - Performing well in turnovers leads to an average placement of third place (drafter C finished third)
  - Performing average in turnovers leads to an average placement of between second and third place (drafter A finished first, drafter D finished fourth)
  - Performing badly in turnovers leads to an average of second place (drafter B finished second) 
 
However, this backwards-causality approach is not logical because the appearance of causality is an illusion. Performing badly in turnovers didn't help the drafters succeed- their players getting more minutes per game did, and that harmed their performance in turnovers. In fact, when accounting for minutes per game, the opposite effect is uncovered. Drafter A did better than drafter B and drafter C did better than drafter D

## 2. Return on investment  

What matters is the causal effect of investing in turnovers on overall performance. Let's analyze this, first with intuition and then with some degree of rigor 

### 2A. Thinking it through intuitively 

I can imagine two causal mechanisms by which turnovers would be relatively unimportant 

#### Most Categories 

In the Most Categories context, winning turnovers to go 1-8 instead of 0-9 is unhelpful. A loss is a loss. So why invest in turnovers, which are easiest to win when you are already losing? 

This argument seems plausible on face but is framed incorrectly. Perhaps auto-loss scenarios are not worth considering, but if so, then neither are auto-win scenarios. __The proper way to evaluate category importance is determining how helpful each of them is under the condition that the match-up is close. Winning a consolation category isn't helpful, and neither is running up the score. It is under this framework that the question of turnover importance should be evaluated.__

Given that a match-up is close, neither drafter could be dominating the counting statistics. In that circumstance, there is no reason to believe that either drafter is likely to have such a significant advantage in turnovers that investing in turnovers would be futile to the other. So intuitively, I would guess that turnovers are just as likely to be the deciding factor as any other categories 

#### Relying on a consistent advantage 

In many contexts, drafters need to have some advantage to have any shot at winning. For example, say a league has no playoffs at all, and the top team after the regular season wins the league. Presumably the top drafter would have to come into each matchup with an advantage because they chose better players. 

It would be correct to say that given an across the board advantage, a drafter is more likely to lose turnovers than other categories. However, as discussed before, it is a fallacy to imply anything causal from that statement.

Let's say a drafter has a significant advantage across all categories except turnovers, for which they have a significant disadvantage. It is undeniable that investing in turnovers would become less likely to improve performance, because it would be difficult to overcome their significant disadvantage. However, there is also a countervailing effect: the other counting statistics would also become less rewarding to invest in, because there would be minimal room to improvement. If you were already winning say rebounds nine times out of ten, how much could you really gain by investing more in it? Intuitively, I would guess that these effects are roughly equivalent and so would cancel each other out

### 2B. Modeling the problem mathematically

#### Defining optimal weights 

Let's start by taking a step back and being clear about what category importance and "optimal category weights" really are. The discussion of static ranking lists on the G-score page gives a framework for thinking about proper weighting. It models a situation wherein all players except one have been selected from a pool with arbitrary statistics. The proper weighting is designed so that a player's overall score is proportional to the benefit they incur to the reward function. On an individual category level, the weights then reflect the marginal improvement in the reward function earned by each increment of investment into the categories. This is equivalent to the definition of a partial derivative. So another way to frame the proper weight of a category is the partial derivative of the reward function with respect to investment in that category. 

I calculated this derivative in the paper, and it matches up well with the intuition that category importance should be evaluated under the condition that the match-up is close. For Most Categories, the derivative is the product of two factors
- How likely is it that the other eight categories are tied? I call this situation a "tipping point" for the category
- How likely is it that an incremental improvement in turnovers flips the category from a loss to a win? Or in a more technical sense, what is the probability density of turnovers around zero, conditional on the first criteria?

The intuition here is that the two conditions together specify a situation under which a tiny investment in a category can flip the overall result from a loss to a win. This is analagous to 538's [voter power index](https://projects.fivethirtyeight.com/2016-election-forecast/#tipping-point), a way of quantifying the importance of voters under the electoral college. The importance of a particular voter is equal to the probability that they can flip the result of their state and flipping that state flips the result of the electoral college.

For Each Category and Rotisserrie, the analysis is even simpler. The partial derivative is just the probability density of each category around zero

The most reliable way for a drafter to obtain a consistent advantage is by choosing players who get more playing time than expected. This gives them some advantage in all of the counting statistics. So we can set up the distribution such that one drafter has a constant advantage across all of the counting statistics (disadvantage for turnovers). 

#### Modeling performance across partial derivatives

To estimate the required partial derivatives, we need a distribution that can incorporate how the various categories relate to each other. And for that, we need a correlation matrix. 

Correlation is a measure of how related two metrics are. When two metrics tend to be either both high or both low, they are highly correlated. When they tend to be either high/low or low/high, they are negatively correlated. When they are totally unrelated, they are uncorrelated, or have a correlation of zero.  A correlation matrix contains the pairwise correlations between many metrics. For the fantasy basketball setting, with scores normalized by week-to-week variance (and turnovers muliplied by -1), the correlation matrix is 
                                                        
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

This correlation matrix can be used to parameterize a multivariate normal distribution to approximate the score differential between two teams. (Technically I calculated this as the correlation matrix for individual players and not for differentials between teams. Fortunately the two are equivalent, since correlation, variance, and covariance are all bilinear)

Simulating the multivariate distribution with that correlation matrix, it is easy to confirm the expected result, that turnovers are usually lost in overall wins and won in overall losses

 |        | Points    | Rebounds    | Assists    | Steals    | Blocks    | Threes    | Turnovers    | Free Throw \%   | Field Goal \%   |
 |:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:---------|:---------|
 | Loss | 18.3\% | 28.7\% | 29.2\% | 28.8\% | 34.3\% | 29.2\% | 61.6\% | 39\%   | 36.9\% |
 | Win | 81.6\% | 71.2\% | 70.5\% | 71.1\% | 65.4\% | 71\%   | 38.6\% | 61.1\% | 62.8\% |

To see the influence of advantage states, we can model the distribution with a constant added to all of the counting statistics (including turnovers, for which it is a disadvantage). This will increase the overall probability of winning, and perhaps change importances. 

The probability of both criteria occuring can be estimated by sampling from the distribution many times. For each scenario with five category wins, all of the winning categories are considered tipping points. For each scenario with four category wins, the losing categories are considered tipping points. Then after the tipping points are identified, the probability of the tipping point category being around zero is estimated 

#### Results of the math for Most Categories

 | Likelihood of winning the matchup  | Points    | Rebounds    | Assists    | Steals    | Blocks    | Threes    | Turnovers    | Free Throw \%   | Field Goal \%   |
|-----:|:-------|:------|:------|:------|:------|:------|:------|:---------|:---------|
| 50.0\%   | 10.3\% | 6.8\% | 6.2\% | 9.0\% | 7.1\% | 6.6\% | 7.2\% | 7.1\%    | 7.4\%    |
| 59.7\% | 10.0\% | 7.4\% | 6.7\% | 8.6\% | 5.9\% | 6.8\% | 7.0\% | 6.9\%    | 7.1\%    |
| 68.9\% | 9.1\%  | 6.4\% | 6.1\% | 8.0\% | 5.6\% | 6.0\% | 6.5\% | 6.4\%    | 6.3\%    |
| 77.1\% | 8.4\%  | 5.5\% | 5.1\% | 6.4\% | 4.6\% | 5.3\% | 5.2\% | 5.0\%    | 5.5\%    |
| 83.9\% | 6.5\%  | 4.6\% | 4.3\% | 5.3\% | 3.8\% | 4.1\% | 4.0\% | 4.3\%    | 4.4\%    |

Turnovers have slightly lower importance than average in advantage states, but this effect is slight even when the advantage is extreme. This is likely because no matter how large the advantage state is, if a matchup ends up close, then no drafter could have dominated the counting statistics. Given that the counting statistics were close, one would expect turnovers to also be relatively close. 

#### Results of the math for Each Category and Rotisserie

 | Average category winning %  | Points    | Rebounds    | Assists    | Steals    | Blocks    | Threes    | Turnovers    | Free Throw \%   | Field Goal \%   |
|-----:|:-------|:------|:------|:------|:------|:------|:------|:---------|:---------|
| 50.0\%   | 34.0\% | 30.9\% | 27.7\% | 36.4\% | 30.6\% | 31.0\% | 33.4\% | 33.4\%   | 34.4\%   |
| 54.3\% | 32.3\% | 30.9\% | 29.1\% | 36.4\% | 30.8\% | 30.3\% | 33.7\% | 33.4\%   | 34.4\%   |
| 58.5\% | 29.6\% | 29.1\% | 27.3\% | 33.4\% | 28.2\% | 28.9\% | 30.9\% | 33.4\%   | 34.4\%   |
| 62.3\% | 27.2\% | 27.0\% | 24.8\% | 29.2\% | 25.8\% | 26.0\% | 27.0\% | 33.4\%   | 34.4\%   |
| 65.8\% | 22.8\% | 23.2\% | 21.7\% | 24.5\% | 23.0\% | 22.1\% | 24.2\% | 33.4\%   | 34.4\%   |

Again, turnovers decline in importance at a similar rate as the other counting statistics. One way to explain this is that while it becomes harder to win turnovers as your advantage increases, you also lose room for improvement in the non-turnover counting statistics. If you want to eke out a small additional advantage, dominating the counting statistics to such an extreme degree that they are auto-wins may be just as difficult as making a few improbable turnover wins more possible. 

Interestingly, it turns out that the percentage statistics have outsize importance in situations where one drafter has a playing time advantage. In retrospect this is obvious: with the counting statistics largely shored up, the percentage statistics, which are unbiased by playing time, are still just as difficult to win and become relatively more important. 

## 3. One other argument: turnovers are volatile

One final argument is that turnovers are hard to predict on a week-to-week basis, and therefore are not worth investing in for head-to-head formats

It is true that turnovers are relatively volatile from week to week. However, this is not unique; all categories have some level of week to week volatility. Turnovers are not even most volatile category. Steals are, by a wide margin. 

G-scores deal with this by incorporating week-to-week variance. They do downweight turnovers relative Z-scores, but not in an extreme way

## 4. Testing 

To some degree, the hypothesis that down-weighting turnovers is uniquely beneficial can be tested. I ran a test with the following setup
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

It should be noted that this test does not cover argument the circumstance under which one drafter has an overall advantage, because all player statistics are known beforehand, making it imposssible for any drafter to have a surprisingly good team. There remains the possibility that in a real league, with some uncertainty about how players are going to perform, the best-positioned drafters will have a significant advantage in general. If the heuristic of section 2B is wrong or misleading, perhaps turnovers become significantly less important in that case

## 5. Conclusion

I have not seen a convincing argument that turnovers should be down-weighted to an extreme degree. That's why I've set the default to treating turnovers like every other category. 

Still, absence of evidence is not evidence of absence, and there might be some more nuanced reason that turnovers should be downweighted not captured here. If you want to use this website and want to downweight turnovers, feel free to manually set the turnover multiplier on the parameters page.

Also, keep in mind that many other drafters ignore turnovers. Winning them reliably may take only a small investment while grabbing every player whose strength is turnovers would be overkill. So in practice you might want to downweight turnovers a bit, perhaps to $50\%$
