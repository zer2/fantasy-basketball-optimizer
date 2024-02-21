# The curious case of turnovers 

Conventional wisdom dictates that turnovers should be valued less than other categories when evaluating players. As such, many analysts recommend down-weighting the category by $75\%$ or more by default. There is a seemingly good reason for this, and I understand why it has been the orthodoxy for so long. However, I believe that as well-entrenched this idea is, it is also wrong. 

It is natural to look at results and conclude that turnovers are not important to invest in. Succesful teams often lose turnovers, and poorly performing teams often win them. Of course, seeing that, anyone would naturally start to believe that investing in turnovers is not very helpful. But that is not exactly logical. What matters is the causal relationship between investing in turnovers and doing well: on the margins, does investing in turnovers yield a similar return to investing in other categories? My analysis suggests that it does, which is why this website's default is to treat turnovers like every other category (multiplied by -1 of course). 

My thinking on this topic is not entirely rigorous, which is why I am not including it in the paper or incorporating it into H-scoring. Still, I realize that if I am going to disregard conventional wisdom, I should at least explain my thought my process. I will do that here

## 1. Correlation does not imply causation

It is often noted that teams which perform poorly in turnovers tend to do better overall and vice versa. 

The statement of fact is probably true. However, concluding from it that investing in turnovers does not increase overall win probability is a fallacy because [correlation does not imply causation](https://en.wikipedia.org/wiki/Correlation_does_not_imply_causation). __It may be true that doing well in turnovers is associated with losing overall. However, that does not necessarily mean anything about the causal relationship between the two. Investing in turnovers could still increase the probability of winning as much as investing in any of the other categories does or more.__ 

To see how this fallacy can manifest itself in real data, consider the following simplified results table 

| | Drafter A | Drafter B | Drafter C | Drafter D| 
|:-------|:-------|:-------|:-------|:-------|
|Turnover weight | $100\%$ | $0\%$ | $100\%$ | $0\%$ |
|Player m/g | $40$ | $40$ | $20$ | $20$ |
|Result- turnovers | $20$ | $30$ | $10$ | $20$ | 
|Result- placement | $1$st | $2$nd | $3$rd | $4$th |

Teams that invested more in turnovers made fewer turnovers. Also, teams which made fewer turnovers did worse overall. One might naively conclude from this that investing in turnovers led to worse performance. However- upon closer inspection of the data, it is apparent that investing in turnovers actually improved performance 

<iframe width = "672" height = "378" src="https://github.com/zer2/Fantasy-Basketball--in-progress-/assets/17816840/b167980e-3947-4abe-a4a8-72ae22cff6d1"> </iframe>

This happened because there was a third cause, minutes per game, which both made teams succesful and increased the number of turnovers. It was not making many turnovers which helped teams do well- it was the conditions required for doing well that caused more turnovers. 

This example is not necessarily illustrative of actual fantasy basketball. The point is just that it could be, so we cannot draw conclusions directly from correlations

## 2. Return on investment  

What actually matters is the causal effect of investing in turnovers on overall performance. Let's analyze this, first with intuition and then with some degree of rigor 

### 2A. The intuitive approach

I can imagine two causal mechanisms by which turnovers would be relatively unimportant 

#### Most Categories 

In the Most Categories context, winning turnovers to go 1-8 instead of 0-9 is unhelpful. A loss is a loss. So why invest in turnovers, which are easiest to win when a team is already losing? 

This argument seems plausible on face but is framed incorrectly. Perhaps auto-loss scenarios are not worth considering, but if so, then neither are auto-win scenarios. __The proper way to evaluate category importance is determining how helpful each of them is under the condition that the match-up is close. Winning a consolation category isn't helpful, and neither is running up the score.__

Given that a match-up is close, neither drafter could be dominating the counting statistics. In that circumstance, there is no reason to expect either drafter to have such a significant advantage in turnovers that investing in it would be futile for the other. So intuitively, I would guess that turnovers are just as likely to be the deciding factor as any other category 

<iframe width = "672" height = "378" src="https://github.com/zer2/Fantasy-Basketball--in-progress-/assets/17816840/b0373be6-ce75-4c55-89fc-8ad29939ce24"> </iframe>

#### Relying on a consistent advantage 

In many contexts, drafters need to have some advantage to have any shot at winning. For example, say a league has no playoffs at all, and the top team after the regular season wins the league. Presumably the top drafter would have to come into each matchup with an advantage because they chose better players. 

It would be correct to say that given an across the board advantage, a drafter is more likely to lose turnovers than other categories. However, as already established, correlation does not imply causation. What matters is whether they can improve their overall performances through turnovers as much as through other categories. 

Let's say a drafter has a significant advantage across all categories except turnovers, for which they have a significant disadvantage. One would expect that investing in turnovers would become less likely to translate to more category wins, since the investment would only matter if an opponent had an anomolously bad week for turnovers. However, the other counting statistics would also become less rewarding to invest in, since those investments would only matter if an opponent had an anomolously good week for that category. Ultimately, all of the counting statistics would be less rewarding to invest in. So why single out turnovers? 

### 2B. Modeling the problem mathematically

#### Defining optimal weights 

Let's start by taking a step back and being clear about what category importance and "optimal category weights" really are. The discussion of static ranking lists on the G-score page gives a framework for thinking about proper weighting. It models a situation wherein all players except one have been selected from a pool with arbitrary statistics. The proper weighting is designed so that a player's overall score is proportional to the benefit they incur to the reward function. On an individual category level, the weights then reflect the marginal improvement in the reward function earned by each increment of investment into the categories. This is equivalent to the definition of a partial derivative. So another way to frame the proper weight of a category is the partial derivative of the reward function with respect to investment in that category. 

I calculated this derivative in the paper, and it matches up well with the intuition that category importance should be evaluated under the condition that the match-up is close. For Most Categories, the derivative is the product of two factors
- How likely is it that the other eight categories are tied? I call this situation a "tipping point" for the category
- How likely is it that an incremental improvement in turnovers flips the category from a loss to a win? Or in a more technical sense, what is the probability density of turnovers around zero, conditional on the first criteria?

The intuition here is that the two conditions together specify a situation under which a tiny investment in a category can flip the overall result from a loss to a win. This is analagous to 538's [voter power index](https://projects.fivethirtyeight.com/2016-election-forecast/#tipping-point), a way of quantifying the importance of voters under the electoral college. The importance of a particular voter is equal to the probability that they can flip the result of their state and flipping that state flips the result of the electoral college.

For Each Category and Rotisserrie, the analysis is even simpler. The partial derivative is just the probability density of each category around zero.

The most reliable way for a drafter to obtain a consistent advantage is by choosing players who get more playing time than expected. This gives them some advantage in all of the counting statistics. So we can set up the distribution such that one drafter has a constant advantage across all of the counting statistics (disadvantage for turnovers). 

#### Modeling performance across partial derivatives

To estimate the required partial derivatives, we need a distribution that can incorporate how the various categories relate to each other. And for that, we need a correlation matrix. 

Correlation is a measure of how related two metrics are. When two metrics tend to be either both high or both low, they are highly correlated. When they tend to be either high/low or low/high, they are negatively correlated. When they are totally unrelated, they are uncorrelated, or have a correlation of zero.  A correlation matrix contains the pairwise correlations between many metrics. For the 2023 season, with scores normalized by week-to-week variance (and turnovers muliplied by -1), I calculated the correlation matrix to be
                                                        
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

As expected, turnovers are negatively correlated with the other counting statistics to a unique degree.

This correlation matrix can be used to parameterize a multivariate normal distribution to approximate the score differential between two teams. (Technically I calculated this as the correlation matrix for individual players and not for differentials between teams. Fortunately the two are equivalent, since correlation, variance, and covariance are all bilinear)

To model the advantage state, we can use a small positive number as the mean for the non-turnover counting statistic differentials, and negative the same number for the mean of the turnover differential

#### Results of the math for Most Categories

The probability of both derivative criteria occuring can be estimated by sampling from the distribution many times. For each scenario with five category wins, all of the winning categories are considered tipping points. For each scenario with four category wins, the losing categories are considered tipping points. Then after the tipping points are identified, the probability of the tipping point category being around zero is estimated 

 | Likelihood of winning the matchup  | Points    | Rebounds    | Assists    | Steals    | Blocks    | Threes    | Turnovers    | Free Throw \%   | Field Goal \%   |
|-----:|:-------|:------|:------|:------|:------|:------|:------|:---------|:---------|
| 50.0\%   | 10.3\% | 6.8\% | 6.2\% | 9.0\% | 7.1\% | 6.6\% | 7.2\% | 7.1\%    | 7.4\%    |
| 59.7\% | 10.0\% | 7.4\% | 6.7\% | 8.6\% | 5.9\% | 6.8\% | 7.0\% | 6.9\%    | 7.1\%    |
| 68.9\% | 9.1\%  | 6.4\% | 6.1\% | 8.0\% | 5.6\% | 6.0\% | 6.5\% | 6.4\%    | 6.3\%    |
| 77.1\% | 8.4\%  | 5.5\% | 5.1\% | 6.4\% | 4.6\% | 5.3\% | 5.2\% | 5.0\%    | 5.5\%    |
| 83.9\% | 6.5\%  | 4.6\% | 4.3\% | 5.3\% | 3.8\% | 4.1\% | 4.0\% | 4.3\%    | 4.4\%    |

As expected intuitively, turnovers are about as important as other categories. They are slightly on the lower end under advantage states, but this effect is minimal even when the advantage states are extreme

#### Results of the math for Each Category and Rotisserie

Analyzing for this case is easier, since the partial derivative is just the density of the corresponding probability density function around zero. This can be directly estimated from the same experiment

 | Average category winning %  | Points    | Rebounds    | Assists    | Steals    | Blocks    | Threes    | Turnovers    | Free Throw \%   | Field Goal \%   |
|-----:|:-------|:------|:------|:------|:------|:------|:------|:---------|:---------|
| 50.0\%   | 34.0\% | 30.9\% | 27.7\% | 36.4\% | 30.6\% | 31.0\% | 33.4\% | 33.4\%   | 34.4\%   |
| 54.3\% | 32.3\% | 30.9\% | 29.1\% | 36.4\% | 30.8\% | 30.3\% | 33.7\% | 33.4\%   | 34.4\%   |
| 58.5\% | 29.6\% | 29.1\% | 27.3\% | 33.4\% | 28.2\% | 28.9\% | 30.9\% | 33.4\%   | 34.4\%   |
| 62.3\% | 27.2\% | 27.0\% | 24.8\% | 29.2\% | 25.8\% | 26.0\% | 27.0\% | 33.4\%   | 34.4\%   |
| 65.8\% | 22.8\% | 23.2\% | 21.7\% | 24.5\% | 23.0\% | 22.1\% | 24.2\% | 33.4\%   | 34.4\%   |

As expected, turnovers decline in importance at a similar rate as the other counting statistics. 

It is interesting to note that the percentage statistics have outsize importance in situations where one drafter has a playing time advantage. With the counting statistics largely shored up, the percentage statistics, which are unbiased by playing time, are still just as difficult to win and become relatively more important. 

## 3. One other argument: turnovers are volatile

One final argument is that turnovers are hard to predict on a week-to-week basis, and therefore are not worth investing in for head-to-head formats.

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
