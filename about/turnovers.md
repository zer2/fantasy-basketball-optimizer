# The curious case of turnovers 

Turnovers are weird. Typically you want your players to play more so that they can add points, blocks, steals, etc. But the best thing for the turnovers category is having your players ride the bench. 

Including the turnovers category in player rankings punishes the stars for being stars and rewards irrelevant players who don't play much. This feels wrong to many, including professional analysts. As a result, they often recommend manually down-weighting the category by $75\%$ or even $100\%$ relative to what Z-scores would otherwise tell them.

There are real arguments to be made for this approach. However, my own analysis has led me to believe that none of the arguments are strong enough to warrant such an extreme treatment. As such, the website's default is to weigh turnovers as normal. 

My thinking on this topic is not entirely rigorous, which is why I am not including it in the paper or incorporating it into H-scoring. Still, I realize that deciding how to treat turnovers is an important part of fantasy drafting, and that my default goes against the grain of community wisdom. So I will go through the arguments against considering turnovers equally here, and explain why they are not convincing to me

## 1. The first argument: strong turnover teams tend to lose 

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
 
However, this backwards-causality approach is not logical because the appearance of causality is an illusion. Performing badly in turnovers didn't help the drafters succeed- their players getting more minutes per game did, and that harmed their performance in turnovers. In fact, when accounting for minutes per game, the opposite effect is uncovered. Drafter A did better than drafter B and drafter C did better than drafter D. 

## 2. The second argument: Winners won't win turnovers anyway

A second argument is that because winning teams tend to perform badly in turnovers, they will usually be so far behind in turnovers that investing in turnovers is futile for them. 

There are two versions of this argument. One is that specifically for Most Categories, winning a matchup is only possible when a drafter is doing well overall. The second is that over the course of an entire season, winning teams are banking on having a consistent advantage, which makes it difficult to do well in turnovers

### 2A. Getting over the hump in Most Categories

In the Most Categories context, winning turnovers to go 1-8 instead of 0-9 is unhelpful. A loss is a loss. So why invest in turnovers, which are easiest to win when you are already losing? 

This argument seems plausible on face but is framed incorrectly. Winning a consolation category isn't helpful, but neither is running up the score. __Only close match-ups, for which outcomes are uncertain, are relevant. The most important categories are those which can generate wins in match-ups that can go either way.__ 

At least for me it is hard to intuit whether turnovers are more or less important than other categories under the condition of a close match. Fortunately, the question can be modeled mathematically.  

#### Math

Let's start by taking a step back and being clear about what category importance and "optimal category weights" really are. The discussion of static ranking lists on the G-score page gives a framework for thinking about proper weighting. It models a situation wherein all players except one have been selected from a pool with arbitrary statistics. The proper weighting is designed so that a player's overall score is proportional to the benefit they incur to the reward function. On an individual category level, the weights then reflect the marginal improvement in the reward function earned by each increment of investment into the categories. This is equivalent to the definition of a partial derivative. So another way to frame the proper weight of a category is the partial derivative of the reward function (in this case, the probability of winning a matchup) with respect to investment in that category. 

I calculated this derivative in the paper, and it matches up well with the intuition that category importance should be evaluated under the condition that the match-up is close. The derivative is the product of two factors
- How likely is it that the other eight categories are tied? I call this situation a "tipping point" for the category
- How likely is it that an incremental improvement in turnovers flips the category from a loss to a win? Or in a more technical sense, what is the probability density of turnovers around zero, conditional on the first criteria?

The intuition here is that the two conditions together specify a situation under which a tiny investment in a category can flip the overall result from a loss to a win. This is analagous to 538's [voter power index](https://projects.fivethirtyeight.com/2016-election-forecast/#tipping-point), a way of quantifying the importance of voters under the electoral college. The importance of a particular voter is equal to the probability that they can flip the result of their state and flipping that state flips the result of the electoral college.

We can estimate these partial derivatives. First, we need a distribution that can incorporate how the various categories relate to each other. And for that, we need a correlation matrix. 

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

The probability of both criteria occuring can be estimated by sampling from the distribution many times. For each scenario with five category wins, all of the winning categories are considered tipping points. For each scenario with four category wins, the losing categories are considered tipping points. Then after the tipping points are identified, the probability of the tipping point category being around zero is estimated 

#### Conclusion of the math

The resulting approximate importances are 

 | Points    | Rebounds    | Assists    | Steals    | Blocks    | Threes    | Turnovers    | Free Throw \%   | Field Goal \%   |
|:------|:------|:------|:------|:------|:------|:------|:---------|:---------|
| 10.3\% | 6.8\% | 6.2\% | 9.0\% | 7.1\% | 6.6\% | 7.2\% | 7.1\%    | 7.4\%    |

The turnover weight is in line with the weights of the other categories. 

This means that for a single Most Categories match-up, if neither drafter has an advantage in any category, an investment in turnovers is roughly as likely as an investment in any other category to flip a loss to a win.

Upon reflection, this is intuitively reasonable. __For a matchup to be close, neither drafter can dominate the non-turnover counting statistics. Given that, it seems unlikely that one drafter would have such a massive lead in turnovers that it would be unsurmountable.__ 


### 2B. Relying on a consistent advantage  

One might note that the math in the last section was predicated on neither drafter having an advantage in any category coming into the week. That assumption is arguably problematic, because in many contexts, drafters need to have some advantage to have any shot at winning. For example, say a league has no playoffs at all, and the top team after the regular season wins. Presumably the top drafter will have to come into each matchup with an advantage because they chose better players. This argument applies to both Most Categories and Each Category.

The most reliable way for a drafter to obtain a consistent advantage is by choosing players who get more playing time than expected. This gives them some advantage in all of the counting statistics. 

The math from the previous section can be expanded to the situation where one drafter has an advantage in the counting stats coming into the week. I added a small constant to each of the counting stats (including turnovers), then re-ran the experiment, noting what percent of matchups were overall victories for each advantage state. The resulting category importances were as follows

For Most Categories 

 | Likelihood of winning the matchup  | Points    | Rebounds    | Assists    | Steals    | Blocks    | Threes    | Turnovers    | Free Throw \%   | Field Goal \%   |
|-----:|:-------|:------|:------|:------|:------|:------|:------|:---------|:---------|
| 50\%   | 10.3\% | 6.8\% | 6.2\% | 9.0\% | 7.1\% | 6.6\% | 7.2\% | 7.1\%    | 7.4\%    |
| 59.7\% | 10.0\% | 7.4\% | 6.7\% | 8.6\% | 5.9\% | 6.8\% | 7.0\% | 6.9\%    | 7.1\%    |
| 68.9\% | 9.1\%  | 6.4\% | 6.1\% | 8.0\% | 5.6\% | 6.0\% | 6.5\% | 6.4\%    | 6.3\%    |
| 77.1\% | 8.4\%  | 5.5\% | 5.1\% | 6.4\% | 4.6\% | 5.3\% | 5.2\% | 5.0\%    | 5.5\%    |
| 83.9\% | 6.5\%  | 4.6\% | 4.3\% | 5.3\% | 3.8\% | 4.1\% | 4.0\% | 4.3\%    | 4.4\%    |

It does appear to be the case that with an increasing advantage, turnovers become less likely to be a tipping point relative to other categories. However this effect is small even when the advantage is extreme. Intuitively this makes sense because no matter how large the advantage state is, if the matchup ended up close then no drafter could have dominated the counting statistics. Given that the counting statistics were close, one would expect turnovers to also be relatively close. 

We can also do this analysis for Each Category. For Each Category, the derivative of the reward function relative to investment in a category is just the probability density around zero of the team differential distribution.  The results are as follows

 | Average category winning %  | Points    | Rebounds    | Assists    | Steals    | Blocks    | Threes    | Turnovers    | Free Throw \%   | Field Goal \%   |
|-----:|:-------|:------|:------|:------|:------|:------|:------|:---------|:---------|
| 50\%   | 34.0\% | 30.9\% | 27.7\% | 36.4\% | 30.6\% | 31.0\% | 33.4\% | 33.4\%   | 34.4\%   |
| 54.3\% | 32.3\% | 30.9\% | 29.1\% | 36.4\% | 30.8\% | 30.3\% | 33.7\% | 33.4\%   | 34.4\%   |
| 58.5\% | 29.6\% | 29.1\% | 27.3\% | 33.4\% | 28.2\% | 28.9\% | 30.9\% | 33.4\%   | 34.4\%   |
| 62.3\% | 27.2\% | 27.0\% | 24.8\% | 29.2\% | 25.8\% | 26.0\% | 27.0\% | 33.4\%   | 34.4\%   |
| 65.8\% | 22.8\% | 23.2\% | 21.7\% | 24.5\% | 23.0\% | 22.1\% | 24.2\% | 33.4\%   | 34.4\%   |

There is an interesting takeaway from this analysis, but it has nothing to do with turnovers! It turns out that the percentage statistics have outsize importance in situations where one drafter has a playing time advantage. In retrospect this is obvious: with the counting statistics largely shored up, the percentage statistics, which are unbiased by playing time, are still just as difficult to win and become relatively more important. 

All in all, what does this tell us about turnovers? 

It tells us that __investment in turnovers has approximately equivalent returns on investment as investment in other categories, even for a very good team__. The idea that teams that are in position to win benefit less from investment in turnovers than in other categories is not supported. 

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

It should be noted that this test does not cover argument 2B because all player statistics are known beforehand, making it imposssible for any drafter to have a surprisingly good team. There remains the possibility that in a real league, with some uncertainty about how players are going to perform, the best-positioned drafters will have a significant advantage in general. If the heuristic of section 2B is wrong or misleading, perhaps turnovers become significantly less important in that case

## 5. Conclusion

I have not seen a convincing mathematical argument that turnovers should be down-weighted to an extreme degree. That's why I've set the default to treating turnovers like every other category. 

Still, absence of evidence is not evidence of absence, and there might be some more nuanced reason that turnovers should be downweighted not captured here. If you want to use this website and want to downweight turnovers, feel free to manually set the turnover multiplier on the parameters page

