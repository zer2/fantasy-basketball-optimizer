Turnovers are a unique category because turnovers are a negative asset and therefore are inversely correlated to other categories. That is, winning turnovers make it less likely that you will win other categories. For this reason, many fantasy basketball analysts recommend down-weighting the turnovers category to a low weight like $25\%$ or even $0\%$.

Unfortunately, I had to ignore the concept of correlations between categories in the paper for technical reasons (it makes the math impossible, in a sense). I had to treat turnovers like every other category. Still, I realize that understanding how to treat turnovers is an important part of drafting strategy, and I did set the default weighting for turnovers to be $25\\%$. Below is a justification for that decision 

## Part 1: Correlation and tipping points 

Correlation is a measure of how related two metrics are. When two metrics tend to be either both high or both low, they are highly correlated. When they tend to be either high/low or low/high, they are negatively correlated. when they are totally unrelated, they are uncorrelated, or have a correlation of zero. 

A correlation matrix contains the pairwise correlations between many metrics. For the fantasy basketball setting, with scores normalized by week-to-week variance, the correlation matrix is 
                                                        
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

 It is clear that the turnovers category is uniquely negatively correlated to the other categories. 

### Most categories 

One way of thinking about the importance of turnovers is to break it down into two factors
- How likely is it that the other eight categories are tied?
- How likely is it that an incremental improvement in turnovers flips the category from a loss to a win? Or in a more technical sense, what is the probability density of turnovers around zero, conditional on the first criteria? 

Under those two conditions, investment in turnovers will matter. If either of them are not met, investing in turnovers would not help, because the match-up would be won or lost anyway. 

The probability of both criteria occuring can be estimated by approximating the values of all categories are multivariate normals with mean zero and sampling from the distribution many times. The result is 

 | Points    | Rebounds    | Assists    | Steals    | Blocks    | Threes    | Turnovers    | Free Throw \%   | Field Goal \%   |
|:------|:------|:------|:------|:------|:------|:------|:---------|:---------|
| 10.8\% | 7.7\%  | 7.1\%  | 8.8\%  | 6.5\%  | 6.7\%  | 7.1\%  | 7.2\%     | 6.8\%     |

So turnovers actually end up having a low-end likelihood of mattering, but not in a unique way compared to other categories

### Each category

Each Category does not have the same concept of a "tipping point", so this effect is not relevant 


## Part 2: The need for consistent advantage

Most players are interested in winning their league, not just doing well generally. This may seem like a pedantic distinction but it is actually quite important. Winning a league requires winning head-to-head matchups consistently, and in the Each Categories context, preferably by large margins. This is difficult to accomplish while also trying to win the turnovers category. 

Note what happens to the tipping point probabilities when we start with some advantage


 |        | Points    | Rebounds    | Assists    | Steals    | Blocks    | Threes    | Turnovers    | Free Throw \%   | Field Goal \%   |
|----:|:------|:------|:------|:------|:------|:------|:------|:---------|:---------|
| 0   | 10.8\% | 7.7\%  | 7.1\%  | 8.8\%  | 6.5\%  | 6.7\%  | 7.1\%  | 7.2\%     | 6.8\%     |
| 0.5 | 7.1\%  | 5.1\%  | 4.4\%  | 5.7\%  | 4.2\%  | 5.0\%  | 3.1\%  | 4.6\%     | 4.6\%     |
| 1   | 1.9\%  | 1.9\%  | 2.0\%  | 2.5\%  | 1.7\%  | 2.2\%  | 0.4\%  | 1.6\%     | 1.8\%     |'

While all categories become less likely to matter individually in this scenario, turnovers become almost vanishly unlikely to matter. This is because with a consistent advantage in other categories, you are almost definitely not winning turnovers anyway.

There isn't an easy way to translate this analysis for Each Category, but in concept, the same idea should hold. Under the condition that the drafter is doing well and has a good shot at winning the entire league, they are likely not winning turnovers

## Part 3: Testing 

I ran a simple test to determine if down-weighting turnovers is a good idea. To do this test, I split up all drafters into two groups of six- one treated turnovers normally, the other down-weighted it. The team with the highest regular season record won. The results are as follows for the down-weighter's win rate:

| Weight |  Most Categories | Each Category |
|----:|:------|:------|
| 0\% | 11.34\% | 13.38\% |
| 0.1\% | 10.08\% | 12.26\% |
| 0.2\% | 9.76\% | 11.93\% |
| 0.3\% | 9.5\% | 11.63\% |
| 0.4\% | 9.14\% | 11.33\% |

There is a clear trend for both formats that down-weighting turnovers improves chances of winning the league

## Part 4: Takeaways

I think turnovers ought to be downweighted somewhat most of the time. However, the appropriate re-weighting for turnovers depends heavily on context. 

-Format 
  -Head to head, regular season: If your league has a top-heavy reward structure and reaching the playoffs is difficult, then it makes sense to ignore turnovers and hope that other categories propel you to the playoffs. If making it to the playoffs is relatively easy or the payout structure is even, then you might be able to do well even without significant good luck for most of the season, so you shouldn't downweight turnovers too much 
  -Head to head, playoffs: Playoffs are likely evenly matched because they are always between strong teams. Turnovers are still relatively unimportant because they are unlikely to be the tipping point, but not as unimportant as when you are relying on a consistent advantage in other categories
  -Rotissierie: Rotisserie has no playoffs, so you need exceptional performances across the board for the entire season to win. It is difficult to do this while also winning turnovers, so turnovers should be  down-weighted. [Micah Lamdin](https://hashtagbasketball.com/fantasy-basketball/content/how-to-play-fantasy-basketball-rotisserie) is one analyst who agrees with this
-Other drafters
  -If other drafters are ignoring turnovers, you might be able to gain a small advantage by being the one drafter that does keep them in mind, without having to sacrifice much in value from the other categories
  -If other drafters are weighing turnovers highly, it means low-turnover players will be hard to find anyway, so it makes sense to deprioritize them 
