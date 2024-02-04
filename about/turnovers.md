# The curious case of turnovers

Turnovers are a unique category because turnovers are a negative asset and therefore are inversely correlated to other categories. That is, winning turnovers make it less likely that you will win other categories. For this reason, many fantasy basketball analysts recommend down-weighting the turnovers category to a low weight like $25\%$ or even $0\%$.

I think that there is a real argument to be made here, but it is overblown. For this reason the website default is to weigh turnovers like every other category. 

Unfortunately, I had to ignore the concept of correlations between categories in the paper for technical reasons (it makes the math impossible, in a sense). So my argument here is not enormously rigorous, and not built into the logic of H-scoring. Still, I realize that understanding how to treat turnovers is an important part of drafting strategy, and my default may be controversial. So I will lay out a heuristic justification here, going through the main arguments that are made against turnovers, then testing the hypothesis 

## Argument 1: Turnovers are volatile

One argument is that turnovers are hard to predict on a week-to-week basis, and therefore are not worth investing in. 

It is true that turnovers are relatively volatile. However, this is not unique, and turnovers are not the most volatile category: steals are by a large margin. To deal with this, all category scores should be adjusted by the week-to-week variance, which is the idea of G-scores. This does end up slightly down-weighting turnovers, though not to an extreme degree

## Argument 2: Players need to play 

Another possible argument for downweighting turnovers, specifically for Most Categories, is that if you are going to win most categories your players will have to play a lot and therefore you will probably lose turnovers anyway, so it does not make sense to invest in them. This argument does make some sense, and can be analyzed mathematically. 

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

 It is clear that the turnovers category is uniquely negatively correlated to the counting statistics, which lends some credence to the idea that it is hard to win many categories without losing turnovers. However, that is not the whole story. 

One way of thinking about the importance of turnovers is to break it down into two factors
- How likely is it that the other eight categories are tied?
- How likely is it that an incremental improvement in turnovers flips the category from a loss to a win? Or in a more technical sense, what is the probability density of turnovers around zero, conditional on the first criteria? 

Under those two conditions, investment in turnovers will matter. If either of them are not met, investing in turnovers would not help, because the match-up would be won or lost anyway. 

The probability of both criteria occuring can be estimated by approximating the values of all categories are multivariate normals with mean zero and sampling from the distribution many times. I tried this with 2023 data and got 

 | Points    | Rebounds    | Assists    | Steals    | Blocks    | Threes    | Turnovers    | Free Throw \%   | Field Goal \%   |
|:------|:------|:------|:------|:------|:------|:------|:---------|:---------|
| 10.8\% | 7.7\%  | 7.1\%  | 8.8\%  | 6.5\%  | 6.7\%  | 7.1\%  | 7.2\%     | 6.8\%     |

So turnovers actually end up having a low-ish likelihood of mattering, though not in a unique way compared to other categories. 

It might be surprising that turnovers are not markedly less important than the others in importance given the argument made earlier. The logical oversight was that if your players are playing far more than your opponents' players, you are likely winning the matchup no matter what and no categories matter. The only important scenarios are those in which the other eight categories are tied, meaning some counting statistics must be won and some must be lost. Take an arbitrary example of a tipping point scenario
- Won: Rebounds, Steals, Three, Free Throw %
- Lost: Points, Assists, Blocks, Field Goal %

Is it easy to tell who has an advantage in turnovers? Each drafter won three counting statistics, which are all highly correlated with playing time

## Argument 3: Need for consistent advantage 

One might note that the math in the last section was predicated on neither player having an advantage in any category coming into the week. That assumption is arguably problematic, because in many contexts, drafters need to have some advantage to have any shot at winning. For example, say a league has no playoffs at all, and the top team after the regular season wins. Presumably the top drafter will have come into each matchup with an advantage because they chose better players. 

Note what happens to the tipping point probabilities from the last section when we start with some advantage. The advantages are per player and per category

 | Advantage  | Points    | Rebounds    | Assists    | Steals    | Blocks    | Threes    | Turnovers    | Free Throw \%   | Field Goal \%   |
|----:|:------|:------|:------|:------|:------|:------|:------|:---------|:---------|
| 0   | 10.8\% | 7.7\%  | 7.1\%  | 8.8\%  | 6.5\%  | 6.7\%  | 7.1\%  | 7.2\%     | 6.8\%     |
| 0.5 | 7.1\%  | 5.1\%  | 4.4\%  | 5.7\%  | 4.2\%  | 5.0\%  | 3.1\%  | 4.6\%     | 4.6\%     |
| 1   | 1.9\%  | 1.9\%  | 2.0\%  | 2.5\%  | 1.7\%  | 2.2\%  | 0.4\%  | 1.6\%     | 1.8\%     |'

While all categories become less likely to matter individually in this scenario, the importance of turnovers vanishes in the most extreme way. This should lend credence to the idea that turnovers don't matter when a drafter is playing to their upside. 

However, again there are catches 
- First, the advantages in the table above are rather extreme and unlikely. the $1$ advantage scenario where turnovers are vanishingly unimportant translates to an overall $94\%$ chance of winning, which is unrealistically high
- Second, even in the case that a drafter does have such an extreme advantage, again, none of the categories are particularly important in that scenario. The drafter is very likely to win no matter what, does it make sense to strategize around that?
- Third, a setup where a drafter needs a significant advantage coming into every week in order to win is rare. Usually, around half of all teams get into the playoffs, so a team with only medium luck in terms of player performance can make it. Then the playoff matches are usually relatively fair, and come down to weekly variance
  
## Testing 

To some degree, we can test the hypothesis that down-weighting turnovers is a good idea. I ran a test with the following setup
- For each category, across 1,000 experiments: 
 - Divide all drafters into two groups of six
 - One group downweights the category by a certain factor. The other does not
 - Many seasons are simulated, by sampling from an the actual season of data 
 - The team with the highest regular season record wins

Down-weighting a single category can be a good strategy in general- it is essnetially punting. If down-weighting turnovers is a uniquely important measure to take, then the reward for down-weighting it should be greater than doing so for other categories

The results are as follows for the down-weighter's win rate:

| Turnovers | | |
|----:|:------|:------|
| Weight |  Most Categories | Each Category |
| 0\% | 11.3\% | 13.4\% |
| 10\% | 10.1\% | 12.3\% |
| 20\% | 9.8\% | 11.9\% |
| 30\% | 9.5\% | 11.6\% |
| 40\% | 9.1\% | 11.3\% |


Punting or soft-punting turnovers is about as beneficial as punting other categories. With this evidence, there is no reason to treat it differently. 

It should be noted that this test does not cover the third argument because all player statistics are known beforehand. There remains the possibility that in a real league, with some uncertainty about how players are going to perform, the best-positioned drafters will have a significant advantage in general. If having this advantage is necessary, then there might still be good reason to down-weight turnovers. 

## Takeaways

I think it makes sense to downweight turnovers to some degree when you need your players to outperform across the board to win, e.g. in Rotisserie. To what degree I am not sure- it depends on how confident you are about your predictions of player performance. Otherwise, I think it is best to weigh turnovers at least close to as highly as other categories
