# The curious case of turnovers

The turnovers category is unique because unlike all of the other counting statistics, for turnovers, fewer is better. This uniqueness makes turnovers a weird category to strategize around and leads many fantasy basketball analysts to recommend down-weighting the turnovers category to a low weight like $25\%$ or even $0\%$ relative to what Z-scores would otherwise tell them. 

There are real arguments to be made for this approach. However, my own analysis has led me to believe that none of the arguments are strong enough to warrant treating turnovers in such a radically different way. As such, the website's default is to weigh turnovers as normal. 

Unfortunately, I had to ignore some relevant aspects of the math in the paper for technical reasons (it makes the math impossible, in a sense). So my investigation into the arguments around turnovers has not been enormously rigorous, and is not built into the logic of H-scoring. Still, I realize that deciding how to treat turnovers is an important part of drafting strategy, and that my default may be controversial. So I will go through the arguments that are made in favor of auto-punting turnovers here and explain why they don't convince me

## 1. Empirical evidence 

Punting turnovers is often justified by the empirical fact that succesful teams tend to do poorly in turnovers.

I don't dispute the fact. I do posit that inverting the logic to claim that doing poorly in turnovers leads to success is a fallacy of [reverse causality](https://en.wikipedia.org/wiki/Correlation_does_not_imply_causation), which is a fallacy because correlation does not imply causation. Two separate things might be associated with each other without causing each other- in fact, one of them might be actively working against the other despite the correlation in incidence. 

If this concept is difficult to grasp intuitively, you shouldn't feel bad, because it haunts professional researchers and science reporters too. My favorite example of this is the oft-reported link between diet soda and obesity. Many studies have found a strong association between drinking diet soda and becoming obese, leading many researchers to suggest that some hidden mechanism makes drinking diet soda unhealthy. However, a recent [meta-analysis](https://academic.oup.com/nutritionreviews/article/71/7/433/1807065) suggests that this link is most likely ephemeral, and the result of a reverse-causality fallacy. Observationally, people who drink diet soda are more likely to gain weight. But those are also the same people most likely to be worried about their weight because of individual risk factors such as their activity level or a genetic propensity to be overweight. When controlling for those factors, the link disappears or even reverses. In other words- it is not drinking diet soda that makes people obese, but the risk factors for becoming obese that make someone drink diet soda. If anything, drinking diet soda likely decreases obesity risk. It only appears to be the reverse in the data because of a failure to control for countervailing factors. 

The fantasy basketball equivalent of this statement is that it is not de-prioritizing turnovers that leads to success, rather, it is the conditions that lead to success which also lead to poor scores for turnovers. To see how this can look misleading in results data, consider the following simplified results table 

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

## 2. The second argument: playing to your outs 

The most common theoretical argument for downweighting turnovers, specifically for Most Categories, is that winning turnovers requires losing other categories. Stated slightly more strategically, the idea is as follows
- A drafter's goal is to win the overall matchup
- Winning turnovers generally only happens when the drafter's players are playing fewer minutes than their opponents. In this situation, the drafter is most likely losing the matchup
- Ergo, winning the turnovers category is only relevant when the matchup is already lost, and therefore is not valuable 

This is essentially borrowing the concept of [playing to your outs](https://articles.starcitygames.com/articles/learning-to-truly-play-to-your-outs/) from strategy games. However, applying the concept here is a fallacy and utilizes a false leap in logic

### Intuition 

The concept of playing to your outs is mostly invoked in cases where a player is likely to lose. When a player is likely to win, a complementary concept is at play: [win-more](https://articles.starcitygames.com/magic-the-gathering/win-more-in-commander-magic-what-it-is-and-isnt/). The idea of win-more is that there is no point strategizing around situations which are guaranteed wins for the same reason that there is no point strategizing around situations that are guaranteed losses. Adding it to the concept of playing to your outs, the full picture is that one should not strategize around any scenario where the outcome is guaranteed; rather, one should strategize around scenarios where the outcome hangs in the balance and could be affected by strategic decision-making. 

In the context of fantasy basketball, scenarios where the drafter has a significant disadvantage in terms of playing time may be irrelevant because the drafter is going to lose no matter what. If that is the case, then the reverse argument should also apply: if the drafter has a significant advantage in playing time, they are going to win no matter what, and those scenarios should be excluded too. The only relevant scenarios to strategize around are those where the outcome is close. 

For the outcome to be close, each drafter must win at least two of the non-turnover counting statistics (otherwise one drafter would have five category wins already without turnovers, and their victory would be assured). It is not clear, at least to me, if any drafter has a strong advantage in turnovers given that context. 

### Math 

The intuitive version of the argument is not entirely clear. Can a mathematical approach bring clarity? 

Correlation is a measure of how related two metrics are. When two metrics tend to be either both high or both low, they are highly correlated. When they tend to be either high/low or low/high, they are negatively correlated. when they are totally unrelated, they are uncorrelated, or have a correlation of zero. 

A correlation matrix contains the pairwise correlations between many metrics. For the fantasy basketball setting, with scores normalized by week-to-week variance (and turnovers muliplied by -1), the correlation matrix is 
                                                        
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

 It is clear that the turnovers category is uniquely negatively correlated to the counting statistics, which lends some credence to the idea that it is hard to win many categories without losing turnovers. 
 
However, does that necessarily translate to a decreased optimal weight on turnovers? To answer this question we must first take a step back and be clear about what it means for a weight to be optimal.

The discussion of static ranking lists on the G-score page gives a framework for thinking about proper weighting. It models a situation wherein all players except one have been selected from a pool with arbitrary statistics. The proper weighting is designed so that a player's overall score is proportional to the benefit they incur to the reward function. On an individual category level, the weights then reflect the marginal improvement in the reward function earned by each increment of investment into the categories. This is equivalent to the definition of a partial derivative. So another way to frame the proper weight of a category is the partial derivative of the reward function (in this case, the probability of winning a matchup) with respect to investment in that category. 
 
 I calculated this derivative for the Most Categories context in the paper, and it boiled down to two factors multiplied together: 
- How likely is it that the other eight categories are tied? I call this situation a "tipping point" for the category
- How likely is it that an incremental improvement in turnovers flips the category from a loss to a win? Or in a more technical sense, what is the probability density of turnovers around zero, conditional on the first criteria? 

This aligns well with the intution that strategies should focus on scenarios which are close matchups, ignoring scenarios where the outcome is already guaranteed. The two implicit conditions in the derivative are equivalent to calculating the probability that an incremental investment in a category flips the result of the overall outcome. 

The probability of both criteria occuring can be estimated by approximating the values of all categories as multivariate normals with mean zero and sampling from the distribution many times. For each scenario with five category wins, all of the winning categories are considered tipping points. For each scenario with four category wins, the losing categories are considered tipping points. Then after the tipping points are identified, the probability of the tipping point category being around zero is estimated. I tried this with 2023 data and got 

 | Points    | Rebounds    | Assists    | Steals    | Blocks    | Threes    | Turnovers    | Free Throw \%   | Field Goal \%   |
|:------|:------|:------|:------|:------|:------|:------|:---------|:---------|
| 10.4\% | 7.2\%  | 7.0\%  | 9.2\%  | 6.1\%  | 6.5\%  | 7.4\%  | 6.7\%     | 6.6\%     |

Turnovers end up having approximately average importance based on this approximation

## 3. The third argument: banking on overperformance

One might note that the math in the last section was predicated on neither drafter having an advantage in any category coming into the week. That assumption is arguably problematic, because in many contexts, drafters need to have some advantage to have any shot at winning. For example, say a league has no playoffs at all, and the top team after the regular season wins. Presumably the top drafter will have to come into each matchup with an advantage because they chose better players. 

### Intuition 

Assume that drafter A comes into a week against drafter B with higher expected values across all counting statistics, giving them an advantage in all of them except turnovers. Drafter A is highly likely to win the matchup.

Again only the tipping point scenarios have relevance, because if drafter A wins or loses across the board, no marginal difference in decision-making could have changed the outcome. There are two relevant questions
- Is it less likely for turnovers to be a potential tipping point than other categories? Maybe. I don't see any particular reason to expect this 
- Is the probability density of turnovers being tied, given a tipping point, particularly low? Again, in all tipping point scenarios, drafter A must have lost at least two counting statistics. Given that this is the case, it seems unlikely that drafter A ended up with an enormous advantage in overall playing time. Even if they did- obviously some of the counting statistics bucked the trend, and drafter A scored fewer of them despite having more playing time. Why couldn't the same happen to turnovers? 

It might be the case that with higher expected values for counting statistics, the relative importance of turnovers decreases. But if that is the case it is not clear to what degree based on intuition. 

### Math

The math from the last section can be expanded to the situation where one drafter has an advantage in the counting stats coming into the week.

 | Likelihood of winning overall  | Points    | Rebounds    | Assists    | Steals    | Blocks    | Threes    | Turnovers    | Free Throw \%   | Field Goal \%   |
|-----:|:-------|:------|:------|:------|:------|:------|:------|:---------|:---------|
| 50.0   | 10.5\% | 6.8\% | 6.9\% | 9.2\% | 6.4\% | 6.7\% | 6.8\% | 7.0\%    | 7.2\%    |
| 59.9 | 10.5\% | 7.2\% | 6.4\% | 8.5\% | 6.1\% | 6.6\% | 7.1\% | 6.4\%    | 6.8\%    |
| 69.1 | 9.9\%  | 6.9\% | 6.2\% | 7.8\% | 5.4\% | 5.7\% | 6.6\% | 6.2\%    | 6.3\%    |
| 77.2 | 8.1\%  | 5.6\% | 5.0\% | 6.1\% | 5.1\% | 5.0\% | 5.1\% | 5.0\%    | 5.3\%    |
| 84.0   | 6.8\%  | 4.3\% | 4.5\% | 4.4\% | 3.8\% | 4.4\% | 4.1\% | 4.0\%    | 4.0\%    |

I calculated the advantage by adding a small constant to all of the counting stats (including turnovers), then observing what percentage of the corresponding simulations were victories. 

It does appear to be the case that with an increasing advantage, turnovers become less likely to be a tipping point relative to other categories. However this effect is small even when the advantage is extreme

## 4. The fourth argument: turnovers are volatile

One final argument is that turnovers are hard to predict on a week-to-week basis, and therefore are not worth investing in. 

It is true that turnovers are relatively volatile from week to week. However, this is not unique; all categories have some level of week to week volatility. Turnovers are not even most volatile category. Steals are, by a wide margin. 

G-scores deal with this by incorporating week-to-week variance. They do downweight turnovers relative Z-scores, but not in an extreme way

## 5. Testing 

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

