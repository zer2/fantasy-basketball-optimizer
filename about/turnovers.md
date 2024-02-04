# The curious case of turnovers

Turnovers are a unique category because they are a negative asset and therefore are inversely correlated to other categories. That is, winning turnovers make it harder to win other categories and vice versa. For this reason, many fantasy basketball analysts recommend down-weighting the turnovers category to a low weight like $25\%$ or even $0\%$ relative to what Z-scores would otherwise tell them. 

I think that there is a real argument to be made here, but it is overblown. For this reason the website default is to weigh turnovers like every other category. 

Unfortunately, I had to ignore the concept of correlations between categories in the paper for technical reasons (it makes the math impossible, in a sense). So my argument here is not enormously rigorous, and not built into the logic of H-scoring. Still, I realize that understanding how to treat turnovers is an important part of drafting strategy, and my default may be controversial. So I will lay out a heuristic justification here, going through the main arguments that are made and why they don't convince me, then testing the hypothesis that punting turnovers is uniquely beneficial

## 1. The first argument: turnovers are volatile

One argument is that turnovers are hard to predict on a week-to-week basis, and therefore are not worth investing in. 

It is true that turnovers are relatively volatile from week to week. However, this is not unique; all categories have some level of week to week volatility. Turnovers are not even most volatile category- steals are by a large margin. 

To deal with this, all category scores should be adjusted by the week-to-week variance, which is the idea of G-scores. There is no reason to treat turnover volatility different from the volatility of other categories

## 2. The second argument: playing to your outs 

### Intuition 

A second argument for downweighting turnovers, specifically for Most Categories, is that winning turnovers requires having your players play fewer minutes, which is a losing strategy overall. This is essentially borrowing the concept of [playing to your outs](https://articles.starcitygames.com/articles/learning-to-truly-play-to-your-outs/) from strategy games. The idea is that guaranteed-lose situations should be ignored in decision-making. The argument has intuitive appeal, but it is also a fallacy.

The concept of playing to your outs is mostly invoked in cases where a player is likely to lose. When a drafter is likely to win, a complementary concept is at play: [win-more](https://articles.starcitygames.com/magic-the-gathering/win-more-in-commander-magic-what-it-is-and-isnt/). The idea of win-more is that there is no point strategizing around games which you are definitely going to win, because you are going to win either way. Adding it to the concept of playing to your outs, the full picture is that one should not strategize around any scenario where the outcome is guaranteed; rather, one should strategize around scenarios where the outcome hangs in the balance and could be affected by strategic decision-making. In the case of fantasy basketball, scenarios where the drafter has a significant disadvantage in terms of playing time are largely irrelevant because they are going to lose no matter what. But the flipside scenario is also irrelevant: if the drafter has a significant advantage in playing time, they are going to win no matter what. The only relevant scenarios to strategize around are those where the outcome is close, and for the outcome to be close, each drafter must win some share of the counting statistics, making it unclear which team has the advantage in turnovers. 

One might counter with the empirical fact that succesful teams tend to do poorly in turnovers. I believe that this is the truth, but inverting the logic to claim that doing poorly in turnovers leads to success is a fallacy of [questionable cause](https://en.wikipedia.org/wiki/Questionable_cause). Simplifying this drastically, consider the following results:

| | Drafter A | Drafter B | Drafter C | Drafter D| 
|:-------|:-------|:-------|:-------|:-------|
|Turnover weight | 0 | 1 | 0 | 1 |
|Player m/g | 40 | 40 | 20 | 20 |
|Result- turnovers | Bad | Middling | Middling | Good | 
|Result- championship | Won | Won | Lost | Lost |

Ignoring the player m/g numbers, one might naively infer two effects
- Investing in turnovers lead to a better performance in turnovers. Evidence:
  - Drafters B and D invested in turnovers, and were middling/good in them
  - Drafters A and C did not invest in turnovers, and were middling/bad in them 
- Performing badly in turnovers improves the probability of victory. Evidence:
  - Drafters A and B won, and were middling/bad in turnovers
  - Drafters C and D lost, and were middling/good in turnovers
 
Putting these two arguments together makes a seemingly solid case that investing in turnovers leads to bad performance. However, this backwards-causality approach was not logical. It can easily be seen by viewing the results holistically that turnover weight was irrelevant; the better-supported driver of victory was minutes per game. 

### Math 

The intuitive version of the argument is not entirely clear.  Can a mathematical approach bring clarity? 

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

 It is clear that the turnovers category is uniquely negatively correlated to the counting statistics, which lends some credence to the idea that it is hard to win many categories without losing turnovers. 
 
 However, does that necessarily translate to a decreased importance. One way of thinking about the importance of turnovers is to break it down into two factors
- How likely is it that the other eight categories are tied?
- How likely is it that an incremental improvement in turnovers flips the category from a loss to a win? Or in a more technical sense, what is the probability density of turnovers around zero, conditional on the first criteria? 

Under those two conditions, investment in turnovers will matter. If either of them are not met, investing in turnovers would not help, because the match-up would be won or lost anyway. 

The probability of both criteria occuring can be estimated by approximating the values of all categories as multivariate normals with mean zero and sampling from the distribution many times. I tried this with 2023 data and got 

 | Points    | Rebounds    | Assists    | Steals    | Blocks    | Threes    | Turnovers    | Free Throw \%   | Field Goal \%   |
|:------|:------|:------|:------|:------|:------|:------|:---------|:---------|
| 10.8\% | 7.7\%  | 7.1\%  | 8.8\%  | 6.5\%  | 6.7\%  | 7.1\%  | 7.2\%     | 6.8\%     |

So turnovers actually end up having a low-ish likelihood of mattering, though not in a unique way compared to other categories

## 3. The third argument: Banking on overperformance

One might note that the math in the last section was predicated on neither player having an advantage in any category coming into the week. That assumption is arguably problematic, because in many contexts, drafters need to have some advantage to have any shot at winning. For example, say a league has no playoffs at all, and the top team after the regular season wins. Presumably the top drafter will have come into each matchup with an advantage because they chose better players. 

Note what happens to the tipping point probabilities from the last section when we start with some advantage.

 | Likelihood of winning overall  | Points    | Rebounds    | Assists    | Steals    | Blocks    | Threes    | Turnovers    | Free Throw \%   | Field Goal \%   |
|-----:|:-------|:------|:------|:------|:------|:------|:------|:---------|:---------|
| 50.0\%   | 10.4\% | 7.2\% | 7.0\% | 9.2\% | 6.1\% | 6.5\% | 7.4\% | 6.7\%    | 6.6\%    |
| 61.4\% | 9.9\%  | 7.0\% | 6.7\% | 8.6\% | 6.8\% | 6.5\% | 6.7\% | 6.3\%    | 6.3\%    |
| 72.1\% | 8.0\%  | 5.9\% | 5.8\% | 7.1\% | 5.8\% | 6.0\% | 5.6\% | 5.8\%    | 5.6\%    |
| 81.0\%   | 6.9\%  | 5.1\% | 4.9\% | 6.0\% | 4.3\% | 4.5\% | 4.5\% | 4.6\%    | 4.5\%    |
| 87.7\% | 5.0\%  | 4.0\% | 3.3\% | 4.2\% | 2.9\% | 3.7\% | 3.2\% | 3.6\%    | 3.3\%    |

I calculated the advantage by adding a small constant to all of the counting stats, then observing what percentage of the corresponding simulations were victories. Even with the significant advantage in the counting statistics, turnovers still are not drastically less important than other categories 

## 4. Testing 

To some degree, we can test the hypothesis that down-weighting turnovers is a good idea. I ran a test with the following setup
- For each category
  - Divide all drafters into two groups of six
  - One group downweights the category by a certain factor. The other does not
  - For each sequential arrangement of seats (6 down-weighters/6 normals, or 1 normal/6 down-weighters/5 normals, etc. )
    - One thousand seasons are simulated by sampling from the actual season of data
    - The team with the highest regular season record wins

Down-weighting a single category can be a good strategy in general- it is essentially punting. If down-weighting turnovers is a uniquely important measure to take, then we should expect the benefit of down-weighting turnovers to be greater than the benefits of down-weighting other categories

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

It should be noted that this test does not cover the third argument because all player statistics are known beforehand. There remains the possibility that in a real league, with some uncertainty about how players are going to perform, the best-positioned drafters will have a significant advantage in general. 

## 5. Conclusion

I have not seen a convincing mathematical argument that turnovers should be down-weighted to an extreme degree. That's why I've set the default to treating turnovers like every other category. 

Still, absence of evidence is not evidence of absence, and there might be some more nuanced reason that turnovers should be downweighted not captured here. If you want to use this website and want to downweight turnovers, feel free to manually set the turnover multiplier on the parameters page

