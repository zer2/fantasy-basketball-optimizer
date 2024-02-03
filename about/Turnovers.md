Turnovers are a unique category because turnovers are a negative asset and therefore are inversely correlated to other categories. That is, winning turnovers make it less likely that you will win other categories. For this reason, many fantasy basketball analysts recommend down-weighting the turnovers category to a low weight like $25\%$ or even $0\%$.

Unfortunately, I had to ignore the concept of correlations between categories in the paper for technical reasons (it makes the math impossible, in a sense). I had to treat turnovers like every other category. Still, I realize that understanding how to treat turnovers is important, so I thought I would lay out my less tructured thoughts on the category here. 

Long story short I think down-weighting turnovers does make sense, though to what degree depends on context

## Part 1: Covariance and tipping points 
                                                        

Choosing from top players and weeks randomly (with turnovers inverted), the correlation matrix looks like this 

 |        | pts    | trb    | ast    | stl    | blk    | fg3    | tov    | ft_pct   | fg_pct   |\n|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:---------|:---------|\n| pts    | 100.0% | 47.2%  | 57.7%  | 40.1%  | 18.6%  | 63.0%  | -66.5% | 17.5%    | 19.0%    |\n| trb    | 47.2%  | 100.0% | 24.1%  | 21.6%  | 46.9%  | 2.5%   | -41.4% | -20.9%   | 27.5%    |\n| ast    | 57.7%  | 24.1%  | 100.0% | 41.1%  | -4.1%  | 35.5%  | -63.0% | 11.0%    | -9.2%    |\n| stl    | 40.1%  | 21.6%  | 41.1%  | 100.0% | 8.9%   | 28.6%  | -36.0% | 6.5%     | -6.8%    |\n| blk    | 18.6%  | 46.9%  | -4.1%  | 8.9%   | 100.0% | -7.4%  | -12.5% | -14.8%   | 24.0%    |\n| fg3    | 63.0%  | 2.5%   | 35.5%  | 28.6%  | -7.4%  | 100.0% | -34.2% | 21.0%    | -11.6%   |\n| tov    | -66.5% | -41.4% | -63.0% | -36.0% | -12.5% | -34.2% | 100.0% | -4.7%    | 1.2%     |\n| ft_pct | 17.5%  | -20.9% | 11.0%  | 6.5%   | -14.8% | 21.0%  | -4.7%  | 100.0%   | -13.8%   |\n| fg_pct | 19.0%  | 27.5%  | -9.2%  | -6.8%  | 24.0%  | -11.6% | 1.2%   | -13.8%   | 100.0%   |

One way of thinking about category importance for Most Categories is to break it down into two factors
- How likely is it that the other eight categories are tied?
- How likely is it that an incremental improvement in turnovers flips the category from a loss to a win?

Under those two conditions, investment in turnovers will matter. If either of them are not met, investing in turnovers would not help, because the match-up would be won or lost anyway. 

These probabilities can be estimated by approximating the values of all categories are multivariate normals with mean zero. The result is


## Part 2: The need for consistent advantage

Most players are interested in winning their league, not just doing well generally. This may seem like a pedantic distinction but it is actually quite important. Winning a league requires winning head-to-head matchups consistently, and in the Each Categories context, preferably by large margins. This is difficult to accomplish while also trying to win the turnovers category. 

Note what happens to the tipping point probabilities when we start with some advantage

While all categories become less likely to matter individually in this scenario, turnovers become almost vanishly unlikely to matter. This is because with a consistent advantage in other categories, you are almost definitely not winning turnovers anyway

## Part 3: Testing 

## Part 4: Takeaways

The appropriate re-weighting for turnovers depends heavily on context. 

-Format 
  -Head to head, regular season: If your league has a top-heavy reward structure and reaching the playoffs is difficult, then it makes sense to ignore turnovers and hope that other categories propel you to the playoffs. If making it to the playoffs is relatively easy or the payout structure is even, then you might be able to do well even without significant good luck for most of the season, so you shouldn't downweight turnovers too much 
  -Head to head, playoffs: Playoffs are likely evenly matched because they are always between strong teams. Turnovers are still relatively unimportant because they are unlikely to be the tipping point, but not as unimportant as when you are relying on a consistent advantage in other categories
  -Rotissierie: Rotisserie has no playoffs, so you need exceptional performances across the board for the entire season to win. It is difficult to do this while also winning turnovers, so turnovers should be  down-weighted. [Micah Lamdin](https://hashtagbasketball.com/fantasy-basketball/content/how-to-play-fantasy-basketball-rotisserie) is one analyst who agrees with this
-Other drafters
  -If other drafters are ignoring turnovers, you might be able to gain a small advantage by being the one drafter that does keep them in mind, without having to sacrifice much in value from the other categories
  -If other drafters are weighing turnovers highly, it means low-turnover players will be hard to find anyway, so it makes sense to deprioritize them 
