# Dynamic strategy with H-scoring

Every seasoned fantasy manager knows that performing well at the highest level of competition requires adapting to draft circumstances on the fly. Picking directly from a static ranking list can easily lead to an unbalanced, incohesive team. 

In general, the way that the best managers adapt is more of an art than a science. They might "punt" some number of categories, then choose players in such a way that they are above average in the other categories. The idea is that since margins do not matter, winning many categories by small margins is worth losing a few by large margins.

In the [paper](https://arxiv.org/abs/2307.02188), I derive an algorithm called H-scoring to translate this intuition into a rigorous procedure. By framing draft strategy as an optimization problem, it implicitly understands how to rebalance teams for optimal performance, sacrificing some categories and shoring up others. While imperfect, I believe that the logic is sound, and evidence suggests that it works, at least in a simplified context. 

Below is a summary of how the algorithm is designed

## 1. The H-scoring approach

Dynamic drafting is a fundamentally more difficult proposition than static drafting. More information about drafting context is helpful, but figuring out the right way to incorporate it into decision-making is tricky. 

The most challenging aspect of the problem is accounting for future draft picks. They are neither completely under the manager's control (since the manager does not know which players will be available later) nor completely random (since the manager will decide which players to take of those available). Instead, future draft picks fall somewhere between the two extremes. 

H-scores' solution to this dilemma is conceiving of future picks as being controlled by a weight vector, by which aggregate statistics of future picks can be approximated. For example, the  algorithm might give seven categories $14\%$ weight each and two categories $1\%$ each, for a total of $100\%$. The algorithm would then assume that the manager would use the custom weighing to value future picks. That would lead the algorithm to approximate statistics for future picks with a slight upwards bias in the seven up-weighted categories and a large downwards bias in the two down-weighted categories. By this mechanism, the manager has some measure of control over future picks. 

Of course, being able to map a weight vector $j$ to approximate statistics for future picks is not enough for optimal draft strategy. The manager needs to operate the other way around, and pick both a player $p$ from the available candidate pool and a weight vector $j$ for future picks such that the probability of winning (H-score) is optimized. 

Doing all of this algorithmically is far from a trivial task. It can be accomplished by first building a full model linking $p$ and $j$ to H-score, then applying mathematical tools to the model to discover choices of $p$ and $j$ that mazimize the reward function

## 2. Calculating H-score based on $p$ and $j$

Writing out a single equation for H-score is cumbersome because the methodology behind it has so many steps. Instead, it is easiest to understand by starting from the ultimate goal of the metric and working backwards to successively fill in gaps.

We'll do this for counting statistics, which are easier to work with than the percentage statistics. The percentage statistics can be dealt with in an analogous way.

### 2a. Defining the objective function

In the static ranking context the expected number of category wins was a reasonable objective even for Most Categories, since strategizing how to win only five out of nine categories was impossible. In the dynamic context, more information is available, and using the appropriate objective function for the format is warranted. 

Another consideration is that the manager will have some control over the aggregate statistics of their team, so the objective function should be expressed as a function of team composition. 

 With
- $w_{c,o}(X)$ as the probability of winning category $c$ against opponent $o$ based on the manager's aggregate team statistics $X$ 
- $|C|$ as the number of categories
- $|T|$ as the number of teams

A sensible choice for $H_e(X)$, the objective function relative to the team $A$'s stat distribution $X$ for Each Category, is

$$
H_e(X) = \frac{ \sum_{c,o} w_{c,o}(X)}{|C| * (|T| - 1)}
$$

Its Most Categories counterpart $H_m(x)$ must be slighly more complicated, since it involves the probability of winning the majority of categories. It can be written as

$$
H(j)  = \frac{ \sum_o w_{1,o}(X) * w_{2,o}(X) * w_{3,o}(X) * w_{4,o}(X) * w_{5,o}(X) * (1-w_{6,o}(X)) * (1-w_{7,o}(X)) * (1-w_{8,o}(X)) * (1- w_{9,o}(X)) + \cdots}{|T| -1}
$$

Where there is a term for each scenario including five or more scenario wins. $1-w(X)$ represents a category that is lost, in the sense that a $0.8$ or $80\%$ chance of winning translates to a $1.0 - 0.8 = 0.2$ or $20\%$ chance of losing.

You may also note that this formula assumes that all categories are independent from each other, given a choice of $X$. Ideally the equation would allow for correlations and compute the joint probability of each scenario. However, this turns out to be massively difficult- more on that in the limitations section

### 2b. Calculating $W_{c,o}(X)$

The discussion of static ranking lists established that point differentials between teams can be modeled as Bell curves. This can be applied in the dynamic context as well. One necessary modification is that player selections do not contribute to variance in the same way, since the manager controls their own future picks, and previous picks are known. With the assumption that the manager's own future draft picks contribute no variance, the resulting curve can be defined in the following way
- The mean is $X - X_{o_\mu}$, where $X_{o_\mu}$ is the expected value of $X$ for opponent $o$
- The variance is (roughly) $(M- N) * m_{\sigma}^2 + 2 * N * m_{\tau}^2$, where $M$ is the number of players that each team will eventually have and $N$ is the total that the opponent has picked so far. 

### 2c. Breaking down $X$ and $x_{o_\mu}$

$X$ and $X_\mu$ are not particularly helpful in and of themselves, because it is not obvious how to estimate them. They are more helpful after being decomposed into components for each stage of the draft

- $X = X_s + X_p + X_{\phi}$ where
  - $X_s$ is the aggregate statistics of team $A$'s already selected players: 
  - $X_p$ is the statistics of the candidate player
  - $X_{\phi}$ is the expected statistics of unchosen players
- $X_{o_\mu} = X_{o_\theta} + X_{o_\phi}$ where
  - $X_{o_\theta}$ is the aggregate statistics of players drafted up to the round player $p$ is being drafted
  - $X_{o_\phi}$ is the expected aggregate statistics of players drafted past the round player $p$ is being drafted

  Then define 
  - $X_\delta = X_{\phi} - X_{o_\phi}$. In other words, how we expect the statistics of the manager's future picks to differ from what would otherwise would be expected

This allows us to redefine the Bell curve's mean to $X_s + X_p - X_{o_\theta} + X_\delta$. $X_s$ is known to the manager. Values of $X_p$ are known as a function of candidate player. $m_{\sigma}$ and $m_{\tau}$ are easily estimated, as discussed in the static context. 

So every component of the equation can be accounted for, except for $X_{o_\theta}$ and $X_\delta$. The next sections describes how to find them

### 2d. Estimating $X_{\theta}$

$X_{o_\theta}$ depends on the opponent. If the opponent has already chosen one more player than the current manager, $X_{o_\theta}$ is already known exactly. If they have instead chosen the same number of players, then some level of interpolation must be used to fill in the last player's statistics. 

There are many possible ways to make this interpolation. H-scoring's method is ranking all available players by generic value (either G-score or Z-score) and then finding the ones that would be taken by the end of the round, if all managers were picking purely by generic value. Then it takes the average stats of all of them to fill in the gap left by the extra player

### 2e. Estimating $X_\delta$

A natural choice for modeling the statistics of draft picks is the [multivariate normal distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution). It has two useful properties
- It can incorporate correlations between different categories. This is essential because it allows the algorithm to understand that some combinations of categories are easier to jointly optimize than others, e.g. prioriting both rebounds and blocks is easier than prioritizing assists and turnovers
- It is relatively easy to work with. A more complicated function, while perhaps more suited to real data, would make the math much more complicated

It is simple to derive a parameterization for $X_\delta$ when it ignores player position and draft circumstances. One could simply compute the mean, variance, and correlations of real player data. However, player position and draft circumstances both present complications
- If raw player statistics were used to calculate correlations, the model would believe that it could choose a team full of e.g. all point guards to punt Field Goal % to an extreme degree. In reality teams need to choose players from a mix of positions. Player position can be accounted for by subtracting out position means. E.g. if Centers get $+0.5$ rebounds on average, a center's rebound number could be adjusted by $-0.5$. Or, since there are some flex spots making position requirements not entirely rigid, the adjustment number could be scaled by some constant which is $\leq 1$, which we call $\nu$. So if $\nu$ is $0.8$, the previous example would instead lead to $0.4$ rebounds being subtracted out from centers' numbers
- Draft circumstances can be accounted for with two conditions, one dealing with the pool of available players and the other dealing with how players in $X_\delta$ are chosen
  - All players above a certain threshold of general value have been picked, as an approximation of the fact that more valuable players will be taken earlier. We can approximate this threshold to be $0$, since $X_\delta$ is defined relative to what is expected, and should have $0$ value when $j = v$ 
  - The chosen player is the highest-scoring of those remaining based on the aforementioned weight vector $j$. This reflects that the manager will choose the best players according to their own weight vector in the future

This scenario provides a launching point for calculating the expected value of future draft picks, $X_\delta(j)$. The calculation involves many steps of linear algebra, the details of which are in the paper. The result is as follows

Defining 
- $\Sigma$ as the [covariance matrix](https://en.wikipedia.org/wiki/Covariance_matrix) across players after being adjusted for position 
- $v$ as a weighting that we expect other managers to use, perhaps corresponding to Z-score or G-score
- $N$ as how many players the manager has already selected
- $\omega$ as a parameter controlling how well punting strategies are expected to work generally
- $\gamma$ as a complement to $\omega$, controlling how much general value needs to be sacrificed in order to find the player that optimizes for the punting strategy

Then
$$
X_\delta(j) = \left( 12 - N \right) \Sigma \left( v j^T - j v^T \right) \Sigma \left( - \gamma j - \omega v \right) \frac{
   \sqrt{\left(j -  \frac{v v^T \Sigma j}{v^T \Sigma v} \right) ^T \Sigma \left( j -  \frac{v v^T \Sigma j}{v^T \Sigma v}  \right) }
  }{j^T \Sigma j v^T \Sigma v - \left( v^T \Sigma j \right) ^2} 
$$

Super simple and easy to calculate, right :stuck_out_tongue:. $X_\delta(j)$ is obviously too complicated to evaluate repeatedly by hand. Fortunately it is almost trivial for computers to do it for you, as implemented on this website.

It should be noted that this calculation is *very* rough because it uses many layers of approximation. Still, it captures the main effects that are important: higher weight for a category increases the expectation for that category, weights that are more different from standard weights lead to more extreme statistics, and some combinations of categories work better together than others

For intuition, see an animated version of how the algorithm works for two categories below 

<iframe width = "896" height = "504" src="https://github.com/zer2/Fantasy-Basketball--in-progress-/assets/17816840/7ca9674a-8780-4839-9bbb-02025bf33f6f"> </iframe>

In this example $j$ weighs $C_1$ above $C_2$, so the algorithm's contour lines are askew from the line of general value. The algorithm can sacrifice a small amount of general value by moving to the left in order to find a player that fits $j$ better, with a high value for $C_1$ and a low value for $C_2$

## 3. Optimizing for $p$ and $j$

The equations in the preceding sections provide a full picture of how to map $p$ and $j$ to an H-score. The next step is finding the best possible values of $p$ and $j$.

There are a finite number of potential players $p$, so the manager can simply try each of them. However $j$ presents a problem because trying all values of $j$ is not possible, since there are infinite choices for $j$. Even if the manager were to simplify it to e.g. $10$ choices of weight per category, there would still be $\approx 10^8$ options to look through, which is a lot!

Instead of looking through all the options for $j$ at random for each possible choice of $p$, we can use a method called [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent). Essentially, gradient descent conceives of the solution space as a multi-dimensional mountain, and repeatedly moves in the direction of the highest or lowest slope to eventually reach a peak or valley. See a demonstration from youtube below, of gradient descent finding a minimum from various starting points

<iframe width = "800" height = "450" padding = "none" scrolling="no" src="https://www.youtube.com/embed/kJgx2RcJKZY"> </iframe>

You may recognize that this method doesn't guarantee finding the absolute minimum or maximum, it just keeps going until it gets stuck. While this is not ideal it is also impossible to avoid, since there is no guaranteed way to find the optimal point unless the space has a special property ([convexity](https://en.wikipedia.org/wiki/Convex_function)) which $H(j)$ does not have.

Another downside of gradient descent is that it necessitates recalculating the slope every time it moves, which takes time. Computers can do this calculation fairly quickly but the temporal cost of doing it many times in a row does add up, especially when we are running the process separately to optimize $j$ based on choice of $p$.

After performing gradient descent, each player $p$ is paired with an optimal or close to optimal $j$. One of those pairs has the highest H-score. The player $p$ associated with that pair is the one most recommended by the H-score algorithm

## 4. Results

Simulations were performed to test how well managers would do using H-score for each of their picks. Detailed results are included in the paper. To summarize them, the H-score algorithm won up to $24\%$ of the time in Each Category and up to $43\%$ of the time in Most Category against managers drafting via G-score. These simulations do not have other managers punting, so they may not be perfectly reflective of real fantasy basketball, but they do provide evidence that the algorithm is appropriate.

The behavior of the algorithm is interesting, and I encourage you to look through figures 19 through 25 in the paper which describe it. I will also highlight one particular figure, which demonstrates how the algorithm implicitly handles the concept of punting

<iframe  width = "1000" height = "500" padding = "none" scrolling="no" src="https://github.com/zer2/Fantasy-Basketball--in-progress-/assets/17816840/926f3396-acaf-426a-a8bc-108b66bbb900"> </iframe>
<iframe  width = "1000" height = "500" padding = "none" scrolling="no" src="https://github.com/zer2/Fantasy-Basketball--in-progress-/assets/17816840/41bc0dad-aa23-434b-9cbb-b037de2ed11d"> </iframe>

This shows a heavily bimodal distribution of category win rates, with a sharp peak at 0% and another at 75% or so. It seems that the algorithm is roughly bifurcating between categories that it is attempting to compete in and categories which it is strategically sacrificing. However, the bifurcation is not strictly binary: a sizable portion of the distribution’s density is between 5% and 35%, indicating that some categories are being only partially sacrificed, with the algorithm implicitly still hoping to win those categories sometimes. It turns out that even the categories with a $<5\%$ win chance are not being _entirely_ ignored by the model either. Their weights remain slightly above $0\%$ even in the most extreme cases. So while punt vs. not punt is a discrete decision being made by the model, there is also a secondary decision of to what degree the punt is taken, and that decision is made along a spectrum. 

## 5. Limitations

H-scoring is sophisticated but it is not a panacea and could be improved in many ways
- The algorithm implicitly assumes that other managers will pick players generically for their future picks, which is not always reasonable. It can adjust to some degree by the end of the draft when it knows nearly full teams, but that is not a perfect solution. One way that this could be a serious problem is if another manager has the exact same punt build as you, and the algorithm does not realize that the players it wants most will be harder to get as a result. Manual intervention would help in that case 
- As noted earlier, H-scoring does not take into account correlations between weekly values for a category. For example, even controlling for team composition, a team is likely to have many turnovers during a week when they also have many assists, so they are relatively unlikely to win both assists and turnovers on the same week. This matters for the Most Categories format because it can influence the probabilities of various scenarios. It is difficult to account for because it would require computing the joint cumulative density function of $X$, which is computationally extremely expensive. With the scipy implementation of multivariate normal, it would take many hours to run the algorithm for a single pick. The efficiency can be improved somewhat with smart coding logic but is still prhibitively time-consuming to implement, let alone test it across many seasons and picks. 
- The approximation of future draft statistics smooths out outlier players. A particular strategy might seem to be weak in general, but a single outlier player can make it viable in a way that the approximation does not capture. So if you have an idea for a particular build that relies heavily on a small number of unusual players, it might be better than H-scoring would suggest 
- The algorithm understands that it cannot pick all players of the same position with future picks through the $\nu$ parameter, but it does not adjust H-scores by position, even if the top scorers are heavily tilted towards some positions over others. It works this way because in my simulations, the greedy heuristic of simply taking the highest-scoring available player that can fit on the team at every stage of the draft does fine, and I have not found a value-above-replacement system which improves algorithm performance. However, this may be impractical for real drafting. Real fantasy basketball has no concrete rules around team construction and most managers want to avoid accidentally constructing unbalanced teams. So you might want to pick players of new positions even if they have slightly lower H-scores e.g. If the algorithm is leaning towards centers to align with its punting strategy, and it finds a point guard that is only slightly below the top pick in terms of overall H-score, you might want to pick it
- The extension of H-scoring to Rotisserie implemented in this tool is experimental and likely unreliable. For Rotisserie, week-to-week variance does not matter, which puts more emphasis on uncertainty in season-long averages. H-scoring's framework does not account for that kind of uncertainty 
