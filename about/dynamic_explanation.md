# Dynamic strategy with H-scoring

Every seasoned fantasy drafter knows that performing well at the highest level of competition requires adapting to draft circumstances on the fly. Picking directly from a static ranking list can easily lead to an unbalanced, incohesive team. 

As it currently stands, the way that drafters adapt is generally more of an art than a science. Drafters might "punt" some number of categories, then choose players in such a way that they are above average in the other categories. This is a sensible approach, but it does not have a strong mathematical backbone. 

In the [paper](https://arxiv.org/abs/2307.02188), I derive an algorithm called H-scoring to translate this intuition into a rigorous procedure. By framing draft strategy as an optimization problem, it implicitly understands how to rebalance teams for optimal performance, sacrificing some categories and shoring up others. While imperfect, I believe that the logic is sound, and evidence suggests that it works at least in a simplified context. 

Below is a summary of how the algorithm is designed

## 1. The H-scoring approach

Dynamic drafting is a fundamentally more difficult proposition than static drafting. More information about drafting context is helpful, but figuring out the right way to incorporate it into decision-making is tricky. 

The most challenging aspect of the problem is accounting for future draft picks. They are neither completely under the drafter's control (since the drafter does not know which players will be available later) nor completely random (since the drafter will decide which players to take of those available). Instead, future draft picks fall somewhere between the two extremes. 

H-scores' solution to this dilemma is conceiving of future picks as being controlled by a weight vector, by which aggregate statistics of future picks can be approximated. For example, the  algorithm might give seven categories $14\%$ weight each and two categories $1\%$ each, for a total of $100\%$. The algorithm would then assume that the drafter would use the custom weighing to value future picks. That would lead the algorithm to approximate statistics for future picks with a slight upwards bias in the seven up-weighted categories and a large downwards bias in the two down-weighted categories. By this mechanism, the drafter has some measure of control over future picks. 

Of course, being able to map a weight vector $j$ to approximate statistics for future picks is not enough for optimal draft strategy. The drafter needs to operate the other way around, and pick both a player $p$ from the available candidate pool and a weight vector $j$ for future picks such that the probability of winning (H-score) is optimized. 

This is far from a trivial task. It can be accomplished by first building a full model linking $p$ and $j$ to H-score, then applying mathematical tools to the model to discover choices of $p$ and $j$ that mazimize the reward function

## 2. Calculating H-score based on $p$ and $j$

Writing out a single equation for H-score is cumbersome because the methodology behind it has so many steps. Instead, it is easiest to understand by starting from the ultimate goal of the metric and working backwards to successively fill in gaps 

### 2a. Defining the objective function

In the static ranking context the expected number of category wins was a reasonable objective even for Most Categories, since strategizing how to win only five out of nine categories was impossible. In the dynamic context, more information is available, and using the appropriate objective function for the format is warranted. 

Another consideration is that the drafter will have some control over the aggregate statistics of their team so the objective function should be expressed as a function of team composition. 

Define $H(X)$ as the objective function relative to the team $A$'s stat distribution $X$. With $w_c(X)$ as the probability of winning category $c$ based on $X$ and $|C|$ as the number of categories, the objective function for the Each Category format is simply 

$$
H(X) = \frac{ \sum_c w_c(X)}{|C|}
$$

For Most Categories, $H(x)$ is slighly more complicated, since it is the probability of winning the majority of categories. It can be written as

$$
H(j)  = w_1(X) * w_2(X) * w_3(X) * w_4(X) * w_5(X) * (1-w_6(X)) * (1-w_7(X)) * (1-w_8(X)) * (1- w_9(X)) + \cdots
$$

Where there is a term for each scenario including five or more scenario wins. $1-w(X)$ represents a category that is lost, in the sense that a $0.8$ or $80\%$ chance of winning translates to a $0.2$ or $20\%$ chance of losing.

You may also note that this formula assumes that all categories are independent from each other, given a choice of $X$. Ideally the equation would allow for correlations and compute the joint probability of each scenario. However, this turns out to be massively difficult- more on that in the limitations section

### 2b. Calculating $W_c(X)$

The discussion of static ranking lists established that point differentials between teams can be modeled as Bell curves. This can be applied in the dynamic context as well. One necessary modification is that players on team $A$ do not contribute to player-to-player variance, since they are under control of the drafter. The resulting curve can then be defined in the following way
- The mean is $X - X_{\mu}$, where $X_{\mu}$ is the expected value of $X$ for other teams
- The variance is $N * m_{\sigma}^2 + 2 * N * m_{\tau}^2$

### 2c. Breaking down $X$ and $x_\mu$

$X$ and $X_\mu$ are not particularly helpful in and of themselves, because it is not obvious how to estimate them. They are more helpful after being decomposed into components for each stage of the draft

- $X = X_s + X_p + X_{\phi} + X_\delta$ where
  - $X_s$ is the aggregate statistics of team $A$'s already selected players: 
  - $X_p$ is the statistics of the candidate player
  - $X_{\phi}$ is the expected statistics of unchosen players
  - $X_\delta$ is a difference factor to adjust for future picks being different from expected
- $X_\mu = X_{\theta} + X_{\phi}$ where
  - $X_{\theta}$ is the expected aggregate statistics of players drafted up to the round player $p$ is being drafted

This allows the Bell curve's parameters to be redefined as follows 
- The mean is $X_s + X_p - X_{\theta} + X_\delta$
- The variance is $N * m_{\sigma}^2 + 2 * N * m_{\tau}^2$

$X_s$ is known to the drafter. Values of $X_p$ are known as a function of candidate player. $m_{\sigma}$ and $m_{\tau}$ are easily estimated, as discussed in the static context. 

So every component of the equation can be accounted for, except for $X_{\theta}$ and $X_\delta$. The next sections describes how to find them

### 2d. Estimating $X_{\theta}$

$X_{\theta}$ can be estimated by finding the averages of all players drafted up to a certain round, based on a heuristic metric like G-score or Z-score. Obviously this is imprecise since real drafters won't draft according to any one statistic entirely, but it should still capture most of the effect of earlier draft picks tending to be more valuable, which is the main value of involving $X_{\theta}$

### 2e. Estimating $X_\delta$

A natural choice for modeling the statistics of draft picks is the [multivariate normal distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution). It has two useful properties
- It can incorporate correlations between different categories. This is essential because it allows the algorithm to understand that some combinations of categories are easier to jointly optimize than others, e.g. prioriting both rebounds and blocks is easier than prioritizing assists and turnovers
- It is relatively easy to work with. A more complicated function, while perhaps more suited to real data, would make the math much more complicated

It is simple to derive a parameterization for $X_\delta$ when it ignores player position and draft circumstances. One could simply compute the mean, variance, and correlations of real player data. However, player position and draft circumstances both present complications
- If raw player statistics were used to calculate correlations, the model would believe that it could choose a team full of e.g. all point guards to punt Field Goal % to an extreme degree. In reality teams need to choose players from a mix of positions. Player position can be accounted for by subtracting out position means. E.g. if Centers get $+0.5$ rebounds on average, a center's rebound number could be adjusted by $-0.5$. Or, since there are some flex spots making position requirements not entirely rigid, the adjustment number could be scaled by some constant which is $\leq 1$, which we call $\nu$. So if $\nu$ is $0.8$, the previous example would instead lead to $0.4$ rebounds being subtracted out from centers' numbers
- Draft circumstances can be accounted for with two conditions, one dealing with the pool of available players and the other dealing with how players in $X_\delta$ are chosen
  - All players above a certain threshold of general value have been picked, as an approximation of the fact that more valuable players will be taken earlier. We can approximate this threshold to be $0$, since $X_\delta$ is defined relative to what is expected, and should have $0$ value when $j = v$ 
  - The chosen player is the highest-scoring of those remaining based on the aforementioned weight vector $j$. This reflects that the drafter will choose the best players according to their own weight vector in the future

This scenario provides a launching point for calculating the expected value of future draft picks, $X_\delta(j)$. The calculation involves many steps of linear algebra, the details of which are in the paper. The result is as follows

Defining 
- $\Sigma$ as the [covariance matrix](https://en.wikipedia.org/wiki/Covariance_matrix) across players after being adjusted for position 
- $v$ as a weighting that we expect other drafters to use, perhaps corresponding to Z-score or G-score
- $N$ as how many players the drafter has already selected
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

## 3. Optimizing for $p$ and $j$

The equations in the preceding sections provide a full picture of how to map $p$ and $j$ to an H-score. The next step is finding the best possible values of $p$ and $j$.

There are a finite number of potential players $p$, so the drafter can simply try each of them. However $j$ presents a problem because trying all values of $j$ is not possible, since there are infinite choices for $j$. Even if the drafter were to simplify it to e.g. $10$ choices of weight per category, there would still be $\approx 10^8$ options to look through, which is a lot!

Instead of looking through all the options for $j$ at random for each possible choice of $p$, we can use a method called [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent). Essentially, gradient descent conceives of the solution space as a multi-dimensional mountain, and repeatedly moves in the direction of the highest or lowest slope to eventually reach a peak or valley. See a demonstration from youtube below, of gradient descent finding a minimum from various starting points

<iframe width = "800" height = "450" padding = "none" scrolling="no" src="https://www.youtube.com/embed/kJgx2RcJKZY"> </iframe>

You may recognize that this method doesn't guarantee finding the absolute minimum or maximum, it just keeps going until it gets stuck. While this is not ideal it is also impossible to avoid, since there is no guaranteed way to find the optimal point unless the space has a special property ([convexity](https://en.wikipedia.org/wiki/Convex_function)) which $H(j)$ does not have.

Another downside of gradient descent is that it necessitates recalculating the slope every time it moves, which takes time. Computers can do this calculation fairly quickly but the temporal cost of doing it many times in a row does add up, especially when we are running the process seperately to optimize $j$ based on choice of $p$.

After performing gradient descent, each player $p$ is paired with an optimal or close to optimal $j$. One of those pairs has the highest H-score. The player $p$ associated with that pair is the one most recommended by the H-score algorithm. 

## 4. Results

Simulations were performed to test how well drafters would do using H-score for each of their picks. Detailed results are included in the paper. To summarize them, the H-score algorithm won up to $24\%$ of the time in Each Category and up to $43\%$ of the time in Most Category against G-score drafters. These simulations do not have other drafters punting, so they may not be perfectly reflective of real fantasy basketball, but they do provide evidence that the algorithm is appropriate.

The behavior of the algorithm is interesting, and I encourage you to look through figures 19 through 25 in the paper which describe it. I will also highlight one particular figure, which demonstrates how the algorithm implicitly handles the concept of punting

<iframe  width = "1000" height = "500" padding = "none" scrolling="no" src="https://github.com/zer2/Fantasy-Basketball--in-progress-/assets/17816840/926f3396-acaf-426a-a8bc-108b66bbb900"> </iframe>
<iframe  width = "1000" height = "500" padding = "none" scrolling="no" src="https://github.com/zer2/Fantasy-Basketball--in-progress-/assets/17816840/41bc0dad-aa23-434b-9cbb-b037de2ed11d"> </iframe>

This shows a heavily bimodal distribution of category win rates, with a sharp peak at 0% and another at 75% or so. It seems that the algorithm is roughly bifurcating between categories that it is attempting to compete in and categories which it is strategically sacrificing. However, the bifurcation is not strictly binary: a sizable portion of the distribution’s density is between 5% and 35%, indicating that some categories are being only partially sacrificed, with the algorithm implicitly still hoping to win those categories sometimes. It turns out that even the categories with a $<5\%$ win chance are not being _entirely_ ignored by the model either. Their weights remain slightly above $0\%$ even in the most extreme cases. So while punt vs. not punt is a discrete decision being made by the model, there is also a secondary decision of to what degree the punt is taken, and that decision is made along a spectrum. 

## 5. Limitations

H-scoring is sophisticated but it is not a panacea and could be improved in many ways
- The algorithm does not adjust for the choices of other drafters. If you notice another drafter pursuing a particular punting strategy, you might want to avoid that strategy for yourself so that you do not compete for the same players
- As noted earlier, H-scoring does not take into account correlations between weekly values for a category. For example, even controlling for team composition, a team is likely to have many turnovers during a week when they also have many assists, so they are relatively unlikely to win both assists and turnovers on the same week. This matters for the Most Categories format because it can influence the probabilities of various scenarios. It is difficult to account for because it would require computing the joint cumulative density function of $X$, which is computationally extremely expensive. With the scipy implementation of multivariate normal, it would take many hours to run the algorithm for a single pick. The efficiency can be improved somewhat with smart coding logic but is still prhibitively time-consuming to implement, let alone test it across many seasons and picks
  - Many fantasy basketball ranking sites largely ignore turnovers because they are so highly correlated with other categories, particularly assists. Modeling the effect of correlations would be able to test if that was truly appropriate, so it is unfortunate that doing so is infeasible. In the absence of evidence for auto-punting turnovers, H-scores treat them like any other category. It should also be noted that turnovers are already downweighted by G-scores and H-scores since they have high week-to-week variance, so the algorithm will value them somewhere between full Z-score weight and zero anyway 
- The approximation of future draft statistics smooths out outlier players. A particular strategy might seem to be weak in general, but a single outlier player can make it viable in a way that the approximation does not capture. So if you have an idea for a particular build that relies heavily on a small number of unusual players, it might be better than H-scoring would suggest 
- The algorithm understands that it cannot pick all players of the same position with future picks through the $\nu$ parameter, but it does not adjust H-scores by position, even if the top scorers are heavily tilted towards some positions over others. It works this way because in my simulations, the greedy heuristic of simply taking the highest-scoring available player that can fit on the team at every stage of the draft does fine, and I have not found a value-above-replacement system which improves algorithm performance. However, this may be impractical for real drafting. Real fantasy basketball has no concrete rules around team construction and most drafters want to avoid accidentally constructing unbalanced teams. So you might want to pick players of new positions even if they have slightly lower H-scores e.g. If the algorithm is leaning towards centers to align with its punting strategy, and it finds a point guard that is only slightly below the top pick in terms of overall H-score, you might want to pick it. 
- The extension of H-scoring to Rotisserie implemented in this tool is not described in the paper. It is similar to the Each Category algorithm, except that week-to-week variance is set to zero and it is assumed that other drafters will be drafting based on Z-scores. It has not been verified in the Rotisserie context, so there is even more reason for skepticism when interpreting its results than for the other formats
