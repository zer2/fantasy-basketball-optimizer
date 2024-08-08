# Dynamic strategy with H-scoring

Every seasoned fantasy manager knows that performing well at the highest level of competition requires adapting to draft circumstances on the fly. Picking directly from a static ranking list can easily lead to an unbalanced, incohesive team. 

In general, the way that the best managers adapt is more of an art than a science. They might "punt" some number of categories, then choose players in such a way that they are above average in the other categories. The idea is that since margins do not matter, winning many categories by small margins is worth losing a few by large margins.

In the [paper](https://arxiv.org/abs/2307.02188), I derive an algorithm called H-scoring to translate this intuition into a rigorous procedure. By framing draft strategy as an optimization problem, it implicitly understands how to rebalance teams for optimal performance, sacrificing some categories and shoring up others. While imperfect, I believe that the logic is sound, and evidence suggests that it works, at least in a simplified context. 

For details of how the algorithm works, see the paper. For a much simplified version, with less math, see below 

## 1. The H-scoring approach

Dynamic drafting is a fundamentally more difficult proposition than static drafting. More information about drafting context is helpful, but figuring out the right way to incorporate it into decision-making is tricky. 

The most challenging aspect of the problem is accounting for future draft picks. They are neither completely under the manager's control (since the manager does not know which players will be available later) nor completely random (since the manager will decide which players to take of those available). Instead, future draft picks fall somewhere between the two extremes. 

H-scores' solution to this dilemma is conceiving of future picks as being controlled by a weight vector, by which aggregate statistics of future picks can be approximated. For example, the algorithm might give seven categories $14\%$ weight each and two categories $1\%$ each, for a total of $100\%$. The algorithm would then assume that the manager would use the custom weighing to value future picks. That would lead the algorithm to approximate statistics for future picks with a slight upwards bias in the seven up-weighted categories and a large downwards bias in the two down-weighted categories. By this mechanism, the manager has some measure of control over future picks. 

Of course, being able to map a weight vector $j$ to approximate statistics for future picks is not enough for optimal draft strategy. The manager needs to operate the other way around, and pick both a player $p$ from the available candidate pool and a weight vector $j$ for future picks such that the probability of winning (H-score) is optimized. 

Doing all of this algorithmically is far from a trivial task. It can be accomplished by first building a full model linking $p$ and $j$ to H-score, then applying mathematical tools to the model to discover choices of $p$ and $j$ that mazimize the reward function

## 2. Calculating H-score based on $p$ and $j$

Writing out a single equation for H-score is cumbersome because the methodology behind it has so many steps. Instead, it is easiest to understand by starting from the ultimate goal of the metric and working backwards to successively fill in gaps.

### 2a. Defining the objective function

In optimization, the objective function is the thing we want to maximize. 
- For Each Categories, it is just the expected number of categories won against an arbitrary opponent. This is easy to calculate when victory probabilities are known for each category
- For Most Categories, it is the probability of winning a majority of categories. This is slightly harder to calculate but can still be done by explicitly checking every possibility 

### 2b. Calculating the probability of winning a category

The objective functions defined above both rely on per-category victory probabilities. So we will need to calculate them. 

The discussion of static ranking lists established that point differentials between teams can be modeled as Bell curves. Then, CDFs of those Bell curves can be translated into victory probabilities. This can be applied in the dynamic context as well. 

To parameterize a Bell curve, we need two things: the mean of the distribution $\mu$ and its variance $\sigma^2$. Then, we will have what we need to calculate victory probabilities, with the equation 

$$
CDF(0) = \frac{1}{2}\left[ 1 + \frac{2}{\sqrt{\pi}}* \frac{- \mu }{ \sigma} \right]
$$

### 2c. Calculating the mean and variance of a category differential

$\sigma^2$ of the category differential is not too hard to estimate, assuming that all players contribute the same amount of week-to-week variance. We just add up the week-to-week variance ($m_\tau^2$) and the variance contributions of unknown players on team $B$ ($m_\sigma^2$) to get 

$$
2N m_\tau^2 + (N-K-1) m_\sigma^2
$$

Where $N$ is the number of players per team, and $K$ is the number of players that have been selected already.

Calculating $\mu$ of the category differential is where H-scoring gets complicated. If all players were already chosen, it would be easy to calculate $\mu$ by subtracting the mean of team $A$ with the mean of team $B$. But players yet to be picked complicate the situation. 

Remember that H-scoring gives the manager heuristic control over future picks via the $j$ vector, and we need to account for that. We can start with the difference between team $A$ and $B$ of players already selected, and add an additional factor $X_{\delta}(j)$ to reflect how the manager can manipulate the expected statistics of their future player

### 2d. Estimating $X_{\delta}$

The important question is how the statistics of future draft picks are expected to change based on the manager's choice of $j$. Obviously increasing the weight of a category will increase it in $X_{\delta}$, but it is unclear by how much. 

Assisted by many layers of approximations, the math in the paper concludes that

$$
X_\delta(j) = \left( 12 - N \right) \Sigma \left( v j^T - j v^T \right) \Sigma \left( - \gamma j - \omega v \right) \frac{
   \sqrt{\left(j -  \frac{v v^T \Sigma j}{v^T \Sigma v} \right) ^T \Sigma \left( j -  \frac{v v^T \Sigma j}{v^T \Sigma v}  \right) }
  }{j^T \Sigma j v^T \Sigma v - \left( v^T \Sigma j \right) ^2} 
$$

Super simple and easy to calculate, right :stuck_out_tongue:. $X_\delta(j)$ is obviously too complicated to evaluate repeatedly by hand. Fortunately it is almost trivial for computers to do it for you, as implemented on this website.

It should be noted that this calculation is *very* rough because it uses many layers of approximation. Still, it captures the main effects that are important: higher weight for a category increases the expectation for that category, weights that are more different from standard weights lead to more extreme statistics, and some combinations of categories work better together than others

For visual intuition on how this equation works, see an animated version for two categories below 

<iframe width = "896" height = "504" src="https://github.com/zer2/Fantasy-Basketball--in-progress-/assets/17816840/7ca9674a-8780-4839-9bbb-02025bf33f6f"> </iframe>

In this example $j$ weighs $C_1$ above $C_2$, so the algorithm's contour lines are askew from the line of general value. The algorithm realizes that it can sacrifice a small amount of general value by moving to the left in order to find a player that fits $j$ better, with a high value for $C_1$ and a low value for $C_2$

### 2f. Adjusting for position

It would be a mistake to ignore position, since otherwise the H-scoring algorithm would believe that it could make a team full of only point guards. To get around this, H-scoring adjusts $X_\delta$ with a three-step process

1. Based on current weights, figure out how best to allocate previously chosen players. For example if a build based on $j$ prioritizes point guards heavily, then the algorithm will want to consider a previously chosen player eligible as either SG or PG as a SG
2. Decide how to allocate remaining flex spots. E.g. divvying up two utility spots with one point guard and one small forward 
3. Based on the computed future positions, add a bonus based on position means. E.g. if the algorithm has decided that its two remaining slots will go to point guards, it adds two times the average point guards' stats (normalized to 0) to its total 

## 3. Optimizing for $p$ and $j$

The equations in the preceding sections provide a full picture of how to map $p$ and $j$ to an H-score. The next step is finding the best possible values of $p$ and $j$.

There are a finite number of potential players $p$, so the manager can simply try each of them. However $j$ presents a problem because trying all values of $j$ is not possible, since there are infinite choices for $j$. Even if the manager were to simplify it to e.g. $10$ choices of weight per category, there would still be $\approx 10^8$ options to look through, which is a lot!

Instead of looking through all the options for $j$ at random for each possible choice of $p$, we can use a method called [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent). Essentially, gradient descent conceives of the solution space as a multi-dimensional mountain, and repeatedly moves in the direction of the highest or lowest slope to eventually reach a peak or valley. See a demonstration from youtube below, of gradient descent finding a minimum from various starting points

<iframe width = "800" height = "450" padding = "none" scrolling="no" src="https://www.youtube.com/embed/kJgx2RcJKZY"> </iframe>

You may recognize that this method doesn't guarantee finding the absolute minimum or maximum, it just keeps going until it gets stuck. While this is not ideal it is also impossible to avoid, since there is no guaranteed way to find the optimal point unless the space has a special property ([convexity](https://en.wikipedia.org/wiki/Convex_function)) which $H(j)$ does not have.

Another downside of gradient descent is that it necessitates recalculating the slope every time it moves, which takes time. Computers can do this calculation fairly quickly but the temporal cost of doing it many times in a row does add up, especially when we are running the process separately to optimize $j$ based on choice of $p$.

After performing gradient descent, each player $p$ is paired with an optimal or close to optimal $j$. One of those pairs has the highest H-score. The player $p$ associated with that pair is the one most recommended by the H-score algorithm

## 4. Results

Simulations were performed to test how well managers would do using H-score for each of their picks. Detailed results are included in the paper. To summarize them, the H-score algorithm won approximately $22\%$ of the time in Each Category and $38\%$ of the time in Most Category against managers drafting via G-score. These simulations do not have other managers punting, so they may not be perfectly reflective of real fantasy basketball, but they do provide evidence that the algorithm is appropriate.

The behavior of the algorithm is interesting, and I encourage you to look through the figures which describe it. I will also highlight one particular figure, which demonstrates how the algorithm implicitly handles the concept of punting

<iframe  width = "800" height = "400" padding = "none" scrolling="no" src="https://github.com/user-attachments/assets/86b10960-6780-42ff-b9dd-53f5058e34ca"> </iframe>

The bimodal distribution of weights demonstates that the algorithm does understand the concept of punting. It is implicitly making a decision to either fight wholeheartedly for a category, or allow the category to fall to the wayside. However, it is worth noting that the algorithm does not bring weights of punted categories alls the way down to zero. Instead, it brings it to around $80\%$, which skews the players it ends up taking, without sacrificing too much overall flexibility

## 5. Limitations

H-scoring is not a panacea and could be improved in many ways
- The algorithm implicitly assumes that other managers will pick players generically for their future picks, which is not always reasonable. It adjusts to some degree by the end of the draft when it knows nearly full teams, but that is not a perfect solution. One way that this could be a serious problem is if another manager has the exact same punt build as you, and the algorithm does not realize that the players it wants most will be harder to get as a result. Manual intervention would help in that case 
- H-scoring does not take into account correlations between weekly values for a category. For example, even controlling for team composition, a team is likely to have many turnovers during a week when they also have many assists, so they are relatively unlikely to win both assists and turnovers on the same week. This matters for the Most Categories format because it can influence the probabilities of various scenarios. It is difficult to account for because it would require computing the joint cumulative density function of $X$, which is computationally extremely expensive. With the scipy implementation of multivariate normal, it would take many hours to run the algorithm for a single pick. The efficiency can be improved somewhat with smart coding logic but is still prhibitively time-consuming to implement, let alone test it across many seasons and picks. 
- The approximation of future draft statistics smooths out outlier players. A particular strategy might seem to be weak in general, but a single outlier player can make it viable in a way that the approximation does not capture. So if you have an idea for a particular build that relies heavily on a small number of unusual players, it might be better than H-scoring would suggest 
- Assuming that all players contribute the same amount of week-to-week variance is problematic, because generally, lower means translate to lower variances. This translates to punted categories having lower variance than expected by the algorithm, throwing off the victory probability calculation 
- H-scoring converts ratio stats into counting stats, which somewhat skews its calculation of win probabilities for ratio stats 
