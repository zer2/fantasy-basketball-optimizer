# Adjusting the projections with a Bayesian prior 

H-scoring is fully reliant on a single set of projections. If a drafter takes a player it projects to be a poor performer highly, the algorithm will not "doubt itself" and consider the possibility that its projections for that player are too low. It will assume that pick was a poor choice and the drafter who took it will have a bad team. 

This inability to doubt itself makes the algorithm overconfident, believing that its own team is very strong, when its own projections may not be better than anybody else's. As a practical matter this can lead the algorithm to think its team is so strong that the only way to improve is to "un-punt" categories it has given up on, which is probably a bad idea in practice. 

The papers assume that player projections are all known and agreed upon by all the drafters, so they don't address this issue. However, it is so important in practice that I've added a module specifically to address it. 

## The adjustment 

Say that $w_c$ is the algorithm's naive guess at how likely it is to win category $c$, before performing gradient descent to optimize a future strategy. Corrected versions are calculated as

$$
w^*_d = \frac{w_d - \beth \left(\frac{ \sum_{c \neq d}  \left( w_c \right) - \frac{n}{2}}{ n^2} \right) }{1 + \frac{\beth}{ n^2}}
$$

Where $n$ is the number of categories and $\beth$ is a parameter- more in that in the justification.

These corrected win rates are then used to reverse engineer an adjusted expectation of the team's current strength, like so: 

$$
x^*_d = \text{CDF}^{-1} \left( w^*_d \right)
$$

## Justification 

Say that we have prior expectations that 
- Our average win rate across all categories is approximately 50%, with Normally distributed error. 
- Our guesses for how often we will win a category are unbiased, but have some Normally distributed error. 

This information provides a Bayesian framework for re-calculating adjusted category-level win rates. 

By Bayes' rule, the probability of a certain set of category win rates being correct is proportional to its likelihood time the prior. In this case, the likelihood is 

$$
\prod_c \phi (\frac{w^*_c - w_c}{\epsilon_a})
$$

And the prior probability is 

$$
\phi \left( \frac{\frac{ \sum_c \left( w^*_c - \frac{1}{2} \right)}{n}}{\epsilon_b} \right) = 
\phi \left( \frac{ \sum_c \left( w^*_c - \frac{1}{2} \right)}{\epsilon_b n} \right)
$$

Multiplying them together yeilds 

$$
\left[ \prod_c \phi \left(\frac{w^*_c - w_c}{\epsilon_a} \right) \right] \left[ \phi \left(\frac{ \sum_c \left( w^*_c - \frac{1}{2} \right)}{\epsilon_b n } \right) \right]
$$

We are only interested in what has the maximal likelihood, not what that likelihood is. So it is fine to convert this to log odds, which are 

$$
\left[ \sum_c \left(\frac{w^*_c - w_c}{\epsilon_a} \right)^2 \right] +  \left(\frac{ \sum_c \left( w^*_c - \frac{1}{2} \right)}{\epsilon_b n} \right)^2 
$$

To optimize this, we set the derivative to zero. Applying the chain rule for category d- 

$$
0 = 2 \left(\frac{w^*_d - w_d}{\epsilon_a} \right) \frac{1}{\epsilon_a} + 2 \left(\frac{ \sum_c \left( w^*_c - \frac{1}{2} \right)}{\epsilon_b n} \right) \frac{1}{\epsilon_b n}
$$

Isolating $w^*_d$- 

$$
2 \left(\frac{w_d}{\epsilon_a^2} \right) - 2 \left(\frac{ \sum_{c \neq d}  \left( w^*_c \right) - \frac{n}{2}}{\epsilon_b n} \right) \frac{1}{\epsilon_b n}= 2 \left(\frac{w^*_d}{\epsilon_a} \right) \frac{1}{\epsilon_a} + 2 \frac{w^*_d}{\epsilon_b^2 n^2}
$$

$$
2 \left(\frac{w_d}{\epsilon_a^2} \right) - 2 \left(\frac{ \sum_{c \neq d}  \left( w^*_c \right) - \frac{n}{2}}{\epsilon_b n} \right) \frac{1}{\epsilon_b n}= w^*_d \left( 2 \frac{1}{\epsilon_a^2} + 2 \frac{1}{\epsilon_b^2 n^2} \right) 
$$

So 

$$
w^*_d = \frac{\frac{w_d}{\epsilon_a^2} - \left(\frac{ \sum_{c \neq d}  \left( w^*_c \right) - \frac{n}{2}}{\epsilon_b^2 n^2} \right) }{\frac{1}{\epsilon_a^2} + \frac{1}{\epsilon_b^2 n^2}}
$$

With $\beth = \frac{\epsilon_a^2}{\epsilon_b^2}$, this is 

$$
w^*_d = \frac{w_d - \beth \left(\frac{ \sum_{c \neq d}  \left( w^*_c \right) - \frac{n}{2}}{ n^2} \right) }{1 + \frac{\beth}{ n^2}}
$$

We don't actually have the other values of $w^*_c$ immediately, but we can approximate them with $w_c$. That makes the final formula 

$$
w^*_d = \frac{w_d - \beth \left(\frac{ \sum_{c \neq d}  \left( w_c \right) - \frac{n}{2}}{ n^2} \right) }{1 + \frac{\beth}{ n^2}}
$$


## Small additional adjustment 

The methodology outlined above has a factor of 

$$
\left(\frac{ \sum_{c \neq d}  \left( w^_c \right) - \frac{n}{2}}{ n^2} \right)
$$

This essentially measures how confident the algorithm is in the team's strength before a punting adjustment is made. Technically this changes for each candidate player, but we don't want it to, since that would mean "punishing" players for being strong without a punting strategy. Instead, the website takes the max value across all candidate players for each category.