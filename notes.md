Rankings and algorithms are based on the methods described in [this paper](https://arxiv.org/abs/2307.02188). As detailed in the paper, the algorithms all utilize many simplifying assumptions and are not guaranteed to 
achieve success. 

The most important weaknesses to keep in mind for H-scoring are 
* There is no implicit handling of value above replacement differences across positions. If the algorithm is
leaning towards centers to align with its punting strategy, you might want to pick players of other positions even
if they have slightly lower H-scores
* The algorithm does not adjust for the choices of other players. If you notice another player pursuing a 
particular punting strategy, you might want to avoid that strategy for yourself
* The extension of H-scoring to Rotisserie implemented in this tool is not described in the paper.
It is similar to the Each Category algorithm, except that week-to-week variance is set to zero and it is assumed
that other drafters will be drafting based on Z-scores. It has not been verified in the Rotisserie context, so there
is even more reason for skepticism when interpreting its results than for the other formats

For more details about the algorithm, please consult the paper or contact me on [reddit](https://www.reddit.com/user/zeros1123)
