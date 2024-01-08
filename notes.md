Rankings and algorithms are based on the methods described in [this paper](https://arxiv.org/abs/2307.02188). As detailed in the paper, the algorithms all utilize many simplifying assumptions and are not guaranteed to 
achieve success. 

The most important weaknesses to keep in mind for H-scoring are 
* There is no implicit handling of value above replacement differences across positions. If the algorithm is
leaning towards centers to align with its punting strategy, you might want to pick players of other positions even
if they have slightly lower H-scores
* The algorithm does not adjust for the choices of other players. If you notice another player pursuing a 
particular punting strategy, you might want to avoid that strategy for yourself
