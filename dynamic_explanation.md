The most important weaknesses to keep in mind for H-scoring are 
* The algorithm does not adjust for the choices of other drafters. If you notice another drafter pursuing a 
particular punting strategy, you might want to avoid that strategy for yourself so that you do not compete
for the same players
* The algorithm understands that it cannot pick all players of the same position with future picks through the $\nu$ parameter, but it does not adjust H-scores by
position, even if the top scorers are heavily tilted towards some positions over others. It works this way because in my simulations, the greedy heuristic of simply taking the highest-scoring available player
that can fit on the team at every stage of the draft does fine, and I have not found a value-above-replacement system which improves
algorithm performance. However, this may be impractical for real drafting. Real fantasy basketball has no concrete rules around team construction
and most drafters want to avoid accidentally constructing unbalanced teams. So you might want to pick players of new positions even if they have slightly lower H-scores
e.g. If the algorithm is leaning towards centers to align with its punting strategy, and it finds a point guard that is only slightly below the top pick in terms of overall
H-score, you might want to pick it. 
* The extension of H-scoring to Rotisserie implemented in this tool is not described in the paper.
It is similar to the Each Category algorithm, except that week-to-week variance is set to zero and it is assumed
that other drafters will be drafting based on Z-scores. It has not been verified in the Rotisserie context, so there
is even more reason for skepticism when interpreting its results than for the other formats
