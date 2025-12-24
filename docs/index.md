This is documentation for a [website](https://fantasy-basketball-optimizer-y9jt7t3ypmiejsyjkeayx6.streamlit.app/) which applies algorithms to category-based fantasy basketball. The algorithms are described in these papers: 

- [Improving algorithms for fantasy basketball](https://arxiv.org/abs/2307.02188)
- [Dynamic algorithms for fantasy basketball](https://arxiv.org/abs/2409.09884)
- [Optimizing for Rotisserie fantasy basketball](https://arxiv.org/abs/2501.00933)

Please note that these algorithms are based on a simplified model of fantasy basketball, and there is no guarantee that using them will lead to success. Don't expect to automatically win your league with the algorithms or even to have a better shot than anyone else. There are many practical issues in real fantasy basketball which the algorithms do not account for, potentially rendering the algorithms' suggestions suboptimal in certain situations. The intent of the papers is to start exploring the math underlying fantasy basketball, and the intent of the website is to offer a window into that math.  I only hope that exploring the website/the papers will be fun and interesting :smile:. 

## Introduction to category-based fantasy basketball

In fantasy leagues, "managers" draft teams of real players before the season begins. E.g. I might take Victor Wembanyama with my first pick, then you take Nikola Jokic, someone else takes Luka Doncic, and so on. At the end of the draft everyone has a 13-player team. 

Throughout the season, teams accumulate statistics based on their real players' performances. E.g. if Wembanyama gets 7 blocks in a game, I get 7 blocks, and if Jokic scores 20 points, you get 20 points. 

In category-based leagues, which most fantasy basketball leagues are, the important thing is winning individual categories. So if my team gets 100 blocks and your team gets 10 blocks, I get a "fantasy point" for winning blocks. If your team gets 300 points and my team gets 299 points, you get an equivalent fantasy point for that; the margin is irrelevant. Generally these match-ups occur on a weekly basis across eight or nine statistical categories, and each team keeps all the fantasy points they win throughout the season. 

Based on that scoring mechanism, what is the best way to choose players during a draft? This is a fascinating question which has no simple answer. Trying to do the best we can is the motivating force behind the papers. 

## The papers  

Broadly, the papers formulate category-based fantasy basketball as a math problem, and provide methods for selecting players based on that formulation. The methods take projections of player performance as an input. 

The [first paper](https://arxiv.org/abs/2307.02188) looks at so-called "static" systems, which try to evaluate players in a vacuum. It makes the simplification of assuming that all players except the one being chosen will be selected randomly, which is a strong assumption, but one that lines up well with the traditional metric of Z-scores. The paper describes an improvement upon Z-scores called G-scores, which are used on the website for describing player values statically. See also the [G-score](gscores.md) section of documentation. 

The [second paper](https://arxiv.org/abs/2409.09884) considers the task of picking players based on context. The trickiest part of this is that the best player to select depends on players that will be selected in the future, which cannot be known for certain. The solution is an algorithm called H-scoring which simultaneously optimizes for a player and a heuristic approach to future draft picks. See also the [H-score](hscores.md) section of documentation. 

The [third paper](https://arxiv.org/abs/2501.00933) attacks the trickiest format, Rotisserie or "Roto". Instead of weekly matchups, Roto has one season-long scoring period in which all managers compete against all other managers. Winning a Roto league requires accumulating the most fantasy points in that one scoring period. The approach of the paper is to model all teams' fantasy point totals as correlated Normal distributions, and use estimates of their order statistics to calculate the probability of ending up with the most fantasy points. See also the [Roto](roto.md) section of documentation. 

## The website 

The central purpose of the website is to apply the methods of the papers to drafts and auctions. It makes a few practical adjustments from the methods described in the papers, particularly [adjusting implicit projections during drafts](projectionadjustment.md). It also provides some related analyses for leagues that are already underway via ["Season Mode"](season.md).

Source code is available [here](https://github.com/zer2/fantasy-basketball-optimizer).
