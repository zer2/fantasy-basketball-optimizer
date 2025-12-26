This is documentation for a [website](https://fantasy-basketball-optimizer-y9jt7t3ypmiejsyjkeayx6.streamlit.app/) which applies algorithms to category-based fantasy basketball. The algorithms are described in these papers: 

- [Improving algorithms for fantasy basketball](https://arxiv.org/abs/2307.02188)
- [Dynamic algorithms for fantasy basketball](https://arxiv.org/abs/2409.09884)
- [Optimizing for Rotisserie fantasy basketball](https://arxiv.org/abs/2501.00933)

Please note that these algorithms are based on a simplified model of fantasy basketball, ignoring many practical considerations, and there is no guarantee that using them will lead to success. Don't expect to automatically win your league with the algorithms or even to have a better shot than anyone else. The intent of the papers is just to start exploring the math underlying fantasy basketball, and the intent of the website is to have fun playing around with that math :smile: 

Source code is available [here](https://github.com/zer2/fantasy-basketball-optimizer).

## Category-based fantasy basketball

In fantasy leagues, "managers" draft teams of real players before the season begins. E.g. I take Victor Wembanyama with my first pick, then you take Nikola Jokic, someone else takes Luka Doncic, and so on. At the end of the draft everyone has a 13-player team. 

Throughout the season, teams accumulate statistics based on their real players' performances. E.g. if I have Wembanyama on my team and he gets four blocks in a game, then my team gets four blocks. 

In category-based leagues, which most fantasy basketball leagues are, the important thing is winning individual categories. Having more of a category means you win it, e.g. if my team gets 50 blocks and your team gets 30 blocks, I get a "fantasy point" for winning blocks. All of the tracked categories, usually eight or nine in total, are scored this way. Generally opponents are rotated weekly, and teams try to accumulate as many fantasy points as they can across all the opponents they face throughout a season.

This simple scoring system belies deceptively tricky mathematics. There is no obvious way to compare the values of categories to each other, making it difficult to quantitatively decide which player is best to take. The motivating force behind the papers is untangling this conundrum and applying rigor to the process of evaluating players. 

## The papers  

Broadly, the papers formulate category-based fantasy basketball as a family of math problems, and provide methods for selecting players based on those formulations. The methods take projections of player performance as an input. 

The [first paper](https://arxiv.org/abs/2307.02188) looks at so-called "static" systems, which estimate player value in a vacuum. It provides a mathematical justification for Z-scoring, the traditional metric used by fantasy basketball analysts, and shows a way to improve it. The improved metric, dubbed G-scoring, is used by the website for describing player value statically. See also the [G-score](gscores.md) section of documentation. 

The [second paper](https://arxiv.org/abs/2409.09884) discusses how to evaluate players with full context taken into account. The most difficult aspect of context to incorporate is selections from later draft rounds, because nobody can be exactly sure which players will be taken by which manager in the future. The paper's solution is an algorithm called H-scoring which optimizes a heuristic approach to future draft picks for each candidate player. Players can then be ranked according to how well the team would be expected to perform if they were taken, given the appropriate follow-up in later draft rounds. H-scores are the default way to evaluate players on the website. See also the [H-score](hscores.md) section of documentation. 

The [third paper](https://arxiv.org/abs/2501.00933) approaches the trickiest format, Rotisserie or "Roto". Instead of weekly matchups, Roto has one season-long scoring period in which all managers compete against all other managers. The manager who wins the most fantasy points in that one scoring period wins the league. Since this format is so different, adapting H-scoring to it requires another layer of mathematical scaffolding. See also the [Roto](roto.md) section of documentation. 

## Extensions

The website extends the papers in a few ways. 

One is that it provides methods for [auction drafts](auctions.md) in addition to typical snake drafts. The papers do not discuss auctions directly, but the snake draft methods can be adapted to work for auctions. 

Another is that it [adjusts implicit projections during drafts](projectionadjustment.md). This is a practical consideration which is very useful during real drafts. 
 
It also provides some related analyses for leagues that are already underway via ["Season Mode"](season.md). The analyses evaluate player acquisitions and trades. 

