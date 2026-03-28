This is documentation for a [website](https://fantasy-basketball-optimizer-281850565831.us-east1.run.app/) which applies an algorithm called [H-scoring](hscores.md) to category-based fantasy basketball. The algorithm provides recommendations for which players to choose based on their statistical profiles and how they synergize with existing teams. 

![alt text](img/main.png)
/// caption
The website in drafting mode, with manual player entry
///

Source code is available [here](https://github.com/zer2/fantasy-basketball-optimizer). Relevant math is described in these papers: 

- [Improving algorithms for fantasy basketball](https://arxiv.org/abs/2307.02188)
- [Dynamic algorithms for fantasy basketball](https://arxiv.org/abs/2409.09884)
- [Optimizing for Rotisserie fantasy basketball](https://arxiv.org/abs/2501.00933)

Please note that the algorithm is based on a simplified model of fantasy basketball, ignoring many practical considerations, and there is no guarantee that using it will lead to success. Don't expect to automatically win your league with it or even to have a better shot than anyone else. The intent of the papers is just to start exploring the math underlying fantasy basketball, and the intent of the website is to have fun playing around with that math :smile:.


## A gentle introduction to category-based fantasy basketball

In fantasy leagues, "managers" draft teams of real players before the season begins. E.g. I take Victor Wembanyama with my first pick, then you take Nikola Jokic, someone else takes Luka Doncic, and so on. At the end of the draft everyone has a 13-player team. 

Throughout the season, teams accumulate statistics based on their real players' performances. E.g. if I have Wembanyama on my team and he gets four blocks in a game, then my team gets four blocks. 

In category-based leagues, which most fantasy basketball leagues are, the important thing is winning individual categories. There are usually eight or nine categories. Most categories are won by the team that gets more of them, e.g. if my team gets 50 blocks and your team gets 30 blocks, I get a "fantasy point" for winning blocks. 

This simple scoring system belies deceptively tricky mathematics. There is no obvious way to compare the values of categories to each other, making it difficult to quantitatively decide which player is best to take, even if player performances can be forecasted reliably. The motivating force behind the papers is untangling this conundrum and applying rigor to the process of evaluating players. 

## The papers  

Broadly, the papers formulate category-based fantasy basketball as a family of math problems and analyze how to approach them. The first paper establishes some simple building blocks for fantasy basketball, while the second and third flesh out the H-scoring algorithm. All of the methods take projections of player performance as an input. 

The [first paper](https://arxiv.org/abs/2307.02188) looks at so-called "static" systems, which estimate player value in a vacuum. It provides a mathematical justification for Z-scoring, the traditional metric used by fantasy basketball analysts, and shows a way to improve it. The improved metric, dubbed G-scoring, is used by the website to statically quantify player and team strength on the category level. See also the [G-score](gscores.md) section of documentation. 

The [second paper](https://arxiv.org/abs/2409.09884) introduces H-scoring as a framework for player selection based on context. The key concept is the idea of optimizing a heuristic approach to future draft picks for each candidate player. Players can then be ranked according to how well the team would be expected to perform if they were taken, given the appropriate follow-up in later draft rounds. The [H-scoring](hscores.md) section of documentation has more detail. 

The [third paper](https://arxiv.org/abs/2501.00933) approaches the Rotisserie or "Roto" format. Since this format is so different from the others, adapting H-scoring to it requires another layer of mathematical scaffolding. See also the [Roto](roto.md) section of documentation. 

## Extensions

The website extends the papers in a few ways. 

One is that it provides methods for [auction drafts](auctions.md) in addition to typical snake drafts. The papers do not discuss auctions directly, but the snake draft methods can be adapted to work for auctions. 

Another is that it [adjusts implicit projections during drafts](projectionadjustment.md). This is a practical consideration which is very useful during real drafts. 
 
It also provides some related analyses for leagues that are already underway via ["Season Mode"](season.md). The analyses evaluate player acquisitions and trades. 

