# FSO website documentation

[The FSO Website](https://fantasy-basketball-optimizer-y9jt7t3ypmiejsyjkeayx6.streamlit.app/) implements the methods described in these papers: 

- [Improving algorithms for fantasy basketball](https://arxiv.org/abs/2307.02188)
- [Dynamic algorithms for fantasy basketball](https://arxiv.org/abs/2409.09884)
- [Optimizing for Rotisserie fantasy basketball](https://arxiv.org/abs/2501.00933)

The methods are designed for category leagues (not points leagues). 

## Introduction to category-based fantasy basketball

In fantasy leagues, "managers" draft teams of real players before the season begins. E.g. I might take Victor Wembanyama with my first pick, then you take Nikola Jokic, someone else takes Luka Doncic, and so on. At the end of the draft everyone has a 13-player team. 

The most common format of fantasy basketball pits teams against each other on a weekly basis during the season. Matchups are scored based on statistical categories such as blocks and rebounds. For each category, the team whose players accumulated more of that category during the week earns a point. 

This differs from point-based formats, which are more common for other fantasy sports like fantasy football. Points-based leagues assign point values directly to scoring categories, e.g. two points for a block, one point for a rebound, et cetera. 

Point-based leagues have an element of simplicity, because there is a clear translation between a player's production and their value. The same cannot be said for category leagues. The simple mechanic of scoring based on categories introduces a wealth of complicated strategy and math. 

## The papers  

Broadly, the papers formulate category-based fantasy basketball as a math problem, and provide methods for selecting players based on that formulation. The methods take projections of player performance as an input. They also invoke many simplifications, which are handy both because real fantasy basketball is extremely complex, and because perfect mathematical solutions are not always easy. 

Because of the reliance on existing projections and the simplifications invoked, these methods should in no way be considered the end-all-be-all or truly "optimal" in any sense. They represent one potential starting point. 

## The website 

The central purpose of the website is to apply the methods of the papers to drafts and auctions. It makes a few practical adjustments from the methods described in the papers, particularly [adjusting implicit projections during drafts](projectionadjustment.md). It also provides some related analyses for leagues that are already underway via 'Season Mode'.

Source code is available [here](https://github.com/zer2/fantasy-basketball-optimizer).
