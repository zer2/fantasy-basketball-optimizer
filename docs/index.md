This is documentation for a [website](https://fantasy-basketball-optimizer-281850565831.us-east1.run.app/) which applies an algorithm called [H-scoring](hscores.md) to category-based fantasy basketball. The algorithm provides recommendations for which players to choose based on their statistical profiles and how they synergize with existing teams. 

H-scoring takes two essential inputs. One is [league context](league-setup.md)- which players are getting selected by which teams. The website allows you to either integrate with a real [fantasy league](league-setup.md#fantasy-provider-connections) or [update picks manually](league-setup.md#manual-entry). The other essential input is [player stats](projections.md). When you connect to a platform, those come from external projections; in manual mode you can use external projections too, or data from past NBA seasons. 

The algorithm is primarily designed for [drafting](drafts.md), but can be applied in the context of [auctions](auctions.md) and [ongoing seasons](season.md) as well. 

![The website in draft mode](img/main.png)
/// caption
The website in drafting mode, with manual player entry
///

Please note that the algorithm is based on a simplified model of fantasy basketball, ignoring many practical considerations, and there is no guarantee that using it will lead to success. Don't expect to automatically win your league with it or even to have a better shot than anyone else.

## Technical reference 

Source code is available [here](https://github.com/zer2/fantasy-basketball-optimizer). Relevant math is described in these papers: 

- [Improving algorithms for fantasy basketball](https://arxiv.org/abs/2307.02188)
- [Dynamic algorithms for fantasy basketball](https://arxiv.org/abs/2409.09884)
- [Optimizing for Rotisserie fantasy basketball](https://arxiv.org/abs/2501.00933)


## Settings glossary

Settings, available in the left sidebar, control the context in which the algorithm operates.  

| Setting | Options | Explained in |
|---|---|---|
| Mode | toggle between Draft/Auction/Season modes | [Draft](drafts.md), [Auction](auctions.md), [Season](season.md) |
| Data source | connect to a platform (Yahoo, Fantrax, ESPN) or enter picks manually | [League Setup](league-setup.md) |
| Drafters and picks | number of drafters and picks per drafter (manual entry) | [League Setup → Manual entry](league-setup.md#manual-entry) |
| Third-round reversal | snake-draft order toggle (manual entry) | [League Setup → Manual entry](league-setup.md#manual-entry) |
| Player stats | configure forward-looking projections, or data from past NBA seasons | [Player Stats](projections.md) |
| Format | toggle between H2H Each Category, H2H Most Categories, or Rotisserie | [H-scoring → Formats & categories](hscores.md#formats-categories) |
| Categories | select statistical categories for scoring | [H-scoring → Formats & categories](hscores.md#formats-categories) |

## Parameter glossary

The website's calculations take a number of user-configurable parameters, available through the left sidebar and explained in relevant documentation sections.

| Parameter | Controls | Explained in |
|---|---|---|
| ω, γ | how aggressively H-scoring punts categories | [H-scoring → Parameters](hscores.md#parameters) |
| Number of iterations | how long the H-scoring algorithm runs | [H-scoring → Parameters](hscores.md#parameters) |
| Position requirements | the roster/position structure a team must satisfy | [H-scoring → Detailed drop-down](hscores.md#detailed-drop-down) |
| υ, ψ | how projections account for injuries and replacement players | [Player Stats → Injury handling](projections.md#injury-handling) |
| ℶ | how strongly a team's projected strength is regressed toward average | [Player Stats → Bayesian strength adjustment](projections.md#bayesian-strength-adjustment) |
| χ, ℵ | Rotisserie projection uncertainty and cross-category correlation | [Player Stats → Rotisserie uncertainty](projections.md#rotisserie-uncertainty) |
| $S_\sigma$ | the spread of auction dollar values across a season (SAVOR) | [Auction Mode → The SAVOR adjustment](auctions.md#the-savor-adjustment) |
| Trade thresholds | which candidate trades are considered and shown | [Season Mode → Trade suggestions](season.md#trade-suggestions) |
