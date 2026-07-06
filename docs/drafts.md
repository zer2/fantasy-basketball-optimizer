# Draft Mode

H-scoring is primarily designed for drafting. For the most part, the website implements the algorithm exactly as it is presented in the papers.

## Speedup behavior

The website does not implement the algorithm exactly because it uses a few 'tricks' to speed up the process. These tricks increase speed significantly, without hurting the robustness of the algorithm in any significant way. 

??? note "What tricks does the website use to speed up the algorithm during drafting?

    The website uses two tricks to speed up the algorithm for drafting. Both are made possible beause for drafting, the top choices are the ones that matter. The difference between the 100th best choice and the 101st is not a big deal. 
    
    The two tricks are: 

    - Running candidates through in batches. Typically, the candidates that end up being relevant as choices for a particular drafting situation also score highly by generic H-score (pre-calculated with no drafting context). To get initial results quickly, the top candidates by initial H-score are fed into the algorithm and returned first. Each batch has 100 candidates, ordered by default H-score. When a new batch comes back with data, it is merged into the existing table, pushing new candidates above candidates already-processed when their H-scores are higher. This behavior will almost never be perceptible to the user, since players with low generic H-score ranks are unlikely to jump to the top. It would be difficult to scroll down fast enough to notice this happening. 
    - The position optimization procedure is not run on every iteration for every player. The ideal arrangement of existing players to positions is unlikely to change quickly while small differences are being made to category weights, so optimizing positions every iteration for every player is unnecessary. The top 30 candidates are checked on every iteration, because they are the ones likely to be picked and it costs little to be precise on such a small group. Everyone else is checked only once every ten iterations. Which players count as the top 30 is itself recomputed every ten iterations from the latest H-scores, so the exactly-optimized group follows the players that matter for the current draft context rather than a fixed pre-draft list.

## Manual entry 

With manual entry, draft picks are entered through the website. 

![Manual draft entry](img/mdraft.png)

The 'Lock in selection' button puts the player shown in the drop-down into the next draft slot. Picks go in a snake order (except when there is a third round reversal) and cannot be skipped. 

The default order in which players are listed in the drop-down is by base H-score. The top player on the list is the default selection. So if 'Lock in selection' is pressed multiple times in succession, available players are taken in base H-score order. 

The table below the player selection drop-down shows which players have been taken by which drafters. 

### Autodrafting

Next to each team name is a button with the letter 'A' in it, for 'Autodrafter'. Click the button to highlight it and make that team an autodrafter. Instead of waiting for a manual input, autodrafters automatically take a player based on H-scoring. 

![Autodraft picture](img/autodraft.png)
/// caption
Team 2 and Team 3 toggled to autodrafting mode. when Team 1 selects a player, Team 2 and Team 3 will automatically make their picks after
/// 


## Live connection

With a live connection, draft selections are provided by the platform. The entire screen becomes a view for candidate evaluation. 

![Live draft candidate view](img/livedraft.png)

The 'Refresh Analysis' button fetches new information on draft picks from the platform and re-runs H-scoring. 