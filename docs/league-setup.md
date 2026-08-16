# League Setup

The H-scoring algorithm factors in which players are selected by other teams. Users can either manually input that information, or integrate with a fantasy provider to load it automatically. 

## Fantasy provider connections

Connecting a provider is the one part of the website that requires signing in with Google. The credentials for a platform are stored against that account, so there has to be an account to store them against. Everything else — entering your own data, uploading projections, drafts, auctions and season tools — works without signing in.

The three platforms currently supported are: 

**Yahoo**: support exists both for pulling existing teams during the season, and for integrating with drafts. This includes mock drafts. To integrate, one must authenticate with Yahoo by following the link on the pop-up generated when Yahoo is selected.

![Yahoo authentication pop-up](img/yahoopop.png)

Once the connection is established, relevant leagues will show up, if any. Mock drafts can also be connected to via manually copy-pasting the code for the mock draft.

FYI there is a bug in the wrapper used for connecting to the Yahoo API, which crashes the app when a drafter has a name that is a pure number. 

![Yahoo league settings](img/yahoosettings.png)

**Fantrax**: support exists both for pulling existing teams during the season, and for integrating with drafts. However this only works with public drafts. 

![Fantrax league settings](img/fantraxsettings.png)

**ESPN**: support only exists for pulling existing teams during the season. Unfortunately, ESPN has no API for draft access. To authenticate to pull a team, a web plug-in is needed. The instructions are on the pop-up generated when ESPN is selected. 

![ESPN authentication pop-up](img/espnpop.png)

## Manual entry

![Manual entry inputs](img/moreinfo.png)

If draft picks are being input manually, a number of additional inputs are required. They are 

- The number of drafters
- The number of picks per drafter
- For snake drafting, a third round reversal toggle. Third round reversal is a common draft setting, wherein the draft order stays the same between the second and third round, instead of snaking. This is designed to limit the advantage of early picks. 
