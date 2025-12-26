# Settings

Settings are available for live connections to fantasy providers, customizable projection uploads, and adapting to different kinds of scoring. They are accessible via the left sidebar.

## League settings

![](img/lsettings.png)

### Fantasy sport

For now, this will just be NBA. MLB or WNBA may be added in the future

### Data integration 

The default option for getting data on drafting situation/teams is entering it manually. 

There is also an option for integrating with a fantasy provider, which allows the website to be used with real fantasy occuring on those platforms. The website will show analysis based on the integrated league. It will never make a pick or take any action itself. 

The three platforms currently supported are: 

**Yahoo**: support exists both for pulling existing teams during the season, and for integrating with drafts. This includes mock drafts. To integrate, one must authenticate with Yahoo by following the link on the pop-up generated when Yahoo is selected.

![](img/yahoopop.png)

Once the connection is established, relevant leagues will show up, if any. Mock drafts can also be connected to via manually copy-pasting the code for the mock draft.

FYI there is a bug in the wrapper used for connecting to the Yahoo API, which crashes the app when a drafter has a name that is a pure number. 

![](img/yahoosettings.png)

**Fantrax**: support exists both for pulling existing teams during the season, and for integrating with drafts. However this only works with public drafts. 

![](img/fantraxsettings.png)

**ESPN**: support only exists for pulling existing teams during the season. Unfortunately, ESPN has no API for draft access. To authenticate to pull a team, a web plug-in is needed. The instructions are on the pop-up generated when ESPN is selected. 

![](img/espnpop.png)

### Mode 

![alt text](img/mode.png)

There are three available modes. 

**Draft mode**: for standard drafts. When drafted players are input manually, only snake drafts are naturally supported. If the drafted players are being pulled via an integration, the website will continually evaluate the best prospects regardless of draft position, so any drafting order will work fine. 

**Auction mode**: for auction drafts. Users must specify that they are doing an auction for either manual entry or platform integration.

**Season mode**: for leagues where teams have already been built. Season mode allows for the inspection of potential free agents, and also includes a trading module. 

### Other info

If draft data is being input manually, the number of drafters, the number of picks per drafter, and the team names of the drafters are also inputs. 

## Player statistics 

Player statistics are an input to the algorithms implemented by the website. They can be sourced either from forward-looking projections or previous NBA seasons.

### Projection

The default for player statistics is to use forward-looking projections. 

![](img/projections.png)

The default projection source is a 50/50 split between ESPN's free forecasts and a modified version of DARKO. The website's version of DARKO projections takes games played and total minutes from the ESPN forecasts, and combines those with DARKO pace and per-possession projections to get per-game projections. This is necessary because DARKO does not forecast games played, and its minute forecasts are designed for the next game only, which is not ideal for fantasy. 

**Note as of December 2025: the ESPN forecasting page currently has bugs, and for that reason the ESPN projections have not been updated since October.**

Two additional kinds of forecasts are also available: those created by Hashtag Basketball, and Basketball Monster (BBM). Both of these projections are paid products, so they cannot be provided through the website. Instead, they must be purchased and uploaded. For HTB, there is no native download option for projections. The projections must be copy pasted into an Excel file and saved as a CSV. For BBM, there is a download option, but only the XLSX download option works. Download it, copy out the projections into Excel, and save them as a CSV UTF-8. Either kind of projection can be manually edited to change projections if desired. 

Also: be careful to download projections for all players instead of just the top players. During a draft, another drafter may take a player outside of the limited projection list, and the website will only have projections for them if they have been provided. 

Projections are combined between different sources by taking weighted means according to the provided weights. If the assigned weights add up to more or less than 1, they will be scaled to add to 1. 

### Historical 

Historical data is available for manual entry drafts. 

![alt text](img/historical.png)

Historical data is available going all the way back to the 1984-85 season, though for any season before 2000-01 player positions will not be available. 

![alt text](img/1984-85.png)
/// caption
H-scores for the 1984-85 season, Each Category. NP means no position
///

Historical data cannot be used when integrated with a fantasy platform, because platforms do not run leagues based on past seasons.  

## Formats & categories

![alt text](img/formats.png)

The website supports the three common category formats for fantasy basketball: H2H Each Category, H2H Most Categories, and Rotisserie. It does not have native support any additional variants, like using a specific category as a tiebreaker. 

It also supports any combination of categories, across the default nine categories and several alternative options.

![alt text](img/categories.png)

For the alternative categories, when using projections, make sure to include them when sourcing the projections. ESPN and DARKO do not forecast them so all of the weight will be from Hashtag or BBM projections. 