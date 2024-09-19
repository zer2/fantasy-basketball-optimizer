# Available data

## Historical season data

Data from old seasons is mostly for fun. See what the algorithms would have recommended for those seasons, given perfect information about player stats! 

The box score data was pulled from the NBA API. 

## Projection sources 

Projections are great for drafting, since they are meant to apply to the upcoming season.

One important thing to note about projections is that they generally do not account for the possibility of long-term injury. Some players are known to be more injury prone, and therefore have significant downside risk that is not accounted for by the projections. App users should keep this in mind when directed to draft potentially injury-prone players.

### Hashtag Basketball 

[Hashtag Basketball](https://hashtagbasketball.com/) provides free projections for the 2024-25 season. 

At the moment, these projections are loaded from Hashtag Basketball manually. So I will not necessarily update them as frequently as the projections are updated on the website itself.

### RotoWire

[Rotowire](https://www.rotowire.com/) is another source of projections. Unlike Hashtag Basketball, their projections are not free, so I have not included them within the app by default. To use them you must buy the projections and download them, so you can upload them to the data tab 

### Basketball Monster

The third projection source is [Basketball Monster](https://basketballmonster.com/). Basketball Monster is also a paid service, and is not included in the app by default. When I downloaded BBM projections, I needed to download them as excel and then save them as CSV to get all of the required information

## Current season data (currently unavailable)

Current season data is pulled on a daily basis from the NBA's API. Using actual season data is a standard and simple method for game-planning for the remainder of the season. 

This data set excludes games when players are injured and provides no forecast for likelihood of future injury. You can write in injury risk in the "No Play %" column if you would like 

## DARKO (currently unavailable)

[DARKO](https://apanalytics.shinyapps.io/DARKO/) is a "machine learning-driven basketball player box-score score projection system." Evidence suggests that DARKO projections are the most 
accurate short-term forecasts available. 

### DARKO-S

DARKO-S is my name for DARKO's 'Daily Player Per-Game Projections'. These forecasts are meant to predict statistics for the next game a player plays

DARKO-S sometimes diverges significantly from long-term expectations. For example, if a player is coming back from injury, DARKO will understandably reduce the players' minutes in its 
forecast, thus driving the players' numbers down across the board. This is useful for day-to-day strategy but not always appropriate for long-term drop decisions. It would be an overreaction
to drop a star player because their numbers are momentarily low after returning from an injury

### DARKO-L

DARKO-L is an adjusted version of Darko's forecast. It uses 
- The average number of minutes each player has played per game so far this season, excluding games where the player was injured
- DARKO's projections of player pace (possessions per 48 minutes), from the per-game projections
- DARKO's estimates of player skill, from 'Current Player Skill Projections'

The idea of DARKO-L is to remove the short-term bias from DARKO-S by anchoring on average minutes played so far this season, rather than a short-term minutes estimate. It is potentially
more useful for making  long-term add/drop decisions than DARKO-S, because it is not affected by recent circumstances such as injuries. However, discounting injuries can also be dangerous. 
If a player is injured to the extent that they will miss much of the remaining fantasy season, then that injury is relevant even in the long-term. For this reason, when using DARKO-S, 
it is wise to temper expectations with your own understanding of player injuriy status
