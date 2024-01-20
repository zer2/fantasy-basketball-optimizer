## Available data

This website pulls data from three sources, all of which have slightly different purposes

### Historical season data

Data from old seasons is mostly for fun. See what the algorithms would have recommended for those seasons, given perfect information about player stats! 

This data was pulled from the NBA API and stored in a one-time load

### Current season data

Current season data is pulled on a daily basis from the NBA's API. Using actual season data is a standard and simple method for game-planning for the remainder of the season. 

This data set excludes games when players are injured and provides no forecast for likelihood of future injury. You can write in injury risk in the "No Play %" column if you would like 

### DARKO

[DARKO](https://apanalytics.shinyapps.io/DARKO/) is a "machine learning-driven basketball player box-score score projection system." Evidence suggests that DARKO projections are the most 
accurate short-term forecasts available. 

It should be noted that DARKO forecasts are meant to predict the next game a player plays, and sometimes diverge significantly from long-term expectations. For example, if a player is 
coming back from injury, DARKO will understandably reduce the players' minutes in its forecast, thus driving the players' numbers down across the board. This is useful for day-to-day
strategy but not always appropriate for long-term drop decisions. It would be an overreaction to drop a star player because their numbers are momentarily low after returning from an injury

DARKO forecasts also do not account for probability of future injury. 

## Additional data aspirations

### Pre-season projections

Ideally, we could provide pre-season projections from sources such as HashTagBasketball and BasketballMonster. However, this is complicated because their projections are paid products. For 
now, you can paste their projections in manually if you have access to them

### Rest-of-season projections

A DARKO-esque system meant for the rest of the fantasy season would be a boon for managers. The author of DARKO has mentioned season-long projections as a future goal, so this will likely be 
available eventually!
