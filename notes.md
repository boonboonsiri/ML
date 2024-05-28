# Idea

Train NHL API using last 7 games as NHL teams are very streaky

Use

- Shot differential (maybe normalize) Shots for / Total Shots for and against
- Best 5 players point totals from the previous 7 games, maybe normalize out of like 50 or something
- Current place in season
- Save percentage
- Home/Away (I think)
- Last 7 win percentage
- Game in season percentage

- Win percentage season culmative
- Current league wide standing


## Useful APIs
https://github.com/Zmalski/NHL-API-Reference?tab=readme-ov-file#get-standings-by-date
- Has a ton of useful data such as l10

Actual game information https://github.com/Zmalski/NHL-API-Reference?tab=readme-ov-file#get-boxscore



Useful later:
curl -L -X GET "https://api-web.nhle.com/v1/partner-game/US/now"
