# Classes for parsing the data
#2624 games in a season
from data import FileManager

from pprint import pprint


def league_percentage_calculator(cur_game, year):
    if year == 2020:
        return cur_game / (56 *32)

    else:
        return cur_game / (82*32)



class Game:
    def __init__(self, game_dict=None):
        # future ideas
        self.distance = None # distance matrix of all teams
        self.backtoback = None
        self.hits = None
        self.penalties = None # giveaways, takeaways

        # house keeping
        self.game_id = None # if all else fails usedful to identify
        self.date = None
        self.home = None
        self.away = None

        self.winner = None # doesn't apply to individual team
        self.game_in_season_percentage = None

        self.home_games_played = None # seasonal data found in standings
        self.home_goals_for = None
        self.home_goals_against = None
        self.home_standing = None
        self.home_season_point_percentage = None
        self.home_season_points = None
        self.home_wins = None
        self.home_ties = None
        self.home_loses = None
        self.home_regulation_wins = None
        self.home_home_wins = None
        self.home_away_wins = None
        self.home_home_games_played = None

        self.home_streak = None
        self.home_streak_code = None
        self.home_goals_for_l10 = None
        self.home_goals_against_l10 = None
        self.home_wins_l10 = None
        self.home_ties_l10 = None
        self.home_loses_l10 = None
        self.home_points_l10 = None # can be accessed in standings
        self.home_save_percentage_l10 = None
        self.home_shots_for_l10 = None
        self.home_shots_against_l10 = None
        self.home_power_play_goals_l10 = None
        self.home_power_plays_l10 = None
        self.home_penalty_kill_goals_against_l10 = None
        self.home_penalty_kills_l10 = None
        self.home_p1 = None
        self.home_p2 = None
        self.home_p3 = None
        self.home_p4 = None
        self.home_p5 = None


        self.away_games_played = None
        self.away_goals_for = None
        self.away_goals_against = None
        self.away_standing = None
        self.away_season_point_percentage = None
        self.away_season_points = None
        self.away_wins = None
        self.away_ties = None
        self.away_loses = None
        self.away_regulation_wins = None
        self.away_home_wins = None
        self.away_away_wins = None
        self.away_home_games_played = None

        self.away_streak = None
        self.away_streak_code = None
        self.away_goals_for_l10 = None
        self.away_goals_against_l10 = None
        self.away_wins_l10 = None
        self.away_ties_l10 = None
        self.away_loses_l10 = None
        self.away_points_l10 = None # can be accessed in standings
        self.away_save_percentage_l10 = None
        self.away_shots_for_l10 = None
        self.away_shots_against_l10 = None
        self.away_power_play_goals_l10 = None
        self.away_power_plays_l10 = None
        self.away_penalty_kill_goals_against_l10 = None
        self.away_penalty_kills_l10 = None
        self.away_p1 = None
        self.away_p2 = None
        self.away_p3 = None
        self.away_p4 = None
        self.away_p5 = None

        if game_dict:
            self.init_dict(game_dict)

    def init_dict(self, game_dict): # please work lol
        for key, value in game_dict.items():
            setattr(self, key, value)

    def to_dict(self) -> dict:
        return {attr: getattr(self, attr) for attr in dir(self) if not attr.startswith("__") and not callable(getattr(self, attr))}


    def __repr__(self) -> str:
        #return str([f'{attr}:{getattr(self, attr)}' for attr in dir(self) if not attr.startswith("__") and not callable(getattr(self, attr))])]
        #return str({attr: getattr(self, attr) for attr in dir(self) if not attr.startswith("__") and not callable(getattr(self, attr))})
        return str(self.to_dict())


    def to_features_v0(self): # return useful features
        home_points = (self.home_p1 + self.home_p2 + self.home_p3 + self.home_p4 + self.home_p5) / (1.5*10*5)
        away_points = (self.away_p1 + self.away_p2 + self.away_p3 + self.away_p4 + self.away_p5) / (1.5*10*5)

        return [
            1-self.home_standing,
            home_points,
            self.home_save_percentage_l10,
            self.home_wins/10,
            self.home_goals_for_l10/(5*10),

            1-self.away_standing,
            away_points,
            self.away_save_percentage_l10,
            self.away_wins/10,
            self.away_goals_for_l10/(5*10),

            self.winner,
        ]


class Parser:
    def __init__(self):
        self.data = None
        self.year = 0
        self.fileManager = FileManager()

        self.last_10 = {} # used to get the last 10 for the season

    def load_data(self, year: int):
        self.data = self.fileManager.read(f"data/api_{year}.json")
        self.year = year

    def load_data_api(self, year, game):
        # load the nessesary data for parsing
        pass

    def parse_game(self, year: int, game_number: int):
        if not self.data or self.year != year:
            self.load_data(year)

        try:
            game_data = self.data['games'][game_number]
        except Exception as e:
            print("MISSING GAME", e, year, game_number)
            return None

        game = Game()

        game.game_id = game_data['id']
        game.home = game_data['homeTeam']['abbrev']
        game.away = game_data['awayTeam']['abbrev']

        game.winner = int(game_data['summary']['linescore']['totals']['home'] >= game_data['summary']['linescore']['totals']['away'])
        game.date = game_data['gameDate']
        game.game_in_season_percentage = league_percentage_calculator(game_number, year)

        try: # Check if valid parsing
            self.parse_standings(game, game_data)
            self.parse_last_10(game, game_data)
        except Exception as e:
            print("Parse Game Exception", e, year, game_number)
            return None

        return game

    def parse_last_10(self, game: Game, game_data: dict):
        home_last_10 = self.get_last_10(game, game.home, game.home_games_played)
        away_last_10 = self.get_last_10(game, game.away, game.away_games_played)
        self.parse_last_10_individually(game, game_data, home_last_10, game.home, away_last_10, game.away)

    def parse_last_10_individually(self, game: Game, game_data: dict, home_last_10: list, home_team: str, away_last_10: list, away_team: str):
        def isHome(g, team): # if the team we're analysing is the home team for that game
            return g['homeTeam']['abbrev'] == team
        # note sog 0, powerPlay 2
        def accumulateValue(g: dict, value: int, team: str, useTeamForStats: bool):
            if useTeamForStats: # ie shots for
                return g['summary']['teamGameStats'][value]['homeValue'] if isHome(g, team) else g['summary']['teamGameStats'][value]['awayValue']
            else: # ie shots against
                return g['summary']['teamGameStats'][value]['awayValue'] if isHome(g, team) else g['summary']['teamGameStats'][value]['homeValue']

        game.home_shots_for_l10 = sum([accumulateValue(g, 0, home_team, True) for g in home_last_10])
        game.home_shots_against_l10 = sum([accumulateValue(g, 0, home_team, False) for g in home_last_10])
        game.home_power_play_goals_l10 = sum([int(accumulateValue(g, 2, home_team, True).split('/')[0]) for g in home_last_10])
        game.home_power_plays_l10 = sum([int(accumulateValue(g, 2, home_team, True).split('/')[1]) for g in home_last_10])
        game.home_penalty_kill_goals_against_l10 = sum([int(accumulateValue(g, 2, home_team, False).split('/')[0]) for g in home_last_10])
        game.home_penalty_kills_l10 = sum([int(accumulateValue(g, 2, home_team, False).split('/')[1]) for g in home_last_10])

        game.away_shots_for_l10 = sum([accumulateValue(g, 0, away_team, True) for g in away_last_10])
        game.away_shots_against_l10 = sum([accumulateValue(g, 0, away_team, False) for g in away_last_10])
        game.away_power_play_goals_l10 = sum([int(accumulateValue(g, 2, away_team, True).split('/')[0]) for g in away_last_10])
        game.away_power_plays_l10 = sum([int(accumulateValue(g, 2, away_team, True).split('/')[1]) for g in away_last_10])
        game.away_penalty_kill_goals_against_l10 = sum([int(accumulateValue(g, 2, away_team, False).split('/')[0]) for g in away_last_10])
        game.away_penalty_kills_l10 = sum([int(accumulateValue(g, 2, away_team, False).split('/')[1]) for g in away_last_10])

        def parse_players( team: str, teams_last_10):
            players_stats = {}
            shots_against = 0
            goalies_goals_against = 0
            for g in teams_last_10:
                players = g['playerByGameStats']['homeTeam'] if isHome(g, team) else g['playerByGameStats']['awayTeam']
                goalies = players['goalies']
                players = players['forwards'] + players['defense']

                for player in players:
                    player_id = player['playerId']
                    points = player['points']

                    if player_id in players_stats:
                        players_stats[player_id] += points
                    else:
                        players_stats[player_id] = points

                for goalie in goalies:
                    shots_against += int(goalie['saveShotsAgainst'].split('/')[1])
                    goalies_goals_against += int(goalie['saveShotsAgainst'].split('/')[0])

            largest_items = sorted(players_stats.values(), reverse=True)[0:5]
            largest_items.append(goalies_goals_against/shots_against)
            return largest_items

        game.home_p1, game.home_p2, game.home_p3, game.home_p4, game.home_p5, game.home_save_percentage_l10 = parse_players(home_team, home_last_10)

        game.away_p1, game.away_p2, game.away_p3, game.away_p4, game.away_p5, game.away_save_percentage_l10 = parse_players(away_team, away_last_10)


    def get_last_10(self, game: Game, team: str, gamesPlayed: int):
        last_10 = []
        if team in self.last_10: # filter the teams games only once
            last_10 = self.last_10[team]
        else:
            last_10 = [game for game in self.data['games'] if ('homeTeam' in game and 'abbrev' in game['homeTeam'] and game['homeTeam']['abbrev'] == team) or ('awayTeam' in game and 'abbrev' in game['awayTeam'] and game['awayTeam']['abbrev'] == team)]
            if len(last_10) == 56: # 2020 season, games out of order because rescheduling covid games
                last_10.sort(key= lambda x: x['gameDate'])
            self.last_10[team] = last_10

        return last_10[gamesPlayed-10:gamesPlayed]


    # WILL BREAK IF TEAM HAS YET TO PLAY A GAME
    def parse_standings(self, game: Game, game_data: dict):
        standings = game_data['standings']['standings']
        home = game.home
        away = game.away

        standings = sorted(standings, key=lambda x: int(x['points']), reverse=True)

        homeIndex = next((i for i, entry in enumerate(standings) if entry.get('teamAbbrev', {}).get('default') == home), None)
        home_standings = standings[homeIndex]

        awayIndex = next((i for i, entry in enumerate(standings) if entry.get('teamAbbrev', {}).get('default') == away), None)
        away_standings = standings[awayIndex]

        game.home_standing = homeIndex/32

        game.home_games_played = home_standings['gamesPlayed']
        game.home_goals_for = home_standings['goalFor']
        game.home_goals_against = home_standings['goalAgainst']
        game.home_season_point_percentage = home_standings['pointPctg']
        game.home_season_points = home_standings['points']
        game.home_wins = home_standings['wins']
        game.home_ties = home_standings['otLosses']
        game.home_loses = home_standings['losses']
        game.home_regulation_wins = home_standings['regulationWins']
        game.home_home_wins = home_standings['homeWins']
        game.home_away_wins = home_standings['roadWins']
        game.home_home_games_played = home_standings['homeGamesPlayed']

        game.home_streak = home_standings['streakCount'] # can be accessed in standings
        game.home_streak_code = home_standings['streakCode']
        game.home_goals_for_l10 = home_standings['l10GoalsFor']
        game.home_goals_against_l10 = home_standings['l10GoalsAgainst']
        game.home_wins_l10 = home_standings['l10Wins']
        game.home_ties_l10 = home_standings['l10OtLosses']
        game.home_loses_l10 = home_standings['l10Losses']
        game.home_points_l10 = home_standings['l10Points']

        game.away_standing = awayIndex/32

        game.away_games_played = away_standings['gamesPlayed']
        game.away_goals_for = away_standings['goalFor']
        game.away_goals_against = away_standings['goalAgainst']
        game.away_season_point_percentage = away_standings['pointPctg']
        game.away_season_points = away_standings['points']
        game.away_wins = away_standings['wins']
        game.away_ties = away_standings['otLosses']
        game.away_loses = away_standings['losses']
        game.away_regulation_wins = away_standings['regulationWins']
        game.away_home_wins = away_standings['homeWins']
        game.away_away_wins = away_standings['roadWins']
        game.away_home_games_played = away_standings['homeGamesPlayed']

        game.away_streak = away_standings['streakCount'] # can be accessed in standings
        game.away_streak_code = away_standings['streakCode']
        game.away_goals_for_l10 = away_standings['l10GoalsFor']
        game.away_goals_against_l10 = away_standings['l10GoalsAgainst']
        game.away_wins_l10 = away_standings['l10Wins']
        game.away_ties_l10 = away_standings['l10OtLosses']
        game.away_loses_l10 = away_standings['l10Losses']
        game.away_points_l10 = away_standings['l10Points']


    def verify_load_data(self):
        self.load_data(2020)
        pprint(self.data['games'][0:10])

    def verify_parse_game(self, year=2020, game_number=400):
        game = self.parse_game(year, game_number)
        pprint(game)

def tester():
    # p = Parser(); p.verify_load_data();
    p = Parser(); p.verify_parse_game()



def main():
    tester()
    # testing()


if __name__ == "__main__":
    main()
