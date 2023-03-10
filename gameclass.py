
import requests
import json
from mlclass import Game

class PlayerStats:

    def __init__(self):
        pass 

    def query(self, playerId, season, gameNumber): #* return player stat for game of season
        pass 
        
class LeagueStats:
    def __init__(self):
        pass

class NHLTeam: # Class to obtain team schedule

# ?startDate=2018-01-09 Start date for the search
# ?endDate=2018-01-12 End date for the search
# ?season=20172018 Returns all games from specified season
# ?gameType=R Restricts results to only regular season games. Can be set to any value from Game Types endpoint
# GET https://statsapi.web.nhl.com/api/v1/schedule?teamId=30 Returns Minnesota Wild games for the current day
    def __init__(self, id):
        self.id = id

        self.team_list = []

    def obtain_game(self, year:int , game_number:int): # Call NHL api and return game information for this year
        pass 

    def format_game(self, gameData):
        pass 

    def calculate_season(self, season:int):
        
        #https://statsapi.web.nhl.com/api/v1/schedule?teamId=30

        for gameNumber in range(82):
            #* Game ingestion pipeline
            game_data = self.obtain_game(season, gameNumber)
            formated_data = self.format_game(game_data)
            game = Game(formated_data)

            pass 
class NHLTeams: # Used to collect information on NHL teams

    
    def __init__(self):
        self.team_ids = [1,5, 6, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
        pass 

    def generate_NHL_teams(self):
        return self.team_ids

    # 
        # nhl_teams = []
        # for team in self.team_ids:
        #     nhl_teams.append(NHLTeam(team))

        # return nhl_teams

#* http://statsapi.web.nhl.com/api/v1/game/2019020001/boxscore
#* To access Databaser
class FileReaderAndWriter:
    def __init__(self):
        pass 


    def get_game(season, game, team_id): #* Try to reuse?
        pass 

    def read_game(self):
        pass 

    def write_game(self):
        pass 

    def read_ml_game(self):
        pass 
    def write_ml_game(self):
        pass 

class FileReaderAndWriter():

    def read(self, filename="default.json"):
        f = open(filename)
        data = json.load(f)
        f.close()

        return data 
    def write(self, data, filename="default.json"):
        # Serializing json
        json_object = json.dumps(data, indent=2)
        
        # Writing to sample.json
        with open(filename, "w") as outfile:
            outfile.write(json_object)


class MLGuesser:
    
    def __init__(self):
        pass 

    def get_current_game_data(year: int, game_number: int):
        pass 
    def calculate(year: int, game_number: int):

        pass 


#* To create database
class MLDataCollector:
    #? http://statsapi.web.nhl.com/api/v1/game/2019020001/boxscore
    # Seasons
        # Teams
            # Game number

    game_url = 'http://statsapi.web.nhl.com/api/v1/game/'
    years = [2019, 2020, 2021, 2022]

    def __init__(self):
        self.data = {} # Empty dictionary


    def read(self, filename="default.json"):
        f = FileReaderAndWriter()
        data = f.read(filename)

        return data 
        
    def write(self, data, filename = "default.json",):
        f = FileReaderAndWriter()
        data = f.write(data, filename)

    def readFile(self):
        return self.read('games.json')
    
    def readYears(self):
        data = {}
        for year in self.years:
            data[year] = json.loads(self.read(f'games_{year}.json')) #? Why do I need this, I should already be loading it as a json but whatever
        return data 
    
    def writeYears(self):
        years = self.years

        full_data = self.readFile()

        for year in years:
            year_arr = full_data[str(year)]
            self.write(json.dumps(year_arr), f'games_{year}.json')

        
    #* Collects user data
    def collect_data(self):
        #* Collects data
        seasons = [2019, 2020, 2021] # Seasons to collect data

        nhlteamsclass = NHLTeams() 
        self.nhl_teams = nhlteamsclass.generate_NHL_teams()

        for season in seasons:

            for team in self.nhl_teams:
                team.calculate_season(season)

    #* MAKES GAMES REQUEST
    def bulk_api_call(self):
        seasons = self.years # Seasons to collect data
        for season in seasons:
            year_data = self.api_call_year(season)
            self.data[season] = year_data 

        self.write(self.data, "games.json")

    def api_call_year(self, year):
        year_data = []

        for game_number in range(0, int(82*32/2)):  
            game_data, failure = self.call_api(year, game_number+1)
            if(game_number % 25 == 1): # Debugging purposes
                print(f'{year} and {game_number}')

            if failure: # If game not found then break
                break 

            year_data.append(game_data)

        return year_data

    def call_api(self, year, game_number): # data, failure
        resp = requests.get(f'http://statsapi.web.nhl.com/api/v1/game/{year}02{str(game_number).zfill(4)}/boxscore')
        return json.loads(resp.content), resp.status_code == 404


#* To create database
class SeasonDataCollector:
    #? http://statsapi.web.nhl.com/api/v1/game/2019020001/boxscore
    # Seasons
        # Teams
            # Game number

    game_url = 'http://statsapi.web.nhl.com/api/v1/game/'
    years = [2019, 2020, 2021, 2022]

    def __init__(self):
        self.data = {} # Empty dictionary


    def read(self, filename="default.json"):
        f = FileReaderAndWriter()
        data = f.read(filename)

        return data 
        
    def write(self, data, filename = "default.json",):
        f = FileReaderAndWriter()
        f.write(data, filename)

    def readFile(self):
        return self.read('games.json')
    
    def readYears(self):
        data = {}
        for year in self.years:
            data[year] = json.loads(self.read(f'season_{year}.json')) #? Why do I need this, I should already be loading it as a json but whatever, probably cause array now that i think of it
        return data 
    
    def writeYears(self):
        years = self.years

        full_data = self.readFile()

        for year in years:
            year_arr = full_data[str(year)]
            self.write(json.dumps(year_arr), f'season_{year}.json')


    #* MAKES GAMES REQUEST
    def bulk_api_call(self):
        seasons = self.years # Seasons to collect data

        for season in seasons:
            year_data = self.api_call_year(season)
            self.data[season] = year_data 

        self.writeYears();


    def api_call_year(self, year):
        year_data = {}

        nhlteamsclass = NHLTeams() 
        nhl_teams = nhlteamsclass.generate_NHL_teams()

        for team in nhl_teams:  
            team_data, failure = self.call_api(year, team)

            print(f'{team}+{year}')

            if failure: # If game not found then break
                break 

            year_data[team] = team_data

        return year_data

    def call_api(self, year, team): # data, failure
        resp = requests.get(f'https://statsapi.web.nhl.com/api/v1/schedule?expand=schedule.linescore&teamId={team}&season={year}{year+1}&gameType=R')
        return json.loads(resp.content), resp.status_code == 404
#* Main

def main():
    # datacollector =  MLDataCollector()
    # data = datacollector.readYears()

    season_collector = SeasonDataCollector()
    season_collector.bulk_api_call()

    game = Game(data, 2019, 1)


if __name__ == "__main__":
    main()




# #* TESTING
# params = {
#         "teamId":30, 
#         "startDate":"2018-01-09",
#         "endDate": "2020-01-09",   
#     }
# blah = requests.get("https://statsapi.web.nhl.com/api/v1/schedule", params)
# ?startDate=2018-01-09 Start date for the search
# ?endDate=2018-01-12 End date for the search
# ?season=20172018 Returns all games from specified season
# ?gameType=R Restricts results to only regular season games. Can be set to any value from Game Types endpoint
# GET https://statsapi.web.nhl.com/api/v1/schedule?teamId=30