
import requests


class Game:

    #* Top 3 scorers in points last 10. Method one, just find last 10 games, method 2 keep a point tracker class for every players last 10 games
    #* Save percentage
    #* Shots for
    #* Shots against
    #* Home/Away (1,0)
    #* Goals for
    #* Goals against
    #* Win percentage (Season culmative)
    #* last 10 win count

    def __init__(self, formated_game):
        pass 


    def to_arr(self):
        return []
    def calculate(self):
        pass
    def to_str(self):

        pass


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

    def generate_NHL_teams(self) -> list:
        nhl_teams = []
        for team in self.team_ids:
            nhl_teams.append(NHLTeam(team))

        return nhl_teams


# http://statsapi.web.nhl.com/api/v1/game/2019020001/boxscore

class MLDataCollector:


    def __init__(self):
        pass 


    def get_game(season, game, team_id): #* Try to reuse?
        pass 



    # Seasons
        # Teams
            # Game number
    def read(filename="default.txt"):
        pass 

    def write(filename = "default.txt"):
        pass


    """
    Collects user data
    """
    def collect_data(self):
        #* Collects data
        seasons = [2019, 2020, 2021] # Seasons to collect data

        self.data = {} #* Master object for storing data

        nhlteamsclass = NHLTeams() 
        self.nhl_teams = nhlteamsclass.generate_NHL_teams()

        for season in seasons:

            for team in self.nhl_teams:
                team.calculate_season(season)




class MLGuesser:

    def __init__(self):
        pass 

    def get_current_game_data(year: int, game_number: int):
        pass 
    def calculate(year: int, game_number: int):

        pass 



#* Main

def main():
    pass
    # datacollector =  MLDataCollector()
    # datacollector.collect_data()


# ?startDate=2018-01-09 Start date for the search
# ?endDate=2018-01-12 End date for the search
# ?season=20172018 Returns all games from specified season
# ?gameType=R Restricts results to only regular season games. Can be set to any value from Game Types endpoint
# GET https://statsapi.web.nhl.com/api/v1/schedule?teamId=30

    #* TESTING
    params = {
            "teamId":30, 
            "startDate":"2018-01-09",
            "endDate": "2020-01-09",   
        }
    blah = requests.get("https://statsapi.web.nhl.com/api/v1/schedule", params)

    import pdb; pdb.set_trace()
    




if __name__ == "__main__":
    main()



