from pdb import set_trace
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
    #* Game in the season (by percentage)
    #* Current League wide standing
    #* Shots for? Shots against?
    #*

    data = {}


    def __init__(self, game_data, year, game_number):
        self.game_data = game_data
        self.game = game_data[year][game_number]

        self.calculate_game()

    def calculate_game(self):
        self.home = self.calculate_team(self.game['teams']['home'])
        self.away = self.calculate_team(self.game['teams']['away'])

        self.winner = self.game['team']['home']['teamStats']['teamSkaterStats']['goals'] >= self.game['away']['home']['teamStats']['teamSkaterStats']['goals']
        

    def calculate_team(self, team):
        team_data = {}

        set_trace()

    def calculate_past_game(self, team):
        pass


    def api():
        pass 

    def to_arr(self):
        return []
    def calculate(self):
        pass
