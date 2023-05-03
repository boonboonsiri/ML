from pdb import set_trace

def league_percentage_calculator(cur_game, year):
    if(year == 2022 or year == 2021):
        return cur_game / (82*32)
    elif year == 2020:
        return cur_game / (56 *32)
    else:
       return cur_game / (70*32)
    
#* Old game data, 
class Game:
    
    #* Top 3 scorers in points last 10. Method one, just find last 10 games, method 2 keep a point tracker class for every players last 10 games
    #* Save percentage
    #* Shots for
    #* Shots against
    #* Home/Away (1,0)
    #* Goals for
    #* Goals against
    #* last 10 win count
    #* last 10 tie count
    #* Game in the season (by percentage)
    #*
    #? Need to do seperately
    #* Win percentage (Season culmative)
    #* Current League wide standing


    data = {}


    def __init__(self, game_data, season_data, year, game_number):
        self.game_data = game_data
        self.season_data = season_data
        self.game = game_data[year][game_number]
        self.year = year
        self.game_number = game_number

        self.gamePk = int(f'{year}02{str(game_number).zfill(4)}')

        self.calculate_game()

    def calculate_game(self):
        self.home = self.calculate_team(self.game['teams']['home'], 'home')
        self.away = self.calculate_team(self.game['teams']['away'], 'away')

        self.winner = self.game['teams']['home']['teamStats']['teamSkaterStats']['goals'] >= self.game['teams']['away']['teamStats']['teamSkaterStats']['goals']
        

    def calculate_team(self, team, status):
        team_data = {}
        team_id = team.get('team').get('id')

        skaters = team.get('skaters')
        goalies = team.get('goalies')

        not_status = 'away' if status == 'home' else 'away'

        team_data['is_home'] = status == 'home'
        team_data['id'] = team_id;
        team_data['shots_for'] = 0
        team_data['shots_against'] = 0
        team_data['goals_for'] = 0 
        team_data['goals_against'] = 0
        team_data['last_10_win'] = 0
        team_data['last_10_tie'] = 0
        team_data['game_in_season'] = league_percentage_calculator(self.game_number, self.year)
        team_data['league_standing'] = 0

        team_schedule_index, last_10 = self.last_ten(team_id)

        for game in last_10:
            actual_game = game.get('games')[0]
            team_data['shots_for'] += actual_game['linescore']['teams'][status]['shotsOnGoal']
            team_data['shots_against'] += actual_game['linescore']['teams'][not_status]['shotsOnGoal']

            team_data['goals_for'] += actual_game['linescore']['teams'][status]['goals']
            team_data['goals_against'] += actual_game['linescore']['teams'][not_status]['goals']
            
        set_trace()
        #* Game in the season (by percentage)

        return team_data

    def last_ten(self, team_id): #* Returns last 10 games
        team_schedule = self.season_data[self.year][str(team_id)]
        # team_schedule_index = team_schedule['dates'].index(lambda date: date.get('games')[0]['gamePk'] == self.gamePk)
        generator = (i for i,v in enumerate(team_schedule['dates']) if int(v.get('games')[0]['gamePk']) == self.gamePk+1)
        team_schedule_index = next(generator)
        return team_schedule_index, team_schedule.get('dates')[max(team_schedule_index-10, 0):team_schedule_index]

    def calculate_past_game(self, team):
        pass


    def api():
        pass 

    def to_arr(self):
        return []
    def calculate(self):
        pass


class TeamGame():
    # Calculate team game per season
    def __init__(self):
        pass 

class TrainingGame: # Game information from both teams for the last 10 games, and result, used 
    def __init__(self):
        pass 


