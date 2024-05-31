from game_parser import Parser, Game
from data import FileManager
from pprint import pprint
import requests
import json
from tqdm import tqdm
import multiprocessing
from datetime import datetime, timedelta


# Main interaction for creating, manipulating, reading, and writing games
class GameManager:
    def __init__(self):
        # settings
        self.seasons = [2019, 2020, 2021, 2022, 2023]
        self.multiprocessing = True


    def parse_all(self):
        if self.multiprocessing:
            pool = multiprocessing.Pool(processes=5)  # Number of processes
            pool.map(self.parse_season, self.seasons)
            pool.close()
            pool.join()
        else:
            for season in self.seasons:
                self.parse_season(season)

    def parse_season(self,season: int):
        year_data = {'games': []}
        parser = Parser()
        for game_number in tqdm(range(int(82*32/2)), desc="Processing", unit="iteration"):
            game = parser.parse_game(season, game_number)
            year_data['games'].append(game)

        year_data['games'] = [g.to_dict() if g else None for g in year_data['games']] # convert to dictiontary

        f = FileManager()
        f.write(year_data, f"data/game_{season}.json")




def testing():
    pass

def main():
    # testing()
    g = GameManager(); g.parse_all()
if __name__ == "__main__":
    main()
