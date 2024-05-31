from game_parser import Parser, Game
from data import FileManager
from pprint import pprint
from tqdm import tqdm
import multiprocessing
import torch
import matplotlib.pyplot as plt


# Main interaction for creating, manipulating, reading, and writing games
class GameManager:
    def __init__(self):
        # settings
        self.seasons = [2018]
        self.tensor_seasons = [2018, 2019, 2020, 2021, 2022, 2023]
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

    def read_season(self, year) -> dict:
        f = FileManager()
        season_data = f.read(f"data/game_{year}.json")
        season_data['games'] = [Game(g) if g else None for g in season_data['games']]

        return season_data

    def get_tensors_all_seasons(self): # NOTE MODIFY which features, and name of file
        results = []
        if self.multiprocessing:
            pool = multiprocessing.Pool(processes=5)  # Number of processes
            results = pool.map(self.get_tensors_season, self.tensor_seasons)
            pool.close()
            pool.join()
        else:
            for season in self.tensor_seasons:
                results.append(self.get_tensors_season(season))

        features_results = [result[0] for result in results]
        labels_results = [result[1] for result in results]
        features_tensors = torch.cat(features_results, dim=0)
        labels_tensors = torch.cat(labels_results, dim=0)

        tensor_name = 'v0' #!HERE
        features_file = f'data/{tensor_name}_features_tensor.pt'
        labels_file = f'data/{tensor_name}_labels_tensor.pt'

        # Save tensors to disk
        torch.save(features_tensors, features_file)
        torch.save(labels_tensors, labels_file)

    def get_tensors_season(self, season: int):
        season_data = self.read_season(season)
        tensors_arr = [g.to_features_v0() for g in season_data['games'] if g] #!HERE
        data_torch = torch.tensor(tensors_arr, dtype=torch.float32)
        features_torch = data_torch[:, :-1]
        labels_torch = data_torch[:, -1]
        return features_torch, labels_torch

    def get_tensor(self, season: int, game_number: int):
        season_data = self.read_season(season)
        game = season_data['games'][game_number-1] # - 1 for how I store vs NHL
        tensors_arr =[game.to_features_v0()]
        data_torch = torch.tensor(tensors_arr, dtype=torch.float32)
        features_torch = data_torch[:, :-1]
        labels_torch = data_torch[:, -1]

        return game, features_torch, labels_torch

    def check_spread(self, season=2023):
        features, labels = self.get_tensors_season(season)

        combined = [(feature, label) for feature, label in zip(features, labels)]
        combined.sort(key=lambda x: x[1])

        features = [x[0] for x in combined]
        labels = [x[1] for x in combined]

        # Plot loss versus epoch
        # plt.plot(range(1, len(features)), [], label='Train Loss')
        plt.plot(range(0, len(features)), labels, label='Outcome')
        plt.xlabel('Game')
        plt.ylabel('Various')
        plt.title('Loss vs Epoch')
        plt.legend()
        plt.show()



def testing():
    g = GameManager()
    #season_data = g.read_season(2023)
    #pprint(g.get_tensor(2023, 1244))
    g.check_spread()
    #import pdb; pdb.set_trace()
    #print(season_data['games'][-1])

def main():
    testing()
    #g = GameManager(); g.parse_all() # RE PARSE ALL DATA
    #g = GameManager(); g.get_tensors_all_seasons()
if __name__ == "__main__":
    main()
