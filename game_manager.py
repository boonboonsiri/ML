from game_parser import Parser, Game
from data import FileManager
from pprint import pprint
from tqdm import tqdm
import multiprocessing
import torch
import matplotlib.pyplot as plt
import numpy as np
import random


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

    def check_spread(self, season=2023, to_analyse=[0,1,2]):

        def scatter(combined, slice_length=10):
            def generate_reddish_hex():
                # Generate random values for red, green, and blue components
                red = random.randint(170, 255)  # R: 170-255 for reddish tones
                green = random.randint(0, 100)   # G: 0-100 for darker shades
                blue = random.randint(0, 100)    # B: 0-100 for darker shades

                # Convert RGB values to hexadecimal format
                hex_code = "#{:02x}{:02x}{:02x}".format(red, green, blue)

                return hex_code

            def generate_bluish_hex():
                # Generate random values for red, green, and blue components
                red = random.randint(0, 100)    # R: 0-100 for darker shades
                green = random.randint(0, 100)  # G: 0-100 for darker shades
                blue = random.randint(170, 255) # B: 170-255 for bluish tones

                # Convert RGB values to hexadecimal format
                hex_code = "#{:02x}{:02x}{:02x}".format(red, green, blue)

                return hex_code

            random_indices = sorted(np.random.choice(len(combined), slice_length, replace=False))

            arr = []
            for i in random_indices:
                arr.append(combined[i])


            y = np.array([i for i in range(len(arr[0][0])+1)])

            for elem in arr:
                x = elem[0].tolist() + [elem[1].tolist()]
                x = np.array(x)
                jitter_strength = 0.1
                y_jitter =  np.random.normal(0, jitter_strength, y.shape)
                y_jitter = np.where(x[-1] == 1, -np.abs(y_jitter), np.abs(y_jitter))
                y_jittered = y + y_jitter

                hex = generate_bluish_hex() if x[-1] == 1 else generate_reddish_hex()
                plt.scatter(y_jittered, x,s=3, c=hex)

            plt.ylabel('X-axis Label')
            plt.xlabel('Y-axis Label')
            plt.title('Scatter Plot Example')
            plt.show()

        def get_averages(combined, to_analyse):
            for num in to_analyse:
                combined.sort(key=lambda x: float(x[0][num]))
                total = [c[0][num] for c in combined]
                mean = float(sum(total) / len(total))
                median = float(total[len(total)//2])

                home_wins = [elem[0][num] for elem in combined if elem[1] == 1]
                away_wins = [elem[0][num] for elem in combined if elem[1] == 0]

                mean_home = float(sum(home_wins) / len(home_wins))
                median_home = float(home_wins[len(home_wins)//2])
                mean_away = float(sum(away_wins) / len(away_wins))
                median_away = float(away_wins[len(away_wins)//2])
                #import pdb; pdb.set_trace()

                print(num, mean, median, mean_home, median_home, mean_away, median_away)

        features, labels = self.get_tensors_season(season)
        combined = [(feature, label) for feature, label in zip(features, labels)]
        get_averages(combined, to_analyse)
        scatter(combined)


        combined.sort(key=lambda x: float(x[0][2]))
        combined.sort(key=lambda x: x[1]) # sort by wins loses

        features = [x[0] for x in combined]
        labels = [x[1] for x in combined]

        plt.plot(range(0, len(features)), labels, label='Outcome')

        # Plot each array as a separate line
        for i, _ in enumerate(features[0]):
            arr = [f[i] for f in features]
            if i in to_analyse:
                plt.plot(arr, label=f'Line {i}')

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
