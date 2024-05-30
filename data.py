# All data classes for collecting data

import requests
import json
from pprint import pprint
from tqdm import tqdm
import multiprocessing
from datetime import datetime, timedelta



class FileManager:
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

    def verify(self):
        data = None
        # Define the API endpoint for standings
        url = "https://api.nhle.com/stats/rest/en/team"

        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
        print(type(data))
        self.write(data, 'data/test.json')
        data = self.read('data/test.json')
        print(type(data))


class APIMachine:

    def __init__(self):
        self.seasons = [2019, 2020, 2021, 2022, 2023]
        self.multiprocessing = True

    def api_download_all(self):
        if self.multiprocessing:
            pool = multiprocessing.Pool(processes=5)  # Number of processes
            pool.map(self.handle_season, self.seasons)
            pool.close()
            pool.join()
        else:
            for season in self.seasons:
                self.handle_season(season)

    def handle_season(self,season):
        print(season)
        year_data = self.api_call_year(season)
        f = FileManager()
        f.write(year_data, f"data/api_{season}.json")

    def api_call_year(self, year):
        year_data = {'games': []}

        for game_number in tqdm(range(int(82*32/2)), desc="Processing", unit="iteration"):
            game_data, failure = self.call_game_api(year, game_number)
            if failure: # If game not found then break
                break
            year_data['games'].append(game_data)

        return year_data

    def call_game_api(self, year: int, game_number: int): # data, failure
        url = f'https://api-web.nhle.com/v1/gamecenter/{year}02{str(game_number+1).zfill(4)}/boxscore'
        resp = requests.get(url)
        data = None
        if resp.status_code == 200:
            data = resp.json()
            game_date = data['gameDate'] # subtract 1 day to not include the game (for training future games). THIS FOR A FACT MEANS YOU CAN"T TRAIN GAME 0
            date_obj = datetime.strptime(game_date, '%Y-%m-%d'); new_date = date_obj - timedelta(days=1)
            game_date = new_date.strftime('%Y-%m-%d')

            url = f"https://api-web.nhle.com/v1/standings/{game_date}"
            resp = requests.get(url)

            if resp.status_code == 200:
                standing_data = resp.json()
                data['standings'] = standing_data



        return data, resp.status_code == 404

    def verify_game_api(self, year=2020, game=400):
        data, _ = self.call_game_api(year, game)
        pprint(data)


def testing():
    # f = FileManager(); f.verify()
    #a = APIMachine(); a.verify_game_api(2020, 10)
    #f = FileManager(); a = f.read('test.json')

def main():
    testing()
    #a = APIMachine(); a.api_download_all() # download

if __name__ == "__main__":
    main()
