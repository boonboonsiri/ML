# Classes for parsing the data
#2624 games in a season

def league_percentage_calculator(cur_game, year):
    if(year == 2022 or year == 2021):
        return cur_game / (82*32)
    elif year == 2020:
        return cur_game / (56 *32)
    else:
       return cur_game / (70*32)



def main():
    # testing()


if __name__ == "__main__":
    main()
