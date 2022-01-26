# get nba games of the day
# author @david03sanchez

import pandas as pd
import requests
from bs4 import BeautifulSoup


def games_of_day():
    team_dict = {
     'Hawks': 'ATL', 'Celtics': 'BOS', 'Nets': 'BKN', 'Hornets': 'CHA', 'Bulls': 'CHI', 'Cavaliers': 'CLE',
     'Mavericks': 'DAL', 'Nuggets': 'DEN', 'Pistons': 'DET', 'Warriors': 'GSW', 'Rockets': 'HOU', 'Pacers': 'IND',
     'Clippers': 'LAC', 'Lakers': 'LAL', 'Grizzlies': 'MEM', 'Heat': 'MIA', 'Bucks': 'MIL', 'Timberwolves': 'MIN',
     'Pelicans': 'NOP', 'Knicks': 'NYK', 'Thunder': 'OKC', 'Magic': 'ORL', '76ers': 'PHI', 'Suns': 'PHX',
     'Trail Blazers': 'POR', 'Kings': 'SAC', 'Spurs': 'SAS', 'Raptors': 'TOR', 'Jazz': 'UTA', 'Wizards': 'WAS'}
    todays_games = str(pd.to_datetime("today").tz_localize('UTC').tz_convert('EST').date())
    res = requests.get(f"https://www.nba.com/games?date={todays_games}")
    soup = BeautifulSoup(res.text, 'html.parser')
    team_list = [team_dict[span.get_text()] for span in soup.find_all('span') if span.get_text() in team_dict.keys()]
    daily_games = [x for x in zip(team_list[::2], team_list[1::2])]
    return daily_games
