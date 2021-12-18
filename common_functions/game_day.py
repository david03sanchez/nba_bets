# get nba games of the day
# author @david03sanchez

import pandas as pd
import requests
from bs4 import BeautifulSoup


def games_of_day():
    current_teams = pd.read_csv('C:\\Users\\David\\PycharmProjects\\nba_bets\\common_functions\\current_teams.csv')
    team_dict = dict(zip(list(current_teams["TEAM_MASCOT"]), list(current_teams["TEAM_ABBREVIATION"])))
    todays_games = str(pd.to_datetime("today"))[:10]
    res = requests.get(f"https://www.nba.com/games?date={todays_games}")
    soup = BeautifulSoup(res.text, 'html.parser')
    span_list = []
    for span in soup.find_all('span'):
        span_list.append(span.get_text())
    daily_teams = []
    game_titles = []

    for text in span_list:
        if text in team_dict.keys():
            daily_teams.append(team_dict[text])

    for team in daily_teams:
        if daily_teams.index(team) % 2 == 0:
            game_titles.append((team, daily_teams[daily_teams.index(team) + 1]))

    # return won't output game_titles when main() run?
    # python game_day.py returns nothing?
    return game_titles
