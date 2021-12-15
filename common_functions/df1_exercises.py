import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import csv
#%%
df1 = pd.read_csv('C:\\Users\\David\\PycharmProjects\\nba_bets\\common_functions\\gamedf.csv', index_col=0)

#%%
#total points scored by atlanta hawks 2020-21 regular season
game_date = df1.loc[:,'GAME_DATE']

df2 = df1[("2020-12-22 00:00:00" < game_date) & (game_date < "2021-05-21 00:00:00")]

team_ticker = df2[['TEAM_ABBREVIATION','GAME_DATE', "PTS"]]

atlanta_games = team_ticker[team_ticker["TEAM_ABBREVIATION"] == "ATL"]

atlanta_games['PTS'].sum()
#8186

#%%
#total number of points chicago scored between 1992-1996

df3 = df1[("1992-11-06 00:00:00" < game_date) & (game_date < "1997-06-13 00:00:00")]

bulls_ticker = df3[["TEAM_ABBREVIATION", "PTS"]]

bulls_games = bulls_ticker[bulls_ticker["TEAM_ABBREVIATION"] == "CHI"]

bulls_games["PTS"].sum()
#49251

#%%
#get the daily nba games
def games_of_the_day():
    todays_games = str(pd.to_datetime("today"))[:10]
    res = requests.get(f"https://www.nba.com/games?date={todays_games}")
    soup = BeautifulSoup(res.text, 'html.parser')
    span_list = []
    for team in soup.find_all('span'):
        span_list.append(team.get_text())

    teams = pd.read_csv('C:\\Users\\David\\PycharmProjects\\nba_bets\\common_functions\\nba_teams.csv')
    prefixes = sorted(list(teams["name"]))
    mascot_names = []
    daily_games = []
    game_titles = []

    for team in prefixes:
        mascot_names.append(team.split()[-1])

    for text in span_list:
        if text in mascot_names:
            daily_games.append(text)

    for team in daily_games:
        if daily_games.index(team) % 2 == 0:
            game_titles.append(f'{team} at {daily_games[daily_games.index(team)+1]}')

    return game_titles








