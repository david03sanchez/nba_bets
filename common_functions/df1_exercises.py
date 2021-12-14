import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
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


