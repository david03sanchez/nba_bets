import pandas as pd
import numpy as np
#%%
df1 = pd.read_csv('/home/danny/nba/data/gamedf.csv', index_col=0)
df1.index = pd.to_datetime(df1['GAME_DATE'])
df1.sort_index(inplace=True, ascending=False)
distance_df = pd.read_csv('/home/data/distance_map.csv', index_col=0)
distance_df['regular'] = np.nan
for r in range(0, distance_df.shape[0]):
    team1 = distance_df.iloc[r, 0]
    team2 = distance_df.iloc[r, 1]
    distance_df.iloc[r,-1] = f'{team1}{team2}{team2}{team1}'

distance_df.to_csv('/home/data/distance_map.csv')
#%%
def distanceSlice(game_id, team):
    #TEAM_ABBREVIATION, GAME_ID, GAME_DATE, MATCHUP, DISTANCE
    # game_id = 20300224
    # team = 'ATL'
    subset = df1.loc[:, ['TEAM_ABBREVIATION', 'GAME_ID', 'GAME_DATE', 'MATCHUP']][df1['GAME_ID'] == game_id]
    game_date = pd.to_datetime(subset['GAME_DATE'][0])
    subset['distance'] = np.nan
    homeAwayCheck = True
    awayAwayCheck = True
    homeDistance = 0
    awayDistance = 0
    hometeam = subset[subset['MATCHUP'].str.contains('@') == False].iloc[0, 0]
    awayteam = subset[subset['MATCHUP'].str.contains('@') == True].iloc[0, 0]
    # get all games of hometeam, locate last 2, ['MATCHUP'][1] is previous game, check if home/away
    homeAwayCheck = True if '@' in df1[(df1['TEAM_ABBREVIATION'] == hometeam) & (pd.to_datetime(df1['GAME_DATE']) <= game_date)].iloc[:2, :]['MATCHUP'][1] else False
    awayAwayCheck = True if '@' in df1[(df1['TEAM_ABBREVIATION'] == awayteam) & (pd.to_datetime(df1['GAME_DATE']) <= game_date)].iloc[:2, :]['MATCHUP'][1] else False
    # homeDistance = 0 if homeAwayCheck is False else
    # if hometeam was home last game
    if homeAwayCheck is False:
        homeDistance = 0
    # if hometeam was away last game
    elif homeAwayCheck is True:
        # where the home team played last game
        prevGameLoc = df1[df1['TEAM_ABBREVIATION'] == hometeam].iloc[:2, :]['MATCHUP'][1][-3:]
        homeDistance = distance_df[distance_df['regular'].str.contains(f'{hometeam}{prevGameLoc}')]['distance'].reset_index(drop=True)[0]

    # if awayteam was home last game
    if awayAwayCheck is False:
        awayDistance = 0
    elif awayAwayCheck is True:
        prevGameLoc = df1[df1['TEAM_ABBREVIATION'] == awayteam].iloc[:2, :]['MATCHUP'][1][-3:]
        awayDistance = distance_df[distance_df['regular'].str.contains(f'{awayteam}{prevGameLoc}')]['distance'].reset_index(drop=True)[0]
    subset.loc[subset['TEAM_ABBREVIATION'] == hometeam, 'distance'] = homeDistance
    subset.loc[subset['TEAM_ABBREVIATION'] == awayteam, 'distance'] = awayDistance
    # .reset_index(drop=True, inplace=True)
    return subset[subset['TEAM_ABBREVIATION'] == team].iloc[0,-1]

#%%
# TEST
last10 = df1.iloc[:20,[df1.columns.get_loc(c) for c in ['TEAM_ABBREVIATION', 'GAME_ID', 'GAME_DATE', 'MATCHUP']]].reset_index(drop=True)
last10['distance'] = np.nan

# def addDistance(game_id, team):
#     x = 0
for x in range(last10.shape[0]):
    game_row = last10.iloc[x,:]
    game_row['distance'] = distanceSlice(game_row['GAME_ID'], game_row['TEAM_ABBREVIATION'])
    last10.iloc[x, last10.columns.get_loc('distance')] = game_row['distance']

#%%
def getTeamStats(abv, latestdate):
    # abv = 'ATL'
    # latestdate = '2003-12-14'
    team_subset = df1[(df1['TEAM_ABBREVIATION'] == abv) & (pd.to_datetime(df1['GAME_DATE']) < latestdate)].copy()
    team_subset['GAME_DATE'] = pd.to_datetime(team_subset['GAME_DATE'])
    team_subset.index = team_subset['GAME_DATE']
    team_subset.sort_index(inplace=True, ascending=False)
    stats_columns = ['GAME_ID', 'PTS', 'FGM', 'FGA', 'FG_PCT',
                     'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB',
                     'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'window_sum10', 'window_sum5',
                     'window_sum3', 'location', 'numerical_wins', 'break_days', 'distance', 'PIE', 'CORE_PTS']
    team_subset['numerical_wins'] = np.where(team_subset['WL'] == 'L', 0, 1)
    team_subset['location'] = np.where(team_subset['MATCHUP'].str.contains('@'), -1, 1)
    team_subset = team_subset.iloc[::-1].copy()
    team_subset['window_sum10'] = team_subset['numerical_wins'].rolling(10).sum()
    team_subset['window_sum5'] = team_subset['numerical_wins'].rolling(5).sum()
    team_subset['window_sum3'] = team_subset['numerical_wins'].rolling(3).sum()
    team_subset['LAG_DATA'] = team_subset['GAME_DATE'].shift(1)
    team_subset['break_days'] = team_subset["GAME_DATE"] - team_subset["LAG_DATA"]
    team_subset['break_days'] = team_subset['break_days'].dt.days
    team_subset['distance'] = np.nan
    team_subset['PIE'] = (
        team_subset['PTS'] + team_subset['FGM'] + team_subset['FTM'] - team_subset['FTA'] +
        team_subset['DREB'] + team_subset['OREB'] + team_subset['AST'] + team_subset['STL'] +
        team_subset['BLK'] - team_subset['PF'] - team_subset['TOV'])
    team_subset['CORE_PTS'] = team_subset['PTS']
    team_subset = team_subset.iloc[-11:, [team_subset.columns.get_loc(c) for c in stats_columns]]
    team_subset.iloc[:, 1:18] = team_subset.iloc[:, 1:18].ewm(halflife=7).mean()
    for x in range(11):
        game_row = team_subset.iloc[x, :]
        game_row['distance'] = distanceSlice(game_row['GAME_ID'], abv)
        team_subset.iloc[x, team_subset.columns.get_loc('distance')] = game_row['distance']

    return team_subset