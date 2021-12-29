import pandas as pd
import numpy as np
from common_functions.utils import DataObject
from multiprocessing import get_context
import matplotlib.pyplot as plt
#%%
df1 = pd.read_csv('/home/danny/nba/data/gamedf.csv',index_col = 0)

def getTeamStats(abv, latestdate):
    abv = 'ATL'
    latestdate = '2003-12-14'
    # edit this function for additional feature eng
    team_subset = df1[df1['TEAM_ABBREVIATION'] == abv].copy()
    team_subset['GAME_DATE'] = pd.to_datetime(team_subset['GAME_DATE'])
    team_subset.index = team_subset['GAME_DATE']
    team_subset.sort_index(inplace=True, ascending=False)
    colnames = team_subset.columns
    stats_columns = ['PTS', 'FGM', 'FGA', 'FG_PCT',
                     'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB',
                     'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'window_sum10', 'window_sum5',
                     'window_sum3', 'location', 'numerical_wins', 'break_days']
    date_subset = team_subset[team_subset['GAME_DATE'] < latestdate].copy()
    date_subset['numerical_wins'] = np.where(date_subset['WL'] == 'L', 0, 1)
    date_subset['location'] = np.where(date_subset['MATCHUP'].str.contains('@'), -1, 1)
    date_subset = date_subset.iloc[::-1].copy()
    date_subset['window_sum10'] = date_subset['numerical_wins'].rolling(10).sum()
    date_subset['window_sum5'] = date_subset['numerical_wins'].rolling(5).sum()
    date_subset['window_sum3'] = date_subset['numerical_wins'].rolling(3).sum()
    date_subset['LAG_DATA'] = date_subset['GAME_DATE'].shift(1)
    date_subset['break_days'] = date_subset["GAME_DATE"] - date_subset["LAG_DATA"]
    date_subset['break_days'] = date_subset['break_days'].dt.days
    current_stats = date_subset.iloc[-11:, [date_subset.columns.get_loc(c) for c in stats_columns]].copy()
    current_stats['PIE'] = (
            current_stats['PTS'] + current_stats['FGM'] + current_stats['FTM'] - current_stats[
        'FTA'] + current_stats['DREB'] +
            current_stats['OREB'] + current_stats['AST'] + current_stats['STL'] + current_stats[
                'BLK'] - current_stats['PF'] - current_stats['TOV'])
    current_stats['CORE_PTS'] = current_stats['PTS']
    current_stats.iloc[:, 0:18] = current_stats.iloc[:, 0:18].ewm(halflife=7).mean()

    return current_stats

def getOverUnder(gameid):
    try:
        target_game = df1[df1['GAME_ID'] == gameid]  # contains target
        # target_game = df1[df1['GAME_ID'] == 29900545] #contains target
        if target_game.shape[0] != 2:
            return None
        relevant_teams = target_game['TEAM_ABBREVIATION'].tolist()
        match_location_away = target_game.loc[target_game['MATCHUP'].str.contains('@')]
        match_location_home = target_game.loc[~target_game['MATCHUP'].str.contains('@')]
        target_game_date = match_location_home['GAME_DATE']
        # match_outcome_home = np.where(match_location_away['WL'] == 'W',0,1) #0 if away team wins
        spread = match_location_home.iloc[0, match_location_home.columns.get_loc('PTS')] + \
                 match_location_away.iloc[0, match_location_away.columns.get_loc('PTS')]
        game_date = match_location_away['GAME_DATE'].values[0]
        home_team = match_location_away['MATCHUP'].str.extract(r'((?<=@.)\S{3})')[0].tolist()
        away_team = [x for x in relevant_teams if x not in home_team]
        home_df = getTeamStats(home_team[0], game_date)
        away_df = getTeamStats(away_team[0], game_date)
        # normalized_hdf = (home_df - home_df.min()) / (home_df.max() - home_df.min())
        # normalized_adf = (away_df - away_df.min()) / (away_df.max() - away_df.min())
        if home_df.shape == (11, 25) and away_df.shape == (11, 25):
            output = [target_game_date, spread, home_df, away_df]
        else:
            return None
    except:
        return None
    return output

def get_optimization(indf):
    all_games_ids = indf['GAME_ID'].unique()
    pool = get_context("fork").Pool(22) #change to number of cores on machine
    optimization_result = pool.map(getOverUnder, all_games_ids)
    pool.close()
    return optimization_result
