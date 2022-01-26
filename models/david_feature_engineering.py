import pandas as pd
import numpy as np
from multiprocessing import get_context
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
df1 = pd.read_csv('/home/danny/nba/data/gamedf.csv', index_col=0)
df1.index = pd.to_datetime(df1['GAME_DATE'])
df1.sort_index(inplace=True, ascending=False)
distance_df = pd.read_csv('/home/data/distance_map.csv', index_col=0)
#%%
def distanceSlice(game_id, team):
    #TEAM_ABBREVIATION, GAME_ID, GAME_DATE, MATCHUP, DISTANCE
    # game_id = 22100560
    # team = 'ATL'
    subset = df1.loc[:, ['TEAM_ABBREVIATION', 'GAME_ID', 'GAME_DATE', 'MATCHUP']][df1['GAME_ID'] == game_id]
    game_date = pd.to_datetime(subset['GAME_DATE'][0])
    subset['distance'] = np.nan
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
        prevGameLoc = df1[(df1['TEAM_ABBREVIATION'] == hometeam) & (pd.to_datetime(df1['GAME_DATE']) <= game_date)].iloc[:2, :]['MATCHUP'][1][-3:]
        homeDistance = distance_df[distance_df['regular'].str.contains(f'{hometeam}{prevGameLoc}')]['distance'].reset_index(drop=True)[0]

    # if awayteam was home last game
    if awayAwayCheck is False:
        awayDistance = 0
    elif awayAwayCheck is True:
        prevGameLoc = df1[(df1['TEAM_ABBREVIATION'] == awayteam) & (pd.to_datetime(df1['GAME_DATE']) <= game_date)].iloc[:2, :]['MATCHUP'][1][-3:]
        awayDistance = distance_df[distance_df['regular'].str.contains(f'{hometeam}{prevGameLoc}')]['distance'].reset_index(drop=True)[0]
    subset.loc[subset['TEAM_ABBREVIATION'] == hometeam, 'distance'] = homeDistance
    subset.loc[subset['TEAM_ABBREVIATION'] == awayteam, 'distance'] = awayDistance
    # .reset_index(drop=True, inplace=True)
    return subset[subset['TEAM_ABBREVIATION'] == team].iloc[0,-1]

def getTeamStats(abv, latestdate):
    # abv = 'ATL'
    # latestdate = '2021-01-06'
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
    team_subset = team_subset.iloc[-60:, [team_subset.columns.get_loc(c) for c in stats_columns]]
    # x = 2
    for x in range(60):
        game_row = team_subset.iloc[x, :]
        game_row['distance'] = distanceSlice(game_row['GAME_ID'], abv)
        team_subset.iloc[x, team_subset.columns.get_loc('distance')] = game_row['distance']

    return team_subset

def getOverUnder(gameid):
    try:
        target_game = df1[df1['GAME_ID'] == gameid].reset_index(drop=True)  # contains target
        # target_game = df1[df1['GAME_ID'] == 22100560].reset_index(drop=True) #contains target
        if target_game.shape[0] != 2: #if all-star or other special game
            return None
        game_date = pd.to_datetime(target_game.loc[:,'GAME_DATE'].iloc[0])
        over_under = target_game.loc[~target_game['MATCHUP'].str.contains('@')]['PTS'].iloc[0] - target_game.loc[target_game['MATCHUP'].str.contains('@')]['PTS'].iloc[0]
        home_team = target_game.loc[~target_game['MATCHUP'].str.contains('@')]['TEAM_ABBREVIATION'].iloc[0]
        away_team = target_game.loc[target_game['MATCHUP'].str.contains('@')]['TEAM_ABBREVIATION'].iloc[0]
        home_df = getTeamStats(home_team, game_date)
        away_df = getTeamStats(away_team, game_date)
        if home_df.shape == (11, 28) and away_df.shape == (11, 28):
            output = [game_date, over_under, home_df, away_df]
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
    complete_dataset = []
    for val in optimization_result:
        if val != None:
            complete_dataset.append(val)
    return complete_dataset
#%%
complete_dataset = get_optimization(df1)

#%%
train_labels = []
train_features = []
test_labels = []
test_features = []

for r in range(0,len(complete_dataset)):
    print(r)
    if pd.to_datetime(complete_dataset[r][0]) < pd.to_datetime('2020-01-01'):
        train_labels.append(complete_dataset[r][1])
        home_row = complete_dataset[r][2].to_numpy().flatten('F')
        away_row = complete_dataset[r][3].to_numpy().flatten('F')
        both_row = np.concatenate((home_row,away_row))
        train_features.append(both_row)
    else:
        test_labels.append(complete_dataset[r][1])
        home_row = complete_dataset[r][2].to_numpy().flatten('F')
        away_row = complete_dataset[r][3].to_numpy().flatten('F')
        both_row = np.concatenate((home_row,away_row))
        test_features.append(both_row)

num_features = 550
train_labels = np.array(train_labels)
train_features = np.array(train_features)
test_labels = np.array(test_labels)
test_features = np.array(test_features)
trainlab = np.nan_to_num(train_labels)
trainset = np.nan_to_num(train_features)
testlab = np.nan_to_num(test_labels)
testset = np.nan_to_num(test_features)

dtrain = xgb.DMatrix(trainset, label=trainlab)
np.random.seed(12)

params = {}
params['eval_metric'] = 'mae'
params['tree_method'] = 'gpu_hist'
# params['colsample_bytree'] = .849
# params['gamma'] = .07
params['learning_rate'] = .01
params['max_depth'] = 5
# params['early_stopping_rounds'] = 30
params['objective'] = 'reg:squarederror'
# params['scale_pos_weight'] = 2

num_round = 1200

bst = xgb.train(params, dtrain,num_round)

dtest = xgb.DMatrix(testset)
predictions = bst.predict(dtest)
pred_df = pd.DataFrame([predictions,test_labels]).transpose()
pred_df.columns = ['predictions','labels']
print(mean_squared_error(pred_df['labels'], pred_df['predictions'], squared=False))
print(mean_absolute_error(pred_df['labels'], pred_df['predictions']))