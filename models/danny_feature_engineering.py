import pandas as pd
import numpy as np
from common_functions.utils import DataObject
from multiprocessing import get_context
env = 'linux'
#%%
if env == 'mac':
    root_data_dir = '/Users/danny/nba_bets/data/'
elif env == 'linux':
    root_data_dir = '/home/danny/nba/data/'


def getTeamStats(abv, latestdate):
    # edit this function for additional feature eng
    team_subset = df1[df1['TEAM_ABBREVIATION'] == abv].copy()
    team_subset['GAME_DATE'] = pd.to_datetime(team_subset['GAME_DATE'])
    team_subset.index = team_subset['GAME_DATE']
    team_subset.sort_index(inplace=True, ascending=False)
    colnames = team_subset.columns
    stats_columns = ['PTS', 'FGM', 'FGA', 'FG_PCT',
                     'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB',
                     'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF']
    date_subset = team_subset[team_subset['GAME_DATE'] < latestdate].copy()
    date_subset['numerical_wins'] = np.where(date_subset['WL'] == 'L', 0, 1)
    date_subset['location'] = np.where(date_subset['MATCHUP'].str.contains('@'), -1, 1)
    date_reversed = date_subset.iloc[::-1].copy()
    date_reversed['window_sum10'] = date_reversed['numerical_wins'].rolling(10).sum()
    date_reversed['window_sum5'] = date_reversed['numerical_wins'].rolling(5).sum()
    date_reversed['window_sum3'] = date_reversed['numerical_wins'].rolling(3).sum()
    stats_columns.extend(['window_sum10', 'window_sum5', 'window_sum3', 'location', 'numerical_wins', 'break_days'])
    date_subset = date_reversed.copy()
    date_subset['LAG_DATA'] = date_subset['GAME_DATE'].shift(1)
    date_subset['break_days'] = date_subset["GAME_DATE"] - date_subset["LAG_DATA"]
    date_subset['break_days'] = date_subset['break_days'].dt.days
    current_stats = date_subset.iloc[-11:, [date_subset.columns.get_loc(c) for c in stats_columns]].copy()
    base_points = current_stats['PTS']
    current_stats['PIE'] = (
            current_stats['PTS'] + current_stats['FGM'] + current_stats['FTM'] - current_stats[
        'FTA'] + current_stats['DREB'] +
            current_stats['OREB'] + current_stats['AST'] + current_stats['STL'] + current_stats[
                'BLK'] - current_stats['PF'] - current_stats['TOV'])
    current_stats['CORE_PTS'] = base_points
    current_stats.iloc[:, 0:18] = current_stats.iloc[:, 0:18].ewm(halflife=7).mean()
    return current_stats


def getTeamStatsold(abv,latestdate):
    #edit this function for additional feature eng
    team_subset = df1[df1['TEAM_ABBREVIATION'] == abv].copy()
    # team_subset = df1[df1['TEAM_ABBREVIATION'] == 'ATL'].copy()
    # latestdate = '2000-01-19'
    team_subset['GAME_DATE'] = pd.to_datetime(team_subset['GAME_DATE'])
    team_subset.index = team_subset['GAME_DATE']
    team_subset.sort_index(inplace=True, ascending=False)
    colnames = team_subset.columns
    stats_columns = ['PTS', 'FGM', 'FGA', 'FG_PCT',
                     'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB',
                     'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF']
    date_subset = team_subset[team_subset['GAME_DATE'] < latestdate].copy()
    date_subset['numerical_wins'] = np.where(date_subset['WL'] == 'L', 0, 1)
    date_subset['location'] = np.where(date_subset['MATCHUP'].str.contains('@'),-1,1)
    date_reversed = date_subset.iloc[::-1].copy()
    date_reversed['window_sum10'] = date_reversed['numerical_wins'].rolling(10).sum()
    date_reversed['window_sum5'] = date_reversed['numerical_wins'].rolling(5).sum()
    date_reversed['window_sum3'] = date_reversed['numerical_wins'].rolling(3).sum()
    stats_columns.extend(['window_sum10', 'window_sum5', 'window_sum3','location','numerical_wins'])
    date_subset = date_reversed.copy()

    current_stats = date_subset.iloc[-11:, [date_subset.columns.get_loc(c) for c in stats_columns]].copy()
    base_points = current_stats['PTS']
    current_stats['PIE'] = (
                current_stats['PTS'] + current_stats['FGM'] + current_stats['FTM'] - current_stats[
            'FTA'] + current_stats['DREB'] +
                current_stats['OREB'] + current_stats['AST'] + current_stats['STL'] + current_stats[
                    'BLK'] - current_stats['PF'] - current_stats['TOV'])
    current_stats['CORE_PTS'] = base_points
    current_stats.iloc[:,0:18] = current_stats.iloc[:,0:18].ewm(halflife=7).mean()
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
        if home_df.shape == (11, 26) and away_df.shape == (11, 26):
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


#read in data
df1 = pd.read_csv(root_data_dir + 'gamedf.csv',index_col = 0)


optimization_result = get_optimization(df1)

complete_dataset = []
for val in optimization_result:
    if val != None :
        complete_dataset.append(val)


train_labels = []
train_features = []
test_labels = []
test_features = []

for r in range(0,len(complete_dataset)):
    print(r)
    if (pd.to_datetime(complete_dataset[r][0]) < '2020-01-01').bool():
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

#%%
trainlab = np.nan_to_num(train_labels)
trainset = np.nan_to_num(train_features)
testlab = np.nan_to_num(test_labels)
testset = np.nan_to_num(test_features)

from tensorflow.keras.layers import Dense, Dropout,Conv1D, MaxPooling1D, Flatten, GlobalAvgPool1D
from tensorflow.keras import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l1



model = Sequential()
model.add(Dense((num_features), input_dim=(num_features), activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(500,activation='relu'))
model.add(Dense(500,activation='relu'))
model.add(Dense(500,activation='relu'))
model.add(Dense(500,activation='relu'))
model.add(Dense(500,activation='relu'))
model.add(Dense(500,activation='relu'))
model.add(Dense(500,activation='relu'))
model.add(Dense(500,activation='relu'))
# model.add(Dropout(0.4))
model.add(Dense(1, activation='linear'))
model.summary()

model.compile(loss='mean_absolute_error', optimizer=optimizers.Adam(lr=0.000001), metrics=['mae'])
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=15)
#
# opt = SGD(lr=0.001, momentum=0.9)
# model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])


history = model.fit(trainset, trainlab, epochs=4000, batch_size=20000, callbacks=[early_stop],
                    validation_split=.2,shuffle=False)

from matplotlib import pyplot
pyplot.subplot(212)
pyplot.title('MAE')
pyplot.plot(history.history['mae'], label='train')
pyplot.plot(history.history['val_mae'], label='test')
pyplot.legend()
pyplot.show()


from sklearn.metrics import mean_squared_error, mean_absolute_error
predictions = model.predict(testset).flatten()
pred_df = pd.DataFrame([predictions,testlab]).T
pred_df.columns = ['predictions','labels']
print(mean_squared_error(pred_df['labels'], pred_df['predictions'], squared=False))
print(mean_absolute_error(pred_df['labels'], pred_df['predictions']))
22.03
17.17
21.53
16.80
21.54
16.69
20.96
16.39
20.68
16.2
20.63
16.18
20.59
16.12

#bench
20.27
15.91

#%%
trainlab = np.nan_to_num(train_labels)
trainset = np.nan_to_num(train_features)
testlab = np.nan_to_num(test_labels)
testset = np.nan_to_num(test_features)


import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

dtrain = xgb.DMatrix(trainset, label=trainlab)

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