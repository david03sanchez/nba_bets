import pandas as pd
import numpy as np
from multiprocessing import get_context
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

df1 = pd.read_csv('/home/data/gamedf.csv',index_col = 0)

ranking_map = pd.read_csv('/home/danny/nba/data/ranking_map.csv',index_col=0)

def getTeamStats(abv, latestdate):
    # abv = 'ATL'
    # latestdate = '2003-12-14'
    # edit this function for additional feature eng
    team_subset = df1[(df1['TEAM_ABBREVIATION'] == abv) & (pd.to_datetime(df1['GAME_DATE']) < latestdate)].copy()
    team_subset['GAME_DATE'] = pd.to_datetime(team_subset['GAME_DATE'])
    team_subset.index = team_subset['GAME_DATE']
    team_subset.sort_index(inplace=True, ascending=False)
    stats_columns = ['PTS', 'FGM', 'FGA', 'FG_PCT',
                     'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB',
                     'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'window_sum10', 'window_sum5',
                     'window_sum3', 'location', 'numerical_wins', 'break_days']
    team_subset['numerical_wins'] = np.where(team_subset['WL'] == 'L', 0, 1)
    team_subset['location'] = np.where(team_subset['MATCHUP'].str.contains('@'), -1, 1)
    team_subset = team_subset.iloc[::-1].copy()
    team_subset['window_sum10'] = team_subset['numerical_wins'].rolling(10).sum()
    team_subset['window_sum5'] = team_subset['numerical_wins'].rolling(5).sum()
    team_subset['window_sum3'] = team_subset['numerical_wins'].rolling(3).sum()
    team_subset['LAG_DATA'] = team_subset['GAME_DATE'].shift(1)
    team_subset['break_days'] = team_subset["GAME_DATE"] - team_subset["LAG_DATA"]
    team_subset['break_days'] = team_subset['break_days'].dt.days
    team_subset = team_subset.iloc[-30:, [team_subset.columns.get_loc(c) for c in stats_columns]]
    team_subset['PIE'] = (
            team_subset['PTS'] + team_subset['FGM'] + team_subset['FTM'] - team_subset[
        'FTA'] + team_subset['DREB'] +
            team_subset['OREB'] + team_subset['AST'] + team_subset['STL'] + team_subset[
                'BLK'] - team_subset['PF'] - team_subset['TOV'])
    team_subset['CORE_PTS'] = team_subset['PTS']
    team_subset['year_norm'] = team_subset.index.year - 1983
    rank_sub = ranking_map[ranking_map['TEAM_ABBREVIATION'] == abv][['PTS_Rank','AST_Rank','Index_Date']]
    rank_sub['Index_Date'] = pd.to_datetime(rank_sub['Index_Date'])
    rank_sub.set_index('Index_Date',inplace=True)
    team_subset = team_subset.join(rank_sub,how='left')
    team_subset.fillna(0,inplace=True)
    return team_subset

def getOverUnder(gameid):
    try:
        target_game = df1[df1['GAME_ID'] == gameid].reset_index(drop=True)  # contains target
        # target_game = df1[df1['GAME_ID'] == 29900549].reset_index(drop=True) #contains target
        if target_game.shape[0] != 2: #if all-star or other special game
            return None
        game_date = target_game.loc[0,'GAME_DATE']
        # over_under = target_game['PTS'].sum()
        over_under = target_game.loc[~target_game['MATCHUP'].str.contains('@')]['PTS'].iloc[0] - target_game.loc[target_game['MATCHUP'].str.contains('@')]['PTS'].iloc[0]
        away_team = target_game.loc[target_game['MATCHUP'].str.contains('@')]['TEAM_ABBREVIATION'].iloc[0]
        home_team = target_game.loc[~target_game['MATCHUP'].str.contains('@')]['TEAM_ABBREVIATION'].iloc[0]
        away_team = target_game.loc[target_game['MATCHUP'].str.contains('@')]['TEAM_ABBREVIATION'].iloc[0]
        home_df = getTeamStats(home_team, game_date)
        away_df = getTeamStats(away_team, game_date)
        if home_df.shape == (30, 29) and away_df.shape == (30, 29):
            output = [game_date, over_under, home_df, away_df]
        else:
            return None
    except:
        return None
    return output
# getOverUnder(29900549)
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

complete_dataset = get_optimization(df1)

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

num_features = len(home_row)
train_labels = np.array(train_labels)
train_features = np.array(train_features)
test_labels = np.array(test_labels)
test_features = np.array(test_features)
trainlab = np.nan_to_num(train_labels)
trainset = np.nan_to_num(train_features)
testlab = np.nan_to_num(test_labels)
testset = np.nan_to_num(test_features)

#%%

class trainRegressionModels:

    def __init__(self, trainset,trainlab):
        self.trainset = trainset
        self.trainlab = trainlab
        self.model_list = []

    def train_individual_model(self,learning_rate=.01,max_depth=5,num_round=1200):
        params = {}
        params['eval_metric'] = 'mae'
        params['tree_method'] = 'gpu_hist'
        params['learning_rate'] = learning_rate
        params['max_depth'] = max_depth
        params['early_stopping_rounds'] = 100
        params['objective'] = 'reg:squarederror'

        dtrain = xgb.DMatrix(self.trainset, label=self.trainlab)
        np.random.seed(12)
        print('training model and adding to ensemble')
        bst = xgb.train(params, dtrain, num_round)
        self.model_list.append(bst)
        return None

    def save_model(self,path='/home/danny/data/xgb_ensemble_spread.pkl'):
        import pickle
        output = open(path, 'wb')
        # Pickle dictionary using protocol 0.
        pickle.dump(self.model_list, output)

    def predict_validation(self,testset):
        import pickle
        pkl_file = open('/home/danny/data/xgb_ensemble_spread.pkl', 'rb')
        model_list = pickle.load(pkl_file)
        dtest = xgb.DMatrix(testset)
        complete_predictions = pd.DataFrame()
        for m in range(0,len(model_list)):
            print(m)
            current_bst = model_list[m]
            predictions = current_bst.predict(dtest)
            pred_df = pd.DataFrame([predictions, test_labels]).transpose()
            pred_df.columns = ['predictions', 'labels']
            complete_predictions = pd.concat([complete_predictions,pred_df],axis=1)
            print(mean_squared_error(pred_df['labels'], pred_df['predictions'], squared=False))
            print(mean_absolute_error(pred_df['labels'], pred_df['predictions']))
        return complete_predictions

ensemble_model = trainRegressionModels(trainset,trainlab)
ensemble_model.train_individual_model()
ensemble_model.train_individual_model(learning_rate=.01,max_depth=4,num_round=1200)
ensemble_model.train_individual_model(learning_rate=.01,max_depth=6,num_round=1200)
ensemble_model.train_individual_model(learning_rate=.01,max_depth=7,num_round=1200)
ensemble_model.train_individual_model(learning_rate=.01,max_depth=3,num_round=1200)
ensemble_model.train_individual_model(learning_rate=.1,max_depth=2,num_round=400)



ensemble_model.save_model()


finaldf = ensemble_model.predict_validation(testset)
preds = finaldf.loc[:,finaldf.columns.str.contains('predictions')].copy()
preds['composite'] = preds.mean(axis=1)
preds['label'] = finaldf.iloc[:,1]

print(mean_squared_error(preds['label'], preds['composite'], squared=False))
print(mean_absolute_error(preds['label'], preds['composite']))
#bench
# 14.133108750969141
# 11.147844716005732
# 11.13


#%%

from common_functions.utils import overunder_driver
from common_functions.game_day import games_of_day
from itertools import chain

inputlist = games_of_day()
print(inputlist)


def get_games(away_team, home_team):
    # away_team = 'MIN'
    # home_team = 'LAL'
    game_date = pd.to_datetime('today')
    away_stats = getTeamStats(away_team, game_date)
    home_stats = getTeamStats(home_team, game_date)

    home_row = home_stats.to_numpy().flatten('F')
    away_row = away_stats.to_numpy().flatten('F')

    both_row = np.concatenate((home_row, away_row))
    both_row = both_row.reshape(1,-1)

    import pickle
    pkl_file = open('/home/danny/data/xgb_ensemble_spread.pkl', 'rb')
    model_list = pickle.load(pkl_file)
    dtest = xgb.DMatrix(both_row)
    complete_predictions = []
    for m in range(0, len(model_list)):
        print(m)
        current_bst = model_list[m]
        predictions = current_bst.predict(dtest)[0]
        complete_predictions.append(predictions)

    composite_prediction = np.mean(complete_predictions)
    composite_variance = np.std(complete_predictions)

    return composite_prediction, composite_variance

def get_output_row(away,home):
    c_game,c_game_variance = get_games(away, home)

    c_row = pd.DataFrame([c_away,c_home,c_game,c_game_variance]).T
    return c_row

outdf = pd.DataFrame()
for i in inputlist:
    c_away = i[0]
    c_home = i[1]
    current_row = get_output_row(c_away,c_home)
    outdf = pd.concat([outdf,current_row])

outdf.columns = ['away','home','spread','standard dev']
outdf['t-stat'] = outdf['spread'] / outdf['standard dev']