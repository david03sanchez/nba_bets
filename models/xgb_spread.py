import pandas as pd
import numpy as np
from common_functions.utils import DataObject
env = 'linux'
#%%
if env == 'mac':
    root_data_dir = '/Users/danny/nba_bets/data/'
elif env == 'linux':
    root_data_dir = '/home/danny/nba/data/'

#read in data
df1 = pd.read_csv(root_data_dir + 'gamedf.csv',index_col = 0)
dataset_object = DataObject(df1)
optimization_result = dataset_object.get_optimization(label_function='spread')

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
        home_row = complete_dataset[r][2].to_numpy().flatten()
        away_row = complete_dataset[r][3].to_numpy().flatten()
        both_row = np.concatenate((home_row,away_row))
        train_features.append(both_row)
    else:
        test_labels.append(complete_dataset[r][1])
        home_row = complete_dataset[r][2].to_numpy().flatten()
        away_row = complete_dataset[r][3].to_numpy().flatten()
        both_row = np.concatenate((home_row,away_row))
        test_features.append(both_row)

train_labels = np.array(train_labels)
train_features = np.array(train_features)
test_labels = np.array(test_labels)
test_features = np.array(test_features)


#%%
import xgboost as xgb
from sklearn.metrics import mean_squared_error

dtrain = xgb.DMatrix(train_features, label=train_labels)

param = {'max_depth': 3, 'eta': .1, 'objective': 'reg:squarederror'}
param['tree_method'] = 'gpu_hist'
param['eval_metric'] = 'mae'

num_round = 200
bst = xgb.train(param, dtrain, num_round)

dtest = xgb.DMatrix(test_features)
predictions = bst.predict(dtest)
pred_df = pd.DataFrame([predictions,test_labels]).transpose()
pred_df.columns = ['predictions','labels']
pred_df['correct_call'] = np.where(np.sign(pred_df['predictions']) == np.sign(pred_df['labels']), 1, 0)
mean_squared_error(predictions, test_labels)**.5
pred_df['correct_call'].mean()
pred_df['const'] = 1
bst.save_model(root_data_dir + 'spreadmodel.bst')

import statsmodels.api as sm
model = sm.OLS(pred_df['labels'], pred_df[['predictions','const']])
results = model.fit()
print(results.summary())


#%%
import xgboost as xgb
# bst = xgb.Booster()
# bst.load_model('/home/danny/nba/overundermodel.bst')
df1 = pd.read_csv(root_data_dir + 'gamedf.csv',index_col = 0)
scoring_object = DataObject(df1)
team_list = scoring_object.get_team_list()



home_stats = scoring_object.getTeamStats('MIL','2021-11-09')
away_stats = scoring_object.getTeamStats('PHI','2021-11-09')

home_stats_flat = home_stats.to_numpy().reshape(1,-1)
away_stats_flat = away_stats.to_numpy().reshape(1,-1)

score_row = np.concatenate((home_stats_flat,away_stats_flat),axis=1)
score_row_inverse = np.concatenate((away_stats_flat,home_stats_flat),axis=1)
bst.predict(xgb.DMatrix(score_row))
bst.predict(xgb.DMatrix(score_row_inverse))
