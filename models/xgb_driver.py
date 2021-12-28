import xgboost as xgb
import pandas as pd
import numpy as np
from common_functions.utils import overunder_driver




# df1 = pd.read_csv('/Users/danny/nba_bets/data/gamedf.csv', index_col=0)
# subset = df1.loc[df1['TEAM_ABBREVIATION'] == 'GSW',:]
inputlist = [('HOU','CHA'),('CHI','ATL'),('BOS','MIN'),
             ('UTA','SAS'),('MEM','PHX'),('DAL','POR'),('BKN','LAC')
             # ('UTA','MIN'),('DEN','NOP'),('BKN','HOU'),
             # ('ORL','SAC'),('POR','GSW'),('BOS','LAC')
             ]
md = overunder_driver(env = 'linux')
team_list = md.get_team_list()

game_df = md.get_df(inputlist)
over_under_df = game_df.loc[:,['away_team','home_team','over_under','over_under_inverse']]
ar_list, home_df, away_df = md.verify_game('HOU','CHA')

#home team odds + spread
## if negative go with away team

#todo add streak against each team per row plus current streak to values
#also add

