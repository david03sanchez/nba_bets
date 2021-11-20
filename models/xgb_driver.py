import xgboost as xgb
import pandas as pd
import numpy as np
from common_functions.utils import model_driver


# df1 = pd.read_csv('/Users/danny/nba_bets/data/gamedf.csv', index_col=0)
# subset = df1.loc[df1['TEAM_ABBREVIATION'] == 'GSW',:]
inputlist = [('HOU','NYK'),('MIA','WAS'),('NOP','IND'),
             ('OKC','BOS'),('CHA','ATL'),('ORL','MIL'),
             ('MEM', 'MIN'), ('PHI', 'POR'), ('UTA', 'SAC')
             ]
md = model_driver(env = 'linux')
team_list = md.get_team_list()

game_df = md.get_df(inputlist)
game_df['spread'] = game_df['spread']
over_under_df = game_df.loc[:,['away_team','home_team','over_under','over_under_inverse']]
ar_list, home_df, away_df = md.verify_game('CHA','ATL')

#home team odds + spread
## if negative go with away team
## if positive go with home team
