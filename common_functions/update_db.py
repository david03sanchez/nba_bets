import pandas as pd
from nba_api.stats.static import teams
import sys
env = 'linux'
#%%
if env == 'mac':
    root_data_dir = '/Users/danny/nba_bets/data/'
elif env == 'linux':
    root_data_dir = '/home/danny/nba/data/'  # todo move the models to this directory on linux
    sys.path.append('/home/danny/nba_bets')
from common_functions.utils import getTeamDF

#%%


all_games = pd.DataFrame()
nba_teams_table = pd.DataFrame(teams.get_teams())
all_abs = nba_teams_table['abbreviation'].unique()

for abv in all_abs:
    print(abv)
    gamedf = getTeamDF(abv)
    all_games = pd.concat([all_games,gamedf],axis=0)


all_games.to_csv(root_data_dir + 'gamedf.csv')
