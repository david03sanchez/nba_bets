import pandas as pd
from nba_api.stats.static import teams
import sys
from multiprocessing import get_context
env = 'linux'
#%%
if env == 'mac':
    root_data_dir = '/Users/danny/nba_bets/data/'
elif env == 'linux':
    root_data_dir = '/home/data/'
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

df1 = pd.read_csv('/home/danny/nba/data/gamedf.csv',index_col = 0)


def get_ranking(input_tup):
    abv = input_tup[0]
    index_date = input_tup[1]
    try:
        # abv = 'ATL'
        # index_date = '2003-12-13'
        window_start_date = pd.to_datetime(index_date) - pd.DateOffset(days=60)
        points_df = df1[
            (pd.to_datetime(df1['GAME_DATE']) > window_start_date) & (pd.to_datetime(df1['GAME_DATE']) < index_date)].copy()
        offense_df = points_df.groupby('TEAM_ABBREVIATION').sum()[['PTS', 'AST']]
        offense_df.sort_values('PTS', ascending=False, inplace=True)
        offense_df.reset_index(inplace=True, drop=False)
        offense_df['PTS_Rank'] = range(1, offense_df.shape[0] + 1)
        offense_df.sort_values('AST', ascending=False, inplace=True)
        offense_df['AST_Rank'] = range(1, offense_df.shape[0] + 1)
        relevant_rank = offense_df[offense_df['TEAM_ABBREVIATION'] == abv][['PTS_Rank', 'AST_Rank']]
        relevant_rank['TEAM_ABBREVIATION'] = [abv]
        relevant_rank['Index_Date'] = [index_date]
        relevant_rank.reset_index(inplace=True, drop=True)
        return relevant_rank
    except:
        relevant_rank = pd.DataFrame(columns=['PTS_Rank','AST_Rank'],index=[index_date]).fillna(0)
        relevant_rank['TEAM_ABBREVIATION'] = [abv]
        relevant_rank['Index_Date'] = [index_date]
        relevant_rank.reset_index(inplace=True,drop=True)
        return

all_pairs = df1[['TEAM_ABBREVIATION','GAME_DATE']].drop_duplicates()
pair_tuples = tuple(zip(all_pairs['TEAM_ABBREVIATION'],all_pairs['GAME_DATE']))
get_ranking(pair_tuples[98324])

pool = get_context("fork").Pool(22) #change to number of cores on machine
optimization_result = pool.map(get_ranking,pair_tuples)
pool.close()
complete_dataset = pd.concat(optimization_result)
complete_dataset.reset_index(drop=True,inplace=True)
complete_dataset.to_csv('/home/danny/nba/data/ranking_map.csv')