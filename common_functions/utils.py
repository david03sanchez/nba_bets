from multiprocessing import Pool
import pandas as pd
from itertools import chain
import pandas as pd
import numpy as np
import xgboost as xgb

# helper functions

def getTeamDF(abbreviation):
    from nba_api.stats.endpoints import leaguegamefinder
    from nba_api.stats.static import teams
    all_nba_teams = pd.DataFrame(teams.get_teams())
    relevant_id = all_nba_teams[all_nba_teams['abbreviation'] == abbreviation]['id'].tolist()
    gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=relevant_id)
    games_dict = gamefinder.get_normalized_dict()
    games_df = pd.DataFrame(games_dict['LeagueGameFinderResults'])
    return games_df

def transformStats(current_stats_df):
    base_points = current_stats_df['PTS']
    current_stats_df['PIE'] = (current_stats_df['PTS']+current_stats_df['FGM']+current_stats_df['FTM']-current_stats_df['FTA']+current_stats_df['DREB']+
                               current_stats_df['OREB']+current_stats_df['AST']+current_stats_df['STL']+current_stats_df['BLK']-current_stats_df['PF']-current_stats_df['TOV'])
    current_stats_df = current_stats_df.ewm(halflife=4).mean()
    current_stats_df['CORE_PTS'] = base_points
    return current_stats_df

#game object: receives the fully downloaded csv with all games

class DataObject:
    def __init__(self, inputdf):
        self.df1 = inputdf

    def getTeamStats(self,abv,latestdate):
        team_subset = self.df1[self.df1['TEAM_ABBREVIATION'] == abv].copy()
        # team_subset = df1[df1['TEAM_ABBREVIATION'] == 'ATL'].copy()
        # latestdate = '2000-01-19'
        team_subset['GAME_DATE'] = pd.to_datetime(team_subset['GAME_DATE'])
        team_subset.index = team_subset['GAME_DATE']
        team_subset.sort_index(inplace=True, ascending=False)
        colnames = team_subset.columns
        stats_columns = ['PTS', 'FGM', 'FGA', 'FG_PCT',
                         'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB',
                         'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF']
        date_subset = team_subset[team_subset['GAME_DATE'] < latestdate]

        current_stats = date_subset.iloc[0:11, [date_subset.columns.get_loc(c) for c in stats_columns]]
        game_data = date_subset.iloc[0:11, [date_subset.columns.get_loc(c) for c in ['WL', 'TEAM_ID', 'GAME_ID']]]
        game_data['Key'] = game_data['TEAM_ID'].astype(str) + "-" + game_data['GAME_ID'].astype(str)
        current_stats = transformStats(current_stats)
        return current_stats

    def __getSpread__(self,gameid):
        try:
            target_game = self.df1[self.df1['GAME_ID'] == gameid]  # contains target
            # target_game = df1[df1['GAME_ID'] == 29900545] #contains target
            if target_game.shape[0] != 2:
                return None
            relevant_teams = target_game['TEAM_ABBREVIATION'].tolist()
            match_location_away = target_game.loc[target_game['MATCHUP'].str.contains('@')]
            match_location_home = target_game.loc[~target_game['MATCHUP'].str.contains('@')]
            target_game_date = match_location_home['GAME_DATE']
            # match_outcome_home = np.where(match_location_away['WL'] == 'W',0,1) #0 if away team wins
            spread = match_location_home.iloc[0, match_location_home.columns.get_loc('PTS')] - \
                     match_location_away.iloc[0, match_location_away.columns.get_loc('PTS')]
            game_date = match_location_away['GAME_DATE'].values[0]
            home_team = match_location_away['MATCHUP'].str.extract(r'((?<=@.)\S{3})')[0].tolist()
            away_team = [x for x in relevant_teams if x not in home_team]
            home_df = self.getTeamStats(home_team[0], game_date)
            away_df = self.getTeamStats(away_team[0], game_date)
            normalized_hdf = (home_df - home_df.min()) / (home_df.max() - home_df.min())
            normalized_adf = (away_df - away_df.min()) / (away_df.max() - away_df.min())
            if home_df.shape == (11, 20) and away_df.shape == (11, 20):
                output = [target_game_date, spread, home_df, away_df]
            else:
                return None
        except:
            return None
        return output

    def __getOverUnder__(self,gameid):
        try:
            target_game = self.df1[self.df1['GAME_ID'] == gameid]  # contains target
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
            home_df = self.getTeamStats(home_team[0], game_date)
            away_df = self.getTeamStats(away_team[0], game_date)
            normalized_hdf = (home_df - home_df.min()) / (home_df.max() - home_df.min())
            normalized_adf = (away_df - away_df.min()) / (away_df.max() - away_df.min())
            if home_df.shape == (11, 20) and away_df.shape == (11, 20):
                output = [target_game_date, spread, home_df, away_df]
            else:
                return None
        except:
            return None
        return output

    def get_optimization(self,label_function='spread'):
        if label_function == 'spread':
            in_func = self.__getSpread__
        elif label_function == 'over_under':
            in_func = self.__getOverUnder__
        all_games_ids = self.df1['GAME_ID'].unique()
        pool = Pool(22)
        optimization_result = pool.map(in_func, all_games_ids)  ##When iterated over produces [2, 3, 4]
        pool.close()
        pool.join()
        return optimization_result


    def get_team_list(self):
        subset = self.df1[self.df1['GAME_DATE'] > '2020-01-01'].copy()
        team_list = pd.DataFrame(subset.loc[:, ['TEAM_ABBREVIATION', 'TEAM_NAME']].drop_duplicates())
        team_list.sort_values('TEAM_NAME',inplace=True)
        return team_list

class model_driver:
    def __init__(self, game_date=pd.to_datetime('today')):
        self.game_date = game_date
        self.bst_spread = xgb.Booster()
        self.bst_spread.load_model('/home/danny/nba/spreadmodel.bst')
        self.bst_overunder = xgb.Booster()
        self.bst_overunder.load_model('/home/danny/nba/overundermodel.bst')
        df1 = pd.read_csv('/home/danny/nba_bets/data/gamedf.csv', index_col=0)
        self.scoring_object = DataObject(df1)

    def get_games(self,away_team,home_team):
        self.away_stats = self.scoring_object.getTeamStats(away_team,self.game_date)
        self.home_stats = self.scoring_object.getTeamStats(home_team,self.game_date)
        home_stats_flat = self.home_stats.to_numpy().reshape(1, -1)
        away_stats_flat = self.away_stats.to_numpy().reshape(1, -1)

        score_row = np.concatenate((home_stats_flat, away_stats_flat), axis=1)
        score_row_inverse = np.concatenate((away_stats_flat, home_stats_flat), axis=1)
        spread_val = self.bst_spread.predict(xgb.DMatrix(score_row))
        spread_inverse = self.bst_spread.predict(xgb.DMatrix(score_row_inverse))
        over_under_val = self.bst_overunder.predict(xgb.DMatrix(score_row))
        over_under_inverse = self.bst_overunder.predict(xgb.DMatrix(score_row_inverse))
        out_list = [spread_val,spread_inverse,over_under_val,over_under_inverse]
        return out_list

    def get_df(self,inputlist):
        self.inputlist = inputlist
        outlist = []
        for i in inputlist:
            c_away = i[0]
            c_home = i[1]
            c_game = list(chain.from_iterable(self.get_games(c_away,c_home)))
            c_pair = [c_away,c_home]
            c_pair.extend(c_game)
            outrow = c_pair
            outlist.append(outrow)
        outdf = pd.DataFrame(outlist)
        outdf.columns = ['away_team','home_team','spread',
                         'spread_inverse','over_under','over_under_inverse']
        return outdf
    def get_team_list(self):
        return self.scoring_object.get_team_list()
