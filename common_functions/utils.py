from multiprocessing import Pool
import pandas as pd

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
    current_stats_df['PIE'] = (current_stats_df['PTS']+current_stats_df['FGM']+current_stats_df['FTM']-current_stats_df['FTA']+current_stats_df['DREB']+
                               current_stats_df['OREB']+current_stats_df['AST']+current_stats_df['STL']+current_stats_df['BLK']-current_stats_df['PF']-current_stats_df['TOV'])
    current_stats_df = current_stats_df.ewm(halflife=4).mean()
    return current_stats_df

# class complete_dataset(df1)

class DataObject:
    def __init__(self, inputdf):
        self.df1 = inputdf

    def __getTeamStats__(self,abv,latestdate):
        team_subset = self.df1[self.df1['TEAM_ABBREVIATION'] == abv].copy()
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
            home_df = self.__getTeamStats__(home_team[0], game_date)
            away_df = self.__getTeamStats__(away_team[0], game_date)
            normalized_hdf = (home_df - home_df.min()) / (home_df.max() - home_df.min())
            normalized_adf = (away_df - away_df.min()) / (away_df.max() - away_df.min())
            if home_df.shape == (11, 19) and away_df.shape == (11, 19):
                output = [target_game_date, spread, home_df, away_df]
            else:
                return None
        except:
            return None
        return output

    def get_optimization(self,label_function='spread'):
        if label_function == 'spread':
            in_func = self.__getSpread__
        all_games_ids = self.df1['GAME_ID'].unique()
        pool = Pool(22)
        optimization_result = pool.map(in_func, all_games_ids)  ##When iterated over produces [2, 3, 4]
        pool.close()
        pool.join()
        return optimization_result

    def get_latest_stats(self,abv, latestdate):
        teamsubset = self.df1
        gamedf = teamsubset[teamsubset['TEAM_ABBREVIATION'] == abv]
        team_subset = gamedf.copy()
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

    def get_team_list(self):
        subset = self.df1[self.df1['GAME_DATE'] > '2020-01-01'].copy()
        team_list = pd.DataFrame(subset.loc[:, ['TEAM_ABBREVIATION', 'TEAM_NAME']].drop_duplicates())
        team_list.sort_values('TEAM_NAME',inplace=True)
        return team_list