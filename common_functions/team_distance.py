# this code was used to generate a distance map of all nba teams' arenas referred to as distance_map.csv
import pandas as pd
import numpy as np
import googlemaps
from itertools import combinations

df1 = pd.read_csv('/home/data/gamedf.csv', index_col=0)
df1['GAME_DATE'] = pd.to_datetime(df1['GAME_DATE'])
team_names = df1[df1['GAME_DATE'] > '12-31-2020']['TEAM_NAME'].unique()
team_abv = df1[df1['GAME_DATE'] > '12-31-2020']['TEAM_ABBREVIATION'].unique()
team_addy = ['1 State Farm Dr, Atlanta, GA 30303', '100 Legends Wy, Boston, MA 02114', '1 Center Court, Cleveland, OH 44115',
       '1501 Dave Dixon Dr, New Orleans, LA 70113', '1901 W Madison St, Chicago, IL 60612', '2500 Victory Ave, Dallas, TX 75219',
       '1000 Chopper Cir, Denver, CO 80204', '1 Warriors Way, San Francisco, CA 94158', '1510 Polk St, Houston, TX 77002',
       '1111 S Figueroa St, Los Angeles, CA 90015', '1111 S Figueroa St, Los Angeles, CA 90015', '601 Biscayne Blvd, Miami, FL 33132',
       '1111 Vel R. Phillips Ave, Milwaukee, WI 53203', '600 N 1st Ave, Minneapolis, MN 55403', '620 Atlantic Ave, Brooklyn, NY 11217',
       '4 Pennsylvania Plaza, New York, NY 10001', '400 W Church St Suite 200, Orlando, FL 32801', '125 S Pennsylvania St, Indianapolis, IN 46204',
       '3601 S Broad St, Philadelphia, PA 19148', '201 E Jefferson St, Phoenix, AZ 85004', '1 N Center Ct St, Portland, OR 97227',
       '500 David J Stern Walk, Sacramento, CA 95814', '1 AT&T Center Parkway, San Antonio, TX 78219', '100 W Reno Ave, Oklahoma City, OK 73102',
       '40 Bay St., Toronto, ON M5J 2X2, Canada', '301 S Temple, Salt Lake City, UT 84101', '191 Beale St, Memphis, TN 38103',
       '601 F St NW, Washington, DC 20004', '2645 Woodward Ave, Detroit, MI 48201', '333 E Trade St, Charlotte, NC 28202']

current_teams = pd.DataFrame(data={'TEAM_NAME': team_names, 'TEAM_ABBREVIATION': team_abv, 'TEAM_ADDY': team_addy})
combination_list = list(combinations(current_teams['TEAM_ABBREVIATION'], 2))
combination_df = pd.DataFrame(combination_list, columns=['team1', 'team2'])
combination_df['distance'] = np.nan


def get_distance(abv1, abv2):
    addy1 = current_teams.loc[current_teams['TEAM_ABBREVIATION'] == abv1, :].iloc[0, 2]
    addy2 = current_teams.loc[current_teams['TEAM_ABBREVIATION'] == abv2, :].iloc[0, 2]
    gmaps = googlemaps.Client(key="AIzaSyD3JHU1gaLCSzAUxtfzl_xoSW6cPJ8FGTU")
    json_response = gmaps.distance_matrix(addy1, addy2)
    raw_string = json_response['rows'][0]['elements'][0]['distance']['text'].replace(' ', '')
    float_string = ''.join([x for x in raw_string if x.isalpha() is False and x != ','])
    distance_miles = float(float_string) * 0.62137
    return distance_miles


for r in range(0, combination_df.shape[0]):
    print(r)
    team1 = combination_df.iloc[r, 0]
    team2 = combination_df.iloc[r, 1]
    c_distance = get_distance(team1, team2)
    combination_df.iloc[r, -1] = c_distance

combination_df.to_csv('/home/david/nba_bets/common_functions/distance_map.csv')
