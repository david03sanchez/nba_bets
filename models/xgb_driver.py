import xgboost as xgb
import pandas as pd
import numpy as np
from common_functions.utils import model_driver


inputlist = [('MIL','PHI'),('ATL','UTA'),('POR','LAC')]\
    # ,
    #          ('NOP','DAL'),('MIA','DEN'),('PHX','SAC'),
    #          ('ATL','GSW'),('CHA','LAL')]
md = model_driver()
team_list = md.get_team_list()
game_df = md.get_df(inputlist)

