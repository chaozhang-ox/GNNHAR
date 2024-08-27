"""
Compute the daily variance from 5-min return data
Compute the variance data for multi-horizon and various universes
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats.mstats import winsorize
from sklearn.linear_model import HuberRegressor, LinearRegression, LassoCV, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from multiprocessing import cpu_count, Pool
from joblib import dump
from os.path import join
import os
from datetime import datetime
from sklearn import preprocessing
from numpy import linalg as LA
import scipy

DJIA_stocks_l = ['MMM', 'AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 'HD', 'HON', 'IBM', 'GS', 'NKE',
                 'INTC', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'PG', 'CRM', 'TRV', 'UNH', 'VZ', 'WMT']
DJIA_stocks_l.sort()

SP100_stocks_l = ['AAPL', 'ABT', 'ACN', 'ADBE', 'ADP', 'AMGN', 'AMT', 'AMZN', 'AXP', 'BA', 'BAC', 'BDX', 'BMY',
                 'BSX', 'C', 'CAT', 'CB', 'CI', 'CMCSA', 'CME', 'COP', 'COST', 'CRM', 'CSCO', 'CVS', 'CVX', 'D',
                 'DHR', 'DIS', 'DUK', 'FIS', 'FISV', 'GE', 'GILD', 'GOOG', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'INTU',
                 'ISRG', 'JNJ', 'JPM', 'KO', 'LLY', 'LMT', 'LOW', 'MA', 'MCD', 'MDT', 'MMM', 'MO', 'MRK', 'MS',
                 'MSFT', 'NFLX', 'NKE', 'NVDA', 'ORCL', 'PEP', 'PFE', 'PG', 'PNC', 'QCOM', 'SBUX', 'SO', 'SYK',
                 'T', 'TGT', 'TJX', 'TMO', 'TXN', 'UNH', 'UNP', 'UPS', 'USB', 'VZ', 'WFC', 'WMT']
SP100_stocks_l.sort()


data_name_dic = {'DJIA': DJIA_stocks_l, 'SP100': SP100_stocks_l}

def load_data(path):
    ret_data = pd.read_csv(join(path, 'Data', 'data_5min.csv'))

    stocks_l = [i for i in ret_data.columns if i not in ['Date', 'Time']]
    ret_data[stocks_l] *= 100

    # winsorize the data to avoid measurement errors in LOBSTER
    up = 99.5
    low = 0.5
    for clm in ret_data.columns:
        if clm not in ['Date', 'Time']:
            max_p = np.nanpercentile(ret_data[clm], up)
            min_p = np.nanpercentile(ret_data[clm], low)

            ret_data.loc[ret_data[clm] > max_p, clm] = max_p
            ret_data.loc[ret_data[clm] < min_p, clm] = min_p

    return ret_data


# compute the variance of the data
def compute_variance(sub_data):
    stocks_l = [i for i in sub_data.columns if i not in ['Date', 'Time']]
    sq_data = sub_data[stocks_l] ** 2
    var_sum = sq_data.sum(min_count=1)
    var_sum = pd.DataFrame(var_sum).T
    return var_sum


# compute the variance for different horizons and universes
def Compute_Horizon(path, univese, ret_vol, horizon):
    if ret_vol == 'ret':
        daily_var_data = pd.read_csv(join(path, 'Data', 'daily_return.csv'), index_col=0)
    elif ret_vol == 'var':
        daily_var_data = pd.read_csv(join(path, 'Data', 'daily_variance.csv'), index_col=0)
    else:
        print('Please choose ret or var')
        return
        
    var_data = 0
    for i in range(horizon):
        var_data += daily_var_data.shift(-i)
    var_data.dropna(inplace=True)
    var_univ = var_data[data_name_dic[univese]]
    var_univ.to_csv(join(path, 'Data', f'{univese}_{ret_vol}_FH{horizon}.csv'))


if __name__ == '__main__':
    path = 'your_local_path'

    ret_data = load_data(path)
    stocks_l = [i for i in ret_data.columns if i not in ['Date', 'Time']]
    date_l = list(set(ret_data['Date'].tolist()))
    date_l.sort()

    ### Compute daily return
    daily_return_data = ret_data.groupby(by='Date').sum(min_count=1)
    daily_return_data.index = list(daily_return_data.index)
    daily_return_data.to_csv(join(path, 'Data', 'daily_return.csv'))
    
    ### Compute daily variance
    var_df = ret_data.groupby(by='Date').apply(compute_variance)
    var_df.index = date_l
    
    var_df.columns = stocks_l
    var_df.to_csv(join(path, 'Data', 'daily_variance.csv'))
    
    ### Compute variance over different horizons
    horizon = 5
    for name in ['DJIA30', 'SP100']:
        Compute_Horizon(path, name, 'ret', horizon)
        Compute_Horizon(path, name, 'var', horizon)