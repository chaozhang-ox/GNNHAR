"""
Subsample the minutely data to 5 minutes and merge the data of all stocks in the stock list.
The output data is of the shape (T, N), where T is the number of 5-minute intervals for all trading days and N is the number of stocks.
"""

import argparse
import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
from multiprocessing import cpu_count, Pool
from joblib import dump
from os.path import join
import os
from datetime import datetime
from sklearn import preprocessing
import time

# stock list
stocks_l = ['AAPL', 'MSFT', 'AMZN', 'FB', 'BRK.B', 'JPM', 'GOOG', 'GOOGL', 'JNJ', 'V', 'PG', 'XOM', 'BAC', 'T',
                  'UNH', 'DIS', 'MA', 'INTC', 'VZ', 'HD', 'MRK', 'CVX', 'WFC', 'PFE', 'KO', 'CMCSA', 'CSCO', 'PEP',
                  'BA', 'C', 'WMT', 'ADBE', 'MDT', 'ABT', 'MCD', 'BMY', 'AMGN', 'CRM', 'NVDA', 'PM', 'NFLX', 'ABBV',
                  'ACN', 'COST', 'PYPL', 'TMO', 'AVGO', 'HON', 'UNP', 'NKE', 'UTX', 'ORCL', 'IBM', 'TXN', 'NEE', 'LIN',
                  'SBUX', 'LLY', 'QCOM', 'MMM', 'GE', 'CVS', 'DHR', 'LMT', 'AMT', 'MO', 'LOW', 'USB', 'BKNG', 'AXP',
                  'FIS', 'GILD', 'UPS', 'CAT', 'MDLZ', 'CHTR', 'TFC', 'ANTM', 'GS', 'CI', 'TJX', 'ADP', 'BDX', 'CME',
                  'CB', 'PNC', 'COP', 'INTU', 'ISRG', 'D', 'SPGI', 'FISV', 'DUK', 'SYK', 'SO', 'TGT', 'MS', 'BSX', 'AGN', 'RTN']
stocks_l.sort()


def logret_data(data):
    data['Price'] = (data['ask_1'] + data['bid_1']) / 2
    data['ret'] = np.log(data['Price']).diff()
    return data[['time', 'ret']][1:]


# sanity check for the data and save the good data
def data_sanity(path, ticker):
    ticker_path = join(path, 'LOBData', ticker)
    files = os.listdir(ticker_path)
    files = [i for i in files if i.endswith('.csv')]
    files.sort()

    data_l = []

    for file in files:
        date = file.split('_')[1]
        data = pd.read_csv(join(ticker_path, file))
        # 07-03, 12-24 are half trading days, removed from the data
        if len(data) == 391:
            if '07-03' in date or '12-24' in date:
                pass
            elif (np.abs(data['ask_1'] / data['bid_1'] - 1) < 0.5).all():
                ret_data = logret_data(data)
                ret_data['date'] = date
                data_l.append(ret_data)
            # if the spread is too large, print the data to check
            else:
                a = data['ask_1'] / data['bid_1']
                idx = a.argmax()
                print(data[idx-5:idx+5])
                print(data[-10:])
                ret_data = logret_data(data)
                ret_data['date'] = date
                data_l.append(ret_data)
        else:
            print('- ' * 10 + date + ' Missing')
            print(len(data))

    all_data = pd.concat(data_l)
    all_data.to_csv(join(path, 'Minute_Data', ticker+'.csv'), index=False)
    return all_data


# merge the minute data of all stocks in the stock list
def merge_data(path):
    data_l = []
    tickers = os.listdir(join(path, 'Minute_Data'))
    tickers.sort()
    for i in tickers:
        ticker = i.split('.csv')[0]
        print(ticker)
        all_data = pd.read_csv(join(path, 'Minute_Data', ticker+'.csv'))
        print(all_data.shape)
        month_l = list(set([i[:7] for i in all_data['date']]))
        month_l.sort()
        print(len(month_l))
        all_data.rename(columns={'ret':ticker}, inplace=True)
        data_l.append(all_data[['date', 'time', ticker]])

    from functools import reduce
    df = reduce(lambda df1, df2: pd.merge(df1, df2, on=['date', 'time'], how='outer'), data_l)
    df.rename(columns={'date':'Date', 'time':'Time'}, inplace=True)
    print(df)
    os.makedirs(join(path, 'Data'), exist_ok=True)
    df.to_csv(join(path, 'Data', 'data_1min.csv'), index=False)


# subsample the data to 5 minutes
def data_subsample(path):
    df = pd.read_csv(join(path, 'Data', 'data_1min.csv'))
    df_gb = df.groupby(by=df.index // 5)
    df_5min = df_gb.sum(min_count=1)
    df_5min['Date'] = df['Date'].to_list()[4::5]
    df_5min['Time'] = df['Time'].to_list()[4::5]
    clms = [i for i in df_5min.columns if i not in ['Date', 'Time']]
    clms.sort()
    df_5min = df_5min[['Date', 'Time'] + clms]
    os.makedirs(join(path, 'Data'), exist_ok=True)
    df_5min.to_csv(join(path, 'Data', 'data_5min.csv'), index=False)


if __name__ == '__main__':
    path = 'your local path for storing minutely LOBSTER data and processed data'
    # under the path, the code will create 3 folders: 
    # 1. LOBSTER: the raw LOBSTER data, 
    # 2. Minute_Data: the sanity checked minutely return data for each stock separately,
    # 3. Data: processed panel data, like 1-min, 5-min returns, realized variance

    for ticker in stocks_l:
        print(' * ' * 20 + ticker + ' * ' * 20)
        all_data = data_sanity(path, ticker)
        print(all_data)

    merge_data(path)
    data_subsample(path)