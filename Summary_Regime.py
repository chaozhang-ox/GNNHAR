"""
Summarize the results of the forecast models, based on different regimes.
We split the testing period into 2 sub-periods, based on the volatility of the SPY. 
"""

import os
from os.path import *
from MCS import *

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

path = 'your_local_path'
sum_path = join(path, 'Var_Results_Sum')


def load_RV_SPY(horizon):
    spy_rv_df = pd.read_csv(join(path, 'Data', f'SPY_var_FH{horizon}.csv'), index_col=0)
    return spy_rv_df


# split the testing period into 2 sub-periods, based on the volatility of the SPY
# q is the percentile of the SPY volatility, default is 90%. 
# In other words, we consider the high-volatility period as the top 10% of the SPY volatility, and the rest as the low-volatility period.
def split_period(spy_rv_df, test_pred_df, q=90.0):
    spy_rv_df = spy_rv_df.loc[test_pred_df.index]
    perc_choice = np.percentile(spy_rv_df['SPY'], q)
    low_vol_dates = spy_rv_df[spy_rv_df['SPY'] < perc_choice].index.tolist()
    high_vol_dates = spy_rv_df[spy_rv_df['SPY'] >= perc_choice].index.tolist()
    return low_vol_dates, high_vol_dates


def load_data(universe, horizon):
    var_df = pd.read_csv(join(path, 'Data', f'{universe}_var_FH{horizon}.csv'), index_col=0)
    var_df.fillna(method="ffill", inplace=True)
    vech_df = var_df[var_df.index <= '2021-07-01']
    vech_df = vech_df.sort_index(axis=1)
    return vech_df


def QLIKE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)


def Loss(vech_df, test_pred_df, spy_rv_df):
    low_vol_dates, high_vol_dates = split_period(spy_rv_df, test_pred_df)
    test_df = vech_df.loc[test_pred_df.index]
    ticker_l = vech_df.columns.tolist()
    test_pred_df.columns = ticker_l
    all_df_l = []
    for date_l in [low_vol_dates, high_vol_dates]:
        df_l = []

        for ticker in ticker_l:
            y_true = test_df.loc[date_l, ticker].values
            y_pred = test_pred_df.loc[date_l, ticker].values
            assert (y_pred > 0).all()
            mse = mean_squared_error(y_true, y_pred)
            qlike = QLIKE(y_true, y_pred)

            df_l.append([np.round(mse, 4), np.round(qlike, 4)])

        df = pd.DataFrame(np.array(df_l), index=ticker_l, columns=['MSE', 'QLIKE'])
        all_df_l.append(df)

    return all_df_l


def Result(vech_df, version_name, universe, horizon):
    result_files = [i for i in files if
                    ('_pred' in i) and version_name in i and '_' + universe + '_' in i and f'F{horizon}' in i and 'W22' in i]

    result_files.sort()
    for (i, item) in enumerate(result_files):
        print(i, item)

    spy_rv_df = load_RV_SPY(horizon)

    E_low_l = []
    E_high_l = []
    Q_low_l = []
    Q_high_l = []
    files_l = []

    for filename in result_files:
        test_pred_df = pd.read_csv(join(sum_path, filename), index_col=0)
        test_pred_df = test_pred_df.sort_index(axis=1)
        # print(filename)
        # print(test_pred_df)
        test_pred_df[test_pred_df<=0] = np.nan
        test_pred_df.fillna(method="ffill", inplace=True)

        df_l = Loss(vech_df, test_pred_df, spy_rv_df)
        low_df, high_df = df_l

        E_low_l.append(low_df['MSE'])
        E_high_l.append(high_df['MSE'])

        Q_low_l.append(low_df['QLIKE'])
        Q_high_l.append(high_df['QLIKE'])

        file_key_name = filename.split('_')[2] + '_' + filename.split('_')[3]
        # file_key_name = filename
        files_l.append(file_key_name)

    E_df_low = pd.concat(E_low_l, axis=1)
    E_df_low.columns = files_l
    E_df_high = pd.concat(E_high_l, axis=1)
    E_df_high.columns = files_l

    Q_df_low = pd.concat(Q_low_l, axis=1)
    Q_df_low.columns = files_l
    Q_df_high = pd.concat(Q_high_l, axis=1)
    Q_df_high.columns = files_l
    return E_df_low, E_df_high, Q_df_low, Q_df_high


def norm_loss(df):
    return df.apply(lambda x: x/df['GHAR_iden'], axis=0)


def rank_MCS(loss_df, pval_df):
    loss_mean_df = loss_df.mean(0)
    rank_df = loss_mean_df.rank()
    pval_df = pd.DataFrame(pval_df, columns=['p-value'])
    pval_df['loss'] = loss_mean_df
    pval_df['ratio'] = loss_mean_df / loss_mean_df.loc['GHAR_iden']
    pval_df['rank'] = rank_df
    # model names, may need to modify according to the user's choices
    idx_l = ['GHAR_iden', 'GHAR_iden+glasso', 'MSE_GNNHAR1L', 'MSE_GNNHAR2L', 'MSE_GNNHAR3L', 'QLike_HAR', 'QLike_GHAR', 'QLike_GNNHAR1L', 'QLike_GNNHAR2L', 'QLike_GNNHAR3L']

    return pval_df.loc[idx_l, ['ratio', 'p-value']]


if __name__ == '__main__':
    horizon = 1
    universe = 'DJIA'

    vech_df = load_data(universe, horizon)
    files = os.listdir(sum_path)
    files.sort()

    E_df_low, E_df_high, Q_df_low, Q_df_high = Result(vech_df, 'Forecast_Var', universe, horizon)
    print(' * ' * 10 + '| MSE |' + ' * ' * 10)
    print(E_df_low.mean(0))
    print(E_df_high.mean(0))

    print(' * ' * 10 + '| QLIKE |' + ' * ' * 10)
    print(Q_df_low.mean(0))
    print(Q_df_high.mean(0))

    for E_df in [(E_df_low), (E_df_high)]:
        mcs_E = ModelConfidenceSet(E_df, 0.05, 10000, 2).run()
        sum_E = rank_MCS(E_df, mcs_E.pvalues)
        print(" * * * * * MCS of MSE * * * * * ")
        print(np.round(sum_E, 3))

    for Q_df in [(Q_df_low), (Q_df_high)]:
        mcs_Q = ModelConfidenceSet(Q_df, 0.05, 10000, 2).run()
        sum_Q = rank_MCS(Q_df, mcs_Q.pvalues)
        print(" * * * * * MCS of QLIKE * * * * * ")
        print(np.round(sum_Q, 3))