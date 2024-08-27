"""
Summarize the results of the forecast models, including the MSE, QLIKE, and the MCS tests.
"""

import os
from os.path import *
from MCS import *

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

path = 'your_local_path'
sum_path = join(path, 'Var_Results_Sum')


def load_data(universe, horizon):
    var_df = pd.read_csv(join(path, 'Data', f'{universe}_var_FH{horizon}.csv'), index_col=0)
    var_df.fillna(method="ffill", inplace=True)
    vech_df = var_df[var_df.index <= '2021-07-01']
    vech_df = vech_df.sort_index(axis=1)
    return vech_df


def QLIKE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)


def Loss(vech_df, test_pred_df):
    test_df = vech_df.loc[test_pred_df.index]
    ticker_l = vech_df.columns.tolist()
    test_pred_df.columns = ticker_l
    df_l = []

    for ticker in ticker_l:
        y_true = test_df[ticker].values
        y_pred = test_pred_df[ticker].values
        assert (y_pred > 0).all()
        mse = mean_squared_error(y_true, y_pred)
        qlike = QLIKE(y_true, y_pred)

        df_l.append([np.round(mse, 4), np.round(qlike, 4)])

    df = pd.DataFrame(np.array(df_l), index=ticker_l, columns=['MSE', 'QLIKE'])
    return df


def Result(vech_df, version_name, universe, horizon):
    result_files = [i for i in files if
                    ('_pred' in i) and version_name in i and '_' + universe + '_' in i and f'F{horizon}' in i and 'W22' in i]
    
    result_files.sort()
    for (i, item) in enumerate(result_files):
        print(i, item)

    E_df_l = []
    Q_df_l = []

    files_l = []
    for filename in result_files:
        test_pred_df = pd.read_csv(join(sum_path, filename), index_col=0)
        test_pred_df = test_pred_df.sort_index(axis=1)
        test_pred_df[test_pred_df<=0] = np.nan
        test_pred_df.fillna(method="ffill", inplace=True)

        df = Loss(vech_df, test_pred_df)

        E_df_l.append(df['MSE'])
        Q_df_l.append(df['QLIKE'])

        file_key_name = filename.split('_')[2] + '_' + filename.split('_')[3]
        # file_key_name = filename
        files_l.append(file_key_name)

    E_df = pd.concat(E_df_l, axis=1)
    E_df.columns = files_l
    Q_df = pd.concat(Q_df_l, axis=1)
    Q_df.columns = files_l
    return result_files, E_df, Q_df


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

    result_files, E_df, Q_df = Result(vech_df, 'Forecast_Var', universe, horizon)
    print(E_df.mean(0))
    print(Q_df.mean(0))

    print(' * ' * 30)
    print(norm_loss(E_df).mean(0))
    print(norm_loss(Q_df).mean(0))

    mcs_E = ModelConfidenceSet(E_df, 0.05, 10000, 2).run()
    sum_E = rank_MCS(E_df, mcs_E.pvalues)

    mcs_Q = ModelConfidenceSet(Q_df, 0.05, 10000, 2).run()
    sum_Q = rank_MCS(Q_df, mcs_Q.pvalues)

    print(" * * * * * MCS of MSE * * * * * ")
    print(np.round(sum_E, 3))

    print(" * * * * * MCS of QLIKE * * * * * ")
    print(np.round(sum_Q, 3))

    # print(' * ' * 30)
    mse_ticker_model = norm_loss(E_df).T
    qlike_ticker_model = norm_loss(Q_df).T

    # save results to csv
    mse_ticker_model.to_csv(join(sum_path, 'mse_ticker_model.csv'))
    qlike_ticker_model.to_csv(join(sum_path, 'qlike_ticker_model.csv'))