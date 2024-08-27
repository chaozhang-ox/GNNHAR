"""
Linear models to forecast the realized volatility, including HAR and GHAR. HAR is a special case of GHAR, assuming the adjacency matrix is identity.
"""

import argparse
import os
from os.path import join

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser()
parser.add_argument("--window", type=int, default=22, help="forward-looking period")
parser.add_argument("--horizon", type=int, default=1, help="forecasting horizon")
parser.add_argument("--model_name", type=str, default='GHAR', help="model name")
parser.add_argument("--adj_name", type=str, default='iden+glasso', help="adj choices")
parser.add_argument("--data_name", type=str, default='DJIA', help="data name")
parser.add_argument("--version", type=str, default='Forecast_Var', help="version name")

opt = parser.parse_args()
print(opt)

# Specific version
this_version = '_'.join(
    [opt.version,
     opt.model_name,
     opt.adj_name,
     opt.data_name,
     'W' + str(opt.window),
     'F' + str(opt.horizon)])


path = 'your_local_path'
model_save_path = join('your_model_storage_path', this_version)
os.makedirs(model_save_path, exist_ok=True)


def load_feature_data(universe):
    feature_df = pd.read_csv(join(path, 'Data', f'{universe}_var_FH1.csv'), index_col=0)
    feature_df.fillna(method="ffill", inplace=True)
    feature_df = feature_df[feature_df.index <= '2021-07-01']
    feature_df = feature_df.sort_index(axis=1)
    return feature_df


def load_data(universe, horizon):
    var_df = pd.read_csv(join(path, 'Data', f'{universe}_var_FH{horizon}.csv'), index_col=0)
    var_df.fillna(method="ffill", inplace=True)
    vech_df = var_df[var_df.index <= '2021-07-01']
    vech_df = vech_df.sort_index(axis=1)
    return vech_df


def load_ret(universe):
    ret_df = pd.read_csv(join(path, 'Data', f'{universe}_ret_FH1.csv'), index_col=0)
    ret_df.fillna(method="ffill", inplace=True)
    ret_df = ret_df[ret_df.index <= '2021-07-01']
    ret_df = ret_df.sort_index(axis=1)
    return ret_df


def preprocess_HAR(feature_df, vech_df):
    subdf_l = []
    all_assets_l = [i for i in vech_df.columns if i not in ['Date', 'Time']]
    all_assets_l.sort()

    har_lags = [1, 5, 22]
    for target_var in vech_df:
        subdf = pd.DataFrame()
        subdf['Target'] = vech_df[target_var].copy()
        subdf['Date'] = vech_df.index
        subdf['Ticker'] = target_var
        indpt_df_l = []
        for lag in har_lags:
            tmp_indpdt_df = 0
            for il in range(1, 1+lag):
                tmp_indpdt_df += feature_df[target_var].shift(il)

            indpt_df_l.append(tmp_indpdt_df / lag)

        # reverse the time order
        explain_df = pd.concat(indpt_df_l, axis=1)
        explain_df.columns = ['var+lag%d' % i for i in har_lags]

        subdf = pd.merge(subdf, explain_df, left_index=True, right_index=True)
        subdf.replace([np.inf, -np.inf], np.nan, inplace=True)
        subdf.dropna(inplace=True)
        subdf_l.append(subdf)

    df = pd.concat(subdf_l)
    df.reset_index(drop=True, inplace=True)

    date_l = list(set(df['Date'].tolist()))
    date_l.sort()

    subdf_dic = {}
    for date in date_l:
        subdf = df[df['Date'] == date]
        subdf_dic[date] = subdf

    print('Finish preparation!')
    return subdf_dic, date_l



def preprocess_adj_l(date_l, subdf_dic, adj_df_l):
    new_subdf_l = []
    for date in date_l:
        subdf = subdf_dic[date]
        # print(subdf)
        tmp_subdf_l = []
        clms = [i for i in subdf.columns if 'lag' in i]
        # print(clms)
        for k, adj_df in enumerate(adj_df_l):
            # print(adj_df)
            tmp_subdf = pd.DataFrame(np.dot(adj_df, subdf[clms]), columns=['sec'+str(k)+i for i in clms], index=subdf.index)
            tmp_subdf_l.append(tmp_subdf)
        new_subdf = pd.concat([subdf[['Target', 'Date', 'Ticker']]]+tmp_subdf_l, axis=1)
        new_subdf_l.append(new_subdf)

    df = pd.concat(new_subdf_l)
    df.reset_index(drop=True, inplace=True)
    print('Finish transformation!')
    return df


def df2arr(df, vars_l):
    all_inputs = df[vars_l].values
    all_targets = df[['Target']].values
    return all_inputs, all_targets


def GLASSO_Precision(subret):
    from sklearn.covariance import GraphicalLassoCV
    n = subret.shape[1]
    tickers = subret.columns
    cov = GraphicalLassoCV().fit(subret)
    print('Alpha in GLASSO: %.3f' % cov.alpha_)
    corr = cov.precision_ != 0
    print('Sparsity of Adj: %.3f' % corr.mean())
    corr_adj = corr - np.identity(n)
    d_sqrt_inv = np.diag(np.sqrt(1/(corr_adj.sum(1)+1e-8)))
    adj_df = pd.DataFrame(np.dot(np.dot(d_sqrt_inv, corr_adj), d_sqrt_inv), columns=tickers, index=tickers)
    return adj_df


def Train(ret_df, vech_df, subdf_dic, date, date_l):
    timestamp = date_l.index(date)
    # split time
    s_p = max(timestamp-1000, 0)
    f_p = min(timestamp + opt.window, len(date_l)-1)

    s_date = date_l[s_p]
    f_date = date_l[f_p]

    subret = ret_df[ret_df.index < date]
    subret = subret[subret.index >= s_date]

    subdata = vech_df[vech_df.index < date]
    subdata = subdata[subdata.index >= s_date]
    tickers = subret.columns

    n = vech_df.shape[1]
    adj_name_l = opt.adj_name.split('+')
    adj_df_l = []
    for adj_name in adj_name_l:
        if adj_name == 'iden':
            adj_df = pd.DataFrame(np.identity(n), index=tickers, columns=tickers)
        elif adj_name == 'glasso':
            adj_df = GLASSO_Precision(subret)
        else:
            adj_df = pd.DataFrame(np.zeros((n, n)), index=tickers, columns=tickers)

        adj_df_l.append(adj_df)

    df = preprocess_adj_l(date_l[s_p:f_p+1], subdf_dic, adj_df_l)
    
    vars_l = [i for i in df.columns if 'lag' in i]
    # split data
    train_df = df[df['Date'] >= s_date]
    train_df = train_df[train_df['Date'] < date]
    print(train_df)
    
    test_df = df[df['Date'] >= date]
    test_df = test_df[test_df['Date'] < f_date]
    print(test_df)
    
    train_x, train_y = df2arr(train_df, vars_l)
    test_x, test_y = df2arr(test_df, vars_l)
    
    best_model = LinearRegression()
    best_model.fit(train_x, train_y)
    print(best_model.coef_)
    
    test_pred_df = test_df[['Ticker', 'Date']]
    test_pred_df['Pred_VHAR'] = best_model.predict(test_x)
    test_pred_df = test_pred_df.pivot(index='Date', columns='Ticker', values='Pred_VHAR')
    
    test_pred_df.columns = list(test_pred_df.columns)
    test_pred_df.index = list(test_pred_df.index)
    
    print('Before: %.3f' % test_pred_df.min().min())
    
    # adjust the negative forecasts to the minimum of the training data
    for clm in test_pred_df.columns:
        clm_pred_df = test_pred_df[clm]
        clm_train_df = train_df[train_df['Ticker'] == clm]['Target']
        clm_pred_df[clm_pred_df <= 0] = clm_train_df.min()
        test_pred_df[clm] = clm_pred_df
    
    print('After: %.3f' % test_pred_df.min().min())
    
    save_path = join(path, 'Var_Pred_Results', this_version)
    os.makedirs(save_path, exist_ok=True)
    
    test_pred_df.to_csv(join(save_path, 'Pred_%s.csv' % date))


def connect_pred():
    save_path = join(path, 'Var_Pred_Results', this_version)
    files_l = os.listdir(save_path)
    pred_files = [i for i in files_l if 'Pred_' in i]
    pred_files.sort()
    test_pred_df_l = []
    for i in pred_files:
        test_pred_df = pd.read_csv(join(save_path, i), index_col=0)
        test_pred_df_l.append(test_pred_df)

    test_pred_df = pd.concat(test_pred_df_l)
    print(test_pred_df)

    sum_path = join(path, 'Var_Results_Sum')
    os.makedirs(sum_path, exist_ok=True)
    test_pred_df.to_csv(join(sum_path, this_version + '_pred.csv'))


if __name__ == '__main__':
    feature_df = load_feature_data(opt.universe)
    vech_df = load_data(opt.universe, opt.horizon)
    ret_df = load_ret(opt.universe)

    n = vech_df.shape[1]

    subdf_dic, date_l = preprocess_HAR(feature_df, vech_df)

    print('Training Starts Now ...')
    idx = date_l.index('2011-07-01')

    for date in date_l[idx::opt.window]:
        print(' * ' * 20 + date + ' * ' * 20)
        Train(ret_df, vech_df, subdf_dic, date, date_l)

    connect_pred()