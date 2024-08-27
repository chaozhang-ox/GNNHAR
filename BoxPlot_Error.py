"""
Plot the boxplot of the forecast error and ratio for different models
"""
import os
from os.path import *
from MCS import *

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

path = 'your_local_path'
sum_path = join(path, 'Var_Results_Sum')
plot_path = join(path, 'Var_Results_Plot')


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
    forecast_error = test_pred_df - test_df
    forecast_ratio = test_pred_df / test_df
    return forecast_error, forecast_ratio


def Result(vech_df, version_name, universe, horizon):
    result_files = [i for i in files if
                    ('_pred' in i) and version_name in i and '_' + universe + '_' in i and f'F{horizon}' in i and 'W22' in i]

    result_files.sort()
    for (i, item) in enumerate(result_files):
        print(i, item)

    error_dic = {}
    ratio_dic = {}

    for filename in result_files:
        test_pred_df = pd.read_csv(join(sum_path, filename), index_col=0)
        test_pred_df = test_pred_df.sort_index(axis=1)
        test_pred_df[test_pred_df<=0] = np.nan
        test_pred_df.fillna(method="ffill", inplace=True)

        forecast_error, forecast_ratio = Loss(vech_df/horizon, test_pred_df/horizon)

        file_key_name = filename.split('_')[2] + '_' + filename.split('_')[3]
        error_dic[file_key_name] = forecast_error
        ratio_dic[file_key_name] = forecast_ratio

    return error_dic, ratio_dic


def BoxPlot_Error_Ratio(error_dic, ratio_dic, name, horizon):
    if name == 'Error':
        data_dic = error_dic
    else:
        data_dic = ratio_dic

    pdf_name = join(plot_path, 'BoxPlot_%s_%d.pdf' % (name, horizon))

    cmap = plt.get_cmap("tab10")

    pair_l = [['GHAR_iden', 'QLike_HAR'], ['GHAR_iden+glasso', 'QLike_GHAR'], ['MSE_GNNHAR1L', 'QLike_GNNHAR1L'], ['MSE_GNNHAR2L', 'QLike_GNNHAR2L'], ['MSE_GNNHAR3L', 'QLike_GNNHAR3L']]

    new_df_l = []
    for pair in pair_l:
        df_mse = data_dic[pair[0]]
        df_qli = data_dic[pair[1]]
        new_df = pd.DataFrame([df_mse.values.reshape(-1), df_qli.values.reshape(-1)], index=['MSE', 'QLIKE']).T
        new_df_l.append(new_df)

    all_df = pd.concat(new_df_l, axis=1)
    all_df.columns = [r'HAR$_M$', r'HAR$_Q$', r'GHAR$_M$', r'GHAR$_Q$', r'GNNHAR1L$_M$', r'GNNHAR1L$_Q$', r'GNNHAR2L$_M$', r'GNNHAR2L$_Q$', r'GNNHAR3L$_M$', r'GNNHAR3L$_Q$']

    with PdfPages(pdf_name) as pdf:
        f, ax = plt.subplots()
        box_plot = ax.boxplot(all_df, 0, '', vert=False, whis=0, positions=[1, 1.5, 2.5, 3., 4, 4.5, 5.5, 6., 7., 7.5])
        for median in box_plot['medians'][::2]:
            median.set_color(cmap(0))
        for median in box_plot['medians'][1::2]:
            median.set_color(cmap(1))

        if name == 'Error':
            plt.axvline(x=0, color='grey', linestyle='--')
        else:
            plt.axvline(x=1, color='grey', linestyle='--')

        plt.yticks([1, 1.5, 2.5, 3., 4, 4.5, 5.5, 6., 7., 7.5], all_df.columns)
        plt.tight_layout()
        pdf.savefig()
        plt.close()


if __name__ == '__main__':
    horizon = 1
    universe = 'DJIA'

    vech_df = load_data(universe, horizon)
    files = os.listdir(sum_path)
    files.sort()

    error_dic, ratio_dic = Result(vech_df, 'Forecast_Var', universe, horizon)
    BoxPlot_Error_Ratio(error_dic, ratio_dic, 'Error', horizon)
    BoxPlot_Error_Ratio(error_dic, ratio_dic, 'Ratio', horizon)
