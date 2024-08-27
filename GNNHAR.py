"""
Proposed GNNHAR models to forecast the realized volatility. 
Include HAR, GHAR, GNNHAR1L, GNNHAR2L, and GNNHAR3L, with different loss functions, implemented in PyTorch.
For linear regressions with MSE loss, we also provide another implementation in GHAR.py, through the LinearRegression class in sklearn.
"""

import argparse
import os
from os.path import join

import numpy as np
import pandas as pd
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset
import torch.optim as optim

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

parser = argparse.ArgumentParser()
parser.add_argument("--window", type=int, default=22, help="moving window")
parser.add_argument("--horizon", type=int, default=1, help="forecasting horizon")
parser.add_argument("--valid_len", type=int, default=22, help="validation period")
parser.add_argument("--model_name", type=str, default='GNNHAR1L', help="model name")
parser.add_argument("--adj_name", type=str, default='glasso', help="adj choices")
parser.add_argument("--universe", type=str, default='DJIA', help="data name")
parser.add_argument("--loss", type=str, default='MSE', help="loss function")
parser.add_argument("--n_epochs", type=int, default=5000, help="epochs for training")
parser.add_argument("--n_hid", type=int, default=9, help="hidden neurons")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--ens", type=int, default=0, help="No. Ensemble")
parser.add_argument("--numNN", type=int, default=1, help="number of NNs")
parser.add_argument("--version", type=str, default='Forecast_Var', help="version name")

opt = parser.parse_args()
print(opt)

# Specific version
this_version = '_'.join(
    [opt.version,
     opt.loss,
     opt.model_name,
     opt.adj_name,
     opt.universe,
     'E' + str(opt.n_epochs),
     'H' + str(opt.n_hid),
     'BS' + str(opt.batch_size),
     'LR' + str(opt.lr),
     'W' + str(opt.window),
     'F' + str(opt.horizon),
     'Val' + str(opt.valid_len)])

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


def get_lag_avg(df, lag):
    res = pd.DataFrame(columns=df.columns, index=df.index).fillna(0)
    for l in range(1, lag + 1):
        res += (1 / lag) * df.shift(l)
    return res


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


def GLASSO_Precision(subret):
    from sklearn.covariance import GraphicalLassoCV
    n = subret.shape[1]
    tickers = subret.columns
    cov = GraphicalLassoCV().fit(subret)
    print('Alpha in GLASSO: %.3f' % cov.alpha_)
    corr = cov.precision_ != 0
    print('Sparsity of Adj: %.3f' % corr.mean())
    corr_adj = corr - np.identity(n)
    # adj_df = pd.DataFrame(corr_adj / (corr_adj.sum(1)[:, np.newaxis] + 1e-8), columns=tickers, index=tickers)
    d_sqrt_inv = np.diag(np.sqrt(1/(corr_adj.sum(1)+1e-8)))
    adj_df = pd.DataFrame(np.dot(np.dot(d_sqrt_inv, corr_adj), d_sqrt_inv), columns=tickers, index=tickers)
    return adj_df


class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvLayer, self).__init__()

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        if bias is True:
            self.bias = nn.Parameter(torch.FloatTensor(1, out_features))
            nn.init.ones_(self.bias)
        else:
            self.bias = None

    def forward(self, node_feature, adj):
        h = torch.matmul(node_feature, self.weight)
        output = torch.matmul(adj, h)
        if self.bias is not None:
            return output + self.bias
        return output

# HAR model
class HAR(nn.Module):
    def __init__(self):
        super(HAR, self).__init__()

        self.linear1 = nn.Linear(3, 1, bias=True)
        self.relu = nn.ReLU()

    def forward(self, node_feat, adj):
        # node_feat: (batch_size, N, 3)
        # adj: (N, N)

        H1 = self.linear1(node_feat)
        res = self.relu(H1)

        return res.squeeze(-1)
    

# GHAR model
class GHAR(nn.Module):
    def __init__(self, n_hid):
        super(GHAR, self).__init__()

        self.linear1 = nn.Linear(3, 1, bias=True)

        self.gcn1 = GraphConvLayer(3, n_hid, bias=False)
        self.relu = nn.ReLU()

    def forward(self, node_feat, adj):
        # node_feat: (batch_size, N, 3)
        # adj: (N, N)

        H1 = self.linear1(node_feat)

        H2 = self.gcn1(node_feat, adj)
        res = H1 + H2
        res = self.relu(res)

        return res.squeeze(-1)

# 1-layer GNNHAR
class GNNHAR1L(nn.Module):
    def __init__(self, n_hid):
        super(GNNHAR1L, self).__init__()

        self.linear1 = nn.Linear(3, 1, bias=True)

        self.gcn1 = GraphConvLayer(3, n_hid, bias=False)
        self.mlp1 = nn.Linear(n_hid, 1, bias = False)
        self.relu = nn.ReLU()

    def forward(self, node_feat, adj):
        # node_feat: (batch_size, N, 3)
        # adj: (N, N)

        H1 = self.linear1(node_feat)

        H2 = self.gcn1(node_feat, adj)
        H2 = self.relu(H2)
        H2 = self.mlp1(H2) # (batch_size, N, 1)

        res = H1 + H2
        res = self.relu(res)

        return res.squeeze(-1)


class GNNHAR2L(nn.Module):
    def __init__(self, nhid):
        super(GNNHAR2L, self).__init__()

        self.linear1 = nn.Linear(3, 1, bias=True)

        self.gcn1 = GraphConvLayer(3, nhid, bias=False)
        self.gcn2 = GraphConvLayer(nhid, nhid, bias = False)

        self.mlp1 = nn.Linear(nhid, 1, bias = False)
        self.relu = nn.ReLU()

    def forward(self, node_feat, adj):
        # node_feat: (batch_size, N, 3)
        # adj: (N, N)

        H1 = self.linear1(node_feat)

        # 2-layer of GCN:
        H2 = self.relu(self.gcn1(node_feat, adj))
        H2 = self.relu(self.gcn2(H2, adj))

        # dimension: nhid to 1
        H2 = self.mlp1(H2) # (batch_size, N, 1)

        res = H1 + H2
        res = self.relu(res)

        return res.squeeze(-1)


class GNNHAR3L(nn.Module):
    def __init__(self, nhid):
        super(GNNHAR3L, self).__init__()

        self.linear1 = nn.Linear(3, 1, bias=True)

        self.gcn1 = GraphConvLayer(3, nhid, bias=False)
        self.gcn2 = GraphConvLayer(nhid, nhid, bias = False)
        self.gcn3 = GraphConvLayer(nhid, nhid, bias = False)

        self.mlp1 = nn.Linear(nhid, 1, bias = False)
        self.relu = nn.ReLU()

    def forward(self, node_feat, adj):
        # node_feat: (batch_size, N, 3)
        # adj: (N, N)

        H1 = self.linear1(node_feat)

        # 2-layer of GCN:
        H2 = self.relu(self.gcn1(node_feat, adj))
        H2 = self.relu(self.gcn2(H2, adj))
        H2 = self.relu(self.gcn3(H2, adj))

        # dimension: nhid to 1
        H2 = self.mlp1(H2) # (batch_size, N, 1)

        res = H1 + H2
        res = self.relu(res)

        return res.squeeze(-1)
    

def Compute_Adj(ret_df, vech_df, date, date_l):
    timestamp = date_l.index(date)
    # split time
    s_p = max(timestamp-1000, 0)
    v_p = timestamp - opt.valid_len
    f_p = min(timestamp + opt.window, len(date_l)-1)

    s_date = date_l[s_p]
    v_date = date_l[v_p]
    f_date = date_l[f_p]

    subret = ret_df[ret_df.index < date]
    subret = subret[subret.index >= s_date]

    subdata = vech_df[vech_df.index < date]
    subdata = subdata[subdata.index >= s_date]

    n = vech_df.shape[1]
    adj_name = opt.adj_name
    tickers = subret.columns

    if adj_name == 'glasso':
        adj_df = GLASSO_Precision(subret)
    else:
        adj_df = pd.DataFrame(np.zeros((n, n)), columns=tickers, index=tickers)

    print((s_date, v_date, f_date))
    adj_df = Tensor(adj_df.values)
    return adj_df, s_p, v_p, timestamp, f_p


def df2arr(df, vars_l):
    all_inputs = Tensor(df[vars_l].values)
    all_targets = Tensor(df[['Target']].values)
    return all_inputs, all_targets


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, forecast_y):
        if opt.loss == 'QLike':
            true_fore = outputs / (forecast_y + 1e-4) # stablize the training
            l_v = torch.mean(true_fore - torch.log(true_fore))
        else:
            mseloss = nn.MSELoss()
            l_v = mseloss(outputs, forecast_y)
        return l_v


# Train a single model
def Train_Single(train_loader, valid_loader, model_index, seed, date):
    torch.manual_seed(seed)
    print("------ Model %d Starts with Random Seed %d " % (model_index, seed))
    if opt.model_name == 'HAR':
        model = HAR()
    elif opt.model_name == 'GHAR':
        model = GHAR(opt.n_hid)
    elif opt.model_name == 'GNNHAR1L':
        model = GNNHAR1L(opt.n_hid)
    elif opt.model_name == 'GNNHAR2L':
        model = GNNHAR2L(opt.n_hid)
    elif opt.model_name == 'GNNHAR3L':
        model = GNNHAR3L(opt.n_hid)
    else:
        print('Please choose the correct model')
        return
    
    if cuda:
        model.cuda()

    for parameter in model.parameters():
        print(parameter)

    # optimizer
    loss_function = Loss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-5)
    best_val_mse = 1e8

    train_loss = []
    valid_loss = []
    for epoch in range(opt.n_epochs):  # loop over the dataset multiple times
        epoch_loss_train = []
        epoch_loss_valid = []

        model.train()
        for _, (train_X, train_y) in enumerate(train_loader):
            train_X, train_y = Variable(train_X), Variable(train_y)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            forecast_y = model(train_X, adj_df)
            loss = loss_function(train_y, forecast_y)
            loss.backward()
            optimizer.step()
            epoch_loss_train.append(loss.item())

        # validation data
        model.eval()
        for _, (val_X, val_y) in enumerate(valid_loader):
            val_X, val_y = Variable(val_X), Variable(val_y)

            val_out = model(val_X, adj_df)
            loss = loss_function(val_y, val_out)
            epoch_loss_valid.append(loss.item())

        train_loss_epoch = np.mean(epoch_loss_train)
        valid_loss_epoch = np.mean(epoch_loss_valid)
        train_loss.append(train_loss_epoch)
        valid_loss.append(valid_loss_epoch)

        if epoch % int(opt.n_epochs/10) == 0:
            print("[Epoch %d] [Train Loss: %.4f] [Valid Loss: %.4f]" % (epoch, train_loss_epoch, valid_loss_epoch))

        # if validation loss decreases, save the model parameters
        if loss.item() < best_val_mse:
            best_val_mse = loss.item()
            torch.save(model.state_dict(), join(model_save_path, 'Best_Model' + '_' + date + '_index%d' % model_index))

    train_loss_arr = np.array(train_loss)
    valid_loss_arr = np.array(valid_loss)
    loss_arr = np.stack([train_loss_arr, valid_loss_arr], axis=1)
    loss_df = pd.DataFrame(loss_arr, columns=['Train', 'Valid'])
    loss_df.to_csv(join(model_save_path, 'loss_%s_index%d.csv' % (date, model_index)), index=False)
    return loss_df


def Train(dataset, adj_df, s_p, v_p, timestamp, f_p, targets, date):
    train_idx = range(s_p, v_p)
    val_idx = range(v_p, timestamp)
    test_idx = range(timestamp, f_p)

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_idx), shuffle=False)

    # Train multiple models with different random seeds
    for iii in range(opt.ens*opt.numNN, (opt.ens+1)*opt.numNN):
        seed = np.random.randint(low=1, high=10000)
        loss_df = Train_Single(train_loader, valid_loader, model_index=iii, seed=seed, date=date)

        while (np.abs(loss_df['Valid'].diff()) < 1e-6).mean() > 0.5 or loss_df['Valid'].iloc[-1] > 100:
            print(' * ' * 20)
            print('  Attention!!!   Restart Training!!!  ')
            print(' * ' * 20)
            seed = np.random.randint(low=1, high=10000)
            loss_df = Train_Single(train_loader, valid_loader, model_index=iii, seed=seed, date=date)

    # Forecast testing period
    for iii in range(opt.ens*opt.numNN, (opt.ens+1)*opt.numNN):
        with torch.no_grad():
            if opt.model_name == 'HAR':
                model = HAR()
            elif opt.model_name == 'GHAR':
                model = GHAR(opt.n_hid)
            elif opt.model_name == 'GNNHAR1L':
                model = GNNHAR1L(opt.n_hid)
            elif opt.model_name == 'GNNHAR2L':
                model = GNNHAR2L(opt.n_hid)
            elif opt.model_name == 'GNNHAR3L':
                model = GNNHAR3L(opt.n_hid)
            else:
                print('Please choose the correct model')
                return
    
            model.load_state_dict(torch.load(join(model_save_path, 'Best_Model' + '_' + date + '_index%d' % iii)))
            model.eval()

            if cuda:
                model.cuda()

            for _, (test_X, test_y) in enumerate(test_loader):
                test_X, test_y = Variable(test_X), Variable(test_y)
                forecast_test_y = model(test_X, adj_df)

        y_pred = forecast_test_y.cpu().detach().numpy()
        test_pred_df = pd.DataFrame(y_pred, index=targets.index[test_idx], columns=targets.columns)

        print('Min: %.3f' % test_pred_df.min().min())

        save_path = join(path, 'Var_Pred_Results', this_version)
        os.makedirs(save_path, exist_ok=True)

        test_pred_df.to_csv(join(save_path, 'Pred_%s_Ens%d.csv' % (date, iii)))


# Some trained models may not converge well, we only use the forecasts from those models with a good converge
# This selection is based on the training and validation data, so no look-forward bias
def Screen_Ensemble(date, thres_perc=50):
    loss_l = []
    for j in range(opt.numNN):
        loss_df = pd.read_csv(join(model_save_path, 'loss_%s_index%d.csv' % (date, j)))
        loss_l.append(loss_df['Valid'].iloc[-1])

    threshold_loss = np.percentile(loss_l, thres_perc)
    select_l = []
    for j in range(opt.numNN):
        if loss_l[j] <= threshold_loss:
            select_l.append(j)
        else:
            pass
    return select_l


# Connect the forecasts for each sub-period
# Forecast variance matrix with shape T_t * N; T_t is the length of the entire testing period
def connect_pred():
    save_path = join(path, 'Var_Pred_Results', this_version)
    files_l = os.listdir(save_path)
    dates_l = [i.split('_')[1] for i in files_l if 'Pred_' in i and '_Ens0.csv']
    dates_l = list(set(dates_l))
    dates_l.sort()

    test_pred_df_l = []
    for date in dates_l:
        tmp_pred_df_l = []
        select_l = Screen_Ensemble(date)
        for j in select_l:
            tmp_test_pred_df = pd.read_csv(join(save_path, '_'.join(['Pred', date, 'Ens%d.csv' % j])), index_col=0)
            tmp_pred_df_l.append(tmp_test_pred_df)

        test_pred_df = pd.DataFrame(np.stack(tmp_pred_df_l).mean(0), index=tmp_test_pred_df.index, columns=tmp_test_pred_df.columns)
        test_pred_df_l.append(test_pred_df)

    test_pred_df = pd.concat(test_pred_df_l) * opt.horizon
    print(test_pred_df)

    sum_path = join(path, 'Var_Results_Sum')
    os.makedirs(sum_path, exist_ok=True)
    test_pred_df.to_csv(join(sum_path, this_version + '_pred.csv'))


if __name__ == '__main__':
    feature_df = load_feature_data(opt.universe)
    vech_df = load_data(opt.universe, opt.horizon)
    ret_df = load_ret(opt.universe)

    n = vech_df.shape[1]

    if opt.horizon == 1:
        lag1 = get_lag_avg(feature_df, 1).iloc[22:]
        lag5 = get_lag_avg(feature_df, 5).iloc[22:]
        lag22 = get_lag_avg(feature_df, 22).iloc[22:]
    else:
        e_idx = -opt.horizon + 1
        lag1 = get_lag_avg(feature_df, 1).iloc[22:e_idx]
        lag5 = get_lag_avg(feature_df, 5).iloc[22:e_idx]
        lag22 = get_lag_avg(feature_df, 22).iloc[22:e_idx]

    targets = vech_df.iloc[22:]

    Y, lag1, lag5, lag22 = np.array(targets), np.array(lag1), np.array(lag5), np.array(lag22)

    Y /= opt.horizon

    X = [lag1, lag5, lag22]

    X = np.stack(X, axis=-1)
    X, Y = Tensor(X), Tensor(Y)

    dataset = TensorDataset(X, Y)

    print('Training Starts Now ...')
    date_l = targets.index.tolist()
    idx = date_l.index('2011-07-01')

    for date in date_l[idx::opt.window]:
        print(' * ' * 20 + date + ' * ' * 20)
        adj_df, s_p, v_p, timestamp, f_p = Compute_Adj(ret_df, vech_df, date, date_l)
        Train(dataset, adj_df, s_p, v_p, timestamp, f_p, targets, date)

    connect_pred()