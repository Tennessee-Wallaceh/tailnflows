# type: ignore
from tailnflows.utils import get_project_root
import pandas as pd
import numpy as np
import torch
import json
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from typing import Literal
from torch.nn.functional import softplus
from tailnflows.utils import MarginalTailAdaptiveFlowImport
with MarginalTailAdaptiveFlowImport():
    from synthetic_experiments.data_generators import copula_generator

def get_data_path():
    return f'{get_project_root()}/data'

def get_holidays():
    holidays_file = f'{get_data_path()}/holidays.json'
    with open(holidays_file, mode="r") as j_object:
        holdata = json.load(j_object)
    return holdata['full_day_holidays']

def load_return_data(top_n_symbols):
    target_file = f'{get_data_path()}/sp500_stocks.csv'
    data = pd.read_csv(target_file, index_col='Date', parse_dates=True)

    # select the most traded stocks
    most_traded = data.groupby('Symbol').agg({'Volume': 'mean'})
    volumes = data[['Symbol', 'Volume']].fillna(0)
    traded_days = volumes.groupby('Symbol')['Volume'].apply(lambda col: (col != 0).sum())
    incomplete_sequence = list(traded_days[traded_days < traded_days.max()].index)
    most_traded = most_traded.drop(['GOOGL'] + incomplete_sequence)
    wanted_symbols = list(most_traded.sort_values('Volume', ascending=False).index[:top_n_symbols])

    # filter our holidays and convert to log returns
    holidays = get_holidays()
    log_rets = {}
    for stock in wanted_symbols:
        stock_data = data[data['Symbol'] == stock]
        stock_data = stock_data[stock_data.index.dayofweek < 5] # just working days
        stock_data = stock_data.loc[[d for d in stock_data.index if d not in holidays]]
        stock_data['log_close'] = np.log(stock_data['Close'])
        logret = stock_data['log_close'].diff()
        log_rets[stock] = logret.tail(-1) # first row will be na

    # join data on date index
    joined_data = None
    for symbol, returns in log_rets.items():
        new_data = pd.DataFrame(returns).rename(columns={'log_close': symbol}) 
        if joined_data is None:
            joined_data = new_data
        else:
            joined_data = joined_data.join(new_data)

    return joined_data[wanted_symbols]

DataUse = Literal['test', 'train', 'validate']

class ReturnData(Dataset):
    def __init__(self, dtype, device, random_seed, data_use: DataUse, top_n_symbols=10,  val_prop=0.2, tst_prop=0.4):
        self.top_n_symbols = top_n_symbols
        
        generator = torch.Generator(device=device)
        generator.manual_seed(random_seed)

        joined_data = load_return_data(top_n_symbols)
        n = joined_data.shape[0]
        n_val = int(val_prop * n)
        n_tst = int(tst_prop * n)
        n_trn = n - n_val - n_tst

        x = torch.tensor(joined_data.to_numpy(), dtype=dtype)
        x = x[torch.randperm(x.shape[0], generator=generator)] # shuffle
        x_trn, x_val, x_tst = torch.split(x, [n_trn, n_val, n_tst])
        
        self.dim = x.shape[1]

        trn_mean = x_trn.mean(axis=0)
        trn_std = x_trn.std(axis=0)
        x_trn = (x_trn - trn_mean) / trn_std
        x_val = (x_val - trn_mean) / trn_std
        x_tst = (x_tst - trn_mean) / trn_std

        if data_use == 'train':
            self.return_data = x_trn.to(device)
            self.n = n_trn
        elif data_use == 'test':
            self.return_data = x_tst.to(device)
            self.n = n_tst
        elif data_use == 'validate':
            self.return_data = x_val.to(device)
            self.n = n_val
        else:
            raise Exception(f'Invalid data use: {data_use}')

    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        return self.return_data[idx, :]


class MTAFCopula(Dataset):
    def __init__(self, dtype, device, random_seed, data_use: DataUse, num_samples=20000, dim=8, df=2., num_heavy=4, val_prop=0.2, tst_prop=0.5):
        self.dim = dim
        self.df = df
        self.num_heavy = num_heavy
        self.num_samples = num_samples
        
        generator = torch.Generator(device=device)
        generator.manual_seed(random_seed)

        n_val = int(val_prop * num_samples)
        n_tst = int(tst_prop * num_samples)
        n_trn = num_samples - n_val - n_tst
 
        x =  torch.tensor(
            copula_generator(dim, num_heavy, df, random_seed).get_data(num_samples),
            dtype=dtype,
        )
        x = x[torch.randperm(x.shape[0], generator=generator)] # shuffle
        x_trn, x_val, x_tst = torch.split(x, [n_trn, n_val, n_tst])
        
        self.dim = x.shape[1]

        trn_mean = x_trn.mean(axis=0)
        trn_std = x_trn.std(axis=0)
        x_trn = (x_trn - trn_mean) / trn_std
        x_val = (x_val - trn_mean) / trn_std
        x_tst = (x_tst - trn_mean) / trn_std

        if data_use == 'train':
            self.data = x_trn.to(device)
            self.n = n_trn
        elif data_use == 'test':
            self.data = x_tst.to(device)
            self.n = n_tst
        elif data_use == 'validate':
            self.data = x_val.to(device)
            self.n = n_val
        else:
            raise Exception(f'Invalid data use: {data_use}')

    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        return self.data[idx, :]
    

class NoiseDim(Dataset):
    def __init__(
            self, dtype, device, random_seed, 
            data_use: DataUse, num_samples=20000, d_nuisance=8, df=1., heavy_nuisance=True, 
            val_prop=0.2, tst_prop=0.4
        ):
        self.d_nuisance = d_nuisance
        self.df = df
        self.num_samples = num_samples
        self.heavy_nuisance = heavy_nuisance
        
        generator = torch.Generator(device=device)
        generator.manual_seed(random_seed)

        n_val = int(val_prop * num_samples)
        n_tst = int(tst_prop * num_samples)
        n_trn = num_samples - n_val - n_tst
 
        normal_base = torch.distributions.Normal(loc=0., scale=1.)
        if heavy_nuisance:
            nuisance_base = torch.distributions.StudentT(df=df)
        else:
            nuisance_base = torch.distributions.Normal(loc=0., scale=1.)

        x = torch.zeros([num_samples, d_nuisance + 2])
        x[:,0:d_nuisance] = nuisance_base.sample([num_samples, d_nuisance])
        x[:,d_nuisance] = normal_base.sample([num_samples])
        x[:,d_nuisance + 1] = normal_base.sample([num_samples]) 
        x[:,d_nuisance + 1] *= softplus(x[:,d_nuisance]) # apply the scale

        x = x[torch.randperm(x.shape[0], generator=generator)] # shuffle
        x_trn, x_val, x_tst = torch.split(x, [n_trn, n_val, n_tst])
        
        self.dim = x.shape[1]

        trn_mean = x_trn.mean(axis=0)
        trn_std = x_trn.std(axis=0)
        x_trn = (x_trn - trn_mean) / trn_std
        x_val = (x_val - trn_mean) / trn_std
        x_tst = (x_tst - trn_mean) / trn_std

        if data_use == 'train':
            self.data = x_trn.to(device)
            self.n = n_trn
        elif data_use == 'test':
            self.data = x_tst.to(device)
            self.n = n_tst
        elif data_use == 'validate':
            self.data = x_val.to(device)
            self.n = n_val
        else:
            raise Exception(f'Invalid data use: {data_use}')

    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        return self.data[idx, :]
    
data_sources = {
    'financial_returns': ReturnData,
    'mtaf_copula': MTAFCopula,
    'noise_dim': NoiseDim,
}