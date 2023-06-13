from tailnflows.experiment_utils import get_project_root
import pandas as pd
import numpy as np
import torch
import os

DATA_PATH = os.environ.get('TAILNFLOWS_DATA', None)

def get_data_path():
    if DATA_PATH is None:
        return f'{get_project_root()}/data'
    else:
        return DATA_PATH
    
def get_return_data(wanted_symbols=None, top_n_symbols=10):
    assert wanted_symbols is not None or top_n_symbols is not None, 'Need to pass either top_n_symbols or wanted_symbols!'
    
    target_file = f'{get_data_path()}/VIX.csv'
    vix_data = pd.read_csv(target_file, index_col='Date', parse_dates=True)
    vix_data['Symbol'] = 'VIX'
    target_file = f'{get_data_path()}/sp500_stocks.csv'
    data = pd.concat([
        pd.read_csv(target_file, index_col='Date', parse_dates=True),
        vix_data
    ])
    data = data.loc['2015-1-1':]

    if wanted_symbols is None and top_n_symbols is None:
        wanted_symbols = [
            'AAPL', # apple
            'VIX', # vol index
            'BRK-B', # berkshire hathaway
            'XOM', # exxon mobil
            'GOOG', # google
            'MSFT', # microsoft
            'AMZN', # amazon
            'V', # visa
        ]
    else:
        most_traded = data.groupby('Symbol').agg({'Volume': 'mean'})
        volumes = data[['Symbol', 'Volume']].fillna(0)
        traded_days = volumes.groupby('Symbol')['Volume'].apply(lambda col: (col != 0).sum())
        incomplete_sequence = list(traded_days[traded_days < traded_days.max()].index)
        most_traded = most_traded.drop(['GOOGL'] + incomplete_sequence)
        wanted_symbols = list(most_traded.sort_values('Volume', ascending=False).index[:top_n_symbols])

    log_rets = {}
    for stock in wanted_symbols:
        stock_data = data[data['Symbol'] == stock]
        stock_data = stock_data.asfreq('d')
        stock_data = stock_data[stock_data.index.dayofweek < 5] # just working days (we will still have bank holidays)
        stock_data = stock_data.fillna(method='ffill')
        stock_data['log_close'] = np.log(stock_data['Close'])
        logret = stock_data['log_close'].diff().dropna()
        log_rets[stock] = logret

    joined_data = None
    for symbol, returns in log_rets.items():
        new_data = pd.DataFrame(returns).rename(columns={'log_close': symbol}) 
        if joined_data is None:
            joined_data = new_data
        else:
            joined_data = joined_data.join(new_data)

    x = torch.tensor(joined_data[wanted_symbols].to_numpy())
    x = x[torch.randperm(x.shape[0])] # shuffle

    dim = len(wanted_symbols)

    n = x.shape[0]
    n_val = int(0.2 * n)
    n_tst = int(0.1 * n)
    n_trn = n - n_val - n_tst

    x_trn, x_val, x_tst = torch.split(x, [n_trn, n_val, n_tst])

    # standardise
    trn_mean = x_trn.mean(axis=0)
    trn_std = x_trn.std(axis=0)

    x_trn = (x_trn - trn_mean) / trn_std
    x_val = (x_val - trn_mean) / trn_std
    x_tst = (x_tst - trn_mean) / trn_std

    return x_trn, x_val, x_tst, dim

data_sources = {
    'financial_returns': get_return_data
}