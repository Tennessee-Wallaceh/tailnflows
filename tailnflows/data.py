from tailnflows.utils import get_project_root
import pandas as pd
import numpy as np
import torch
import json


def get_data_path():
    return f'{get_project_root()}/data'

def get_holidays():
    holidays_file = f'{get_data_path()}/holidays.json'
    with open(holidays_file, mode="r") as j_object:
        holdata = json.load(j_object)
    return holdata['full_day_holidays']

def get_return_data(
        dtype,
        random_seed,
        wanted_symbols=None, 
        top_n_symbols=10, 
        standardise=False, 
        val_prop=0.2, 
        tst_prop=0.4,
        test_future=True,
):
    assert wanted_symbols is not None or top_n_symbols is not None, 'Need to pass either top_n_symbols or wanted_symbols!'
    torch.set_default_dtype(dtype)

    target_file = f'{get_data_path()}/VIX.csv'
    vix_data = pd.read_csv(target_file, index_col='Date', parse_dates=True)
    vix_data['Symbol'] = 'VIX'
    target_file = f'{get_data_path()}/sp500_stocks.csv'
    data = pd.concat([
        pd.read_csv(target_file, index_col='Date', parse_dates=True),
        vix_data
    ])

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

    holidays = get_holidays()
    log_rets = {}
    for stock in wanted_symbols:
        stock_data = data[data['Symbol'] == stock]
        stock_data = stock_data[stock_data.index.dayofweek < 5] # just working days
        stock_data = stock_data.loc[[d for d in stock_data.index if d not in holidays]]
        stock_data['log_close'] = np.log(stock_data['Close'])
        logret = stock_data['log_close'].diff()
        log_rets[stock] = logret.tail(-1) # first row will be na

    joined_data = None
    for symbol, returns in log_rets.items():
        new_data = pd.DataFrame(returns).rename(columns={'log_close': symbol}) 
        if joined_data is None:
            joined_data = new_data
        else:
            joined_data = joined_data.join(new_data)


    n = joined_data.shape[0]
    n_val = int(val_prop * n)
    n_tst = int(tst_prop * n)
    n_trn = n - n_val - n_tst
    dim = len(wanted_symbols)
    generator = torch.Generator()
    generator.manual_seed(random_seed)

    if test_future:
        x_tst = torch.tensor(joined_data[wanted_symbols][-n_tst:].to_numpy(), dtype=dtype)
        x = torch.tensor(joined_data[wanted_symbols][:-n_tst].to_numpy(), dtype=dtype)
        x = x[torch.randperm(x.shape[0])] # shuffle
        x_trn, x_val = torch.split(x, [n_trn, n_val])
    else:
        x = torch.tensor(joined_data[wanted_symbols].to_numpy(), dtype=dtype)
        x = x[torch.randperm(x.shape[0])] # shuffle
        x_trn, x_val, x_tst = torch.split(x, [n_trn, n_val, n_tst])

    # standardise
    if standardise:
        trn_mean = x_trn.mean(axis=0)
        trn_std = x_trn.std(axis=0)

        x_trn = (x_trn - trn_mean) / trn_std
        x_val = (x_val - trn_mean) / trn_std
        x_tst = (x_tst - trn_mean) / trn_std

    return x_trn, x_val, x_tst, dim, {'used_symbols': wanted_symbols}

data_sources = {
    'financial_returns': get_return_data
}