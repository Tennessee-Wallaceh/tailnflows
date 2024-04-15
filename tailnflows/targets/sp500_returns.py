from tailnflows.utils import get_project_root, get_data_path
import pandas as pd
import numpy as np
import torch
import json


def load_return_data(top_n_symbols):
    target_file = f"{get_data_path()}/sp500_stocks.csv"
    data = pd.read_csv(target_file, index_col="Date", parse_dates=True)

    # select the most traded stocks
    most_traded = data.groupby("Symbol").agg({"Volume": "mean"})
    volumes = data[["Symbol", "Volume"]].fillna(0)
    traded_days = volumes.groupby("Symbol")["Volume"].apply(
        lambda col: (col != 0).sum()
    )
    incomplete_sequence = list(traded_days[traded_days < traded_days.max()].index)
    most_traded = most_traded.drop(["GOOGL"] + incomplete_sequence)
    wanted_symbols = list(
        most_traded.sort_values("Volume", ascending=False).index[:top_n_symbols]
    )

    # convert to log returns (holidays already removed)
    log_rets = {}
    for stock in wanted_symbols:
        stock_data = data[data["Symbol"] == stock]
        stock_data = stock_data[stock_data.index.dayofweek < 5]  # just working days
        stock_data["log_close"] = np.log(stock_data["Close"])
        logret = stock_data["log_close"].diff()
        log_rets[stock] = logret.tail(-1)  # first row will be na

    # join data on date index
    joined_data = None
    for symbol, returns in log_rets.items():
        new_data = pd.DataFrame(returns).rename(columns={"log_close": symbol})
        if joined_data is None:
            joined_data = new_data
        else:
            joined_data = joined_data.join(new_data)

    return joined_data[wanted_symbols]
