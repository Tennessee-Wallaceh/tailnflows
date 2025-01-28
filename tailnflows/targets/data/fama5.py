from tailnflows.utils import get_data_path
import pandas as pd
import torch


def load_data():
    df = pd.read_csv(
        f"{get_data_path()}/fama5/data.csv",
        index_col='Date', 
        parse_dates=True
    )
    return torch.tensor(df.to_numpy())