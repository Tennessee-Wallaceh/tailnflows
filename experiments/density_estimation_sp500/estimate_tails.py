import torch
from tailnflows.targets.sp500_returns import load_return_data
import tqdm
from tailnflows.utils import add_raw_data
from tailnflows.models.tail_estimation import estimate_df


def estimate_marginal_tails(out_path, dim, tst_ix):
    return_data, wanted_symbols = load_return_data(dim)

    x = torch.tensor(return_data.to_numpy())
    n = x.shape[0]

    trn_val_mask = torch.ones(n, dtype=torch.bool)
    trn_val_mask[tst_ix] = False

    # standardise
    trn_val_mean = x[trn_val_mask, :].mean(axis=0)
    trn_val_std = x[trn_val_mask, :].std(axis=0)
    x = (x - trn_val_mean) / trn_val_std

    dfs = []
    pos_dfs = []
    neg_dfs = []
    loop = tqdm.tqdm(range(x.shape[1]))
    for dim_ix in loop:
        loop.set_description(f"Estimating tails for {wanted_symbols[dim_ix]}")
        dim_x = x[trn_val_mask, dim_ix].cpu()  # standardised x for this dimension
        pos_x = dim_x[dim_x > 0]
        neg_x = dim_x[dim_x < 0]

        try:
            dfs.append(estimate_df(dim_x.abs(), verbose=False))
        except ValueError:
            print(f"ERR: {dim_ix}")
            dfs.append(0)

        try:
            pos_dfs.append(estimate_df(pos_x.abs(), verbose=False))
        except ValueError:
            print(f"ERR: p {dim_ix}")
            pos_dfs.append(0)

        try:
            neg_dfs.append(estimate_df(neg_x.abs(), verbose=False))
        except ValueError:
            print(f"ERR: n {dim_ix}")
            neg_dfs.append(0)

    add_raw_data(
        out_path,
        label="tail_and_scale",
        data={
            "dfs": [float(df) for df in dfs],
            "pos_dfs": [float(df) for df in pos_dfs],
            "neg_dfs": [float(df) for df in neg_dfs],
            "mean": trn_val_mean.cpu().numpy(),
            "std": trn_val_std.cpu().numpy(),
        },
    )


if __name__ == "__main__":
    dim = 300
    return_data, _ = load_return_data(dim)
    x = torch.tensor(return_data.to_numpy())
    n = x.shape[0]

    n_trn = int(n * 0.4)
    n_val = int(n * 0.2)
    n_tst = n - int(n * 0.4) - int(n * 0.2)

    tst_ix = torch.arange(n_tst, n)
    trn_ix, val_ix = torch.split(
        torch.randperm(n_trn + n_val),
        [n_trn, n_val],
    )
    estimate_marginal_tails("sp500_tails", dim, tst_ix)
