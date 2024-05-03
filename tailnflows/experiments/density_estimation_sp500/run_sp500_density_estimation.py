from functools import partial
from dataclasses import dataclass

import torch
import pandas as pd

from tailnflows.targets.sp500_returns import load_return_data
from tailnflows.models.flows import TTF_m, gTAF, mTAF, ModelUse
from tailnflows.train import data_fit
from tailnflows.utils import load_raw_data, add_raw_data

DEFAULT_DTYPE = torch.float32

"""
Model specifications
"""


def base_rqs_spec(depth: int, tail_bound: float):
    return dict(
        rotation=True,
        flow_depth=depth,
        num_bins=3,
        tail_bound=tail_bound,
    )


def ttf_rqs(
    dim: int,
    dfs: list[float],
    pos_dfs: list[float],
    neg_dfs: list[float],
    model_depth: int,
    model_tail_bound: float,
):
    return TTF_m(
        dim,
        use=ModelUse.density_estimation,
        model_kwargs=dict(
            **base_rqs_spec(model_depth, model_tail_bound),
            # TTF specific
            final_affine=True,
            fix_tails=False,
            pos_tail_init=torch.distributions.Uniform(0.05, 1.0).sample([dim]),
            neg_tail_init=torch.distributions.Uniform(0.05, 1.0).sample([dim]),
        ),
    )


def ttf_rqs_fix(
    dim: int,
    dfs: list[float],
    pos_dfs: list[float],
    neg_dfs: list[float],
    model_depth: int,
    model_tail_bound: float,
):
    return TTF_m(
        dim,
        use=ModelUse.density_estimation,
        model_kwargs=dict(
            **base_rqs_spec(model_depth, model_tail_bound),
            # TTF specific
            final_affine=True,
            fix_tails=True,
            pos_tail_init=torch.tensor(
                [1 / df if df != 0.0 else 1e-4 for df in pos_dfs]
            ),
            neg_tail_init=torch.tensor(
                [1 / df if df != 0.0 else 1e-4 for df in neg_dfs]
            ),
        ),
    )


def gtaf_rqs(
    dim: int,
    dfs: list[float],
    pos_dfs: list[float],
    neg_dfs: list[float],
    model_depth: int,
    model_tail_bound: float,
):
    return gTAF(
        dim,
        use=ModelUse.density_estimation,
        model_kwargs=dict(
            **base_rqs_spec(model_depth, model_tail_bound),
            # gTAF specific
            fix_tails=False,
            tail_init=1 / torch.distributions.Uniform(0.05, 1.0).sample([dim]),  # df
        ),
    )


def mtaf_rqs(
    dim: int,
    dfs: list[float],
    pos_dfs: list[float],
    neg_dfs: list[float],
    model_depth: int,
    model_tail_bound: float,
):
    return mTAF(
        dim,
        use=ModelUse.density_estimation,
        model_kwargs=dict(
            **base_rqs_spec(model_depth, model_tail_bound),
            # mTAF specific
            tail_init=torch.tensor(dfs),  # df
        ),
    )


model_definitions = {
    "ttf": ttf_rqs,
    "ttf_fix": ttf_rqs_fix,
    "gtaf": gtaf_rqs,
    "mtaf": mtaf_rqs,
}


"""
Experiment code
"""


def run_experiment(
    out_path,
    dim,
    seed,
    tail_path,
    model_label,
    opt_params,
):
    # general setup
    if torch.cuda.is_available():
        torch.set_default_device("cuda")

    torch.set_default_dtype(DEFAULT_DTYPE)
    torch.manual_seed(seed)

    # prepare data
    tail_and_scale = {
        key: value[:dim]
        for key, value in load_raw_data(tail_path)["tail_and_scale"][0].items()
    }
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

    x_trn = (x[trn_ix] - tail_and_scale["mean"]) / tail_and_scale["std"]
    x_val = (x[val_ix] - tail_and_scale["mean"]) / tail_and_scale["std"]
    x_tst = (x[tst_ix] - tail_and_scale["mean"]) / tail_and_scale["std"]

    # create model and train
    model_fcn = model_definitions[model_label]
    label = model_label

    model = model_fcn(
        dim,
        tail_and_scale["dfs"],
        tail_and_scale["pos_dfs"],
        tail_and_scale["neg_dfs"],
        model_depth=1,
        model_tail_bound=5.0,
    ).to(DEFAULT_DTYPE)
    fit_data = data_fit.train(
        model,
        x_trn.to(DEFAULT_DTYPE),
        x_val.to(DEFAULT_DTYPE),
        x_tst.to(DEFAULT_DTYPE),
        lr=opt_params["lr"],
        num_epochs=opt_params["num_epochs"],
        batch_size=opt_params["batch_size"],
        label=label,
    )

    tst_loss, tst_ix, losses, vlosses, hook_data = fit_data

    add_raw_data(
        out_path,
        label,
        {"dim": dim, "seed": seed, "tst_ll": float(tst_loss)},
        force_write=True,
    )


def configured_experiments():
    model_labels = ["mtaf", "gtaf", "ttf_fix", "ttf"]
    seed = 2
    tail_path = "sp500_tails"
    dim = 5
    opt_params = {"lr": 5e-4, "num_epochs": 10, "batch_size": 100}
    out_path = "2024-05-de-dry-run"

    for model_label in model_labels:
        run_experiment(
            out_path,
            dim,
            seed,
            tail_path,
            model_label,
            opt_params,
        )
