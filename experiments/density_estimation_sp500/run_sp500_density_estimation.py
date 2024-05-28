import torch

from tailnflows.targets.sp500_returns import load_return_data
from tailnflows.models import flows
from tailnflows.train import data_fit
from tailnflows.utils import load_raw_data, add_raw_data

DEFAULT_DTYPE = torch.float32

"""
Model specifications
"""


def base_rqs_spec(dim):
    return flows.base_nsf_transform(
        dim, num_bins=3, tail_bound=5.0, affine_autoreg_layer=True, linear_layer=True
    )


def ttf_rqs(dim, dfs):
    return flows.build_ttf_m(
        dim,
        use="density_estimation",
        base_transformation_init=base_rqs_spec,
        model_kwargs=dict(
            fix_tails=False,
            pos_tail_init=torch.distributions.Uniform(low=0.05, high=1.0).sample([dim]),
            neg_tail_init=torch.distributions.Uniform(low=0.05, high=1.0).sample([dim]),
        ),
    )


def ttf_rqs_fix(dim, dfs):
    return flows.build_ttf_m(
        dim,
        use="density_estimation",
        base_transformation_init=base_rqs_spec,
        model_kwargs=dict(
            fix_tails=True,
            pos_tail_init=torch.tensor([1 / df if df != 0.0 else 1e-4 for df in dfs]),
            neg_tail_init=torch.tensor([1 / df if df != 0.0 else 1e-4 for df in dfs]),
        ),
    )


def gtaf_rqs(dim, dfs):
    return flows.build_gtaf(
        dim,
        use="density_estimation",
        base_transformation_init=base_rqs_spec,
        model_kwargs=dict(
            fix_tails=False,
            tail_init=torch.distributions.Uniform(low=1.0, high=20.0).sample([dim]),
        ),
    )


def mtaf_rqs(dim, dfs):
    return flows.build_mtaf(
        dim,
        use="density_estimation",
        base_transformation_init=base_rqs_spec,
        model_kwargs=dict(fix_tails=True, tail_init=torch.tensor(dfs)),
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
