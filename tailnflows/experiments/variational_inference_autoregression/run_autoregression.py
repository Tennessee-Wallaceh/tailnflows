from functools import partial
from dataclasses import dataclass

import torch
import pandas as pd

from tailnflows.targets.autoregressive import (
    ARParams,
    from_vector_full,
    get_log_likelihood,
    beta_log_prior,
    get_predictive_ll,
    lag_series,
    obs_df_log_prior,
    obs_scale_log_prior,
)
from tailnflows.targets.sp500_returns import load_return_data
from tailnflows.models.flows import TTF_m, gTAF, RQS
from tailnflows.models.flows import ModelUse
from tailnflows.train import variational_fit
from tailnflows.utils import add_raw_data, get_project_root

N_OBS = 100

"""
Model specifications
"""
base_rqs_spec = dict(
    rotation=True,
    flow_depth=5,
    num_bins=5,
    tail_bound=5.0,
)


def rqs(raw_param_dim: int):
    return RQS(
        raw_param_dim, use=ModelUse.variational_inference, model_kwargs=base_rqs_spec
    )


def ttf_rqs(raw_param_dim: int):
    return TTF_m(
        raw_param_dim,
        use=ModelUse.variational_inference,
        model_kwargs=dict(
            **base_rqs_spec,
            # TTF specific
            final_affine=True,
            fix_tails=False,
            pos_tail_init=0.25 * torch.ones([raw_param_dim]),
            neg_tail_init=0.25 * torch.ones([raw_param_dim]),
        ),
    )


def gtaf_rqs(raw_param_dim: int):
    return gTAF(
        raw_param_dim,
        use=ModelUse.variational_inference,
        model_kwargs=dict(
            **base_rqs_spec,
            # gTAF specific
            tail_init=4.0 * torch.ones([raw_param_dim]),  # df
        ),
    )


models = {
    "ttf_rqs": ttf_rqs,
    "rqs": rqs,
    "gtaf_rqs": gtaf_rqs,
}


"""
Experiment fcns
"""


def get_series(
    series_dim: int, n_obs: int, start_point: int, ar_length: int, pred_length: int
):
    return_data, symbols = load_return_data(series_dim)
    raw_series = torch.tensor(return_data.to_numpy())
    series = raw_series[start_point : start_point + n_obs, :]

    input_x, target_x, pred_input = lag_series(series.T, ar_length)
    pred_target = raw_series[
        start_point + n_obs : start_point + n_obs + pred_length, :
    ].T  # the next pred length after the observation sequence

    return input_x, target_x, pred_input, pred_target


@dataclass
class OptConfig:
    lr: float
    batch_size: int
    num_epochs: int


def plot_fit(trained_models, pred_target, plot_label):
    with torch.no_grad():
        fig, (beta_ax, pred_ax, df_ax, scale_ax) = plt.subplots(
            1, 4, figsize=(22, 4), gridspec_kw={"width_ratios": [10, 10, 1, 1]}
        )

        for model_ix, (label, trained_model) in enumerate(trained_models.items()):
            x_q = trained_model[0].sample(20000).cpu()
            ar_params = from_vector_full(x_q, pred_length, base_params)
            samples_df = pd.DataFrame(x_q, columns=ar_params.names)
            samples_df["obs df"] = ar_params.obs_df
            samples_df["obs scale"] = ar_params.obs_scale

            beta_params = [
                pname for pname in ar_params.names if pname.startswith("beta")
            ]
            predictive_params = [
                pname for pname in ar_params.names if pname.startswith("y")
            ]
            coverage_plot(
                samples_df[beta_params],
                beta_ax,
                model_ix,
                label=f"{label} beta samples",
            )
            coverage_plot(samples_df[predictive_params], pred_ax, model_ix)
            coverage_plot(samples_df[[ar_params.names[0]]], df_ax, model_ix)
            coverage_plot(samples_df[[ar_params.names[1]]], scale_ax, model_ix)

            df_ax.set_title(f"df samples")
            beta_ax.set_title(f"beta samples")
            scale_ax.set_title(f"scale samples")
            pred_ax.set_title(f"pred samples")

            # df_ax.set_ylim([0.0, 5.])
            beta_ax.set_ylim([-1.0, 1.0])
            pred_ax.set_ylim(
                [-0.25, 0.25]
            )  # only plot "sensible predicted returns ie within 25%"

        pred_ax.scatter(
            range(len(predictive_params)), pred_target.reshape(-1).cpu(), marker="x"
        )
    plt.savefig(f"{get_project_root()}/experiment_output/{plot_label}.png")
    plt.show()


def run_experiment(
    out_path: str,
    seed: int,
    model_label: str,
    series_dim: int,
    start_point: int,
    ar_length: int,
    pred_length: int,
    opt_params: OptConfig,
):
    if torch.cuda.is_available():
        torch.set_default_device("cuda")

    torch.manual_seed(seed)

    # set up base parameters
    raw_beta_dim = series_dim * series_dim * ar_length
    raw_param_dim = raw_beta_dim + pred_length * series_dim + 2
    base_params = ARParams(
        ar_length=ar_length,
        obs_dim=series_dim,
        betas=torch.rand([raw_beta_dim, series_dim, series_dim, ar_length]),
        obs_df=torch.tensor(2.0),  # overwritten
        obs_scale=torch.tensor(0.5),  # overwritten
        prior_df=torch.tensor(1.0),
        prior_scale=torch.tensor(0.1),  # we don't believe in strong relations
    )

    # map d dimensional reals to structured PredictiveARParams
    # adds a bit of safety
    unc_to_params = partial(
        from_vector_full, pred_length=pred_length, base_ar_parameters=base_params
    )

    # get lagged data
    input_x, target_x, pred_input, pred_target = get_series(
        series_dim,
        N_OBS,
        start_point,
        ar_length,
        pred_length,
    )

    # build autoregressive log probability fcns + build the target
    ll_fcn = get_log_likelihood(
        input_x, target_x
    )  # params -> likelihood of the observations
    pred_ll = get_predictive_ll(pred_input)
    prior_ll = (
        lambda ar_params: beta_log_prior(ar_params)
        + obs_df_log_prior(ar_params)
        + obs_scale_log_prior(ar_params)
    )  # params, params -> prior param ll

    def target(unc_params):
        ar_params = unc_to_params(unc_params)
        return ll_fcn(ar_params) + prior_ll(ar_params) + pred_ll(ar_params)

    # build model
    model = models[model_label](raw_param_dim)

    # run the train
    losses, final_metrics = variational_fit.train(
        model,
        target,
        lr=opt_params["lr"],
        num_epochs=opt_params["num_epochs"],
        batch_size=opt_params["batch_size"],
        label=f"model: {model_label} start: {start_point} d: {raw_param_dim}",
    )

    # save metrics and diagnostics
    add_raw_data(
        out_path + "/metrics",
        model_label,
        {
            "start_point": start_point,
            "psis_k": final_metrics.psis_k,
            "ess": final_metrics.ess,
            "elbo": final_metrics.elbo,
        },
        force_write=True,
    )

    add_raw_data(
        out_path + "/losses",
        model_label,
        {
            "start_point": start_point,
            "samples": losses.cpu().numpy(),
        },
        force_write=True,
    )

    # this can create a potentially large amount of data
    x_q = model.sample(10_000)
    add_raw_data(
        out_path + "/samples",
        model_label,
        {
            "start_point": start_point,
            "samples": x_q.detach().cpu().numpy(),
        },
        force_write=True,
    )


def configured_experiments():
    opt_params = {"lr": 1e-3, "num_epochs": 1e2, "batch_size": 1000}
    seed = 0
    model_label = "ttf_rqs"
    series_dim = 5
    start_point = 100
    ar_length = 1
    pred_length = 5
    out_path = "2024-04-vi-autoreg-dry"

    for model_label in ["ttf_rqs", "rqs", "gtaf_rqs"]:
        run_experiment(
            out_path,
            seed,
            model_label,
            series_dim,
            start_point,
            ar_length,
            pred_length,
            opt_params,
        )
