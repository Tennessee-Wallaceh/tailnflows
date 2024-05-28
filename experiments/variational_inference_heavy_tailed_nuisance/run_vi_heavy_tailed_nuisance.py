import torch

from tailnflows.models import flows
from tailnflows.train import variational_fit
from tailnflows.utils import add_raw_data
from tailnflows.targets.heavy_tailed_nuisance import log_density

"""
Model specifications
"""


def base_rqs_spec(dim):
    return flows.base_nsf_transform(
        dim, num_bins=8, tail_bound=5.0, affine_autoreg_layer=True, linear_layer=True
    )


def ttf_rqs(dim, nuisance_df):
    return flows.build_ttf_m(
        dim,
        use="variational_inference",
        base_transformation_init=base_rqs_spec,
        model_kwargs=dict(
            fix_tails=False,
            pos_tail_init=torch.distributions.Uniform(low=0.05, high=1.0).sample([dim]),
            neg_tail_init=torch.distributions.Uniform(low=0.05, high=1.0).sample([dim]),
        ),
    )


def ttf_rqs_fix(dim, nuisance_df):
    tail_init = 1 / nuisance_df * torch.ones([dim])
    return flows.build_ttf_m(
        dim,
        use="variational_inference",
        base_transformation_init=base_rqs_spec,
        model_kwargs=dict(
            fix_tails=True,
            pos_tail_init=tail_init,
            neg_tail_init=tail_init,
        ),
    )


def gtaf_rqs(dim, nuisance_df):
    return flows.build_gtaf(
        dim,
        use="variational_inference",
        base_transformation_init=base_rqs_spec,
        model_kwargs=dict(
            fix_tails=False,
            tail_init=torch.distributions.Uniform(low=1.0, high=20.0).sample([dim]),
        ),
    )


def mtaf_rqs(dim, nuisance_df):
    df_init = nuisance_df * torch.ones([dim])
    return flows.build_mtaf(
        dim,
        use="variational_inference",
        base_transformation_init=base_rqs_spec,
        model_kwargs=dict(fix_tails=True, tail_init=df_init),
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


def run_experiment(out_path, label, dim, nuisance_df, repeat, experiment_ix=1):
    torch.manual_seed(repeat)

    model = model_fcn(dim, nuisance_df).to(torch.float32)

    losses, final_metrics = variational_fit.train(
        model,
        lambda x: log_density(x, heavy_df=nuisance_df),
        lr=1e-3,
        num_epochs=300,
        batch_size=100,
        label=label,
        seed=repeat,
    )

    add_raw_data(
        out_path,
        label,
        {
            "seed": repeat,
            "dim": dim,
            "nuisance_df": nuisance_df,
            "tst_elbos": final_metrics.elbo,
            "tst_ess": final_metrics.ess,
            "tst_psis_k": final_metrics.psis_k,
        },
        force_write=True,
    )

    add_raw_data(
        out_path + "_losses",
        label,
        {
            "seed": repeat,
            "dim": dim,
            "nuisance_df": nuisance_df,
            "losses": losses.detach().cpu(),
        },
        force_write=True,
    )


def configured_experiments():
    target_dims = [5]
    nuisance_dfs = [1.0, 2.0, 30.0]
    model_labels = model_definitions.keys()
    repeats = 5
    out_path = "2024-05-vi-nuisance"
    experiments = [
        dict(
            out_path=out_path,
            label=label,
            dim=target_dim,
            out_panuisance_df=nuisance_df,
            repeat=repeat,
        )
        for repeat in range(repeats)
        for target_dim in target_dims
        for nuisance_df in nuisance_dfs
        for label in model_labels
    ]

    parallel_runner(run_experiment, experiments, max_runs=4)


if __name__ == "__main__":
    configured_experiments()
