import torch

from tailnflows.models import flows
from tailnflows.train import variational_fit
from tailnflows.utils import add_raw_data, parallel_runner
from tailnflows.targets.heavy_tailed_nuisance import log_density

"""
Model specifications
"""


def base_rqs_spec(dim):
    return flows.base_nsf_transform(
        dim,
        depth=1,
        num_bins=5,
        tail_bound=2.5,
        affine_autoreg_layer=True,
        random_permute=False,
        nn_kwargs=dict(
            use_residual_blocks=True,
            num_blocks=2,
            hidden_features=10 + dim
        ),
    )


def ttf_rqs(dim, nuisance_df):
    # sample using torch for random seed management
    pos_tails = [
        float(tail_p.cpu())
        for tail_p in torch.distributions.Uniform(low=0.05, high=1.00).sample([dim])
    ]
    neg_tails = [
        float(tail_p.cpu())
        for tail_p in torch.distributions.Uniform(low=0.05, high=1.00).sample([dim])
    ]
    return flows.build_ttf_m(
        dim,
        use="variational_inference",
        base_transformation_init=base_rqs_spec,
        model_kwargs=dict(
            fix_tails=False,
            pos_tail_init=pos_tails,
            neg_tail_init=neg_tails,
        ),
        final_rotation='lu'
    )


def ttf_rqs_fix(dim, nuisance_df):
    tail_init = [1 / nuisance_df] * dim
    return flows.build_ttf_m(
        dim,
        use="variational_inference",
        base_transformation_init=base_rqs_spec,
        model_kwargs=dict(
            fix_tails=True,
            pos_tail_init=tail_init,
            neg_tail_init=tail_init,
        ),
        final_rotation='lu'
    )


def gtaf_rqs(dim, nuisance_df):
    dfs = [
        float(df.cpu())
        for df in torch.distributions.Uniform(low=1.0, high=20.0).sample([dim])
    ]
    return flows.build_gtaf(
        dim,
        use="variational_inference",
        base_transformation_init=base_rqs_spec,
        model_kwargs=dict(
            fix_tails=False,
            tail_init=dfs,
        ),
        final_rotation='lu'
    )


def mtaf_rqs(dim, nuisance_df):
    df_init = [nuisance_df] * dim
    return flows.build_mtaf(
        dim,
        use="variational_inference",
        base_transformation_init=base_rqs_spec,
        model_kwargs=dict(fix_tails=True, tail_init=df_init),
        final_rotation='lu'
    )


def nsf(dim, nuisance_df):
    return flows.build_base_model(
        dim,
        use="variational_inference",
        base_transformation_init=base_rqs_spec,
        final_rotation='lu'
    )


model_definitions = {
    "ttf_fix": ttf_rqs_fix,
    "mtaf": mtaf_rqs,
    "nsf": nsf,
    "gtaf": gtaf_rqs,
    "ttf": ttf_rqs,
}


"""
Targets
"""


def build_mixture():
    from torch.distributions import (
        MixtureSameFamily,
        Normal,
        Independent,
        StudentT,
        Categorical,
    )

    comp = Independent(
        StudentT(
            df=torch.tensor([[1.0, 5.0]]),
            loc=torch.tensor(
                [
                    [2.0, 2.0],
                    [-2.0, -2.0],
                ]
            ),
        ),
        1,
    )

    mix = Categorical(torch.tensor([1.0, 1.0]))

    return MixtureSameFamily(mix, comp)


targets = {
    "mixture": lambda x, nuisance_df: build_mixture().log_prob(x),
    "heavy_nuisance": lambda x, nuisance_df: log_density(x, heavy_df=nuisance_df),
}

"""
Experiment code
"""


def run_experiment(
    out_path, model_label, target_label, dim, nuisance_df, loss_label, repeat, opt_params, experiment_ix=0
):
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
        device = "cuda"
    else:
        device = "cpu"

    torch.manual_seed(repeat)

    model_fcn = model_definitions[model_label]

    model = model_fcn(dim, nuisance_df).to(torch.float32).to(device)
    target = targets[target_label]
    target = lambda x: log_density(x, nuisance_df)

    (
        losses,
        final_metrics,
        hook_data,
    ) = variational_fit.train(
        model,
        target,
        label=model_label,
        seed=repeat,
        loss_label=loss_label,
        metric_samples=1_000,
        hook=None,
        **opt_params,
    )

    add_raw_data(
        out_path,
        model_label,
        {
            "seed": repeat,
            "dim": dim,
            "nuisance_df": nuisance_df,
            "loss_label": loss_label,
            "target_label": target_label,
            "tst_elbo": final_metrics.elbo,
            "tst_ess": final_metrics.ess,
            "tst_psis_k": final_metrics.psis_k,
            "tst_cubo": final_metrics.tst_cubo,
        },
        force_write=True,
    )

    add_raw_data(
        out_path + "_losses",
        model_label,
        {
            "seed": repeat,
            "dim": dim,
            "nuisance_df": nuisance_df,
            "loss_label": loss_label,
            "losses": losses.detach().cpu(),
        },
        force_write=True,
    )


def configured_experiments():
    target_dims = [5, 10, 50]
    nuisance_dfs = [0.5]

    model_labels = list(model_definitions.keys())
    model_labels = ["mtaf", "gtaf"]

    opt_params = {
        'ttf': dict(
            lr=1e-3,
            num_epochs=10000,
            batch_size=100
        ),
        'ttf_fix': dict(
            lr=1e-3,
            num_epochs=10000,
            batch_size=100,
            grad_clip_norm=5.,
        ),
        'gtaf': dict(
            lr=1e-3,
            num_epochs=10000,
            batch_size=100,
        ),
        'mtaf': dict(
            lr=1e-3,
            num_epochs=10000,
            batch_size=100,
            grad_clip_norm=5.,
        )
    }

    repeats = 5
    out_path = "2025-01-27-vi-nuisance"
    loss_labels = ["neg_elbo"]
    experiments = [
        dict(
            out_path=out_path,
            model_label=label,
            dim=target_dim,
            nuisance_df=nuisance_df,
            loss_label=loss_label,
            target_label="heavy_nuisance",
            repeat=repeat,
            opt_params=opt_params[label],
        )
        for repeat in range(repeats)
        for target_dim in target_dims
        for nuisance_df in nuisance_dfs
        for label in model_labels
        for loss_label in loss_labels
    ]

    
    for model_label in model_labels:
        assert model_label in model_definitions

    # for experiment in experiments:
    #     run_experiment(**experiment)

    parallel_runner(run_experiment, experiments, max_runs=5)


if __name__ == "__main__":
    configured_experiments()
