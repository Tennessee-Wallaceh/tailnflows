import torch

from tailnflows.models import flows
from tailnflows.train import data_fit
from tailnflows.utils import load_torch_data, add_raw_data

DEFAULT_DTYPE = torch.float64

"""
Model specifications
"""


def base_transformation(dim):
    return flows.base_nsf_transform(
        dim, num_bins=8, tail_bound=3.0, affine_autoreg_layer=True
    )


def normal(dim, nuisance_df, x_trn):
    return flows.build_base_model(
        dim, use="density_estimation", base_transformation_init=base_transformation
    )


def ttf_m(dim, nuisance_df, x_trn):
    return flows.build_ttf_m(
        dim,
        use="density_estimation",
        base_transformation_init=base_transformation,
        model_kwargs=dict(
            fix_tails=False,
            pos_tail_init=torch.distributions.Uniform(low=0.05, high=1.0).sample([dim]),
            neg_tail_init=torch.distributions.Uniform(low=0.05, high=1.0).sample([dim]),
        ),
    )


def ttf_m_fix(dim, nuisance_df, x_trn):
    tail_init = 1 / nuisance_df * torch.ones(dim)
    return flows.build_ttf_m(
        dim,
        use="density_estimation",
        base_transformation_init=base_transformation,
        model_kwargs=dict(
            fix_tails=True, pos_tail_init=tail_init, neg_tail_init=tail_init
        ),
    )


def mtaf(dim, nuisance_df, x_trn):
    df_init = nuisance_df * torch.ones(dim)
    return flows.build_mtaf(
        dim,
        use="density_estimation",
        base_transformation_init=base_transformation,
        model_kwargs=dict(fix_tails=True, tail_init=df_init),
    )


def gtaf(dim, nuisance_df, x_trn):
    return flows.build_gtaf(
        dim,
        use="density_estimation",
        base_transformation_init=base_transformation,
        model_kwargs=dict(
            fix_tails=True,
            tail_init=torch.distributions.Uniform(low=1.0, high=20.0).sample([dim]),
        ),
    )


def m_normal(dim, nuisance_df, x_trn):
    return flows.build_mix_normal(
        dim,
        use="density_estimation",
        base_transformation_init=base_transformation,
    )


def g_normal(dim, nuisance_df, x_trn):
    return flows.build_gen_normal(
        dim,
        use="density_estimation",
        base_transformation_init=base_transformation,
    )


def comet(dim, nuisance_df, x_trn):
    tail_init = nuisance_df * torch.ones(dim)
    return flows.build_comet(
        dim,
        use="density_estimation",
        base_transformation_init=base_transformation,
        model_kwargs=dict(data=x_trn, fix_tails=True, tail_init=tail_init),
    )


model_definitions = {
    "normal": normal,
    "ttf_m": ttf_m,
    "ttf_m_fix": ttf_m_fix,
    "mtaf": mtaf,
    "gtaf": gtaf,
    "m_normal": m_normal,
    "g_normal": g_normal,
    "comet": comet,
}


"""
Experiment code
"""


def load_synthetic_data(dim, nuisance_df, repeat):
    readable_df = str(nuisance_df).replace(".", ",")
    file_name = f"dim-{dim}_v-{readable_df}_repeat-{repeat}"
    try:
        dataset = load_torch_data(f"synthetic_shift/{file_name}")
    except FileNotFoundError:
        print(
            f"No file with that configuration, either sync or generate. ({file_name})"
        )
        return None

    return dataset


def run_experiment(
    out_path,
    dim,
    nuisance_df,
    repeat,
    seed,
    model_label,
    opt_params,
):
    # general setup
    if torch.cuda.is_available():
        torch.set_default_device("cuda")

    torch.set_default_dtype(DEFAULT_DTYPE)
    torch.manual_seed(seed)

    # prepare data
    dataset = load_synthetic_data(dim, nuisance_df, repeat)
    x = dataset["data"]
    trn_ix = dataset["split"]["trn"]
    val_ix = dataset["split"]["val"]
    tst_ix = dataset["split"]["tst"]
    x_trn = x[trn_ix]
    x_val = x[val_ix]
    x_tst = x[tst_ix]

    # create model and train
    model_fcn = model_definitions[model_label]
    label = model_label

    model = model_fcn(dim, nuisance_df, x_trn).to(DEFAULT_DTYPE)

    fit_data = data_fit.train(
        model,
        x_trn.to(DEFAULT_DTYPE),
        x_val.to(DEFAULT_DTYPE),
        x_tst.to(DEFAULT_DTYPE),
        lr=opt_params["lr"],
        num_epochs=opt_params["num_epochs"],
        batch_size=opt_params["batch_size"],
        label=f"{label}-d:{dim}-nu:{nuisance_df:.1f}",
    )

    tst_loss, tst_ix, losses, vlosses, hook_data = fit_data

    add_raw_data(
        out_path,
        label,
        {"dim": dim, "repeat": repeat, "seed": seed, "tst_ll": float(tst_loss)},
        force_write=True,
    )

    add_raw_data(
        out_path + "_losses",
        label,
        {
            "losses": losses.detach().cpu(),
            "vlosses": vlosses.detach().cpu(),
            "tst_ix": tst_ix,
            "seed": seed,
        },
        force_write=True,
    )


def configured_experiments():
    model_labels = model_definitions.keys()
    seed = 2
    opt_params = {"lr": 5e-3, "num_epochs": 10, "batch_size": 100}
    out_path = "2024-05-synth-de-dry"
    nusiance_dfs = [0.5, 1.0, 2.0, 30.0]
    target_d = [5]

    print("running")
    for repeat in range(5):
        for dim in target_d:
            for nuisance_df in nusiance_dfs:
                for model_label in model_labels:
                    run_experiment(
                        out_path,
                        dim,
                        nuisance_df,
                        repeat,
                        seed,
                        model_label,
                        opt_params,
                    )


if __name__ == "__main__":
    configured_experiments()
