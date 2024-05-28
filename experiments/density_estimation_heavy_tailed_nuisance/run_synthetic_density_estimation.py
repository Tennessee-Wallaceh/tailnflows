import torch
from functools import partial
from tailnflows.models import flows
from tailnflows.train import data_fit
from tailnflows.utils import load_torch_data, add_raw_data, parallel_runner

DEFAULT_DTYPE = torch.float32


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
    # "normal": normal,
    # "ttf_m": ttf_m,
    # "ttf_m_fix": ttf_m_fix,
    # "mtaf": mtaf,
    # "gtaf": gtaf,
    # "m_normal": m_normal,
    # "g_normal": g_normal,
    "comet": comet,
}

# ttf tends to converge faster, so lower num epoch
# gtaf sometimes needs a little longer to converge
model_opt_params = {
    "normal": {"lr": 5e-3, "num_epochs": 2000, "batch_size": None, "early_stop_patience": 100},    
    "ttf_m": {"lr": 5e-3, "num_epochs": 2000, "batch_size": None, "early_stop_patience": 100},    
    "ttf_m_fix": {"lr": 5e-3, "num_epochs": 2000, "batch_size": None, "early_stop_patience": 100},    
    "mtaf": {"lr": 5e-3, "num_epochs": 2000, "batch_size": None, "early_stop_patience": 100},    
    "gtaf": {"lr": 5e-3, "num_epochs": 2000, "batch_size": None, "early_stop_patience": 100},    
    "m_normal": {"lr": 5e-3, "num_epochs": 2000, "batch_size": None, "early_stop_patience": 100},    
    "g_normal": {"lr": 5e-3, "num_epochs": 2000, "batch_size": None, "early_stop_patience": 100},    
    "comet": {"lr": 5e-3, "num_epochs": 2000, "batch_size": None, "early_stop_patience": 100},    
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
    experiment_ix=1,
):
    # general setup
    torch.set_default_dtype(DEFAULT_DTYPE)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
        device = "cuda"
    else:
        device = "cpu"

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

    model = model_fcn(dim, nuisance_df, x_trn).to(DEFAULT_DTYPE).to(device)

    fit_data = data_fit.train(
        model,
        x_trn.to(DEFAULT_DTYPE).to(device),
        x_val.to(DEFAULT_DTYPE).to(device),
        x_tst.to(DEFAULT_DTYPE).to(device),
        lr=opt_params["lr"],
        num_epochs=opt_params["num_epochs"],
        batch_size=opt_params["batch_size"],
        early_stop_patience=opt_params["early_stop_patience"],
        label=f"{experiment_ix}|{label}-d:{dim}-nu:{nuisance_df:.1f}",
    )

    tst_loss, tst_ix, losses, vlosses, hook_data = fit_data

    add_raw_data(
        out_path,
        label,
        {"dim": dim, "repeat": repeat, "seed": seed, "tst_ll": float(tst_loss), "df": nuisance_df},
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
    out_path = "2024-05-synth-de-earlystop"
    nuisance_dfs = [0.5, 1.0, 2.0, 30.0]
    target_d = [5, 10]
    num_repeats = 10

    experiments = [
        dict(
            out_path=out_path,
            dim=dim,
            nuisance_df=nuisance_df,
            repeat=repeat,
            seed=seed,
            model_label=model_label,
            opt_params=model_opt_params[model_label],
        )
        for repeat in range(num_repeats)
        for dim in target_d
        for nuisance_df in nuisance_dfs
        for model_label in model_labels
    ]

    parallel_runner(run_experiment, experiments, max_runs=5)

if __name__ == "__main__":
    configured_experiments()
