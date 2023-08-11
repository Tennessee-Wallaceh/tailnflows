# type: ignore
import datetime
import numpy as np
import tqdm
import argparse
import matplotlib.pyplot as plt
import pandas as pd

1
# pytorch
import torch
from torch.optim import Adam

# tuning
from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler

# our modules
from tailnflows.models.density_estimation import get_model, ModelName
from tailnflows.data import data_sources
from tailnflows.experiments.utils import (
    new_experiment,
    write_experiment_details,
    save_model,
)
from tailnflows.utils import get_project_root
from tailnflows.models.base_distribution import TrainableStudentT

from torch.nn.functional import softplus
from tailnflows.models.extreme_transformations import (
    flip,
    EXMarginalTransform,
    TailMarginalTransform,
    MaskedTailAutoregressiveTransform,
)
from tailnflows.models.tail_estimation import estimate_df

precision_types = {
    "float64": torch.float64,
    "float32": torch.float32,
}

tuneables = {
    "lr_grid": tune.grid_search([4e-2, 3e-2, 2e-2, 1e-2, 5e-3, 1e-3]),
    "l2_reg": tune.grid_search([0.0, 1e-3, 1.0, 5.0, 10.0]),
}


def p_to_config(model, name, params, train_config):
    tail_lr = train_config.get("tail_lr", 2e-2)

    first_transform = model._transform._transforms[0]
    base_distribution = model._distribution
    extra_p = {}
    if isinstance(first_transform, TailMarginalTransform):
        if name.startswith("_transform._transforms.0"):
            extra_p = {"lr": tail_lr}

    if isinstance(first_transform, EXMarginalTransform):
        if name.startswith("_transform._transforms.0"):
            extra_p = {"lr": tail_lr}

    if isinstance(base_distribution, TrainableStudentT):
        if name.startswith("_distribution"):
            extra_p = {"lr": tail_lr}

    return {"params": params, **extra_p}


def eval_loss(model, data_loader):
    model.eval()  # set the model to evaluation mode
    # do not calculate gradients in validation stage
    with torch.no_grad():
        loss = 0
        for inputs in data_loader:
            loss += -model.log_prob(inputs).mean()
        loss = loss / len(data_loader)
    return loss


def run_train(train_config, model, data_ref):
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    # setup target data
    trn_data, val_data = ray.get(data_ref)
    train_loader = torch.utils.data.DataLoader(
        trn_data,
        batch_size=train_config["batch_size"],
        shuffle=True,
        generator=torch.Generator(device=device),
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=train_config["batch_size"],
        generator=torch.Generator(device=device),
    )

    model = model.to(trn_data.data.dtype).to(device)  # ensure model is correct dtype

    # preconfigure model
    if train_config["preconfigure"] != "":
        model = preconfigure_model(
            model,
            train_config["preconfigure"],
            torch.concat([trn_data.data, val_data.data]),
        )

    model = model.to(trn_data.data.dtype).to(device)  # rensure model is correct dtype

    # optimizer
    named_params = dict(model.named_parameters())
    optimizer = Adam(
        [
            p_to_config(model, name, params, train_config)
            for name, params in named_params.items()
        ],
        lr=train_config["lr"],
        weight_decay=train_config["l2_reg"],
    )

    # handle checkpointing
    checkpoint = session.get_checkpoint()
    if checkpoint:
        checkpoint_state = checkpoint.to_dict()
        start_epoch = checkpoint_state["epoch"]
        model.load_state_dict(checkpoint_state["model_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    # training loop
    for epoch in range(
        start_epoch, train_config["num_epochs"]
    ):  # loop over the dataset multiple times
        # train
        model.train()  # set the model to training mode
        train_loss = 0.0
        for inputs in train_loader:
            optimizer.zero_grad()
            loss = -model.log_prob(inputs).mean()
            train_loss += loss
            loss.backward()
            optimizer.step()
        train_loss = train_loss / len(train_loader)

        # val
        validation_loss = eval_loss(model, val_loader)

        # checkpoint
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        checkpoint = Checkpoint.from_dict(checkpoint_data)

        session.report(
            {
                "val_loss": validation_loss.item(),
                "trn_loss": train_loss.item(),
            },
            checkpoint=checkpoint,
        )


def run_experiment(
    seed,
    target_name,
    target_kwargs,
    train_config,
    model_name,
    model_kwargs,
    precision="float64",
    subdirectory=None,
):
    # config
    torch.manual_seed(seed)
    data_seed = seed + 1000  # just start as shift of global seed
    tst_data = data_sources[target_name](data_seed, data_use="test", **target_kwargs)
    trn_data = data_sources[target_name](data_seed, data_use="train", **target_kwargs)
    val_data = data_sources[target_name](
        data_seed, data_use="validate", **target_kwargs
    )
    data_ref = ray.put((trn_data, val_data))
    tst_loader = torch.utils.data.DataLoader(
        tst_data, batch_size=train_config["batch_size"]
    )

    # setup model
    model = get_model(ModelName[model_name], trn_data.shape[1], model_kwargs)

    # record experiment details
    if subdirectory is None:
        base_path = f"{get_project_root()}/experiment_output/{target_name}"
    else:
        base_path = (
            f"{get_project_root()}/experiment_output/{subdirectory}/{target_name}"
        )
    experiment_id = new_experiment(base_path)
    subfolder = f"id_{experiment_id}|seed_{seed}"
    path = f"{base_path}/{subfolder}"
    print(f"targetting {target_name} with {model_name} flow, params:[{subfolder}]...")

    details = {
        "seed": seed,
        "target": target_name,
        "target_kwargs": target_kwargs,
        "train_config": train_config,
        "model": model_name,
        "model_kwargs": model_kwargs,
        "run_time": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"),
        "path": path,
        "dim": tst_data.dim,
    }
    write_experiment_details(path, details)

    # configure and run train loop
    _train_fcn = lambda train_config: run_train(
        data_seed,
        device,
        target_name,
        train_config,
        model=model,
        data_ref=data_ref,
        target_kwargs=target_kwargs,
    )

    # run fits
    train_config = {
        option: tuneables.get(val, val) for option, val in train_config.items()
    }
    tuner = tune.Tuner(
        _train_fcn,
        param_space=train_config,
        # scheduler=scheduler,
    )
    results = tuner.fit()

    best_trial = results.get_best_result(metric="val_loss", mode="min", scope="all")
    best_epoch = best_trial.metrics_dataframe.val_loss.argmin()
    best_val_loss = best_trial.metrics_dataframe.val_loss.min()

    checkpoint_data = best_trial.checkpoint.to_dict()
    model.load_state_dict(checkpoint_data["model_state_dict"])
    test_loss = eval_loss(model, tst_loader)
    save_model(path, model)  # save the best model

    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_val_loss:.2f}")
    print(f"Best trial final test loss: {test_loss.item():.2f}")

    # provide some plots
    ax = None  # This plots everything on the same plot
    joint_df = pd.DataFrame()
    for result in results:
        result_df = result.metrics_dataframe
        if result.config == best_trial.config:
            ax = result_df.val_loss.plot(ax=ax, c="orange", label="val_loss")
            ax = result_df.trn_loss.plot(
                ax=ax, c="red", linestyle="--", label="trn_loss"
            )
        else:
            ax = result_df.val_loss.plot(ax=ax, c="blue", alpha=0.5)
            ax = result_df.trn_loss.plot(ax=ax, linestyle="--", c="blue", alpha=0.5)

        for key, val in result.config.items():
            result_df[key] = val
        joint_df = pd.concat([joint_df, result_df])

    joint_df.to_csv(f"{path}/joint_training_data.csv")

    ax.axvline(best_epoch, linestyle="--", c="black", alpha=0.5, label="best_epoch")
    ax.axhline(test_loss, linestyle="--", c="green", alpha=0.5, label="tst_loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(f"best conf: {best_trial.config}")
    ax.legend()
    plt.savefig(f"{path}/tune_figure.png")
    plt.close()

    # save training data as np arrays
    training_data = {
        "best_val_loss": best_val_loss,
        "best_val_epoch": best_epoch,
        "test_loss": test_loss,
    }
    with open(f"{path}/training_data.npy", "wb") as f:
        np.save(f, training_data)


# map a string to a more complicated configuration
target_configs = {
    "top_10": {"top_n_symbols": 10},
    "top_20": {"top_n_symbols": 20},
    "top_50": {"top_n_symbols": 50},
    "top_100": {"top_n_symbols": 100},
}

train_configs = {
    "fixed": {
        "lr": 1e-3,
        "batch_size": 500,
        "num_epochs": 500,
        "l2_reg": 0.0,
        "preconfigure": "",
    },
    "mtaf": {
        "lr": 1e-3,
        "batch_size": 500,
        "num_epochs": 500,
        "l2_reg": 0.0,
        "preconfigure": "mTAF",
    },
    "dextreme": {
        "lr": 1e-3,
        "batch_size": 500,
        "num_epochs": 500,
        "l2_reg": 0.0,
        "preconfigure": "fix_dextreme",
    },
    "high_lr": {
        "lr": 1e-1,
        "batch_size": 500,
        "num_epochs": 500,
        "l2_reg": 0.0,
    },
    "tune_lr": {
        "lr": "lr_grid",
        "batch_size": 500,
        "num_epochs": 500,
    },
    "tail_tune_lr": {
        "lr": 1e-3,
        "tail_lr": "lr_grid",
        "batch_size": 500,
        "num_epochs": 500,
        "l2_reg": 0.0,
    },
    "tune_l2": {
        "lr": 1e-3,
        "l2_reg": "l2_reg",
        "batch_size": 500,
        "num_epochs": 500,
    },
}


def run_preconfigured():
    seed_set = range(10, 100, 10)
    seed_set = [0]
    for seed in seed_set:
        model_specs = [
            ("mTAF", {"num_layers": 1, "tail_bound": 3.0}, "mtaf"),
            ("gTAF", {"tail_bound": 3.0}, "fixed"),
            ("EXF_m", {"tail_bound": 3.0}, "fixed"),
            ("EXF_m", {"tail_bound": 3.0}, "dextreme"),
        ]
        for model, model_kwargs, tconfig in model_specs:
            tc = train_configs[tconfig]
            tc["num_epochs"] = 10
            run_experiment(
                seed=seed,
                target_name="noise_dim",
                target_kwargs={"d_nuisance": 2},
                train_config=tc,
                model_name=model,
                model_kwargs=model_kwargs,
                device="cpu",
                precision="float64",
                subdirectory="noise_dim_new",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-preconfigured_run", action="store_true")
    parser.add_argument("-seed", type=int, action="store", default=0)
    parser.add_argument("-model_name", type=str, action="store", default="TTF")
    parser.add_argument("-target_config", type=str, action="store", default="top_100")
    parser.add_argument("-train_config", type=str, action="store", default="fixed")

    args = parser.parse_args()

    if args.preconfigured_run:
        run_preconfigured()
    else:
        target_kwargs = target_configs[args.target_config]
        train_config = train_configs[args.train_config]

        run_experiment(
            seed=args.seed,
            target_name="financial_returns",
            target_kwargs=target_kwargs,
            train_config=train_config,
            model_name=args.model_name,
            model_kwargs={},
            device="cpu",
            precision="float64",
            subdirectory="rerun_ttf",
        )
