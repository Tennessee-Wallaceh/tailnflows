import torch
import gc
from torch.optim import Adam
import tqdm
import argparse

from tailnflows.utils import add_raw_data, parallel_runner

# model
from nflows.nn.nets import MLP

NUM_OBS = 5000
BATCH_SIZE = 100

activations = {
    "relu": torch.nn.functional.relu,
    "sigmoid": torch.nn.functional.sigmoid,
}


def generate_data(n, d, nuisance_df):
    if nuisance_df > 0.0:
        nuisance_base = torch.distributions.StudentT(df=nuisance_df)
    else:
        nuisance_base = torch.distributions.Normal(loc=0.0, scale=1.0)

    normal_base = torch.distributions.Normal(loc=0.0, scale=1.0)
    x = nuisance_base.sample([n, d])

    noise = normal_base.sample([n, 1])

    y = x[:, [-1]] + noise

    # force no grads
    x = x.detach()
    y = y.detach()

    return x, y


def fit_nn(neural_net, nuisance_df, dim, num_epochs=500, lr=1e-3, label="") -> float:
    # num obs
    x_trn, y_trn = generate_data(NUM_OBS, dim, nuisance_df)
    x_tst, y_tst = generate_data(NUM_OBS, dim, nuisance_df)
    x_val, y_val = generate_data(NUM_OBS, dim, nuisance_df)

    params = list(neural_net.parameters())
    optimizer = Adam(params, lr=lr)

    loop = tqdm.tqdm(range(int(num_epochs)))

    vlosses = torch.zeros(num_epochs)
    min_vloss = torch.tensor(torch.inf)
    tst_loss = torch.tensor(torch.inf)
    for epoch in loop:
        # train step
        optimizer.zero_grad()

        neural_net.train()

        loss = (y_trn - neural_net(x_trn)).square().mean()  # mse

        loss.backward()
        optimizer.step()

        # evaluate
        with torch.no_grad():
            neural_net.eval()
            vloss = (y_val - neural_net(x_val)).square().mean()  # mse
            vlosses[epoch] = vloss
            if vloss <= min_vloss:
                min_vloss = vloss
                tst_loss = (x_tst[:, [-1]] - neural_net(x_tst)).square().mean().detach()

        loop.set_postfix({"loss": f"{loss.detach():.2f} | * {tst_loss:.2f} {label}"})

    return float(tst_loss)


def run_experiment(
    out_path: str,
    seed: int,
    dim: int,
    activation_fcn_name: str,
    hidden_dims: list[int],
    df: float,
    num_epochs: int,
    lr: float,
    verbose: bool = False,
    label: str = "",
    experiment_ix: int = 1,
):
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
        device = "cuda"
    else:
        device = "cpu"

    assert df >= 0.0, "Degree of freedom must be >= 0!"

    torch.manual_seed(seed)

    activation_fcn = activations[activation_fcn_name]

    mlp = MLP([dim], [1], hidden_dims, activation=activation_fcn).to(device)

    if verbose:
        if torch.cuda.is_available():
            phys_device = torch.cuda.device(torch.cuda.current_device())
        else:
            phys_device = "cpu"
        print(
            f"Device: {phys_device}",
        )

    tst_loss = fit_nn(mlp, df, dim, num_epochs, lr, f'{experiment_ix}|{label}')

    # save results
    result = {
        "dim": dim,
        "df": df,
        "seed": seed,
        "activation": activation_fcn_name,
        "tst_loss": float(tst_loss.cpu()),
    }
    add_raw_data(out_path, "", result, force_write=True)


def configured_experiments():
    # python configuration for running number of experiments
    # uses multiprocessing to target multiple runs on 1 GPU so needs
    # to be tuned
    out_path = "2024-05-nn-nonoise-loss"
    seeds = range(3)
    dims = [5, 10, 50, 100]
    dfs = [1.0, 2.0, 5.0, 30.0]
    activations = ["sigmoid", "relu"]
    num_repeats = 10

    experiments = [
        dict(
            out_path=out_path,
            seed=repeat,
            dim=dim,
            activation_fcn_name=activation_fcn,
            hidden_dims=[50, 50],
            df=df,
            num_epochs=5000,
            lr=1e-3,
            verbose=False,
        )
        for repeat in range(num_repeats) 
        for dim in dims
        for df in dfs
        for activation_fcn in activations
    ]

    parallel_runner(
        run_experiment, 
        experiments, 
        max_runs=5
    )


if __name__ == "__main__":
    configured_experiments()