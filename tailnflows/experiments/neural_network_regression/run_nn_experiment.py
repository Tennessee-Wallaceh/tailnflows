import torch
import gc
from torch.optim import Adam
import tqdm
import argparse

from tailnflows.utils import add_raw_data

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
        y_approx = neural_net(x_trn)
        loss = (y_trn - y_approx).square().mean()  # mse

        loss.backward()
        optimizer.step()

        # evaluate
        with torch.no_grad():
            neural_net.eval()
            y_approx = neural_net(x_val)
            vloss = (y_val - y_approx).square().mean()  # mse
            vlosses[epoch] = vloss
            if vloss <= min_vloss:
                min_vloss = vloss
                y_approx = neural_net(x_tst)
                tst_loss = (x_tst[:, [-1]] - y_approx).square().mean().detach()

        loop.set_postfix({"loss": f"{loss.detach():.2f} | * {tst_loss:.2f} {label}"})
        
    return float(tst_loss.cpu())


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
    label:str = "",
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

    tst_loss = fit_nn(mlp, df, dim, num_epochs, lr, label)

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
    import itertools
    import multiprocessing as mp

    max_runs = 2
    out_path = '2024-04-nn-nonoise-loss'
    seeds = range(3)
    dims = [5, 10, 50, 100]
    dfs = [1., 2., 5., 30.]
    activations = ['sigmoid', 'relu']

    experiments = list(itertools.product(seeds, dims, dfs, activations))
    sem = mp.Semaphore(max_runs)
    def wrapped_run(sem, **kwargs):
        with sem: # limit to max runs concurrently
            run_experiment(**kwargs)

    processes = []
    print(f'{len(experiments)} experiments to run...')
    for exp_ix, (seed, dim, df, activation_fcn) in enumerate(experiments): 
        p = mp.Process(target=wrapped_run, args=(sem,), kwargs=dict(
            out_path=out_path,
            seed=seed,
            dim=dim,
            activation_fcn_name=activation_fcn,
            hidden_dims=[50, 50],
            df=df,
            num_epochs=5000,
            lr=1e-3,
            verbose=False,
            label=f'({exp_ix})'
            ))
        p.start()
        processes.append(p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the flexible tails for Normalizing Flows neural network example."
    )
    parser.add_argument(
        "--out_path",
        type=str,
        help="output path, starting from TAILFLOWS_HOME env var",
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed used for data gen + training",
        required=True,
    )
    parser.add_argument("--dim", type=int, help="Dimension of the data", required=True)
    parser.add_argument(
        "--activation_fcn",
        type=str,
        help="Activation fcn in {'sigmoid', 'relu'}",
        required=True,
    )
    parser.add_argument(
        "--df",
        type=float,
        help="The nuisance df, df=0. corresponds to inf (guassian)",
        required=True,
    )
    parser.add_argument(
        "--hidden_dim_str", type=str, help="NN hidden dimensions", default="50-50"
    )
    parser.add_argument(
        "--num_epochs", type=str, help="Number of train loops", default=5000
    )
    parser.add_argument("--lr", type=str, help="Learning rate", default=1e-3)
    parser.add_argument(
        "--verbose", type=bool, help="Set to True for more prints", default=False
    )

    args = parser.parse_args()

    hidden_dims = [int(h_dim) for h_dim in args.hidden_dim_str.split("-")]

    run_experiment(
        out_path=args.out_path,
        seed=args.seed,
        dim=args.dim,
        activation_fcn_name=args.activation_fcn,
        hidden_dims=hidden_dims,
        df=args.df,
        num_epochs=args.num_epochs,
        lr=args.lr,
        verbose=args.verbose,
    )
