import torch
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
    return x, y


def fit_nn(neural_net, nuisance_df, dim, num_epochs=500, lr=1e-3) -> float:
    # num obs
    x_trn, y_trn = generate_data(NUM_OBS, dim, nuisance_df)
    x_tst, y_tst = generate_data(NUM_OBS, dim, nuisance_df)
    x_val, y_val = generate_data(NUM_OBS, dim, nuisance_df)

    params = list(neural_net.parameters())
    optimizer = Adam(params, lr=lr)

    loop = tqdm.tqdm(range(int(num_epochs)))

    vlosses = []
    tst_loss = torch.inf
    for _ in loop:
        # sample without replacement
        rows = torch.randperm(NUM_OBS)[:BATCH_SIZE]
        x_batch = x_trn[rows, :]
        y_batch = y_trn[rows, :]

        # train step
        optimizer.zero_grad()

        neural_net.train()
        y_approx = neural_net(x_batch)
        loss = (y_batch - y_approx).square().mean()  # mse

        loss.backward()
        optimizer.step()

        # evaluate
        with torch.no_grad():
            neural_net.eval()
            y_approx = neural_net(x_val)
            vloss = (y_val - y_approx).square().mean()  # mse
            vlosses.append(vloss)

            if len(vlosses) > 1 and vloss <= min(vlosses):
                y_approx = neural_net(x_tst)
                tst_loss = (y_tst - y_approx).square().mean()

        loop.set_postfix({"loss": f"{loss:.2f} | * {tst_loss:.2f}"})

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
):
    if torch.cuda.is_available():
        torch.set_default_device("cuda")

    assert df >= 0.0, "Degree of freedom must be >= 0!"

    torch.manual_seed(seed)

    activation_fcn = activations[activation_fcn_name]

    mlp = MLP([dim], [1], hidden_dims, activation=activation_fcn)

    if verbose:
        if torch.cuda.is_available():
            device = torch.cuda.device(torch.cuda.current_device())
        else:
            device = "cpu"
        print(
            f"Device: {device}",
        )

    tst_loss = fit_nn(mlp, df, dim, num_epochs, lr)

    # save results
    result = {
        "dim": dim,
        "df": df,
        "seed": seed,
        "activation": activation_fcn_name,
        "tst_loss": tst_loss,
    }
    add_raw_data(out_path, "", result, force_write=True)


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
