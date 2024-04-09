import torch
from torch.distributions.studentT import StudentT
from torch.distributions.normal import Normal
from torch.optim import Adam
import tqdm


def generate_data(n, d, nuisance_df):
    assert d > 1

    if nuisance_df > 0.0:
        nuisance_base = StudentT(df=nuisance_df)
    else:
        nuisance_base = Normal(loc=0.0, scale=1.0)

    normal_base = Normal(loc=0.0, scale=1.0)
    x = nuisance_base.sample([n, d])

    noise = normal_base.sample([n, 1])

    y = x[:, [-1]] + noise
    return x, y


def run_experiment(neural_net, nuisance_df, d=10, lr=1e-3, num_iter=5000, n=5000):
    # num obs
    m = 100  # batch size

    x, y = generate_data(n, d, nuisance_df)
    x_tst, y_tst = generate_data(n, d, nuisance_df)
    x_val, y_val = generate_data(n, d, nuisance_df)

    params = list(neural_net.parameters())
    optimizer = Adam(params, lr=lr)

    loop = tqdm.tqdm(range(int(num_iter)))

    vlosses = []
    tst_ix = None
    tst_loss = torch.inf
    for i in loop:
        rows = torch.randperm(n)[:m]  # sample without replacement
        x_train = x[rows, :]
        y_train = y[rows, :]

        # evaluate loss
        y_approx = neural_net(x_train)

        loss = (y_train - y_approx).square().mean()  # mse

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            y_approx = neural_net(x_val)
            vloss = (y_val - y_approx).square().mean()  # mse
            vlosses.append(vloss)

            if len(vlosses) > 1 and vloss <= min(vlosses):
                y_approx = neural_net(x_tst)
                tst_loss = (y_tst - y_approx).square().mean()
                tst_ix = i

        loop.set_postfix({"loss": f"{loss:.2f} | * {tst_loss:.2f}"})

    return tst_loss, tst_ix
