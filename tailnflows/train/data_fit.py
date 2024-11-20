import torch
from torch.optim import Adam
import tqdm
from torch.utils.data import TensorDataset, DataLoader


def train(
    model,
    x_trn,
    x_val,
    x_tst,
    lr=1e-3,
    num_epochs=500,
    batch_size=100,
    label="",
    hook=None,
    early_stop_patience=None,
    grad_clip=None,
    optimizer=None,
):
    parameters = list(model.parameters())
    if optimizer is None:
        optimizer = Adam(parameters, lr=lr)

    if batch_size is None:
        trn_data = [(x_trn,)]
    else:
        trn_data = DataLoader(
            TensorDataset(x_trn),
            batch_size=batch_size,
            shuffle=True,
        )

    loop = tqdm.tqdm(range(num_epochs))
    losses = torch.empty(num_epochs)
    vlosses = torch.empty(num_epochs)
    hook_data = {}
    tst_loss = torch.tensor(torch.inf)
    best_val_loss = torch.tensor(torch.inf)
    tst_ix = -1
    for epoch in loop:
        # mini batch
        epoch_loss = 0.0
        for (subset,) in trn_data:
            optimizer.zero_grad()
            batch_loss = -model.log_prob(subset).sum()
            trn_loss = batch_loss.mean()
            trn_loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(parameters, grad_clip)

            optimizer.step()
            epoch_loss += batch_loss

        epoch_loss /= x_trn.shape[0]

        with torch.no_grad():
            if hook is not None:
                hook(model, hook_data)

            val_loss = -model.log_prob(x_val).mean()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                tst_loss = -model.log_prob(x_tst).mean()
                tst_ix = epoch

            losses[epoch] = epoch_loss.detach()
            vlosses[epoch] = val_loss.detach()

            loop.set_postfix(
                {
                    "loss": f"{losses[epoch]:.2f} ({vlosses[epoch]:.2f}) {label}: *{tst_loss.detach():.3f} @ {tst_ix}"
                }
            )

            if early_stop_patience is not None and epoch - tst_ix > early_stop_patience:
                break

    return tst_loss.cpu(), tst_ix, losses.cpu(), vlosses.cpu(), hook_data
