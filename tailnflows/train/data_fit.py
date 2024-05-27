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
):
    optimizer = Adam(list(model.parameters()), lr=lr)
    trn_data = DataLoader(TensorDataset(x_trn), batch_size=batch_size)
    loop = tqdm.tqdm(range(num_epochs))
    losses = torch.empty(num_epochs)
    vlosses = torch.empty(num_epochs)
    hook_data = {}
    tst_loss = torch.tensor(torch.inf)
    best_val_loss = torch.tensor(torch.inf)
    tst_ix = -1
    for epoch in loop:
        # mini batch
        for (subset,) in trn_data:
            optimizer.zero_grad()
            trn_loss = -model.log_prob(subset).mean()
            trn_loss.backward()
            optimizer.step()

        with torch.no_grad():
            if hook is not None:
                hook(model, hook_data)

            val_loss = -model.log_prob(x_val).mean()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                tst_loss = -model.log_prob(x_tst).mean().cpu()
                tst_ix = epoch

            losses[epoch] = trn_loss.detach().cpu()
            vlosses[epoch] = val_loss.detach().cpu()

            loop.set_postfix(
                {
                    "loss": f"{losses[-1]:.2f} ({vlosses[-1]:.2f}) {label}: *{tst_loss.detach()/x_trn.shape[-1]:.3f} @ {tst_ix}"
                }
            )

    return tst_loss, tst_ix, losses, vlosses, hook_data
