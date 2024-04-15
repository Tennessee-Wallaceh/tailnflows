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
    record_hook=None,
):
    optimizer = Adam(list(model.parameters()), lr=lr)
    trn_data = DataLoader(TensorDataset(x_trn), batch_size=batch_size)
    loop = tqdm.tqdm(range(num_epochs))
    losses = []
    vlosses = []
    hook_data = {}
    tst_loss = torch.tensor(np.inf)
    tst_ix = -1
    for i in loop:
        # mini batch
        for (subset,) in trn_data:
            optimizer.zero_grad()
            trn_loss = -model.log_prob(subset).mean()
            trn_loss.backward()
            optimizer.step()

        with torch.no_grad():
            if hook is not None:
                record_hook(model, hook_data)

            val_loss = -model.log_prob(x_val).mean()

            if len(vlosses) > 1 and val_loss < min(vlosses):
                tst_loss = -model.log_prob(x_tst).mean().cpu()
                tst_ix = i

            losses.append(trn_loss.cpu())
            vlosses.append(val_loss.cpu())
            loop.set_postfix(
                {
                    "loss": f"{losses[-1]:.2f} ({vlosses[-1]:.2f}) {label}: *{tst_loss.detach():.3f} @ {tst_ix}"
                }
            )

    return tst_loss, tst_ix, losses, vlosses, hook_data
