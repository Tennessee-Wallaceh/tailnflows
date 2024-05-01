import torch
from torch.optim import Adam
import tqdm
from dataclasses import dataclass
from tailnflows.metrics import psis_index, ess


@dataclass
class VIMetrics:
    elbo: float
    psis_k: float
    ess: float


def train(
    model,
    target,
    lr=1e-3,
    num_epochs=500,
    batch_size=100,
    label="",
    seed=0,
    grad_clamp=None,
):
    """
    Runs a variational fit to a potentially unormalised target density.
    Model and target can be defined on either cpu or gpu.
    """

    torch.manual_seed(seed)
    parameters = list(model.parameters())

    # clip grads if needed
    if grad_clamp is not None:
        for p in parameters:
            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, -grad_clamp, grad_clamp))

    optimizer = Adam(parameters, lr=lr)

    loop = tqdm.tqdm(range(int(num_epochs)))
    losses = torch.zeros(int(num_epochs))
    for epoch in loop:
        optimizer.zero_grad()
        x_approx, log_q_x = model.sample_and_log_prob(batch_size)
        neg_elbo_loss = (log_q_x - target(x_approx)).mean()
        neg_elbo_loss.backward()
        optimizer.step()

        with torch.no_grad():
            losses[epoch] = neg_elbo_loss.detach().cpu()
            loop.set_postfix(
                {
                    "elbo": f"{-losses[epoch]:.2f}",
                    "": label,
                }
            )

            # compute test stats at last epoch, using n=10_000
            if epoch == num_epochs - 1:
                tst_ess = ess(model.sample_and_log_prob, target, 10_000)
                tst_ess = tst_ess.detach().cpu().numpy()
                tst_psis = psis_index(model.sample_and_log_prob, target, 10_000)

                loop.set_postfix(
                    {
                        "elbo": f"{-losses[-1]:.3f}",
                        "tst_ess": f"{tst_ess:.3f}",
                        "tst_psis": f"{tst_psis:.3f}",
                        "model": label,
                    }
                )

    final_metrics = VIMetrics(
        -losses[-1],
        tst_psis,
        tst_ess,
    )

    return losses, final_metrics
