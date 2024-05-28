import torch
from torch.optim import Adam
import tqdm
from dataclasses import dataclass
from tailnflows.metrics import psis_index, ess, elbo
from collections import deque

METRIC_REPEATS = 10


@dataclass
class VIMetrics:
    elbo: list[float]
    psis_k: list[float]
    ess: list[float]


def train(
    model,
    target,
    lr=1e-3,
    num_epochs=500,
    batch_size=100,
    label="",
    seed=0,
    grad_clamp=None,
    metric_samples=10_000,  # may need to lower for very high dim
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
        # update step
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

            # compute test stats at last epoch, using larger sample size
            if epoch == num_epochs - 1:
                tst_ess = [
                    ess(model.sample_and_log_prob, target, metric_samples)
                    .detach()
                    .cpu()
                    .numpy()
                    for _ in range(METRIC_REPEATS)
                ]
                tst_psis = [
                    psis_index(model.sample_and_log_prob, target, metric_samples)
                    for _ in range(METRIC_REPEATS)
                ]
                tst_elbo = [
                    elbo(
                        model.sample_and_log_prob,
                        target,
                        metric_samples,
                    )
                    for _ in range(METRIC_REPEATS)
                ]

                loop.set_postfix(
                    {
                        "tst_elbo": f"({min(tst_elbo):.3f}, {max(tst_elbo):.3f})",
                        "tst_ess": f"({min(tst_ess):.3f}, {max(tst_ess):.3f})",
                        "tst_psis": f"({min(tst_psis):.3f}, {max(tst_psis):.3f})",
                        "model": label,
                    }
                )

    final_metrics = VIMetrics(
        tst_elbo,
        tst_psis,
        tst_ess,
    )

    return losses, final_metrics


def train_vrlu(
    model,
    target,
    lr=1e-3,
    num_epochs=500,
    batch_size=100,
    label="",
    seed=0,
    grad_clamp=None,
    metric_samples=10_000,  # may need to lower for very high dim
    alpha=-0.9,
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
    state_d = deque(maxlen=5)

    for epoch in loop:
        # safe update step
        optimizer.zero_grad()
        x_approx, log_q_x = model.sample_and_log_prob(batch_size)
        log_p_x = target(x_approx)
        w = (log_p_x - log_q_x).exp().pow(1 - alpha)
        loss = (w.mean() - 1) / (1 - alpha)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            losses[epoch] = loss.detach().cpu()
            loop.set_postfix(
                {
                    "VRU": f"{losses[epoch]:.4f}",
                    "": label,
                }
            )

            # compute test stats at last epoch, using n=10_000
            if epoch == num_epochs - 1:
                tst_ess = [
                    ess(model.sample_and_log_prob, target, metric_samples)
                    .detach()
                    .cpu()
                    .numpy()
                    for _ in range(METRIC_REPEATS)
                ]
                tst_psis = [
                    psis_index(model.sample_and_log_prob, target, metric_samples)
                    for _ in range(METRIC_REPEATS)
                ]

                loop.set_postfix(
                    {
                        "VRU": f"{losses[-1]:.4f}",
                        "tst_ess": f"({min(tst_ess):.3f}, {max(tst_ess):.3f})",
                        "tst_psis": f"({min(tst_psis):.3f}, {max(tst_psis):.3f})",
                        "model": label,
                    }
                )

    final_metrics = VIMetrics(
        losses[-1],
        tst_psis,
        tst_ess,
    )

    return losses, final_metrics, state_d


def train_safe(
    model,
    target,
    lr=1e-3,
    num_epochs=500,
    batch_size=100,
    label="",
    seed=0,
    grad_clamp=None,
    metric_samples=10_000,  # may need for lower for very high dim
):
    """
    And adjusted train script for debugging
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
    state_d = deque(maxlen=5)

    for epoch in loop:
        # update step
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
            state_d.append(model.state_dict())

            # compute test stats at last epoch, using larger sample size
            if epoch == num_epochs - 1:
                tst_ess = [
                    ess(model.sample_and_log_prob, target, metric_samples)
                    .detach()
                    .cpu()
                    .numpy()
                    for _ in range(METRIC_REPEATS)
                ]
                tst_psis = [
                    psis_index(model.sample_and_log_prob, target, metric_samples)
                    for _ in range(METRIC_REPEATS)
                ]

                loop.set_postfix(
                    {
                        "elbo": f"{-losses[-1]:.3f}",
                        "tst_ess": f"({min(tst_ess):.3f}, {max(tst_ess):.3f})",
                        "tst_psis": f"({min(tst_psis):.3f}, {max(tst_psis):.3f})",
                        "model": label,
                    }
                )

    final_metrics = VIMetrics(
        -losses[-1],
        tst_psis,
        tst_ess,
    )

    return losses, final_metrics, state_d
