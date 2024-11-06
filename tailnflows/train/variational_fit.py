import torch
from torch.optim import Adam
import tqdm
from dataclasses import dataclass
from tailnflows.metrics import ess, elbo, psis_index, bootstrap_metric
from collections import deque
from functools import partial


@dataclass
class VIMetrics:
    elbo: list[float]
    psis_k: list[float]
    ess: list[float]
    tst_cubo: list[float]


def adaptive_f(log_p_x, log_q_x, compute_grad=True):
    log_w = log_p_x - log_q_x
    weights = get_tail_adaptive_weights(log_w)
    loss = (weights * -log_w).sum()

    if compute_grad:
        loss.backward()

    return loss


def get_tail_adaptive_weights(log_w, beta=-1.0):
    # credit: https://github.com/dilinwang820/adaptive-f-divergence
    dx = (log_w - log_w.max()).exp()
    exceedences = (dx.unsqueeze(1) - dx.unsqueeze(0)).sign() > 0
    prob = exceedences.sum(axis=1) / len(log_w)
    wx = (1.0 - prob).pow(beta)
    return (wx / wx.sum()).detach()


def neg_elbo(log_p_x, log_q_x, compute_grad=True):
    log_w = log_p_x - log_q_x

    loss = (-log_w).mean()
    if compute_grad:
        loss.backward()
    return loss


def cubo(log_p_x, log_q_x, compute_grad=True):
    # This is taken from
    # https://github.com/jhuggins/viabel/blob/7c6e5b1b4925fac0e157b7dc99f1329296efee69/viabel/objectives.py#L453
    alpha = 2.0
    log_w = log_p_x - log_q_x
    weights = (alpha * log_w).exp().detach()
    grad_loss = alpha * (weights * log_w).mean()

    if compute_grad:
        grad_loss.backward()

    log_norm = log_w.max()
    scaled_values = torch.exp(log_w - log_norm) ** alpha
    obj_value = torch.log(torch.mean(scaled_values)) / alpha + log_norm

    return obj_value


loss_fcns = {
    "neg_elbo": neg_elbo,
    "cubo": cubo,
}


def train(
    model,
    target,
    lr=1e-3,
    num_epochs=500,
    batch_size=100,
    label="",
    seed=0,
    metric_samples=10_000,  # may need to lower for very high dim
    loss_label="neg_elbo",
    grad_clip_norm=torch.inf,
    weight_decay=0.0,
    hook=None,
    restart_epoch=None,
):
    """
    Runs a variational fit to a potentially unormalised target density.
    Model and target can be defined on either cpu or gpu.
    """

    torch.manual_seed(seed)
    parameters = list(model.parameters())
    optimizer = Adam(parameters, lr=lr, weight_decay=weight_decay)
    loss_fcn = loss_fcns[loss_label]

    loop = tqdm.tqdm(range(int(num_epochs)))
    losses = torch.zeros(int(num_epochs))

    hook_data = []

    tst_elbo = torch.nan
    tst_psis = torch.nan
    tst_ess = torch.nan
    tst_cubo = torch.nan

    for epoch in loop:
        # update step
        try:
            optimizer.zero_grad()
            x_approx, log_q_x = model.sample_and_log_prob(batch_size)
            log_p_x = target(x_approx)
            loss = loss_fcn(log_p_x, log_q_x)

            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, grad_clip_norm)

            if torch.isfinite(grad_norm):
                optimizer.step()

            if restart_epoch is not None and epoch == restart_epoch:
                optimizer = Adam(parameters, lr=lr, weight_decay=weight_decay)
            with torch.no_grad():
                losses[epoch] = loss.detach().cpu()
                loop.set_postfix(
                    {
                        "loss": f"{losses[epoch]:.2f}",
                        "": f"{label}-{loss_label}",
                    }
                )
                if hook is not None:
                    hook_data.append(hook(model, log_p_x, log_q_x))

                # compute test stats at last epoch, using larger sample size
                if epoch == num_epochs - 1:
                    x_approx, log_q_x = model.sample_and_log_prob(metric_samples)
                    log_p_x = target(x_approx)
                    replications = 1000

                    tst_ess = bootstrap_metric(log_p_x, log_q_x, replications, ess)

                    tst_elbo = bootstrap_metric(log_p_x, log_q_x, replications, elbo)

                    tst_cubo = bootstrap_metric(
                        log_p_x,
                        log_q_x,
                        replications,
                        lambda x, y: cubo(x, y, compute_grad=False),
                    )

                    tst_psis = (psis_index(log_p_x, log_q_x),)

                    loop.set_postfix(
                        {
                            "tst_elbo": f"({min(tst_elbo):.3f}, {max(tst_elbo):.3f})",
                            "tst_ess": f"({min(tst_ess):.3f}, {max(tst_ess):.3f})",
                            "tst_cubo": f"({min(tst_cubo):.3f}, {max(tst_cubo):.3f})",
                            "model": f"{label}-{loss_label}",
                        }
                    )

        except ValueError:
            break

        except KeyboardInterrupt:
            break

    final_metrics = VIMetrics(
        tst_elbo,
        tst_psis,
        tst_ess,
        tst_cubo,
    )

    return losses, final_metrics, hook_data
