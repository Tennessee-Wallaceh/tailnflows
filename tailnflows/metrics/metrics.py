import torch
import numpy as np
from tailnflows.metrics.psis import gpdfitnew


def bootstrap_metric(log_p_x, log_q_x, replications, metric):
    samples = len(log_p_x)
    replicate_ix = torch.randint(samples, (replications, samples))
    mean = metric(log_p_x, log_q_x)
    replications = torch.vmap(metric)(log_p_x[replicate_ix], log_q_x[replicate_ix])
    lower_rep, upper_rep = torch.quantile(replications, torch.tensor([0.05, 0.95]))
    lower = 2 * mean - upper_rep
    upper = 2 * mean - lower_rep
    ci = (lower, mean, upper)
    return tuple(float(v.detach().numpy()) for v in ci)


def ess(log_p_x, log_q_x):
    """
    Produces an ESS based sample efficiency metric.
    Usually between 0 and 1, anything not approaching 0 could be
    workable, depending on the situation.
    """
    log_w = log_p_x - log_q_x
    samples = len(log_w)
    log_norm = torch.logsumexp(log_w, 0)
    log_norm_iw = log_w - log_norm
    ess_efficiency = 1 / torch.exp(2 * log_norm_iw).sum()
    ess_efficiency = ess_efficiency / samples
    return ess_efficiency


def elbo(log_p_x, log_q_x):
    log_w = log_p_x - log_q_x
    return log_w.mean()


def entropy(log_p_x, log_q_x):
    return log_q_x.mean()


def marginal_likelihood(log_p_x, log_q_x):
    log_w = log_p_x - log_q_x
    samples = len(log_w)
    return torch.logsumexp(log_w - torch.log(torch.tensor([samples])), dim=0)


def psis_index(log_p_x, log_q_x):
    """
    Produces an PSIS (pareto smoothed importance sampling) score.
    This is the tail index of density ratios q/p, lower is better,
    below 0.7 is considered good enough for importance sampling.
    """
    log_w = log_p_x - log_q_x
    samples = len(log_w)
    M = int(min(3 * np.sqrt(samples), samples / 5))

    max_log_iw = log_w.max()
    log_w -= max_log_iw
    sorted_log_iw = torch.sort(log_w).values  # ascending
    tail_log_iw = sorted_log_iw[-M:]
    threshold = sorted_log_iw[-M - 1]

    tail_iw_exceedences = torch.exp(threshold) * torch.expm1(tail_log_iw - threshold)
    k, _ = gpdfitnew(tail_iw_exceedences.detach().cpu().numpy())
    return k
