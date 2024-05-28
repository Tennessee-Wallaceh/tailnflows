import torch
import numpy as np
from tailnflows.metrics.psis import gpdfitnew


def ess(sample_and_log_prob, target, samples=1000):
    """
    Produces an ESS based sample efficiency metric.
    Usually between 0 and 1, anything not approaching 0 could be
    workable, depending on the situation.
    """
    x_approx, log_q_x = sample_and_log_prob(samples)
    log_p_x = target(x_approx)
    log_iw = log_p_x - log_q_x
    log_norm = torch.logsumexp(log_iw, 0)
    log_norm_iw = log_iw - log_norm
    ess_efficiency = 1 / torch.exp(2 * log_norm_iw).sum()
    ess_efficiency = ess_efficiency / samples
    return ess_efficiency


def elbo(sample_and_log_prob, target, samples=1000):
    """
    Produces an ESS based sample efficiency metric.
    Usually between 0 and 1, anything not approaching 0 could be
    workable, depending on the situation.
    """
    x_approx, log_q_x = sample_and_log_prob(samples)
    log_p_x = target(x_approx)
    log_iw = log_p_x - log_q_x
    return log_iw.mean()


def marginal_likelihood(sample_and_log_prob, target, e_samples=1000):
    x_approx, log_q_x = sample_and_log_prob(e_samples)
    log_p_x = target(x_approx)
    return torch.logsumexp(
        log_p_x - log_q_x - torch.log(torch.tensor([e_samples])), dim=0
    )


def psis_index(sample_and_log_prob, target, samples=1000):
    """
    Produces an PSIS (pareto smoothed importance sampling) score.
    This is the tail index of density ratios q/p, lower is better,
    below 0.7 is considered good enough for importance sampling.
    """
    M = int(min(3 * np.sqrt(samples), samples / 5))

    x_approx, log_q_x = sample_and_log_prob(samples)
    log_p_x = target(x_approx)

    log_iw = log_p_x - log_q_x
    max_log_iw = log_iw.max()
    log_iw -= max_log_iw
    sorted_log_iw = torch.sort(log_iw).values  # ascending
    tail_log_iw = sorted_log_iw[-M:]
    threshold = sorted_log_iw[-M - 1]

    tail_iw_exceedences = torch.exp(threshold) * torch.expm1(tail_log_iw - threshold)
    k, _ = gpdfitnew(tail_iw_exceedences.detach().cpu().numpy())
    return k
