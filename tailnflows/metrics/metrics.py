import torch
import numpy as np
from tailnflows.metrics.psis import gpdfitnew

def ess(sample_and_log_prob, target, e_samples=1000):
    x_approx, log_q_x = sample_and_log_prob(e_samples)
    log_p_x = target(x_approx)
    log_iw = log_p_x - log_q_x
    iw = torch.exp(log_iw)
    norm = torch.sum(iw)
    norm_iw = iw / norm
    ess_efficiency = 1 / (torch.sum(norm_iw ** 2) * e_samples)
    return ess_efficiency

def marginal_likelihood(sample_and_log_prob, target, e_samples=1000):
    x_approx, log_q_x = sample_and_log_prob(e_samples)
    log_p_x = target(x_approx)
    return torch.logsumexp(log_p_x - log_q_x - torch.log(torch.tensor([e_samples])), dim=0)

def psis_index(sample_and_log_prob, target, psis_samples=1000):
    M = int(min(3 * np.sqrt(psis_samples), psis_samples / 5))

    x_approx, log_q_x = sample_and_log_prob(10_000)
    log_p_x = target(x_approx)

    log_iw = (log_p_x - log_q_x).detach().numpy()
    max_log_iw = np.max(log_iw)
    log_iw -= max_log_iw
    sorted_log_iw = np.sort(log_iw) # ascending
    tail_log_iw = sorted_log_iw[-M:]
    threshold = sorted_log_iw[-M - 1]

    tail_iw_exceedences = np.exp(tail_log_iw) - np.exp(threshold)
    k, _ = gpdfitnew(tail_iw_exceedences)
    return k