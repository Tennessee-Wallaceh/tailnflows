from tailnflows.experiments.tailestimation import moments_estimator, hill_estimator, kernel_type_estimator
import numpy as np

def tail_estimate(sorted):
    moment_k_star = moments_estimator(sorted)[3]
    kernel_k_star = kernel_type_estimator(sorted, 10)[3]
    est_tail = 0.
    if moment_k_star > 0. or  kernel_k_star > 0.:
        est_tail = hill_estimator(sorted)[3]
        if est_tail > 10.:
            est_tail = 0.

    return est_tail

def compute_tail_diff(target_x, approx_x):
    return np.abs(
        tail_estimate(np.sort(target_x)[::-1]) - tail_estimate(np.sort(approx_x)[::-1])
    )

def estimate_tail_diffs(model, x_tst):
    pos_areas = []
    neg_areas = []
    nsamples = x_tst.shape[0]
    approx_samps = model.sample(nsamples).detach().numpy()

    for j in tqdm.tqdm(range(dim), leave=False):
        x_tst = x_tst[torch.randperm(x_tst.shape[0])]
        x_tst_marginal = x_tst[:nsamples, j]
        x_app_marginal = approx_samps[:nsamples, j]
    
        marginal_area_pos = compute_tail_diff(
            x_tst_marginal[x_tst_marginal >= 0], 
            x_app_marginal[x_app_marginal >= 0]
        )
        marginal_area_neg = compute_tail_diff(
            np.abs(x_tst_marginal[x_tst_marginal < 0]), 
            np.abs(x_app_marginal[x_app_marginal < 0])
        )
        pos_areas.append(marginal_area_pos)
        neg_areas.append(marginal_area_neg)

    return pos_areas, neg_areas