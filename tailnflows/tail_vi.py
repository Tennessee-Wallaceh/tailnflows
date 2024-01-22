from scipy.stats import linregress


def laplace_approximation(dim, target):
    res = minimize(lambda x: -target(x), torch.zeros([1, dim]), method="bfgs")
    return res.x


def marginal_tail_vi(target, min_log_x=-5.0, max_log_x=10.0, corr_thresh=0.9999):
    # finds
    log_x_ins = torch.linspace(min_log_x, max_log_x, 1000)
    x_ins = log_x_ins.exp()
    log_p = target(x_ins)

    x_us = torch.linspace(1e-5, x_ins.max() * 0.99, 500)
    corrs = []
    slopes = []
    inspected = []
    for x_u in x_us:
        if (log_x_ins > torch.log(x_u)).sum() > 0:
            corr = torch.corrcoef(
                torch.vstack(
                    [
                        log_x_ins[log_x_ins > torch.log(x_u)],
                        log_p[log_x_ins > torch.log(x_u)],
                    ]
                )
            )[0, 1]
            corrs.append(-corr.detach())
            res = linregress(
                log_x_ins[log_x_ins > torch.log(x_u)].detach(),
                log_p[log_x_ins > torch.log(x_u)].detach(),
            )
            slopes.append(-1 / (res.slope + 1))
            inspected.append(x_u)

    # select most correlated point
    select = np.argwhere(torch.tensor(corrs) > corr_thresh)[0]
    if len(select) != 0:
        threshold = torch.log(x_us[select[0]])
    else:
        threshold = None

    if threshold is None:
        return 0.0, None
    else:
        return slopes[select[0]], threshold


def multivariate_tail_vi(target, dim, laplace=True):
    if laplace:
        location = laplace_approximation(5, log_posterior)
    else:
        location = torch.zeros([dim])

    def marginal_target(x, dim_ix, sign):
        full_x = torch.zeros([x.shape[0], dim])
        full_x[:, :] = location
        full_x[:, dim_ix] = sign * x
        return target(full_x)

    pos_tails = []
    neg_tails = []
    for dim_ix in range(dim):
        pos_tails.append(
            marginal_tail_vi(
                lambda x: marginal_target(x, dim_ix, 1), min_log_x=-5.0, max_log_x=10.0
            )[0]
        )
        neg_tails.append(
            marginal_tail_vi(
                lambda x: marginal_target(x, dim_ix, -1), min_log_x=-5.0, max_log_x=10.0
            )[0]
        )

    return pos_tails, neg_tails
