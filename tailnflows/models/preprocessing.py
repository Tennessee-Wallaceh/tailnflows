import torch
from scipy import stats


def estimate_marginal_dfs(x):
    marginal_params = torch.vstack(
        [torch.tensor(stats.t.fit(x_marginal)[0]) for x_marginal in x.split(1, dim=1)]
    )
    return marginal_params


def inverse_and_lad(x, marginal_params):
    # quite hacky torch -> numpy -> torch transformation to use scipy t cdf
    z = torch.hstack(
        [
            torch.tensor(stats.norm.ppf(stats.t.cdf(x_marginal, df=df_marginal)), device=x.device)
            for x_marginal, df_marginal in zip(x.split(1, dim=1), marginal_params)
        ]
    )
    cdf_prob = torch.hstack(
        [
            torch.tensor(stats.t.logpdf(x_marginal, df=df_marginal), device=x.device)
            for x_marginal, df_marginal in zip(x.split(1, dim=1), marginal_params)
        ]
    )

    lad = (cdf_prob - stats.norm.logpdf(z)).sum(dim=1)
    return z, lad
