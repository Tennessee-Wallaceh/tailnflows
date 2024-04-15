import torch


def log_density(x, heavy_df=1.0):
    assert x.shape[1] > 1, "Only defined for dim 2 and above"

    nuisance_base = torch.distributions.StudentT(df=heavy_df)
    log_prob = nuisance_base.log_prob(x[:, :-1]).sum(axis=1)
    log_prob += torch.distributions.Normal(loc=x[:, -2], scale=1.0).log_prob(x[:, -1])

    return log_prob


def generate_data(n, dim, heavy_df=1.0):
    assert dim > 1, "Only defined for dim 2 and above"

    x = torch.zeros([n, dim])

    nuisance_base = torch.distributions.StudentT(df=heavy_df)

    x[:, :-1] = nuisance_base.sample([n, dim - 1])
    x[:, -1] = x[:, -2] + torch.distributions.Normal(loc=0.0, scale=1.0).sample([n])

    return x
