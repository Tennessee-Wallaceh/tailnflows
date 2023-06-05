from posteriordb import PosteriorDatabase
import torch
import torch.distributions as dist
from torch.nn.functional import softplus
import os

def get_target(target_kwargs):
    pdb_path = os.environ['POSTERIOR_DB_PATH']
    my_pdb = PosteriorDatabase(pdb_path)
    model_name = "diamonds"
    model = my_pdb.model(model_name)
    data = my_pdb.data(model_name)

    N = torch.tensor(data.values()["N"])
    k = data.values()["K"]
    y = torch.tensor(data.values()["Y"])
    x = (
        torch.tensor(data.values()["X"]) - torch.tensor(data.values()["X"]).mean(dim=0)
    )[:, 1:]

    def p(params):
        alpha = params[:,0]
        beta = params[:,1:k]
        sigma = softplus(params[:,k]) + 1e-3
        return (
            dist.Independent(dist.Normal(torch.zeros(k-1), torch.ones(k-1)), 1).log_prob(beta) +
            dist.StudentT(3, 8, 10).log_prob(alpha) +
            dist.StudentT(3, 0, 10).log_prob(sigma) - torch.log(torch.tensor(0.5)) +
            dist.Independent(dist.Normal((alpha + x @ beta.T).T, sigma.unsqueeze(1)), 1).log_prob(y)
        )
    
    return p, k+2, 'diamonds'