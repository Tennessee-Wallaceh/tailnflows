from posteriordb import PosteriorDatabase
import torch
import torch.distributions as dist
from torch.nn.functional import softplus
import os 

def get_target(target_kwargs):
    pdb_path = os.environ['POSTERIOR_DB_PATH']
    my_pdb = PosteriorDatabase(pdb_path)
    model_name = "eight_schools"
    data = my_pdb.data(model_name)

    sigma = torch.tensor(data.values()['sigma'])
    y = torch.tensor(data.values()['y'])

    def p(params):
        return (
            dist.Independent(dist.Normal(params, sigma), 1).log_prob(y) +
            dist.Normal(dist.Normal(0, 5).sample(), dist.HalfCauchy(5).sample()).log_prob(params).sum(1)
        )
    
    return p, 8, 'eight_schools'