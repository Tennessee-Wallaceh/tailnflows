import torch
from torch.distributions import Normal, StudentT
from torch.nn.functional import softplus

needed = [
    'd_nuisance',
    'heavy_nuisance',
    'df_nuisance', 
    'scale_model_name',
]
    
def get_target(target_kwargs):
    for kwarg in needed:
        assert kwarg in target_kwargs, f'missing {kwarg} target_kwarg for neals funnel'

    d_nuisance = target_kwargs['d_nuisance']
    heavy_nuisance = target_kwargs['heavy_nuisance']
    df_nuisance = target_kwargs['df_nuisance']
    scale_model_name = target_kwargs['scale_model_name']

    dim = d_nuisance + 2
    if scale_model_name == 'funnel':
        SCALE_MAX = torch.exp(torch.tensor(25.))
        SCALE_MIN = torch.exp(torch.tensor(-25.))
        def scale_model(x):
            # we can sample very large x, so set reasonable maximum
            scale = torch.exp(x[:, d_nuisance])
            with torch.no_grad():
                scale = torch.clip(scale, min=SCALE_MIN, max=SCALE_MAX)
            return scale
                
    elif scale_model_name == 'softplus':
        scale_min = 0.5
        def scale_model(x):
            return scale_min + softplus(x[:, d_nuisance])
    else:
        raise Exception(f'neal funnel scale model {scale_model_name} not supported!')

    if not heavy_nuisance:
        assert df_nuisance is None, f'No df for nuisance when not heavy nuisance!'
        nuisance_dist = Normal(loc=0., scale=1.)
    else:
        nuisance_dist = StudentT(df=df_nuisance)

    def target(x):
        nuisance_log_prob = nuisance_dist.log_prob(x[:, :d_nuisance]).sum(dim=1)
        informative_log_prob = Normal(loc=0., scale=1.).log_prob(x[:, d_nuisance])
        scale = scale_model(x)
        target_log_prob = Normal(loc=0., scale=1.).log_prob(x[:, d_nuisance + 1] / scale)
        return target_log_prob + nuisance_log_prob + informative_log_prob
    
    label = f'd={d_nuisance}|df={df_nuisance}'
    return target, dim, label