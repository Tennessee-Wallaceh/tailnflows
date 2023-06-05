import torch

# nflows dependencies
from nflows.distributions.normal import StandardNormal
from nflows.flows import Flow

# flowtorch dependencies
from flowtorch import bijectors as ft_bijectors
from flowtorch import parameters as ft_parameters
from flowtorch import distributions as ft_distributions

# custom modules
from models.masked_autoregressive import (
    MaskedAutoregressiveTransform,
    Flip,
    Chain,
    Marginal
)
from models.extreme_transformations import (
    SwitchTransform,
    SwitchAffineTransform,
)
from models.base_distribution import TrainableStudentT
from tailnflows.models.rqs import RQSTransform
from tailnflows.models.affine import AffineTransform

def get_model(model_name, dim, model_kwargs={}):
    if model_name == 'TTF_no_spline':
        base_distribution = StandardNormal([dim])
        transform = MaskedAutoregressiveTransform(
            dim, 
            SwitchAffineTransform(),
            hidden_layer_size=dim + 10,
            num_layers=2,
        )
        # nflows implements forward transform as inverse
        transform = Flip(transform)
        model = Flow(distribution=base_distribution, transform=transform)
        def sample_and_log_prob(n):
            return model.sample_and_log_prob(n)
        
    elif model_name == 'TTF':
        hidden_layer_size = model_kwargs.get('hidden_layer_size', dim  * 2)
        num_hidden_layers = model_kwargs.get('num_hidden_layers', 2)
        tail_bound = model_kwargs.get('tail_bound', 5.)
        num_bins = model_kwargs.get('num_bins', 8)
        tail_init = model_kwargs.get('tail_init', None)

        base_distribution = StandardNormal([dim])
        transform = Chain([
            MaskedAutoregressiveTransform(
                dim,
                RQSTransform(tail_bound=tail_bound, num_bins=num_bins),
                hidden_layer_size=hidden_layer_size,
                num_layers=num_hidden_layers,
            ),
            MaskedAutoregressiveTransform(
                dim, 
                SwitchAffineTransform(),
                hidden_layer_size=hidden_layer_size,
                num_layers=num_hidden_layers,
            ),
        ])

    elif model_name == 'TTF_marginal':
        base_distribution = StandardNormal([dim])
        # nflows implements forward transform as inverse
        transform = Chain([
            Marginal(dim, SwitchTransform()),
            MaskedAutoregressiveTransform(
                dim, 
                AffineTransform(),
                hidden_layer_size=dim + 10,
                num_layers=2,
            ),
            MaskedAutoregressiveTransform(
                dim, 
                RQSTransform(tail_bound=5., num_bins=8),
                hidden_layer_size=dim + 10,
                num_layers=2,
            ),
        ])
        
        model = Flow(distribution=base_distribution, transform=transform)
    
    elif model_name == 'ADVI':
        transformation = ft_bijectors.AffineAutoregressive(
            ft_parameters.DenseAutoregressive(hidden_dims=[dim + 10])
        )
        base_dist = torch.distributions.Independent(
            torch.distributions.Normal(torch.zeros(dim), torch.ones(dim)), 
            1
        )
        model = ft_distributions.Flow(base_dist, transformation)
    
    elif model_name == 'ATAF':
        base_dist = TrainableStudentT(dim, init=10.) # init from FTVI code
        bijections =(
            ft_bijectors.SplineAutoregressive(
                ft_parameters.DenseAutoregressive(hidden_dims=[dim + 10]),
                bound=5.,
                count_bins=8,
            ),
            ft_bijectors.AffineAutoregressive(
                ft_parameters.DenseAutoregressive(hidden_dims=[dim + 10])
            )
        )
        transformation = ft_bijectors.Compose(bijections)
        model = ft_distributions.Flow(base_dist, transformation)
        
    else:
        raise ValueError(f'Invalid model name')

    return model