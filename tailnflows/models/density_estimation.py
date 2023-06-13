import torch

# nflows dependencies
from nflows.flows import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.autoregressive import (
    MaskedAffineAutoregressiveTransform, 
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
)
from nflows.transforms.base import CompositeTransform, InverseTransform

from tailnflows.models.extreme_transformations import (
    MaskedTailAutoregressiveTransform, flip
)

# flowtorch dependencies
from flowtorch import bijectors as ft_bijectors
from flowtorch import parameters as ft_parameters
from flowtorch import distributions as ft_distributions

# custom modules
# from tailnflows.models.extreme_transformations import (
#     SwitchTransform,
#     SwitchAffineTransform,
# )
from tailnflows.models.base_distribution import TrainableStudentT

def get_model(dtype, model_name, dim, model_kwargs={}):
    torch.set_default_dtype(dtype)

    if model_name == 'TTF':
        hidden_layer_size = model_kwargs.get('hidden_layer_size', dim  * 2)
        num_hidden_layers = model_kwargs.get('num_hidden_layers', 2)
        tail_bound = model_kwargs.get('tail_bound', 3.)
        num_bins = model_kwargs.get('num_bins', 8)
        tail_init = model_kwargs.get('tail_init', None)

        base_distribution = StandardNormal([dim])
        transform = CompositeTransform([
            flip(MaskedTailAutoregressiveTransform(features=dim, hidden_features=dim  * 2, num_blocks=2)),
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=dim, 
                hidden_features=hidden_layer_size, 
                num_blocks=num_hidden_layers, 
                num_bins=num_bins,
                tails='linear', 
                tail_bound=tail_bound
            )
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
        hidden_layer_size = model_kwargs.get('hidden_layer_size', dim  * 2)
        num_hidden_layers = model_kwargs.get('num_hidden_layers', 2)
        tail_bound = model_kwargs.get('tail_bound', 3.)
        num_bins = model_kwargs.get('num_bins', 8)
        tail_init = model_kwargs.get('tail_init', 10.) # default init from FTVI code

        base_dist = TrainableStudentT(dim, init=tail_init) 
        bijections =(
            ft_bijectors.SplineAutoregressive(
                ft_parameters.DenseAutoregressive(hidden_dims=[dim + 10]),
                bound=tail_bound,
                count_bins=num_bins,
            ),
            ft_bijectors.AffineAutoregressive(
                ft_parameters.DenseAutoregressive(hidden_dims=[dim + 10])
            ),
        )
        transformation = ft_bijectors.Compose(bijections)
        model = ft_distributions.Flow(base_dist, transformation)

    elif model_name == 'nflow_ATAF':
        hidden_layer_size = model_kwargs.get('hidden_layer_size', dim  * 2)
        num_hidden_layers = model_kwargs.get('num_hidden_layers', 2)
        tail_bound = model_kwargs.get('tail_bound', 3.)
        num_bins = model_kwargs.get('num_bins', 8)
        tail_init = model_kwargs.get('tail_init', 10.) # default init from FTVI code

        base_dist = TrainableStudentT(dim, init=tail_init) 
        transform = CompositeTransform([
            MaskedAffineAutoregressiveTransform(features=dim, hidden_features=dim * 2, num_blocks=2),
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=dim, 
                hidden_features=hidden_layer_size, 
                num_blocks=num_hidden_layers, 
                num_bins=num_bins,
                tails='linear', 
                tail_bound=tail_bound
            )
        ])

        model = Flow(distribution=base_dist, transform=transform)
        
    else:
        raise ValueError(f'Invalid model name')

    return model