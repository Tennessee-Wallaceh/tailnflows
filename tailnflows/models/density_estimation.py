import torch

# nflows dependencies
from nflows.flows import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.autoregressive import (
    MaskedAffineAutoregressiveTransform, 
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
)
from nflows.transforms.base import CompositeTransform, InverseTransform

# custom modules
from tailnflows.models.extreme_transformations import (
    MaskedTailAutoregressiveTransform, 
    flip,
    MaskedExtremeAutoregressiveTransform,
    TailMarginalTransform
)
from tailnflows.models.base_distribution import TrainableStudentT

def get_model(dtype, model_name, dim, model_kwargs={}):
    torch.set_default_dtype(dtype)

    if model_name == 'TTF':
        hidden_layer_size = model_kwargs.get('hidden_layer_size', dim  * 2)
        num_hidden_layers = model_kwargs.get('num_hidden_layers', 2)
        tail_bound = model_kwargs.get('tail_bound', 1.)
        num_bins = model_kwargs.get('num_bins', 8)
        tail_init = model_kwargs.get('tail_init', None)

        base_distribution = StandardNormal([dim])
        transform = CompositeTransform([
            # element wise fcn flip, so heavy->light becomes forward (to noise in nflows) transform
            flip(MaskedTailAutoregressiveTransform(features=dim, hidden_features=dim  * 2, num_blocks=2)),
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=dim, 
                hidden_features=hidden_layer_size, 
                num_blocks=num_hidden_layers, 
                num_bins=num_bins,
                tails='linear', 
                tail_bound=tail_bound
            ),
        ])

        if tail_init is not None:
            for _dim in range(dim):
                torch.nn.init.constant_(
                    transform._transforms[0].autoregressive_net.final_layer.bias[_dim * 4], 
                    tail_init
                )
                torch.nn.init.constant_(
                    transform._transforms[0].autoregressive_net.final_layer.bias[_dim * 4 + 1],
                    tail_init
                )
            
        model = Flow(distribution=base_distribution, transform=transform)

    elif model_name == 'EXFLOW':
        hidden_layer_size = model_kwargs.get('hidden_layer_size', dim  * 2)
        num_hidden_layers = model_kwargs.get('num_hidden_layers', 2)
        tail_bound = model_kwargs.get('tail_bound', 1.)
        num_bins = model_kwargs.get('num_bins', 8)
        tail_init = model_kwargs.get('tail_init', None)

        base_distribution = StandardNormal([dim])
        transform = CompositeTransform([
            # element wise fcn flip, so heavy->light becomes forward transform
            # always lightens, will push subgaussian onto RQS
            flip(MaskedExtremeAutoregressiveTransform(features=dim, hidden_features=dim  * 2, num_blocks=2)),
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=dim, 
                hidden_features=hidden_layer_size, 
                num_blocks=num_hidden_layers, 
                num_bins=num_bins,
                tails='linear', 
                tail_bound=tail_bound
            ),
        ])

        model = Flow(distribution=base_distribution, transform=transform)
    
    elif model_name == 'TTF_marginal':
        hidden_layer_size = model_kwargs.get('hidden_layer_size', dim  * 2)
        num_hidden_layers = model_kwargs.get('num_hidden_layers', 2)
        tail_bound = model_kwargs.get('tail_bound', 1.)
        num_bins = model_kwargs.get('num_bins', 8)
        tail_init = model_kwargs.get('tail_init', None)

        base_distribution = StandardNormal([dim])
        transform = CompositeTransform([
            flip(TailMarginalTransform(features=dim)),
            MaskedAffineAutoregressiveTransform(features=dim, hidden_features=dim * 2, num_blocks=2),
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=dim, 
                hidden_features=hidden_layer_size, 
                num_blocks=num_hidden_layers, 
                num_bins=num_bins,
                tails='linear', 
                tail_bound=tail_bound
            ),
        ])

        model = Flow(distribution=base_distribution, transform=transform)

    elif model_name == 'TTF_dextreme':
        hidden_layer_size = model_kwargs.get('hidden_layer_size', dim  * 2)
        num_hidden_layers = model_kwargs.get('num_hidden_layers', 2)
        tail_bound = model_kwargs.get('tail_bound', 1.)
        num_bins = model_kwargs.get('num_bins', 8)
        tail_init = model_kwargs.get('tail_init', None)

        base_distribution = StandardNormal([dim])
        transform = CompositeTransform([
            InverseTransform(MaskedTailAutoregressiveTransform(features=dim, hidden_features=dim  * 2, num_blocks=2)),
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=dim, 
                hidden_features=hidden_layer_size, 
                num_blocks=num_hidden_layers, 
                num_bins=num_bins,
                tails='linear', 
                tail_bound=tail_bound
            ),
        ])

        model = Flow(distribution=base_distribution, transform=transform)

    elif model_name == 'RQS_flow':
        hidden_layer_size = model_kwargs.get('hidden_layer_size', dim  * 2)
        num_hidden_layers = model_kwargs.get('num_hidden_layers', 2)
        tail_bound = model_kwargs.get('tail_bound', 1.)
        num_bins = model_kwargs.get('num_bins', 8)
        
        base_dist = StandardNormal([dim]) 
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

    elif model_name == 'nflow_ATAF':
        hidden_layer_size = model_kwargs.get('hidden_layer_size', dim  * 2)
        num_hidden_layers = model_kwargs.get('num_hidden_layers', 2)
        tail_bound = model_kwargs.get('tail_bound', 1.)
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

    elif model_name == 'nflow_ADVI':
        hidden_layer_size = model_kwargs.get('hidden_layer_size', dim  * 2)
        num_hidden_layers = model_kwargs.get('num_hidden_layers', 2)
        tail_bound = model_kwargs.get('tail_bound', 1.)
        num_bins = model_kwargs.get('num_bins', 8)
        tail_init = model_kwargs.get('tail_init', 10.) # default init from FTVI code

        base_dist = TrainableStudentT(dim, init=tail_init) 
        transform = CompositeTransform([
            MaskedAffineAutoregressiveTransform(features=dim, hidden_features=dim * 2, num_blocks=2),
            MaskedAffineAutoregressiveTransform(features=dim, hidden_features=dim * 2, num_blocks=2),
        ])

        model = Flow(distribution=base_dist, transform=transform)
        
    else:
        raise ValueError(f'Invalid model name')

    return model