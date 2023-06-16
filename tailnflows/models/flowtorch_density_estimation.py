# flowtorch dependencies
import torch

from flowtorch import bijectors as ft_bijectors
from flowtorch import parameters as ft_parameters
from flowtorch import distributions as ft_distributions


def get_model(dtype, model_name, dim, model_kwargs={}):
    torch.set_default_dtype(dtype)

    if model_name == 'ADVI':
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