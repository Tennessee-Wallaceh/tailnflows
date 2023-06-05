import torch
import numpy as np
from nflows.transforms.base import Transform
from nflows.transforms import made as made_module
from flowtorch.parameters import DenseAutoregressive
    
class MaskedAutoregressiveTransform(Transform):
    def __init__(self, dim, transform, kind='nflow', **nn_kwargs):
        super().__init__()
        self.transform = transform
        if kind == 'nflow':
            self.nflow_autoregressive(dim, transform, **nn_kwargs)
        else:
            self.ft_autoregressive(dim, transform, **nn_kwargs)

    def ft_autoregressive(self, dim, transform, num_layers=2, hidden_layer_size=10):
        param_shapes = tuple((dim * shape,) for shape in transform.param_shape)
        self.autoregressive_net = DenseAutoregressive(
            param_shapes=param_shapes,
            input_shape=(dim,),
            context_shape=None,
            hidden_dims=list((hidden_layer_size for _ in range(num_layers))),
        )

        # def call_autoreg(z, context=None):
        #     z = torch.flip(z, dims=(1,))
        #     parameters = self.autoregressive_net(z)
        #     for params, shape in zip(parameters, param_shapes):
        #         torch.flip(params)

        self.parameter_fcn = self.autoregressive_net

    def nflow_autoregressive(self, dim, transform, num_layers=2, hidden_layer_size=10):
        param_shape = transform.param_shape
        # dim -> prod(transform.param_shape) * dim
        parameter_masks = []
        for ix, _ in enumerate(param_shape):
            dim_mask = torch.hstack([
                torch.ones(_shape) if _ix == ix else torch.zeros(_shape) 
                for _ix, _shape in enumerate(param_shape)
            ])
            parameter_masks.append(dim_mask.repeat(dim) > 0)

        self.autoregressive_net = made_module.MADE(
            features=dim,
            hidden_features=hidden_layer_size,
            num_blocks=num_layers,
            output_multiplier=sum(param_shape),
            use_residual_blocks=False,
            # use_batch_norm=True
        )
        def call_autoreg(z, context=None):
            autoregressive_params = self.autoregressive_net(z)
            return (
                autoregressive_params[:, param_mask].reshape(z.shape[0], dim, -1)
                for param_mask in parameter_masks
            )

        self.parameter_fcn = call_autoreg

    def forward(self, z, context=None):
        autoregressive_params = self.parameter_fcn(z, context)
        return self.transform.forward_and_lad(z, *autoregressive_params)

    def inverse(self, x, context=None):
        num_inputs = int(np.prod(x.shape[1:]))
        z = x
        logabsdet = None
        for _ in range(num_inputs):
            autoregressive_params = self.parameter_fcn(z, context)
            z, logabsdet = self.transform.inverse_and_lad(
                z, *autoregressive_params
            )
        return z, logabsdet

class Flip(Transform):
    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def forward(self, z, context=None):
        return self.transform.inverse(z, context)
    
    def inverse(self, z, context=None):
        return self.transform.forward(z, context)


class Marginal(Transform):
    def __init__(self, dim, transform):
        super().__init__()
        self.transform = transform
        self.params = [
            torch.nn.Parameter(torch.zeros(dim * shape,))
            for shape in transform.param_shape
        ]
        for ix, p in enumerate(self.params):
            self.register_parameter(f'p_{ix}', p)

    def forward(self, z, context=None):
        tiled_p = tuple(
            torch.tile(p, (z.shape[0], 1))
            for p in self.params
        )
        x, logabsdet = self.transform.forward_and_lad(z, *tiled_p)
        return x, logabsdet

    def inverse(self, z, context=None):
        tiled_p = tuple(
            torch.tile(p, (z.shape[0], 1))
            for p in self.params
        )
        z, lad = self.transform.inverse_and_lad(z, *tiled_p)
        return z, lad
    

class Chain(Transform):
    """
    Chains a sequence of Transform objects, 
    by just calling the forward and inverse methods of each one.
    """
    def __init__(self, transforms):
        super().__init__()
        self.transforms = torch.nn.ModuleList(transforms)
    
    def parameters(self, recurse: bool = True):
        return [p for transform in self.transforms for p in transform.parameters(recurse)]
    
    def forward(self, x, context=None):
        logabsdet = 0.
        for transform in self.transforms:
            x, lad = transform.forward(x)
            logabsdet += lad
        return x, logabsdet
    
    def inverse(self, x, context=None):
        logabsdet = 0.
        for transform in self.transforms[::-1]:
            x, lad = transform.inverse(x, context=None)
            logabsdet += lad
        return x, logabsdet