import torch
from torch.nn.functional import softplus, relu
from nflows.transforms.autoregressive import AutoregressiveTransform
from nflows.transforms import made as made_module
from nflows.transforms import Transform

MAX_TAIL = 5.
SQRT_2 = torch.sqrt(torch.tensor(2.))
SQRT_PI = torch.sqrt(torch.tensor(torch.pi))
MIN_ERFC_INV = torch.tensor(5e-7)

def _erfcinv(x):
    # with torch.no_grad():
    #     x = torch.clamp(x, min=MIN_ERFC_INV)
    return -torch.special.ndtri(0.5 * x) / SQRT_2

def _shift_power_transform_and_lad(z, tail_param):
    transformed = (SQRT_2 / SQRT_PI) * (torch.pow(1 + z / tail_param, tail_param) - 1)
    lad = (tail_param - 1) * torch.log(1 + z / tail_param)
    lad += torch.log(SQRT_2 / SQRT_PI)
    return transformed, lad

def _shift_power_inverse_and_lad(x, tail_param):
    transformed = (SQRT_PI / SQRT_2) * tail_param * (torch.pow(1 + x, 1 / tail_param) - 1)
    lad = ((1 / tail_param) - 1) * torch.log(1 + x)
    lad -= torch.log(SQRT_2 / SQRT_PI)
    return transformed, lad

def _extreme_transform_and_lad(z, tail_param):
    g = torch.erfc(z / SQRT_2)
    x = (torch.pow(g, -tail_param) - 1) / tail_param
    
    lad = torch.log(g) * (-tail_param -1)
    lad -= 0.5 * torch.square(z) 
    lad += torch.log(SQRT_2 / SQRT_PI)

    return x, lad

def _extreme_inverse_and_lad(x, tail_param):
    inner = 1 + tail_param * x
    g = torch.pow(inner, -1 / tail_param)
    z = SQRT_2 * _erfcinv(g)

    lad = (-1 - 1 / tail_param) * torch.log(inner)
    lad += torch.log(0.5 * SQRT_2 * SQRT_PI)
    lad += torch.square(_erfcinv(g))

    return z, lad


def flip(transform):
    _inverse = transform._elementwise_inverse
    transform._elementwise_inverse = transform._elementwise_forward
    transform._elementwise_forward = _inverse
    return transform

class MaskedTailAutoregressiveTransform(AutoregressiveTransform):
    def __init__(
        self,
        features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        use_residual_blocks=True,
        random_mask=False,
        activation=relu,
        dropout_probability=0.0,
        use_batch_norm=False,
    ):
        self.features = features
        made = made_module.MADE(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=self._output_dim_multiplier(),
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )
        self._epsilon = 1e-3
        super(MaskedTailAutoregressiveTransform, self).__init__(made)

    def _output_dim_multiplier(self):
        return 4

    def _elementwise_forward(self, z, autoregressive_params):
        """light -> heavy"""
        unc_ptail, unc_ntail, unc_scale, shift_param = self._unconstrained_params(autoregressive_params)
        pos_tail_param =  softplus(unc_ptail) - 1. # (-1, inf)
        neg_tail_param =  softplus(unc_ntail) - 1. # (-1, inf)
        scale_param = 1e-2 + softplus(unc_scale)
        tail_param = torch.where(z > 0, pos_tail_param, neg_tail_param)

        sign = torch.sign(z)
        heavy_x, heavy_lad = _extreme_transform_and_lad(torch.abs(z), tail_param)
        light_x, light_lad = _shift_power_transform_and_lad(torch.abs(z), tail_param + 2.)
        
        x = torch.where(tail_param > 0., heavy_x, light_x)
        lad = torch.where(tail_param > 0., heavy_lad, light_lad)

        return sign * x * scale_param + shift_param, (lad + torch.log(scale_param)).sum(axis=1)

    def _elementwise_inverse(self, x, autoregressive_params):
        """heavy -> light"""
        unc_ptail, unc_ntail, unc_scale, shift_param = self._unconstrained_params(autoregressive_params)
        pos_tail_param =  softplus(unc_ptail) - 1. # (-1, inf)
        neg_tail_param =  softplus(unc_ntail) - 1. # (-1, inf)
        scale_param = 1e-2 + softplus(unc_scale)
        
        # affine transform
        x = (x - shift_param) / scale_param
        tail_param = torch.where(x > 0, pos_tail_param, neg_tail_param)

        # tail transform
        sign = torch.sign(x)
        heavy_z, heavy_lad = _extreme_inverse_and_lad(torch.abs(x), torch.abs(tail_param))
        light_z, light_lad = _shift_power_inverse_and_lad(torch.abs(x), tail_param + 2)
        z = torch.where(tail_param > 0., heavy_z, light_z)
        lad = torch.where(tail_param > 0., heavy_lad, light_lad)

        return sign * z, (lad - torch.log(scale_param)).sum(axis=1)

    def _unconstrained_params(self, autoregressive_params):
        autoregressive_params = autoregressive_params.view(
            -1, self.features, self._output_dim_multiplier()
        )
        return (
            autoregressive_params[..., 0], 
            autoregressive_params[..., 1], 
            autoregressive_params[..., 2],
            autoregressive_params[..., 3],
        )
    

class TailMarginalTransform(Transform):
    def __init__(
        self,
        features,
    ):
        self.features = features
        super(TailMarginalTransform, self).__init__()
        self._unc_params = torch.nn.parameter.Parameter(
            torch.zeros(features * self._output_dim_multiplier())
        )

    def _output_dim_multiplier(self):
        return 2

    def forward(self, z, context=None):
        return self._elementwise_forward(z, self._unc_params.repeat(z.shape[0], 1))
    
    def inverse(self, z, context=None):
        return self._elementwise_inverse(z, self._unc_params.repeat(z.shape[0], 1))

    def _elementwise_forward(self, z, autoregressive_params):
        """light -> heavy"""
        unc_ptail, unc_ntail = self._unconstrained_params(autoregressive_params)
        pos_tail_param =  softplus(unc_ptail) - 1. # (-1, inf)
        neg_tail_param =  softplus(unc_ntail) - 1. # (-1, inf)
        tail_param = torch.where(z > 0, pos_tail_param, neg_tail_param)

        sign = torch.sign(z)
        heavy_x, heavy_lad = _extreme_transform_and_lad(torch.abs(z), tail_param)
        light_x, light_lad = _shift_power_transform_and_lad(torch.abs(z), tail_param + 2.)
        
        x = torch.where(tail_param > 0., heavy_x, light_x)
        lad = torch.where(tail_param > 0., heavy_lad, light_lad)

        return sign * x, lad.sum(axis=1)

    def _elementwise_inverse(self, x, autoregressive_params):
        """heavy -> light"""
        unc_ptail, unc_ntail = self._unconstrained_params(autoregressive_params)
        pos_tail_param =  softplus(unc_ptail) - 1. # (-1, inf)
        neg_tail_param =  softplus(unc_ntail) - 1. # (-1, inf)
        tail_param = torch.where(x > 0, pos_tail_param, neg_tail_param)

        # tail transform
        sign = torch.sign(x)
        heavy_z, heavy_lad = _extreme_inverse_and_lad(torch.abs(x), torch.abs(tail_param))
        light_z, light_lad = _shift_power_inverse_and_lad(torch.abs(x), tail_param + 2)
        z = torch.where(tail_param > 0., heavy_z, light_z)
        lad = torch.where(tail_param > 0., heavy_lad, light_lad)

        return sign * z, lad.sum(axis=1)

    def _unconstrained_params(self, autoregressive_params):
        autoregressive_params = autoregressive_params.view(
            -1, self.features, self._output_dim_multiplier()
        )
        return (
            autoregressive_params[..., 0], 
            autoregressive_params[..., 1], 
        )


class MaskedExtremeAutoregressiveTransform(AutoregressiveTransform):
    def __init__(
        self,
        features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        use_residual_blocks=True,
        random_mask=False,
        activation=relu,
        dropout_probability=0.0,
        use_batch_norm=False,
    ):
        self.features = features
        made = made_module.MADE(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=self._output_dim_multiplier(),
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )
        self._epsilon = 1e-3
        super(MaskedExtremeAutoregressiveTransform, self).__init__(made)

    def _output_dim_multiplier(self):
        return 3

    def _elementwise_forward(self, z, autoregressive_params):
        unc_tail, unc_scale, shift_param = self._unconstrained_params(autoregressive_params)
        tail_param =  1e-3 + softplus(unc_tail) # (0, inf)
        scale_param = 1e-2 + softplus(unc_scale)

        sign = torch.sign(z)
        x, lad = _extreme_transform_and_lad(torch.abs(z), tail_param)

        return sign * x * scale_param + shift_param, (lad + torch.log(scale_param)).sum(axis=1)

    def _elementwise_inverse(self, x, autoregressive_params):
        unc_tail, unc_scale, shift_param = self._unconstrained_params(autoregressive_params)
        tail_param =  1e-3 + softplus(unc_tail) # (0, inf)
        scale_param = 1e-2 + softplus(unc_scale)

        # affine transform
        x = (x - shift_param) / scale_param

        # tail transform
        sign = torch.sign(x)
        z, lad = _extreme_inverse_and_lad(torch.abs(x), torch.abs(tail_param))

        return sign * z, (lad - torch.log(scale_param)).sum(axis=1)

    def _unconstrained_params(self, autoregressive_params):
        autoregressive_params = autoregressive_params.view(
            -1, self.features, self._output_dim_multiplier()
        )
        return autoregressive_params[..., 0], autoregressive_params[..., 1], autoregressive_params[..., 2]