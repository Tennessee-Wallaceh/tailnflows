import torch
from torch.nn.functional import softplus, relu
from nflows.transforms.autoregressive import AutoregressiveTransform
from nflows.transforms import made as made_module
from nflows.transforms import Transform
from tailnflows.models.utils import inv_sftplus

MAX_TAIL = 5.0
SQRT_2 = torch.sqrt(torch.tensor(2.0))
SQRT_PI = torch.sqrt(torch.tensor(torch.pi))
MIN_ERFC_INV = torch.tensor(5e-7)


class ExtremeActivation(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.in_dim = dim
        self._unc_mix = torch.nn.Parameter(torch.ones([dim, 3]))
        self._unc_params = torch.nn.Parameter(torch.ones([dim, 2]))
        self.mix = torch.nn.Softmax(dim=1)

    def params(self):
        params = torch.nn.functional.sigmoid(self._unc_params) * 2
        heavy_tail = params[..., 0]
        light_tail = params[..., 1]
        return heavy_tail, light_tail

    def forward(self, z):
        heavy_tail, light_tail = self.params()
        mix = self.mix(self._unc_mix)  # dim x 3

        z_heavy = (
            _extreme_transform_and_lad(z.abs(), heavy_tail)[0] * z.sign()
        )  # batch x dim
        z_light = (
            _extreme_inverse_and_lad(z.abs(), light_tail)[0] * z.sign()
        )  # batch x dim

        combo = mix[:, 0] * z + mix[:, 1] * z_heavy + mix[:, 2] * z_light

        return combo


class ExtremeNetwork(torch.nn.Module):
    def __init__(
        self, features, hidden_features, num_blocks, output_multiplier, **kwargs
    ):
        super().__init__()
        self.base_model = made_module.MADE(
            features=features,
            hidden_features=hidden_features,
            num_blocks=num_blocks,
            output_multiplier=output_multiplier,
            **kwargs
        )
        self.extreme_activation = ExtremeActivation(features * output_multiplier)

    def forward(self, x, context=None):
        param_data = self.base_model(x, context)
        adjusted_param_data = self.extreme_activation(param_data)
        return adjusted_param_data


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
    transformed = (
        (SQRT_PI / SQRT_2) * tail_param * (torch.pow(1 + x, 1 / tail_param) - 1)
    )
    lad = ((1 / tail_param) - 1) * torch.log(1 + x)
    lad -= torch.log(SQRT_2 / SQRT_PI)
    return transformed, lad


def _extreme_transform_and_lad(z, tail_param):
    g = torch.erfc(z / SQRT_2)
    x = (torch.pow(g, -tail_param) - 1) / tail_param

    lad = torch.log(g) * (-tail_param - 1)
    lad -= 0.5 * torch.square(z)
    lad += torch.log(SQRT_2 / SQRT_PI)

    return x, lad


def _extreme_inverse_and_lad(x, tail_param):
    inner = 1 + tail_param * x
    g = torch.pow(inner, -1 / tail_param)
    z = SQRT_2 * _erfcinv(g)

    lad = (-1 - 1 / tail_param) * torch.log(inner)
    lad += torch.square(_erfcinv(g))
    lad += torch.log(SQRT_PI / SQRT_2)

    return z, lad


def _copula_transform_and_lad(u, tail_param):
    inner = torch.pow(1 - u, -tail_param)
    x = (inner - 1) / tail_param
    lad = (-tail_param - 1) * torch.log(1 - u)
    return x, lad


def _copula_inverse_and_lad(x, tail_param):
    u = 1 - torch.pow(tail_param * x + 1, -1 / tail_param)
    lad = (-1 - 1 / tail_param) * torch.log(tail_param * x + 1)
    return u, lad


def _sinh_asinh_transform_and_lad(z, kurtosis_param):
    x = torch.sinh(torch.arcsinh(z) / kurtosis_param)

    lad = torch.log(torch.cosh(torch.arcsinh(z) / kurtosis_param))
    lad -= torch.log(kurtosis_param)
    lad -= 0.5 * torch.log(torch.square(z) + 1)
    return x, lad


def _sinh_asinh_inverse_and_lad(x, kurtosis_param):
    z = torch.sinh(kurtosis_param * torch.arcsinh(x))

    lad = torch.log(torch.cosh(kurtosis_param * torch.arcsinh(x))) + torch.log(
        kurtosis_param
    )
    lad -= 0.5 * torch.log(torch.square(x) + 1)
    return z, lad


def flip(transform):
    _inverse = transform._elementwise_inverse
    transform._elementwise_inverse = transform._elementwise_forward
    transform._elementwise_forward = _inverse
    return transform


class MarginalTailAutoregressiveAffineTransform(AutoregressiveTransform):
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
        fix_tails=False,
        tail_init=None,
    ):
        self.features = features

        made = ExtremeNetwork(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=2,  # shift + scale
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )

        super(MarginalTailAutoregressiveAffineTransform, self).__init__(made)

        if tail_init is None:
            self._unc_ptail = torch.nn.Parameter(torch.ones([features]))
            self._unc_ntail = torch.nn.Parameter(torch.ones([features]))
        else:
            assert torch.Size([features]) == tail_init.shape
            self._unc_ptail = torch.nn.parameter.Parameter(inv_sftplus(tail_init + 1))
            self._unc_ntail = torch.nn.parameter.Parameter(inv_sftplus(tail_init + 1))

        if fix_tails:
            self._unc_ptail.requires_grad = False
            self._unc_ntail.requires_grad = False

    def _output_dim_multiplier(self):
        return 2

    def _elementwise_forward(self, z, autoregressive_params):
        """light -> heavy"""
        shift_param, scale_param = self.constrained_params(autoregressive_params)
        pos_tail_param, neg_tail_param = self.tail_params()
        tail_param = torch.where(
            z > 0,
            pos_tail_param.repeat(z.shape[0], 1),
            neg_tail_param.repeat(z.shape[0], 1),
        )

        sign = torch.sign(z)
        heavy_x, heavy_lad = _extreme_transform_and_lad(torch.abs(z), tail_param)
        light_x, light_lad = _shift_power_transform_and_lad(
            torch.abs(z), tail_param + 2.0
        )

        x = torch.where(tail_param > 0.0, heavy_x, light_x)
        lad = torch.where(tail_param > 0.0, heavy_lad, light_lad)

        return sign * x * scale_param + shift_param, (lad + torch.log(scale_param)).sum(
            axis=1
        )

    def _elementwise_inverse(self, x, autoregressive_params):
        """heavy -> light"""
        shift_param, scale_param = self.constrained_params(autoregressive_params)
        pos_tail_param, neg_tail_param = self.tail_params()

        # affine transform
        x = (x - shift_param) / scale_param
        tail_param = torch.where(
            x > 0,
            pos_tail_param.repeat(x.shape[0], 1),
            neg_tail_param.repeat(x.shape[0], 1),
        )

        # tail transform
        sign = torch.sign(x)
        heavy_z, heavy_lad = _extreme_inverse_and_lad(
            torch.abs(x), torch.abs(tail_param)
        )
        light_z, light_lad = _shift_power_inverse_and_lad(torch.abs(x), tail_param + 2)
        z = torch.where(tail_param > 0.0, heavy_z, light_z)
        lad = torch.where(tail_param > 0.0, heavy_lad, light_lad)

        return sign * z, (lad - torch.log(scale_param)).sum(axis=1)

    def constrained_params(self, autoregressive_params):
        autoregressive_params = autoregressive_params.view(
            -1, self.features, self._output_dim_multiplier()
        )
        shift_param = autoregressive_params[..., 0]
        scale_param = 1e-3 + softplus(autoregressive_params[..., 1])
        return (shift_param, scale_param)

    def tail_params(self):
        pos_tail_param = softplus(self._unc_ptail) - 1.0  # (-1, inf)
        neg_tail_param = softplus(self._unc_ntail) - 1.0  # (-1, inf)
        return (pos_tail_param, neg_tail_param)


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
        unc_ptail, unc_ntail, unc_scale, shift_param = self._unconstrained_params(
            autoregressive_params
        )
        pos_tail_param = softplus(unc_ptail) - 1.0  # (-1, inf)
        neg_tail_param = softplus(unc_ntail) - 1.0  # (-1, inf)
        scale_param = 1e-2 + softplus(unc_scale)
        tail_param = torch.where(z > 0, pos_tail_param, neg_tail_param)

        sign = torch.sign(z)
        heavy_x, heavy_lad = _extreme_transform_and_lad(torch.abs(z), tail_param)
        light_x, light_lad = _shift_power_transform_and_lad(
            torch.abs(z), tail_param + 2.0
        )

        x = torch.where(tail_param > 0.0, heavy_x, light_x)
        lad = torch.where(tail_param > 0.0, heavy_lad, light_lad)

        return sign * x * scale_param + shift_param, (lad + torch.log(scale_param)).sum(
            axis=1
        )

    def _elementwise_inverse(self, x, autoregressive_params):
        """heavy -> light"""
        unc_ptail, unc_ntail, unc_scale, shift_param = self._unconstrained_params(
            autoregressive_params
        )
        pos_tail_param = softplus(unc_ptail) - 1.0  # (-1, inf)
        neg_tail_param = softplus(unc_ntail) - 1.0  # (-1, inf)
        scale_param = 1e-2 + softplus(unc_scale)

        # affine transform
        x = (x - shift_param) / scale_param
        tail_param = torch.where(x > 0, pos_tail_param, neg_tail_param)

        # tail transform
        sign = torch.sign(x)
        heavy_z, heavy_lad = _extreme_inverse_and_lad(
            torch.abs(x), torch.abs(tail_param)
        )
        light_z, light_lad = _shift_power_inverse_and_lad(torch.abs(x), tail_param + 2)
        z = torch.where(tail_param > 0.0, heavy_z, light_z)
        lad = torch.where(tail_param > 0.0, heavy_lad, light_lad)

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
        init=None,
    ):
        self.features = features
        super(TailMarginalTransform, self).__init__()

        if init is None:
            init = torch.distributions.Uniform(-0.5, 0.5).sample(
                [features * self._output_dim_multiplier()]
            )

        assert torch.Size([features * self._output_dim_multiplier()]) == init.shape
        self._unc_params = torch.nn.parameter.Parameter(inv_sftplus(init + 1))

    def _output_dim_multiplier(self):
        return 2

    def forward(self, z, context=None):
        return self._elementwise_forward(z, self._unc_params.repeat(z.shape[0], 1))

    def inverse(self, z, context=None):
        return self._elementwise_inverse(z, self._unc_params.repeat(z.shape[0], 1))

    def _elementwise_forward(self, z, autoregressive_params):
        """light -> heavy"""
        unc_ptail, unc_ntail = self._unconstrained_params(autoregressive_params)
        pos_tail_param = softplus(unc_ptail) - 1.0  # (-1, inf)
        neg_tail_param = softplus(unc_ntail) - 1.0  # (-1, inf)
        tail_param = torch.where(z > 0, pos_tail_param, neg_tail_param)

        sign = torch.sign(z)
        heavy_x, heavy_lad = _extreme_transform_and_lad(torch.abs(z), tail_param)
        light_x, light_lad = _shift_power_transform_and_lad(
            torch.abs(z), tail_param + 2.0
        )

        x = torch.where(tail_param > 0.0, heavy_x, light_x)
        lad = torch.where(tail_param > 0.0, heavy_lad, light_lad)

        return sign * x, lad.sum(axis=1)

    def _elementwise_inverse(self, x, autoregressive_params):
        """heavy -> light"""
        unc_ptail, unc_ntail = self._unconstrained_params(autoregressive_params)
        pos_tail_param = softplus(unc_ptail) - 1.0  # (-1, inf)
        neg_tail_param = softplus(unc_ntail) - 1.0  # (-1, inf)
        tail_param = torch.where(x > 0, pos_tail_param, neg_tail_param)

        # tail transform
        sign = torch.sign(x)
        heavy_z, heavy_lad = _extreme_inverse_and_lad(
            torch.abs(x), torch.abs(tail_param)
        )
        light_z, light_lad = _shift_power_inverse_and_lad(torch.abs(x), tail_param + 2)
        z = torch.where(tail_param > 0.0, heavy_z, light_z)
        lad = torch.where(tail_param > 0.0, heavy_lad, light_lad)

        return sign * z, lad.sum(axis=1)

    def _unconstrained_params(self, autoregressive_params):
        autoregressive_params = autoregressive_params.view(
            -1, self.features, self._output_dim_multiplier()
        )
        return (
            autoregressive_params[..., 0],
            autoregressive_params[..., 1],
        )


class TailAffineMarginalTransform(Transform):
    def __init__(
        self,
        features,
        pos_tail_init,
        neg_tail_init,
        shift_init,
        scale_init,
    ):
        self.features = features
        super(TailAffineMarginalTransform, self).__init__()
        self._unc_pos_tail_params = torch.nn.parameter.Parameter(
            inv_sftplus(pos_tail_init + 1)
        )
        self._unc_neg_tail_params = torch.nn.parameter.Parameter(
            inv_sftplus(neg_tail_init + 1)
        )
        self._unc_scale = torch.nn.parameter.Parameter(inv_sftplus(scale_init))
        self._unc_shift = torch.nn.parameter.Parameter(shift_init)
        # interleave the parameters
        self._unc_params = (
            torch.stack(
                [
                    self._unc_pos_tail_params,
                    self._unc_neg_tail_params,
                    self._unc_scale,
                    self._unc_shift,
                ]
            )
            .t()
            .flatten()
        )

    def _output_dim_multiplier(self):
        return 4

    def forward(self, z, context=None):
        return self._elementwise_forward(z, self._unc_params.repeat(z.shape[0], 1))

    def inverse(self, z, context=None):
        return self._elementwise_inverse(z, self._unc_params.repeat(z.shape[0], 1))

    def _elementwise_forward(self, z, autoregressive_params):
        """light -> heavy"""
        unc_ptail, unc_ntail, unc_scale, shift = self._unconstrained_params(
            autoregressive_params
        )

        pos_tail_param = softplus(unc_ptail) - 1.0  # (-1, inf)
        neg_tail_param = softplus(unc_ntail) - 1.0  # (-1, inf)
        tail_param = torch.where(z > 0, pos_tail_param, neg_tail_param)
        scale = softplus(unc_scale)

        sign = torch.sign(z)
        heavy_x, heavy_lad = _extreme_transform_and_lad(torch.abs(z), tail_param)
        light_x, light_lad = _shift_power_transform_and_lad(
            torch.abs(z), tail_param + 2.0
        )

        x = torch.where(tail_param > 0.0, heavy_x, light_x)
        lad = torch.where(tail_param > 0.0, heavy_lad, light_lad)
        lad += torch.log(scale)

        return sign * x * scale + shift, lad.sum(axis=1)

    def _elementwise_inverse(self, x, autoregressive_params):
        """heavy -> light"""
        unc_ptail, unc_ntail, unc_scale, shift = self._unconstrained_params(
            autoregressive_params
        )

        pos_tail_param = softplus(unc_ptail) - 1.0  # (-1, inf)
        neg_tail_param = softplus(unc_ntail) - 1.0  # (-1, inf)
        tail_param = torch.where(x > 0, pos_tail_param, neg_tail_param)
        scale = softplus(unc_scale)

        x = (x - shift) / scale  # affine

        # tail transform
        sign = torch.sign(x)
        heavy_z, heavy_lad = _extreme_inverse_and_lad(
            torch.abs(x), torch.abs(tail_param)
        )
        light_z, light_lad = _shift_power_inverse_and_lad(torch.abs(x), tail_param + 2)
        z = torch.where(tail_param > 0.0, heavy_z, light_z)
        lad = torch.where(tail_param > 0.0, heavy_lad, light_lad)
        lad -= torch.log(scale)

        return sign * z, lad.sum(axis=1)

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


class CopulaMarginalTransform(Transform):
    def __init__(
        self,
        features,
        pos_tail_init,
        neg_tail_init,
    ):
        self.features = features
        super(CopulaMarginalTransform, self).__init__()
        self._unc_pos_tail_params = torch.nn.parameter.Parameter(
            inv_sftplus(pos_tail_init + 1)
        )
        self._unc_neg_tail_params = torch.nn.parameter.Parameter(
            inv_sftplus(neg_tail_init + 1)
        )
        self._unc_params = (
            torch.stack(
                [
                    self._unc_pos_tail_params,
                    self._unc_neg_tail_params,
                ]
            )
            .t()
            .flatten()
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
        pos_tail_param = softplus(unc_ptail) - 1.0  # (-1, inf)
        neg_tail_param = softplus(unc_ntail) - 1.0  # (-1, inf)
        tail_param = torch.where(z > 0, pos_tail_param, neg_tail_param)

        sign = torch.sign(z)
        x, lad = _copula_transform_and_lad(torch.abs(z), tail_param)

        return sign * x, lad.sum(axis=1)

    def _elementwise_inverse(self, x, autoregressive_params):
        """heavy -> light"""
        unc_ptail, unc_ntail = self._unconstrained_params(autoregressive_params)
        pos_tail_param = softplus(unc_ptail) - 1.0  # (-1, inf)
        neg_tail_param = softplus(unc_ntail) - 1.0  # (-1, inf)
        tail_param = torch.where(x > 0, pos_tail_param, neg_tail_param)

        # tail transform
        sign = torch.sign(x)
        u, lad = _copula_inverse_and_lad(torch.abs(x), torch.abs(tail_param))

        return sign * u, lad.sum(axis=1)

    def _unconstrained_params(self, autoregressive_params):
        autoregressive_params = autoregressive_params.view(
            -1, self.features, self._output_dim_multiplier()
        )
        return (
            autoregressive_params[..., 0],
            autoregressive_params[..., 1],
        )


class EXMarginalTransform(Transform):
    def __init__(
        self,
        features,
        init=1.0,
    ):
        self.features = features
        super(EXMarginalTransform, self).__init__()

        param_dim = features * self._output_dim_multiplier()
        if hasattr(init, "shape") and init.shape == torch.Size([param_dim]):
            self._unc_params = torch.nn.parameter.Parameter(inv_sftplus(init))
        elif not hasattr(init, "shape"):
            self._unc_params = torch.nn.parameter.Parameter(
                inv_sftplus(torch.tensor(init))
                * torch.ones(features * self._output_dim_multiplier())
            )
        else:
            raise Exception("Invalid init for EXMarginalTransform!")

    def _output_dim_multiplier(self):
        return 2

    def forward(self, z, context=None):
        return self._elementwise_forward(z, self._unc_params.repeat(z.shape[0], 1))

    def inverse(self, z, context=None):
        return self._elementwise_inverse(z, self._unc_params.repeat(z.shape[0], 1))

    def _elementwise_forward(self, z, autoregressive_params):
        """light -> heavy"""
        unc_ptail, unc_ntail = self._unconstrained_params(autoregressive_params)
        pos_tail_param = softplus(unc_ptail)  # (0, inf)
        neg_tail_param = softplus(unc_ntail)  # (0, inf)
        tail_param = torch.where(z > 0, pos_tail_param, neg_tail_param)

        sign = torch.sign(z)
        x, lad = _extreme_transform_and_lad(torch.abs(z), tail_param)
        return sign * x, lad.sum(axis=1)

    def _elementwise_inverse(self, x, autoregressive_params):
        """heavy -> light"""
        unc_ptail, unc_ntail = self._unconstrained_params(autoregressive_params)
        pos_tail_param = softplus(unc_ptail)  # (0, inf)
        neg_tail_param = softplus(unc_ntail)  # (0, inf)
        tail_param = torch.where(x > 0, pos_tail_param, neg_tail_param)

        # tail transform
        sign = torch.sign(x)
        z, lad = _extreme_inverse_and_lad(torch.abs(x), torch.abs(tail_param))
        return sign * z, lad.sum(axis=1)

    def _unconstrained_params(self, autoregressive_params):
        autoregressive_params = autoregressive_params.view(
            -1, self.features, self._output_dim_multiplier()
        )
        return (
            autoregressive_params[..., 0],
            autoregressive_params[..., 1],
        )


class PowerMarginalTransform(Transform):
    def __init__(
        self,
        features,
    ):
        self.features = features
        super(PowerMarginalTransform, self).__init__()
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
        pos_tail_param = softplus(unc_ptail)  # (0, inf)
        neg_tail_param = softplus(unc_ntail)  # (0, inf)
        tail_param = torch.where(z > 0, pos_tail_param, neg_tail_param)

        sign = torch.sign(z)
        x, lad = _shift_power_transform_and_lad(torch.abs(z), tail_param)
        return sign * x, lad.sum(axis=1)

    def _elementwise_inverse(self, x, autoregressive_params):
        """heavy -> light"""
        unc_ptail, unc_ntail = self._unconstrained_params(autoregressive_params)
        pos_tail_param = softplus(unc_ptail)  # (-1, inf)
        neg_tail_param = softplus(unc_ntail)  # (-1, inf)
        tail_param = torch.where(x > 0, pos_tail_param, neg_tail_param)

        # tail transform
        sign = torch.sign(x)
        z, lad = _shift_power_inverse_and_lad(torch.abs(x), torch.abs(tail_param))
        return sign * z, lad.sum(axis=1)

    def _unconstrained_params(self, autoregressive_params):
        autoregressive_params = autoregressive_params.view(
            -1, self.features, self._output_dim_multiplier()
        )
        return (
            autoregressive_params[..., 0],
            autoregressive_params[..., 1],
        )


class SinhMarginalTransform(Transform):
    def __init__(
        self,
        features,
    ):
        self.features = features
        super(SinhMarginalTransform, self).__init__()
        self._unc_params = torch.nn.parameter.Parameter(
            inv_sftplus(torch.ones(features * self._output_dim_multiplier()))
        )

    def _output_dim_multiplier(self):
        return 1

    def forward(self, z, context=None):
        return self._elementwise_forward(z, self._unc_params.repeat(z.shape[0], 1))

    def inverse(self, z, context=None):
        return self._elementwise_inverse(z, self._unc_params.repeat(z.shape[0], 1))

    def _elementwise_forward(self, z, autoregressive_params):
        """light -> heavy"""
        unc_kurt = self._unconstrained_params(autoregressive_params)
        kurt_param = softplus(unc_kurt)  # (0, inf)

        x, lad = _sinh_asinh_transform_and_lad(z, kurt_param)
        return x, lad.sum(axis=1)

    def _elementwise_inverse(self, x, autoregressive_params):
        """heavy -> light"""
        unc_kurt = self._unconstrained_params(autoregressive_params)
        kurt_param = softplus(unc_kurt)  # (0, inf)

        z, lad = _sinh_asinh_inverse_and_lad(x, kurt_param)
        return z, lad.sum(axis=1)

    def _unconstrained_params(self, autoregressive_params):
        autoregressive_params = autoregressive_params.view(
            -1, self.features, self._output_dim_multiplier()
        )
        return autoregressive_params[..., 0]


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
        unc_tail, unc_scale, shift_param = self._unconstrained_params(
            autoregressive_params
        )
        tail_param = 1e-3 + softplus(unc_tail)  # (0, inf)
        scale_param = 1e-2 + softplus(unc_scale)

        sign = torch.sign(z)
        x, lad = _extreme_transform_and_lad(torch.abs(z), tail_param)

        return sign * x * scale_param + shift_param, (lad + torch.log(scale_param)).sum(
            axis=1
        )

    def _elementwise_inverse(self, x, autoregressive_params):
        unc_tail, unc_scale, shift_param = self._unconstrained_params(
            autoregressive_params
        )
        tail_param = 1e-3 + softplus(unc_tail)  # (0, inf)
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
        return (
            autoregressive_params[..., 0],
            autoregressive_params[..., 1],
            autoregressive_params[..., 2],
        )


class Marginal(Transform):
    def __init__(self, marginal_transforms):
        self.marginal_transforms = marginal_transforms
        super(Marginal, self).__init__()

    def inverse(self, z, context=None):
        xs = []
        lad = 0.0
        for dim_ix, mt in enumerate(self.marginal_transforms):
            x, _lad = mt.inverse(z[:, [dim_ix]], context)
            xs.append(x)
            lad += _lad
        return torch.hstack(xs), lad

    def forward(self, z, context=None):
        xs = []
        lad = 0.0
        for dim_ix, mt in enumerate(self.marginal_transforms):
            x, _lad = mt.forward(z[:, [dim_ix]], context)
            xs.append(x)
            lad += _lad
        return torch.hstack(xs), lad


def bisection_search(func, x_0_low, x_0_high, target, tol=1e-6, max_iter=100):
    for i in range(max_iter):
        x_mid = 0.5 * (x_0_low + x_0_high)
        f_mid = func(x_mid)
        error = f_mid - target
        if (error.abs() < tol).all():
            return x_mid

        x_0_low = torch.where(error < 0, x_mid, x_0_low)
        x_0_high = torch.where(error > 0, x_mid, x_0_high)

    return 0.5 * (x_0_low + x_0_high)


class Mixture(Transform):
    def __init__(self, transform_1, transform_2):
        self.transform_1 = transform_1
        self.transform_2 = transform_2
        self._unc_mix_param = torch.nn.Parameter(torch.tensor(0.0))
        super(Transform, self).__init__()

    def inverse(self, x, context=None):
        z_1, lad_1 = transform_1.inverse(x, context)
        z_2, lad_2 = transform_2.inverse(x, context)

        mix_param = softplus(self._unc_mix_param)

        z = z_1 * mix_param + z_2 * (1 - mix_param)
        z = lad_1 * mix_param + lad_2 * (1 - mix_param)

        return torch.hstack(xs), lad

    def forward(self, z, context=None):
        x_1, _ = self.transform_1.forward(z)
        x_2, _ = self.transform_2.forward(z)
        x_1_higher = x_1 > x_2
        x_0_high = torch.where(x_1_higher, x_1, x_0)
        x_0_low = torch.where(x_1_higher, x_2, x_1)

        inverse_trans = lambda x: self.inverse(x, context)
        x = bisection_search(
            inverse_trans, x_0_low, x_0_high, target=z, tol=1e-6, max_iter=100
        )  # which x gives z?
        _, inv_lad = inverse_trans(x.detach())
        lad = -inv_lad
        return x, lad
