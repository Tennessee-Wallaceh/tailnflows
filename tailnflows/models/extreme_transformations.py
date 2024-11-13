import torch
from torch.nn.functional import softplus, relu, sigmoid
from nflows.transforms.autoregressive import AutoregressiveTransform
from nflows.transforms import made as made_module
from nflows.transforms import Transform
from tailnflows.models.utils import inv_sftplus, inv_sigmoid
from typing import TypedDict, Optional, Callable
from math import sqrt

from nflows.transforms.splines import rational_quadratic
from nflows.transforms.splines.rational_quadratic import (
    forward_rational_quadratic_spline,
    inverse_rational_quadratic_spline,
    unconstrained_rational_quadratic_spline_forward,
    unconstrained_rational_quadratic_spline_inverse,
)

from tailnflows.models.simple_spline import forward_rqs, inverse_rqs

MAX_TAIL = 5.0
LOW_TAIL_INIT = 0.1
HIGH_TAIL_INIT = 0.9
SQRT_2 = sqrt(2.0)
SQRT_PI = sqrt(torch.pi)
MIN_ERFC_INV = 1e-6
PI = torch.pi


class NNKwargs(TypedDict, total=False):
    hidden_features: int
    num_blocks: int
    use_residual_blocks: bool
    random_mask: bool
    activation: Callable
    dropout_probability: float
    use_batch_norm: bool


class SpecifiedNNKwargs(TypedDict, total=True):
    hidden_features: int
    num_blocks: int
    use_residual_blocks: bool
    random_mask: bool
    activation: Callable
    dropout_probability: float
    use_batch_norm: bool


def configure_nn(nn_kwargs: NNKwargs) -> SpecifiedNNKwargs:
    return {
        "hidden_features": nn_kwargs.get("hidden_features", 5),
        "num_blocks": nn_kwargs.get("num_blocks", 2),
        "use_residual_blocks": nn_kwargs.get("use_residual_blocks", True),
        "random_mask": nn_kwargs.get("random_mask", False),
        "activation": nn_kwargs.get("activation", relu),
        "dropout_probability": nn_kwargs.get("dropout_probability", 0.0),
        "use_batch_norm": nn_kwargs.get("use_batch_norm", False),
    }


class ExtremeActivation(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.in_dim = dim
        self._unc_mix = torch.nn.Parameter(torch.ones([dim, 3]))
        # with torch.no_grad():
        #     self._unc_mix[:, 0] = 5.0  # init at identity
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
            **kwargs,
        )
        self.extreme_activation = ExtremeActivation(features * output_multiplier)

    def forward(self, x, context=None):
        param_data = self.base_model(x, context)
        adjusted_param_data = self.extreme_activation(param_data)
        return adjusted_param_data


def _erfcinv(x):
    with torch.no_grad():
        x = torch.clamp(x, min=MIN_ERFC_INV)
    return -torch.special.ndtri(0.5 * x) / SQRT_2


def _shift_power_transform_and_lad(z, tail_param):
    transformed = (SQRT_2 / SQRT_PI) * (torch.pow(1 + z / tail_param, tail_param) - 1)
    lad = (tail_param - 1) * torch.log(1 + z / tail_param)
    lad += torch.log(torch.tensor(SQRT_2 / SQRT_PI))
    return transformed, lad


def _shift_power_inverse_and_lad(x, tail_param):
    transformed = (
        (SQRT_PI / SQRT_2) * tail_param * (torch.pow(1 + x, 1 / tail_param) - 1)
    )
    lad = ((1 / tail_param) - 1) * torch.log(1 + x)
    lad -= torch.log(torch.tensor(SQRT_2 / SQRT_PI))
    return transformed, lad


def _extreme_transform_and_lad(z, tail_param):
    g = torch.erfc(z / SQRT_2)
    x = (torch.pow(g, -tail_param) - 1) / tail_param

    lad = torch.log(g) * (-tail_param - 1)
    lad -= 0.5 * torch.square(z)
    lad += torch.log(torch.tensor(SQRT_2 / SQRT_PI))

    return x, lad


def _small_erfcinv(log_g):
    """
    Use series expansion for erfcinv(x) as x->0, using subsitution for
    log x^-2
    """
    log_z_sq = 2 * log_g

    inner = torch.log(torch.tensor(2 / PI)) - log_z_sq
    inner -= (torch.log(torch.tensor(2 / PI)) - log_z_sq).log()

    z = inner.pow(0.5) / SQRT_2

    return z


def _extreme_inverse_and_lad(x, tail_param):
    inner = 1 + tail_param * x
    g = torch.pow(inner, -1 / tail_param)
    stable_g = g > MIN_ERFC_INV

    erfcinv_val = torch.zeros_like(x)
    erfcinv_val[stable_g] = _erfcinv(g[stable_g])

    log_g = -torch.log(inner[~stable_g]) / tail_param[~stable_g]
    erfcinv_val[~stable_g] = _small_erfcinv(log_g)

    z = SQRT_2 * erfcinv_val

    lad = (-1 - 1 / tail_param) * torch.log(inner)
    lad += torch.square(erfcinv_val)
    lad += torch.log(torch.tensor(SQRT_PI / SQRT_2))

    return z, lad


def neg_extreme_transform_and_lad(z, tail_param):
    def _small_erfcinv(log_z):
        inner = torch.log(torch.tensor(2 / torch.pi)) - 2 * log_z
        inner -= (torch.log(torch.tensor(2 / torch.pi)) - 2 * log_z).log()
        return inner.pow(0.5) / SQRT_2

    erfc_val = torch.erfc(z / SQRT_2)
    g = erfc_val.pow(-tail_param)

    stable_g = g > MIN_ERFC_INV

    erfcinv_val = torch.zeros_like(z)
    erfcinv_val[stable_g] = _erfcinv(g[stable_g])

    log_z = -torch.log(z[~stable_g])
    log_z += -z[~stable_g].square() / 2
    log_z += torch.log(torch.tensor(SQRT_2 / SQRT_PI))
    log_z *= -tail_param[~stable_g]

    erfcinv_val[~stable_g] = _small_erfcinv(log_z)

    x = -erfcinv_val * 2 / (SQRT_PI * tail_param)

    lad = torch.square(erfcinv_val) - 0.5 * torch.square(z)
    lad += torch.log(torch.tensor(SQRT_2 / SQRT_PI))
    lad += (-1 - tail_param) * torch.log(erfc_val)

    return x, lad


def _tail_switch_transform(z, pos_tail, neg_tail, shift, scale):
    sign = torch.sign(z)
    tail_param = torch.where(z > 0, pos_tail, neg_tail)
    heavy_tail = tail_param > 0
    heavy_x, heavy_lad = _extreme_transform_and_lad(
        torch.abs(z[heavy_tail]), tail_param[heavy_tail]
    )
    light_x, light_lad = neg_extreme_transform_and_lad(
        torch.abs(z[~heavy_tail]), tail_param[~heavy_tail]
    )

    lad = torch.zeros_like(z)
    x = torch.zeros_like(z)

    x[heavy_tail] = heavy_x
    x[~heavy_tail] = light_x

    lad[heavy_tail] = heavy_lad
    lad[~heavy_tail] = light_lad

    lad += torch.log(scale)
    return sign * x * scale + shift, lad


def _tail_affine_transform(z, pos_tail, neg_tail, shift, scale):
    sign = torch.sign(z)
    tail_param = torch.where(z > 0, pos_tail, neg_tail)
    x, lad = _extreme_transform_and_lad(torch.abs(z), tail_param)
    lad += torch.log(scale)
    return sign * x * scale + shift, lad


def _tail_affine_inverse(x, pos_tail, neg_tail, shift, scale):
    # affine
    x = (x - shift) / scale

    # tail transform
    sign = torch.sign(x)
    tail_param = torch.where(x > 0, pos_tail, neg_tail)

    z, lad = _extreme_inverse_and_lad(torch.abs(x), torch.abs(tail_param))

    lad -= torch.log(scale)
    return sign * z, lad


def _tail_forward(z, pos_tail, neg_tail):
    sign = torch.sign(z)
    tail_param = torch.where(z > 0, pos_tail, neg_tail)
    x, lad = _extreme_transform_and_lad(torch.abs(z), tail_param)
    return sign * x, lad.sum(axis=1)


def _tail_inverse(x, pos_tail, neg_tail):
    sign = torch.sign(x)
    tail_param = torch.where(x > 0, pos_tail, neg_tail)
    z, lad = _extreme_inverse_and_lad(torch.abs(x), torch.abs(tail_param))
    return sign * z, lad.sum(axis=1)


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


def _asymmetric_scale_transform_and_lad(z, pos_scale, neg_scale):
    sq_plus_1 = (z.square() + 1.0).sqrt()
    a = pos_scale + neg_scale
    b = pos_scale - neg_scale

    pos_x = pos_scale * (sq_plus_1 + z)
    neg_x = neg_scale * (z - sq_plus_1)
    x = 0.5 * (pos_x + neg_x - b)

    lad = torch.log1p((b / a) * (z / sq_plus_1))
    lad -= torch.log(torch.tensor(2.0))
    return x, lad


def _asymmetric_scale_inverse_and_lad(x, pos_scale, neg_scale):
    a = pos_scale + neg_scale
    b = pos_scale - neg_scale
    disc = a**2 - b**2

    z_dash = a * b + 2 * a * x
    term_2 = (a**2 + 4 * b * x + 4 * x**2).sqrt()

    z = (z_dash - torch.sign(b) * term_2) / disc

    lad = torch.log(2 * a - torch.sign(b) * (2 * b + 4 * x) / term_2)
    lad -= torch.log(disc)
    return z, lad


def two_scale_affine_forward(z, shift, scale_neg, scale_pos, bound=torch.tensor(1.0)):
    # build batch x dim x knots arrays
    derivatives = torch.ones([*z.shape, 3])
    derivatives[:, :, 0] = scale_neg
    derivatives[:, :, -1] = scale_pos

    input_knots = torch.zeros([*z.shape, 3])
    input_knots[:, :, 0] = -bound
    input_knots[:, :, -1] = bound

    output_knots = torch.zeros([*z.shape, 3])
    output_knots[:, :, 0] = -bound
    output_knots[:, :, -1] = bound

    neg_region = z < -bound
    pos_region = z > bound
    body = ~torch.logical_or(neg_region, pos_region)
    neg_scale_ix = (neg_region * torch.arange(z.shape[-1]))[neg_region]
    pos_scale_ix = (pos_region * torch.arange(z.shape[-1]))[pos_region]

    x = torch.empty_like(z)
    lad = torch.empty_like(z)

    x[neg_region] = (z[neg_region] + bound) * scale_neg[neg_scale_ix] - bound
    x[pos_region] = (z[pos_region] - bound) * scale_pos[pos_scale_ix] + bound
    lad[neg_region] = -torch.log(scale_neg[neg_scale_ix])
    lad[pos_region] = -torch.log(scale_pos[pos_scale_ix])

    body_x, body_lad = forward_rqs(
        z[body], input_knots[body], output_knots[body], derivatives[body]
    )
    x[body] = body_x
    # this has already been inverted, so undo for subsequent inversion
    lad[body] = -body_lad

    x += shift

    return x, lad


def two_scale_affine_inverse(x, shift, scale_neg, scale_pos, bound=torch.tensor(1.0)):
    # build batch x dim x knots arrays
    derivatives = torch.ones([*x.shape, 3])
    derivatives[:, :, 0] = scale_neg
    derivatives[:, :, -1] = scale_pos

    input_knots = torch.zeros([*x.shape, 3])
    input_knots[:, :, 0] = -bound
    input_knots[:, :, -1] = bound

    output_knots = torch.zeros([*x.shape, 3])
    output_knots[:, :, 0] = -bound
    output_knots[:, :, -1] = bound

    # undo shift
    x -= shift

    # regions and place holders
    neg_region = x < -bound
    pos_region = x > bound
    body = ~torch.logical_or(neg_region, pos_region)
    neg_scale_ix = (neg_region * torch.arange(x.shape[-1]))[neg_region]
    pos_scale_ix = (pos_region * torch.arange(x.shape[-1]))[pos_region]

    z = torch.empty_like(x)
    lad = torch.empty_like(x)

    # scales
    z[neg_region] = (x[neg_region] + bound) / scale_neg[neg_scale_ix] - bound
    z[pos_region] = (x[pos_region] - bound) / scale_pos[pos_scale_ix] + bound
    lad[neg_region] = torch.log(scale_neg[neg_scale_ix])
    lad[pos_region] = torch.log(scale_pos[pos_scale_ix])

    # body
    body_z, body_lad = inverse_rqs(
        x[body], input_knots[body], output_knots[body], derivatives[body]
    )
    z[body] = body_z
    lad[body] = body_lad

    return z, lad


def flip(transform):
    """
    if it is an autoregressive transform change around the element wise transform,
    to preserve the direction of the autoregression. Otherwise, we can flip the full
    transformation.
    """
    if issubclass(type(transform), AutoregressiveTransform):
        _inverse = transform._elementwise_inverse
        transform._elementwise_inverse = transform._elementwise_forward
        transform._elementwise_forward = _inverse
    else:
        _inverse = transform.inverse
        transform.inverse = transform.forward
        transform.forward = _inverse

    return transform


class TailMarginalTransform(Transform):
    def __init__(
        self,
        features,
        pos_tail_init=None,
        neg_tail_init=None,
    ):
        self.features = features

        super(TailMarginalTransform, self).__init__()

        # init with heavy tail, otherwise heavy targets may fail to fit
        if pos_tail_init is None:
            pos_tail_init = torch.distributions.Uniform(
                LOW_TAIL_INIT, HIGH_TAIL_INIT
            ).sample([features])

        if neg_tail_init is None:
            neg_tail_init = torch.distributions.Uniform(
                LOW_TAIL_INIT, HIGH_TAIL_INIT
            ).sample([features])

        assert torch.Size([features]) == pos_tail_init.shape
        assert torch.Size([features]) == neg_tail_init.shape

        self._unc_pos_tail = torch.nn.parameter.Parameter(inv_sftplus(pos_tail_init))
        self._unc_neg_tail = torch.nn.parameter.Parameter(inv_sftplus(neg_tail_init))

    def forward(self, z, context=None):
        pos_tail_param = softplus(self._unc_pos_tail)
        neg_tail_param = softplus(self._unc_neg_tail)
        x, lad = _tail_forward(z, pos_tail_param, neg_tail_param)
        return x, lad

    def inverse(self, x, context=None):
        pos_tail_param = softplus(self._unc_pos_tail)
        neg_tail_param = softplus(self._unc_neg_tail)
        z, lad = _tail_inverse(x, pos_tail_param, neg_tail_param)
        return z, lad

    def fix_tails(self):
        # freeze only the parameters related to the tail
        self._unc_pos_tail.requires_grad = False
        self._unc_neg_tail.requires_grad = False


class AffineMarginalTransform(Transform):
    def __init__(
        self,
        features,
        shift_init=None,
        scale_init=None,
    ):
        self.features = features
        super(AffineMarginalTransform, self).__init__()

        # random inits if needed
        if shift_init is None:
            shift_init = torch.zeros([features])

        if scale_init is None:
            scale_init = torch.ones([features])

        assert torch.Size([features]) == shift_init.shape
        assert torch.Size([features]) == scale_init.shape

        # convert to unconstrained versions
        self._unc_shift = torch.nn.parameter.Parameter(shift_init)
        self._unc_scale = torch.nn.parameter.Parameter(inv_sftplus(scale_init))

    def forward(self, z, context=None):
        shift = self._unc_shift
        # scale = softplus(self._unc_scale)
        scale = 1e-3 + softplus(self._unc_scale)

        x = z * scale + shift
        lad = torch.log(scale).sum()
        return x, lad

    def inverse(self, x, context=None):
        """heavy -> light"""
        shift = self._unc_shift
        scale = 1e-3 + softplus(self._unc_scale)

        z = (x - shift) / scale
        lad = -torch.log(scale).sum()
        return z, lad


class RQSMarginalTransform(Transform):
    def __init__(
        self,
        features,
        num_bins=10,
        tail_bound=1.0,
        min_bin_width=rational_quadratic.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=rational_quadratic.DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=rational_quadratic.DEFAULT_MIN_DERIVATIVE,
    ):
        self.features = features

        super(RQSMarginalTransform, self).__init__()

        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.tails = "linear"
        self.tail_bound = tail_bound
        self.spline_fn = (
            unconstrained_rational_quadratic_spline_forward,
            unconstrained_rational_quadratic_spline_inverse,
        )
        self.spline_kwargs = {
            "tails": self.tails,
            "left_tail_bound": -self.tail_bound,
            "right_tail_bound": self.tail_bound,
        }
        self.param_dim = self.num_bins * 3 - 1  # per dim
        self._unc_widths = torch.zeros([1, self.features, self.num_bins])
        self._unc_heights = torch.zeros([1, self.features, self.num_bins])
        self._unc_derivatives = torch.zeros([1, self.features, self.num_bins - 1])

    def forward(self, inputs, context=None):
        batch_dim = inputs.shape[0]
        outputs, logabsdet = self.spline_fn[0](
            inputs=inputs,
            unnormalized_widths=self._unc_widths.repeat(batch_dim, 1, 1),
            unnormalized_heights=self._unc_heights.repeat(batch_dim, 1, 1),
            unnormalized_derivatives=self._unc_derivatives.repeat(batch_dim, 1, 1),
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **self.spline_kwargs,
        )

        return outputs, logabsdet.sum(axis=1)

    def inverse(self, inputs, context=None):
        batch_dim = inputs.shape[0]
        outputs, logabsdet = self.spline_fn[1](
            inputs=inputs,
            unnormalized_widths=self._unc_widths.repeat(batch_dim, 1, 1),
            unnormalized_heights=self._unc_heights.repeat(batch_dim, 1, 1),
            unnormalized_derivatives=self._unc_derivatives.repeat(batch_dim, 1, 1),
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **self.spline_kwargs,
        )

        return outputs, logabsdet.sum(axis=1)


class TailAffineMarginalTransform(Transform):
    def __init__(
        self,
        features,
        pos_tail_init=None,
        neg_tail_init=None,
        shift_init=None,
        scale_init=None,
    ):
        self.features = features
        super(TailAffineMarginalTransform, self).__init__()

        # random inits if needed
        if pos_tail_init is None:
            pos_tail_init = torch.distributions.Uniform(
                LOW_TAIL_INIT, HIGH_TAIL_INIT
            ).sample([features])

        if neg_tail_init is None:
            neg_tail_init = torch.distributions.Uniform(
                LOW_TAIL_INIT, HIGH_TAIL_INIT
            ).sample([features])

        if shift_init is None:
            shift_init = torch.zeros([features])

        if scale_init is None:
            scale_init = torch.ones([features])

        assert torch.Size([features]) == pos_tail_init.shape
        assert torch.Size([features]) == neg_tail_init.shape
        assert torch.Size([features]) == shift_init.shape
        assert torch.Size([features]) == scale_init.shape

        # convert to unconstrained versions
        self._unc_pos_tail = torch.nn.parameter.Parameter(inv_sftplus(pos_tail_init))
        self._unc_neg_tail = torch.nn.parameter.Parameter(inv_sftplus(neg_tail_init))
        self._unc_shift = torch.nn.parameter.Parameter(shift_init)
        self._unc_scale = torch.nn.parameter.Parameter(inv_sftplus(scale_init))

    def forward(self, z, context=None):
        pos_tail = softplus(self._unc_pos_tail)
        neg_tail = softplus(self._unc_neg_tail)
        shift = self._unc_shift
        scale = 1e-3 + softplus(self._unc_scale)

        x, lad = _tail_affine_transform(z, pos_tail, neg_tail, shift, scale)
        return x, lad.sum(axis=1)

    def inverse(self, x, context=None):
        """heavy -> light"""
        pos_tail = softplus(self._unc_pos_tail)
        neg_tail = softplus(self._unc_neg_tail)
        shift = self._unc_shift
        scale = 1e-3 + softplus(self._unc_scale)

        z, lad = _tail_affine_inverse(x, pos_tail, neg_tail, shift, scale)
        return z, lad.sum(axis=1)

    def fix_tails(self):
        # freeze only the parameters related to the tail
        self._unc_pos_tail.requires_grad = False
        self._unc_neg_tail.requires_grad = False


class TailSwitchMarginalTransform(Transform):
    def __init__(
        self,
        features,
        pos_tail_init=None,
        neg_tail_init=None,
        shift_init=None,
        scale_init=None,
    ):
        self.features = features
        super(TailSwitchMarginalTransform, self).__init__()

        # random inits if needed
        if pos_tail_init is None:
            pos_tail_init = torch.distributions.Uniform(
                LOW_TAIL_INIT, HIGH_TAIL_INIT
            ).sample([features])

        if neg_tail_init is None:
            neg_tail_init = torch.distributions.Uniform(
                LOW_TAIL_INIT, HIGH_TAIL_INIT
            ).sample([features])

        if shift_init is None:
            shift_init = torch.zeros([features])

        if scale_init is None:
            scale_init = torch.ones([features])

        assert torch.Size([features]) == pos_tail_init.shape
        assert torch.Size([features]) == neg_tail_init.shape
        assert torch.Size([features]) == shift_init.shape
        assert torch.Size([features]) == scale_init.shape

        # convert to unconstrained versions
        self._unc_pos_tail = torch.nn.parameter.Parameter(
            inv_sftplus(pos_tail_init + 1)
        )
        self._unc_neg_tail = torch.nn.parameter.Parameter(
            inv_sftplus(neg_tail_init + 1)
        )
        self._unc_shift = torch.nn.parameter.Parameter(shift_init)
        self._unc_scale = torch.nn.parameter.Parameter(inv_sftplus(scale_init))

    def forward(self, z, context=None):
        pos_tail = softplus(self._unc_pos_tail) - 1.0
        neg_tail = softplus(self._unc_neg_tail) - 1.0
        shift = self._unc_shift
        scale = 1e-3 + softplus(self._unc_scale)

        x, lad = _tail_switch_transform(z, pos_tail, neg_tail, shift, scale)
        return x, lad.sum(axis=1)

    def inverse(self, x, context=None):
        raise NotImplementedError

    def fix_tails(self):
        # freeze only the parameters related to the tail
        self._unc_pos_tail.requires_grad = False
        self._unc_neg_tail.requires_grad = False


class MaskedTailSwitchAffineTransform(AutoregressiveTransform):
    def __init__(
        self,
        features,
        context_features=None,
        nn_kwargs={},
    ):
        self.features = features

        nn_kwargs = configure_nn(nn_kwargs)
        made = made_module.MADE(
            features=features,
            context_features=context_features,
            output_multiplier=self._output_dim_multiplier(),
            **nn_kwargs,
        )
        super(MaskedTailSwitchAffineTransform, self).__init__(autoregressive_net=made)

    def _output_dim_multiplier(self):
        return 4

    def _elementwise_forward(self, z, autoregressive_params):
        """light -> heavy"""
        unc_pos_tail, unc_neg_tail, unc_scale, shift_param = self._unconstrained_params(
            autoregressive_params
        )

        pos_tail = 1.5 * sigmoid(unc_pos_tail) - 1.0
        neg_tail = 1.5 * sigmoid(unc_neg_tail) - 1.0
        shift = shift_param
        scale = softplus(unc_scale)

        x, lad = _tail_switch_transform(z, pos_tail, neg_tail, shift, scale)

        return x, lad.sum(axis=1)

    def _elementwise_inverse(self, x, autoregressive_params):
        raise NotImplementedError

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


class MaskedAutoregressiveTailAffineMarginalTransform(AutoregressiveTransform):
    def __init__(
        self,
        features,
        context_features=None,
        nn_kwargs={},
    ):
        self.features = features

        nn_kwargs = configure_nn(nn_kwargs)
        made = made_module.MADE(
            features=features,
            context_features=context_features,
            output_multiplier=self._output_dim_multiplier(),
            **nn_kwargs,
        )
        super(MaskedAutoregressiveTailAffineMarginalTransform, self).__init__(
            autoregressive_net=made
        )

    def _output_dim_multiplier(self):
        return 4

    def _elementwise_forward(self, z, autoregressive_params):
        """light -> heavy"""
        unc_pos_tail, unc_neg_tail, unc_scale, shift_param = self._unconstrained_params(
            autoregressive_params
        )

        pos_tail = softplus(unc_pos_tail)
        neg_tail = softplus(unc_neg_tail)
        shift = shift_param
        scale = softplus(unc_scale)

        x, lad = _tail_affine_transform(z, pos_tail, neg_tail, shift, scale)
        return x, lad.sum(axis=1)

    def _elementwise_inverse(self, x, autoregressive_params):
        """heavy -> light"""
        unc_pos_tail, unc_neg_tail, unc_scale, shift_param = self._unconstrained_params(
            autoregressive_params
        )
        pos_tail = softplus(unc_pos_tail)
        neg_tail = softplus(unc_neg_tail)
        shift = shift_param
        scale = softplus(unc_scale)

        z, lad = _tail_affine_inverse(x, pos_tail, neg_tail, shift, scale)
        return z, lad.sum(axis=1)

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


class TailScaleShiftMarginalTransform(Transform):
    """
    A two tail scale version of the tail and scale transform.
    """

    def __init__(
        self,
        features,
        pos_tail_init=None,
        neg_tail_init=None,
        shift_init=None,
        pos_scale_init=None,
        neg_scale_init=None,
    ):
        self.features = features
        super(TailScaleShiftMarginalTransform, self).__init__()

        # random inits if needed
        if pos_tail_init is None:
            pos_tail_init = torch.distributions.Uniform(
                LOW_TAIL_INIT, HIGH_TAIL_INIT
            ).sample([features])

        if neg_tail_init is None:
            neg_tail_init = torch.distributions.Uniform(
                LOW_TAIL_INIT, HIGH_TAIL_INIT
            ).sample([features])

        if shift_init is None:
            shift_init = torch.zeros([features])

        if pos_scale_init is None:
            pos_scale_init = torch.ones([features])

        if neg_scale_init is None:
            neg_scale_init = torch.ones([features])

        assert torch.Size([features]) == pos_tail_init.shape
        assert torch.Size([features]) == neg_tail_init.shape
        assert torch.Size([features]) == shift_init.shape
        assert torch.Size([features]) == pos_scale_init.shape
        assert torch.Size([features]) == neg_scale_init.shape

        # convert to unconstrained versions
        self._unc_pos_tail = torch.nn.parameter.Parameter(inv_sftplus(pos_tail_init))
        self._unc_neg_tail = torch.nn.parameter.Parameter(inv_sftplus(neg_tail_init))
        self._unc_shift = torch.nn.parameter.Parameter(shift_init)
        self._unc_pos_scale = torch.nn.parameter.Parameter(inv_sftplus(pos_scale_init))
        self._unc_neg_scale = torch.nn.parameter.Parameter(inv_sftplus(neg_scale_init))

    def forward(self, x, context=None):
        pos_tail = softplus(self._unc_pos_tail)
        neg_tail = softplus(self._unc_neg_tail)
        shift = self._unc_shift
        pos_scale = 1e-5 + softplus(self._unc_pos_scale)
        neg_scale = 1e-5 + softplus(self._unc_neg_scale)

        scale_z, scale_lad = two_scale_affine_inverse(x, shift, neg_scale, pos_scale)
        z, tail_lad = _tail_inverse(scale_z, pos_tail, neg_tail)

        return z, tail_lad + scale_lad.sum(axis=1)

    def inverse(self, z, context=None):
        """light -> heavy"""
        pos_tail = softplus(self._unc_pos_tail)
        neg_tail = softplus(self._unc_neg_tail)
        shift = self._unc_shift
        pos_scale = 1e-5 + softplus(self._unc_pos_scale)
        neg_scale = 1e-5 + softplus(self._unc_neg_scale)

        tail_x, tail_lad = _tail_forward(z, pos_tail, neg_tail)
        x, scale_lad = two_scale_affine_forward(tail_x, shift, neg_scale, pos_scale)

        return x, tail_lad + scale_lad.sum(axis=1)


class CopulaMarginalTransform(Transform):
    def __init__(
        self,
        features,
        pos_tail_init=None,
        neg_tail_init=None,
    ):
        self.features = features
        super(CopulaMarginalTransform, self).__init__()
        if pos_tail_init is None:
            pos_tail_init = torch.distributions.Uniform(
                LOW_TAIL_INIT, HIGH_TAIL_INIT
            ).sample([features])

        if neg_tail_init is None:
            neg_tail_init = torch.distributions.Uniform(
                LOW_TAIL_INIT, HIGH_TAIL_INIT
            ).sample([features])

        assert torch.Size([features]) == pos_tail_init.shape
        assert torch.Size([features]) == neg_tail_init.shape

        # convert to unconstrained versions
        self._unc_pos_tail = torch.nn.parameter.Parameter(inv_sftplus(pos_tail_init))
        self._unc_neg_tail = torch.nn.parameter.Parameter(inv_sftplus(neg_tail_init))

    def forward(self, u, context=None):
        """light -> heavy"""
        tail_param = torch.where(
            u > 0, softplus(self._unc_pos_tail), softplus(self._unc_neg_tail)
        )
        sign = torch.sign(u)
        x, lad = _copula_transform_and_lad(torch.abs(u), tail_param)
        return sign * x, lad.sum(axis=1)

    def inverse(self, x, context=None):
        """heavy -> light"""
        tail_param = torch.where(
            x > 0, softplus(self._unc_pos_tail), softplus(self._unc_neg_tail)
        )
        sign = torch.sign(x)
        u, lad = _copula_inverse_and_lad(torch.abs(x), tail_param)
        return sign * u, lad.sum(axis=1)


class MaskedExtremeAutoregressiveTransform(AutoregressiveTransform):
    def __init__(
        self,
        features,
        nn_kwargs,
        context_features=None,
    ):
        self.features = features
        made = made_module.MADE(
            features=features,
            context_features=context_features,
            output_multiplier=self._output_dim_multiplier(),
            **nn_kwargs,
        )
        # init at low value
        made.final_layer.bias = torch.nn.Parameter(
            -2 * torch.ones_like(made.final_layer.bias)
        )
        super(MaskedExtremeAutoregressiveTransform, self).__init__(made)

    def _output_dim_multiplier(self):
        return 4

    def _elementwise_forward(self, z, autoregressive_params):
        unc_pos_tail, unc_neg_tail, unc_scale, shift_param = self._unconstrained_params(
            autoregressive_params
        )
        shift = shift_param
        pos_tail = softplus(unc_pos_tail)  # (0, inf)
        neg_tail = softplus(unc_neg_tail)  # (0, inf)
        scale = softplus(unc_scale)

        x, lad = _tail_affine_transform(z, pos_tail, neg_tail, shift, scale)
        return x, lad.sum(axis=1)

    def _elementwise_inverse(self, x, autoregressive_params):
        unc_pos_tail, unc_neg_tail, unc_scale, shift_param = self._unconstrained_params(
            autoregressive_params
        )
        shift = shift_param
        pos_tail = softplus(unc_pos_tail)  # (0, inf)
        neg_tail = softplus(unc_neg_tail)  # (0, inf)
        scale = softplus(unc_scale)

        z, lad = _tail_affine_inverse(x, pos_tail, neg_tail, shift, scale)
        return z, lad.sum(axis=1)

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
