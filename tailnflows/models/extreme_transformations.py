import torch
from torch.nn.functional import softplus, relu, sigmoid
from nflows.transforms.autoregressive import AutoregressiveTransform
from nflows.transforms import made as made_module
from nflows.transforms import Transform
from tailnflows.models.utils import inv_sftplus, inv_sigmoid
from typing import TypedDict, Optional, Callable
from math import sqrt
from tailnflows.models.simple_spline import (
    univariate_forward_rqs,
    univariate_inverse_rqs,
)

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


def _small_erfcinv(log_g):
    """
    Use series expansion for erfcinv(x) as x->0
    """
    log_z_sq = 2 * log_g

    inner = torch.log(torch.tensor(2 / PI)) - log_z_sq
    inner -= (torch.log(torch.tensor(2 / PI)) - log_z_sq).log()

    z = inner.sqrt() / SQRT_2

    return z


def _stable_erfcinv(x, log_x):
    with torch.no_grad():
        standard_x = torch.clamp(x, min=MIN_ERFC_INV, max=None)
        small_log_x = torch.clamp(log_x, min=None, max=torch.tensor(MIN_ERFC_INV, device=log_x.device).log())

    return torch.where(
        x > MIN_ERFC_INV,
        -torch.special.ndtri(0.5 * standard_x) / SQRT_2,
        _small_erfcinv(small_log_x),
    )


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


def _extreme_inverse_and_lad(x, tail_param):
    inner = 1 + tail_param * x
    g = torch.pow(inner, -1 / tail_param)
    log_g = -torch.log(inner) / tail_param
    erfcinv_val = _stable_erfcinv(g, log_g)

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


def _g_transform_and_lad(z, p, kappa, epsilon, delta):
    """Apply alternative gumbel transformation."""
    x = kappa * torch.pow(z.square() + delta, p / 2.0) + epsilon * z
    derivative = epsilon + kappa * p * z * torch.pow(z.square() + delta, p / 2.0 - 1)
    lad = torch.log(torch.abs(derivative))
    return x, lad


def _g_inverse_and_lad(x, p, kappa, epsilon, delta, num_iter: int = 50):
    """Inverse alternative gumbel transform via Newton iterations."""
    z = x / (epsilon + kappa)
    for _ in range(num_iter):
        f = kappa * torch.pow(z.square() + delta, p / 2.0) + epsilon * z - x
        df = epsilon + kappa * p * z * torch.pow(z.square() + delta, p / 2.0 - 1)
        z_next = z - f / df
        if torch.max(torch.abs(z_next - z)) < 1e-8:
            z = z_next
            break
        z = z_next
    lad = -torch.log(torch.abs(epsilon + kappa * p * z * torch.pow(z.square() + delta, p / 2.0 - 1)))
    return z, lad


def _exp_tail_transform_and_lad(x, lam):
    y = (torch.exp(lam * x) - 1.0) / lam
    lad = lam * x
    return y, lad


def _exp_tail_inverse_and_lad(y, lam):
    x = torch.log1p(lam * y) / lam
    lad = -torch.log1p(lam * y)
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
        self.shift = torch.nn.parameter.Parameter(shift_init)
        self._unc_scale = torch.nn.parameter.Parameter(inv_sftplus(scale_init))

    @property
    def pos_tail(self):
        return softplus(self._unc_pos_tail)

    @property
    def neg_tail(self):
        return softplus(self._unc_neg_tail)

    @property
    def scale(self):
        return 1e-3 + softplus(self._unc_scale)

    def forward(self, z, context=None):
        """light -> heavy"""
        x, lad = _tail_affine_transform(
            z, self.pos_tail, self.neg_tail, self.shift, self.scale
        )
        return x, lad.sum(axis=1)

    def inverse(self, x, context=None):
        """heavy -> light"""
        z, lad = _tail_affine_inverse(
            x, self.pos_tail, self.neg_tail, self.shift, self.scale
        )
        return z, lad.sum(axis=1)

    def fix_tails(self):
        # freeze only the parameters related to the tail
        self._unc_pos_tail.requires_grad = False
        self._unc_neg_tail.requires_grad = False


class AsymmetricTailAffineMarginalTransform(Transform):
    def __init__(
        self,
        features,
        pos_tail_init=None,
        neg_tail_init=None,
        shift_init=None,
        scale_init=None,
    ):
        self.features = features
        super(AsymmetricTailAffineMarginalTransform, self).__init__()

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
        self.shift = torch.nn.parameter.Parameter(shift_init)
        self._unc_scale = torch.nn.parameter.Parameter(inv_sftplus(scale_init))

    @property
    def pos_tail(self):
        return softplus(self._unc_pos_tail)

    @property
    def neg_tail(self):
        return softplus(self._unc_neg_tail)

    @property
    def scale(self):
        return 1e-3 + softplus(self._unc_scale)

    def inverse(self, z, context=None):
        """light -> heavy"""
        
        # tail transform
        rescale = torch.tensor(SQRT_PI / SQRT_2) # ensures agreement on lad at the origin 
        pos_x, pos_lad = _extreme_inverse_and_lad(torch.abs(z) / rescale, self.pos_tail)
        neg_x, neg_lad = _extreme_transform_and_lad(torch.abs(z) * rescale, self.neg_tail)
        neg_x = -neg_x

        x = torch.where(z > 0, pos_x, neg_x)
        lad = torch.where(
            z > 0, 
            pos_lad - torch.log(rescale), 
            neg_lad + torch.log(rescale)
        )

        # affine
        x = x * self.scale + self.shift

        lad -= torch.log(self.scale)
        lad.sum(axis=1)
        return z, lad.sum(axis=1)

    def forward(self, x, context=None):
        """
        Data -> Noise
        +tail heavy -> light
        -tail light -> heavy
        """
        # affine
        x = (x - self.shift) / self.scale

        # tail transform
        rescale = torch.tensor(SQRT_PI / SQRT_2) # ensures agreement on lad at the origin 
        pos_z, pos_lad = _extreme_inverse_and_lad(torch.abs(x) / rescale, self.pos_tail)
        neg_z, neg_lad = _extreme_transform_and_lad(torch.abs(x) * rescale, self.neg_tail)
        neg_z = -neg_z

        z = torch.where(x > 0, pos_z, neg_z)
        lad = torch.where(
            z > 0, 
            pos_lad - torch.log(rescale), 
            neg_lad + torch.log(rescale)
        )

        lad -= torch.log(self.scale)

        return z, lad.sum(axis=1)


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
            ).sample(torch.Size([features]))

        if neg_tail_init is None:
            neg_tail_init = torch.distributions.Uniform(
                LOW_TAIL_INIT, HIGH_TAIL_INIT
            ).sample(torch.Size([features]))

        if shift_init is None:
            shift_init = torch.zeros([features])

        if scale_init is None:
            scale_init = torch.ones([features])

        assert torch.Size([features]) == pos_tail_init.shape
        assert torch.Size([features]) == neg_tail_init.shape
        assert torch.Size([features]) == shift_init.shape
        assert torch.Size([features]) == scale_init.shape

        # convert to unconstrained versions
        self._pos_tail = torch.nn.parameter.Parameter(pos_tail_init)
        self._neg_tail = torch.nn.parameter.Parameter(neg_tail_init)
        self.shift = torch.nn.parameter.Parameter(shift_init)
        self._unc_scale = torch.nn.parameter.Parameter(inv_sftplus(scale_init - 1e-3))

    @staticmethod
    def _tail_parameterisation(z):

        out, _ = torch.vmap(
            univariate_forward_rqs, in_dims=(0, None, None, None), out_dims=(0, 0)
        )(
            z.reshape(-1, 1).clamp(min=-0.5 + 1e-6, max=0.5 - 1e-6),
            torch.tensor([[-0.5, 0.0, 0.5]]),
            torch.tensor([[-0.5, 0.0, 0.5]]),
            torch.tensor([[1.0, 0.2, 1.0]]),
        )

        out = torch.where(
            z.abs() < 0.5,
            out.reshape(-1),
            z,
        )
        return out

    def forward(self, x, context=None):
        # affine
        x = (x - self.shift) / self.scale

        sign = torch.sign(x)
        tail_param = torch.where(x > 0, self.pos_tail, self.neg_tail)

        # negative tail param as this is being applied in data -> noise
        # so data has tail_param tail
        z, lad = TailSwitchMarginalTransform._transformation(torch.abs(x), -tail_param)
        lad -= torch.log(self.scale)
        return sign * z, lad.sum(dim=1)

    def inverse(self, z, context=None):
        sign = torch.sign(z)
        tail_param = torch.where(
            z > 0,
            self.pos_tail,
            self.neg_tail,
        )
        x, lad = TailSwitchMarginalTransform._transformation(torch.abs(z), tail_param)
        lad += torch.log(self.scale)
        x = sign * x * self.scale + self.shift
        return x, lad.sum(dim=1)

    @property
    def scale(self):
        return 1e-3 + softplus(self._unc_scale)

    @property
    def pos_tail(self):
        return self._tail_parameterisation(self._pos_tail)

    @property
    def neg_tail(self):
        return self._tail_parameterisation(self._neg_tail)

    def fix_tails(self):
        # freeze only the parameters related to the tail
        self._unc_pos_tail.requires_grad = False
        self._unc_neg_tail.requires_grad = False

    @staticmethod
    def extreme_transform(z, tau):
        # tau > 1
        tail_param = (tau - 1).abs()
        g = torch.erfc(z / SQRT_2)
        x = (torch.pow(g, -tail_param) - 1) / tail_param
        x *= SQRT_PI / SQRT_2

        lad = torch.log(g) * (-tail_param - 1)
        lad -= 0.5 * torch.square(z)

        return x, lad

    @staticmethod
    def asymp_h_transform(z):
        g = torch.erfc(z / SQRT_2)
        return -torch.log(g) * SQRT_PI / SQRT_2

    @staticmethod
    def inter_h_transform(z, tau):
        # tau in [0, 1]
        tail_param = tau - 1
        g = torch.erfc(z / SQRT_2)
        x = -_erfcinv(g.pow(-tail_param))
        x *= SQRT_2 / tail_param

        lad = torch.log(g) * (-tail_param - 1)
        lad -= 0.5 * torch.square(z)
        lad += 0.5 * tail_param.square() * x.square()

        return x, lad

    @staticmethod
    def inter_l_transform(z, tau):
        # tau in [-1, 0]
        tail_param = -tau - 1  # tail_param = 1 - (tau - 1)
        inner = torch.erfc(-tail_param * z / SQRT_2)

        g = torch.pow(inner, -1 / tail_param)
        log_g = -torch.log(inner) / tail_param

        erfcinv_val = _stable_erfcinv(g, log_g)

        x = SQRT_2 * erfcinv_val

        lad = -0.5 * tail_param.square() * z.square()
        lad += (-1 - 1 / tail_param) * torch.log(inner)
        lad += torch.square(erfcinv_val)

        return x, lad

    @staticmethod
    def asymp_l_transform(z):
        g = torch.exp(-(SQRT_2 / SQRT_PI) * z)
        return SQRT_2 * _erfcinv(g)

    @staticmethod
    def extreme_inverse(z, tau):
        # tau < -1
        tail_param = -tau - 1

        inner = 1 + tail_param * (SQRT_2 / SQRT_PI) * z

        g = torch.pow(inner, -1 / tail_param)
        log_g = -torch.log(inner) / tail_param

        erfcinv_val = _stable_erfcinv(g, log_g)
        x = SQRT_2 * erfcinv_val

        lad = (-1 - 1 / tail_param) * torch.log(inner)
        lad += torch.square(erfcinv_val)

        return x, lad

    @staticmethod
    def _transformation(z, tau):
        if tau.shape[0] == 1:  # fixed parameter for each observation
            tau = tau.repeat((z.shape[0], 1))

        assert (
            z.shape[1] == tau.shape[1]
        ), f"Tail parameter must be 2D [1, {z.shape[1]}] or {z.shape[1]}"

        heavy_tail = tau > 1
        light_tail = tau < -1
        iheavy = torch.logical_and(~heavy_tail, ~light_tail)
        iheavy = torch.logical_and(iheavy, tau >= 0)
        ilight = torch.logical_and(~heavy_tail, ~light_tail)
        ilight = torch.logical_and(ilight, tau < 0)

        heavy_x, heavy_lad = TailSwitchMarginalTransform.extreme_transform(
            z[heavy_tail], tau[heavy_tail]
        )
        iheavy_x, iheavy_lad = TailSwitchMarginalTransform.inter_h_transform(
            z[iheavy], tau[iheavy]
        )
        ilight_x, ilight_lad = TailSwitchMarginalTransform.inter_l_transform(
            z[ilight], tau[ilight]
        )
        light_x, light_lad = TailSwitchMarginalTransform.extreme_inverse(
            z[light_tail], tau[light_tail]
        )

        x = torch.ones_like(z)
        lad = torch.ones_like(z)

        for index, x_val, lad_val in (
            (heavy_tail, heavy_x, heavy_lad),
            (iheavy, iheavy_x, iheavy_lad),
            (ilight, ilight_x, ilight_lad),
            (light_tail, light_x, light_lad),
        ):
            x[index] = x_val
            lad[index] = lad_val
        return x, lad


class SmoothTailSwitchMarginalTransform(Transform):
    def __init__(
        self,
        features,
        pos_tail_init=None,
        neg_tail_init=None,
        shift_init=None,
        scale_init=None,
    ):
        self.features = features
        super(SmoothTailSwitchMarginalTransform, self).__init__()

        # random inits if needed
        if pos_tail_init is None:
            pos_tail_init = torch.distributions.Uniform(
                LOW_TAIL_INIT, HIGH_TAIL_INIT
            ).sample(torch.Size([features]))

        if neg_tail_init is None:
            neg_tail_init = torch.distributions.Uniform(
                LOW_TAIL_INIT, HIGH_TAIL_INIT
            ).sample(torch.Size([features]))

        if shift_init is None:
            shift_init = torch.zeros([features])

        if scale_init is None:
            scale_init = torch.ones([features])

        assert torch.Size([features]) == pos_tail_init.shape
        assert torch.Size([features]) == neg_tail_init.shape
        assert torch.Size([features]) == shift_init.shape
        assert torch.Size([features]) == scale_init.shape

        # convert to unconstrained versions
        self.pos_tail = torch.nn.parameter.Parameter(pos_tail_init)
        self.neg_tail = torch.nn.parameter.Parameter(neg_tail_init)
        self.shift = torch.nn.parameter.Parameter(shift_init)
        self._unc_scale = torch.nn.parameter.Parameter(inv_sftplus(scale_init - 1e-3))

        self._tail_forward = torch.vmap(
            SmoothTailSwitchMarginalTransform.univariate_smooth_ex_transform,
            in_dims=(1, 0, 0),
            out_dims=(1, 1),
        )

        self._tail_inverse = torch.vmap(
            SmoothTailSwitchMarginalTransform.univariate_smooth_ex_inverse,
            in_dims=(1, 0, 0),
            out_dims=(1, 1),
        )

    def forward(self, x, context=None):
        # affine
        x = (x - self.shift) / self.scale
        z, lad = self._tail_inverse(x, self.pos_tail, self.neg_tail)
        lad -= torch.log(self.scale)
        return z, lad.sum(dim=1)

    def inverse(self, z, context=None):
        x, lad = self._tail_forward(z, self.pos_tail, self.neg_tail)
        lad += torch.log(self.scale)
        x = x * self.scale + self.shift
        return x, lad.sum(dim=1)

    @property
    def scale(self):
        return 1e-3 + softplus(self._unc_scale)

    def fix_tails(self):
        # freeze only the parameters related to the tail
        self._unc_pos_tail.requires_grad = False
        self._unc_neg_tail.requires_grad = False

    @staticmethod
    def _real_transformation(z, pos_tau, neg_tau):
        tau = torch.where(z > 0, pos_tau, neg_tau)
        x, lad = SmoothTailSwitchMarginalTransform._tail_transformation(z.abs(), tau)
        return x * z.sign(), lad

    @staticmethod
    def _tail_transformation(z, tau):
        heavy_tail = tau > 1
        light_tail = tau < -1
        iheavy = torch.logical_and(~heavy_tail, ~light_tail)
        iheavy = torch.logical_and(iheavy, tau >= 0)
        ilight = torch.logical_and(~heavy_tail, ~light_tail)
        ilight = torch.logical_and(ilight, tau < 0)

        heavy_x, heavy_lad = TailSwitchMarginalTransform.extreme_transform(
            z, torch.clamp(tau, min=1.0 + 1e-6, max=None)
        )
        iheavy_x, iheavy_lad = TailSwitchMarginalTransform.inter_h_transform(
            z, torch.clamp(tau, min=0.0 - 1e-6, max=1.0 + 1e-6)
        )
        ilight_x, ilight_lad = TailSwitchMarginalTransform.inter_l_transform(
            z, torch.clamp(tau, min=-1.0 - 1e-6, max=0.0)
        )
        light_x, light_lad = TailSwitchMarginalTransform.extreme_inverse(
            z, torch.clamp(tau, min=None, max=-1.0 - 1e-6)
        )

        x = torch.ones_like(z)
        lad = torch.ones_like(z)

        for index, x_val, lad_val in (
            (heavy_tail, heavy_x, heavy_lad),
            (iheavy, iheavy_x, iheavy_lad),
            (ilight, ilight_x, ilight_lad),
            (light_tail, light_x, light_lad),
        ):
            x = torch.where(index, x_val, x)
            lad = torch.where(index, lad_val, lad)

        return x, lad

    @staticmethod
    def univariate_smooth_ex_transform(z, pos_tau, neg_tau):
        knot_z = torch.tensor([-1.6, 1.6]).reshape(-1, 1)

        # calculate spline data
        knot_x, forward_lad = SmoothTailSwitchMarginalTransform._real_transformation(
            knot_z,
            pos_tau,
            neg_tau,
        )

        # prepare data for spline
        input_knots = knot_z.squeeze().repeat((z.shape[0], 1))
        output_knots = knot_x.squeeze().repeat((z.shape[0], 1))
        derivatives = forward_lad.squeeze().repeat((z.shape[0], 1)).exp()

        spline_x, spline_lad = univariate_forward_rqs(
            z.clamp(min=knot_z[0, 0] + 1e-6, max=knot_z[1, 0] - 1e-6),
            input_knots,
            output_knots,
            derivatives,
        )

        ex_x, ex_lad = SmoothTailSwitchMarginalTransform._real_transformation(
            z, pos_tau, neg_tau
        )

        within = torch.logical_and(
            z < knot_z[1],
            z > knot_z[0],
        )
        x = torch.where(within, spline_x, ex_x)
        lad = torch.where(within, spline_lad, ex_lad)

        return x, lad

    @staticmethod
    def univariate_smooth_ex_inverse(x, pos_tau, neg_tau):
        knot_z = torch.tensor([-1.6, 1.6]).reshape(-1, 1)

        # calculate spline data
        knot_x, forward_lad = SmoothTailSwitchMarginalTransform._real_transformation(
            knot_z,
            pos_tau,
            neg_tau,
        )

        # prepare data for spline
        input_knots = knot_z.squeeze().repeat((x.shape[0], 1))
        output_knots = knot_x.squeeze().repeat((x.shape[0], 1))
        derivatives = forward_lad.squeeze().repeat((x.shape[0], 1)).exp()

        spline_z, spline_lad = univariate_inverse_rqs(
            x.clamp(min=output_knots[:, 0] + 1e-6, max=output_knots[:, 1] - 1e-6),
            input_knots,
            output_knots,
            derivatives,
        )

        # inverse is negative tail param
        ex_z, ex_lad = SmoothTailSwitchMarginalTransform._real_transformation(
            x, -pos_tau, -neg_tau
        )

        within = torch.logical_and(
            x < output_knots[:, 1],
            x > output_knots[:, 0],
        )
        z = torch.where(within, spline_z, ex_z)
        lad = torch.where(within, spline_lad, ex_lad)

        return z, lad


class InterpMarginalTransform(Transform):
    def __init__(
        self,
        features,
        pos_tail_init=None,
        neg_tail_init=None,
        shift_init=None,
        scale_init=None,
    ):
        self.features = features
        super(InterpMarginalTransform, self).__init__()

        # random inits if needed
        if pos_tail_init is None:
            pos_tail_init = torch.distributions.Uniform(
                LOW_TAIL_INIT, HIGH_TAIL_INIT
            ).sample(torch.Size([features]))

        if neg_tail_init is None:
            neg_tail_init = torch.distributions.Uniform(
                LOW_TAIL_INIT, HIGH_TAIL_INIT
            ).sample(torch.Size([features]))

        if shift_init is None:
            shift_init = torch.zeros([features])

        if scale_init is None:
            scale_init = torch.ones([features])

        assert torch.Size([features]) == pos_tail_init.shape
        assert torch.Size([features]) == neg_tail_init.shape
        assert torch.Size([features]) == shift_init.shape
        assert torch.Size([features]) == scale_init.shape

        # convert to unconstrained versions
        self.pos_tail = torch.nn.parameter.Parameter(pos_tail_init)
        self.neg_tail = torch.nn.parameter.Parameter(neg_tail_init)
        self.shift = torch.nn.parameter.Parameter(shift_init)
        self._unc_scale = torch.nn.parameter.Parameter(inv_sftplus(scale_init - 1e-3))

    def inverse(self, z, context=None):
        sign = torch.sign(z)
        tail_param = torch.where(
            z > 0,
            self.pos_tail,
            self.neg_tail,
        )
        x, lad = InterpMarginalTransform._transformation(torch.abs(z), tail_param)
        lad += torch.log(self.scale)
        x = sign * x * self.scale + self.shift
        return x, lad.sum(dim=1)

    def forward(self, x, context=None):
        # affine
        x = (x - self.shift) / self.scale

        sign = torch.sign(x)
        tail_param = torch.where(x > 0, self.pos_tail, self.neg_tail)

        z, lad = InterpMarginalTransform._transformation(
            torch.abs(x), tail_param, inverse=True
        )
        lad -= torch.log(self.scale)
        return sign * z, lad.sum(dim=1)

    @staticmethod
    def interpolated_transformation(z, tail_param):
        # interpolated
        intermediate_tail_p = (0.5 * (tail_param + 1)).abs()  # always in [0, 1]
        _y = torch.exp(-torch.tensor(SQRT_2 / SQRT_PI) * z)
        stable_y = _y > MIN_ERFC_INV

        erfcinv_y = torch.zeros_like(_y)
        erfcinv_y[stable_y] = _erfcinv(_y[stable_y])
        erfcinv_y[~stable_y] = _small_erfcinv(-z[~stable_y])

        low = SQRT_2 * erfcinv_y

        dlow_dx = torch.exp(erfcinv_y.square() - torch.tensor(SQRT_2 / SQRT_PI) * z)

        high = torch.where(
            torch.erfc(z / SQRT_2) > MIN_ERFC_INV,
            -torch.erfc(z / SQRT_2).log(),
            z.log() + 0.5 * z.square() + torch.log(torch.tensor(SQRT_PI / SQRT_2)),
        ) * torch.tensor(SQRT_PI / SQRT_2)
        dhigh_dx = torch.where(
            z < 10,
            torch.exp(-0.5 * z.square()) / torch.erfc(z / SQRT_2),
            torch.tensor(SQRT_PI / SQRT_2) * z,
        )

        intermediate_x = intermediate_tail_p * high + (1 - intermediate_tail_p) * low

        # this could be unstable
        intermediate_lad = intermediate_tail_p * dhigh_dx
        intermediate_lad += (1 - intermediate_tail_p) * dlow_dx
        intermediate_lad = intermediate_lad.log()

        return intermediate_x, intermediate_lad

    @staticmethod
    def bisection_search(func, z_0_low, z_0_high, target, tol=1e-6, max_iter=100):
        for i in range(max_iter):
            z_mid = 0.5 * (z_0_low + z_0_high)
            f_mid = func(z_mid)
            error = target - f_mid
            if (error.abs() < tol).all():
                return z_mid

            z_0_low = torch.where(error < 0, z_0_low, z_mid)
            z_0_high = torch.where(error > 0, z_0_high, z_mid)

        return 0.5 * (z_0_low + z_0_high)

    @staticmethod
    def interpolated_inverse(x, tail_param):
        interpolated_transformation = (
            InterpMarginalTransform.interpolated_transformation
        )
        # interpolated
        low, _ = interpolated_transformation(x, torch.ones_like(x) * -1)
        high, _ = interpolated_transformation(x, torch.ones_like(x))

        z_0_low = torch.minimum(low, high)
        z_0_high = torch.maximum(low, high)

        def forward(z):
            return interpolated_transformation(z, tail_param)[0]

        z = InterpMarginalTransform.bisection_search(forward, z_0_low, z_0_high, x)
        _, inv_lad = interpolated_transformation(z.detach(), tail_param)
        lad = -inv_lad
        return z, lad

    @staticmethod
    def _transformation(z, tail_param, inverse=False):
        interpolated_inverse = InterpMarginalTransform.interpolated_inverse
        interpolated_transformation = (
            InterpMarginalTransform.interpolated_transformation
        )

        if inverse:
            tail_param = -tail_param

        if tail_param.shape[0] == 1:  # fixed parameter for each observation
            tail_param = tail_param.repeat((z.shape[0], 1))

        assert (
            z.shape[1] == tail_param.shape[1]
        ), f"Tail parameter must be 2D [1, {z.shape[1]}] or {z.shape[1]}"

        heavy_tails = tail_param > 1
        light_tails = tail_param < -1

        # fatten the tails
        heavy_tail_p = (tail_param - 1).abs()  # lambda - 1 should always be positive
        g = torch.erfc(z / SQRT_2)
        heavy_x = (torch.pow(g, -heavy_tail_p) - 1) / heavy_tail_p
        heavy_x *= torch.tensor(SQRT_PI / SQRT_2)

        heavy_lad = torch.log(g) * (-heavy_tail_p - 1)
        heavy_lad -= 0.5 * torch.square(z)

        # lighten the tails
        light_tail_p = (
            torch.abs(tail_param) - 1
        ).abs()  # |lambda| - 1 should always be positive
        inner = 1 + torch.tensor(SQRT_2 / SQRT_PI) * light_tail_p * z
        g = torch.pow(inner, -1 / light_tail_p)
        stable_g = g > MIN_ERFC_INV

        erfcinv_val = torch.zeros_like(z)
        erfcinv_val[stable_g] = _erfcinv(g[stable_g])
        log_g = -torch.log(inner[~stable_g]) / tail_param[~stable_g]
        erfcinv_val[~stable_g] = _small_erfcinv(log_g)
        light_x = SQRT_2 * erfcinv_val

        light_lad = (-1 - 1 / light_tail_p) * torch.log(inner)
        light_lad += torch.square(erfcinv_val)

        if inverse:
            intermediate_x, intermediate_lad = interpolated_inverse(z, -tail_param)
        else:
            intermediate_x, intermediate_lad = interpolated_transformation(
                z, tail_param
            )

        # implicitly, where not heavy or light tailed it is intermediate
        x = torch.where(heavy_tails, heavy_x, intermediate_x)
        x = torch.where(light_tails, light_x, x)

        lad = torch.where(heavy_tails, heavy_lad, intermediate_lad)
        lad = torch.where(light_tails, light_lad, lad)

        return x, lad

    @property
    def scale(self):
        return 1e-3 + softplus(self._unc_scale)

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


class GTransform(Transform):
    """Alternative Gumbel transform producing exponential tails."""

    def __init__(self, features, p=1.0, kappa=1.0, epsilon=1.0, delta=1.0):
        self.features = features
        super().__init__()
        self.p = torch.tensor(p)
        self._unc_kappa = torch.nn.Parameter(inv_sftplus(torch.tensor(kappa)))
        self._unc_epsilon = torch.nn.Parameter(inv_sftplus(torch.tensor(epsilon)))
        self._unc_delta = torch.nn.Parameter(inv_sftplus(torch.tensor(delta)))

    @property
    def kappa(self):
        return softplus(self._unc_kappa)

    @property
    def epsilon(self):
        return softplus(self._unc_epsilon)

    @property
    def delta(self):
        return softplus(self._unc_delta)

    def forward(self, z, context=None):
        x, lad = _g_transform_and_lad(z, self.p, self.kappa, self.epsilon, self.delta)
        return x, lad.sum(axis=1)

    def inverse(self, x, context=None):
        z, lad = _g_inverse_and_lad(x, self.p, self.kappa, self.epsilon, self.delta)
        return z, lad.sum(axis=1)


class ExpTailTransform(Transform):
    """Transform exponential tail to Frechet tail."""

    def __init__(self, features, lam=1.0):
        self.features = features
        super().__init__()
        self._unc_lam = torch.nn.Parameter(inv_sftplus(torch.tensor(lam)))

    @property
    def lam(self):
        return softplus(self._unc_lam)

    def forward(self, x, context=None):
        y, lad = _exp_tail_transform_and_lad(x, self.lam)
        return y, lad.sum(axis=1)

    def inverse(self, y, context=None):
        x, lad = _exp_tail_inverse_and_lad(y, self.lam)
        return x, lad.sum(axis=1)


class FullTailMarginalTransform(Transform):
    """Composite transform applying G then exponential Frechet T."""

    def __init__(self, features, *, p=1.0, kappa=1.0, epsilon=1.0, delta=1.0, lam=1.0):
        super().__init__()
        self.features = features
        self.g = GTransform(features, p=p, kappa=kappa, epsilon=epsilon, delta=delta)
        self.t = ExpTailTransform(features, lam=lam)

    def forward(self, z, context=None):
        x, lad1 = self.g.forward(z, context)
        y, lad2 = self.t.forward(x, context)
        return y, lad1 + lad2

    def inverse(self, y, context=None):
        x, lad2 = self.t.inverse(y, context)
        z, lad1 = self.g.inverse(x, context)
        return z, lad1 + lad2


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
