from enum import Enum
from typing import TypedDict, Optional, Type, Callable, Literal
import torch

from torch.nn.functional import logsigmoid

# nflows dependencies
from nflows.flows import Flow
from nflows.distributions import Distribution
from nflows.distributions.normal import StandardNormal
from nflows.transforms.autoregressive import (
    MaskedAffineAutoregressiveTransform,
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
    MaskedUMNNAutoregressiveTransform,
)
from nflows.transforms.lu import LULinear
from nflows.transforms.base import CompositeTransform, InverseTransform
from nflows.transforms.nonlinearities import Logit
from nflows.transforms import Permutation
from nflows.transforms.normalization import BatchNorm
from nflows.transforms.base import Transform
from nflows.transforms.standard import IdentityTransform
from nflows.transforms.orthogonal import HouseholderSequence
from nflows.transforms.permutations import RandomPermutation

# custom modules
from tailnflows.models.extreme_transformations import (
    configure_nn,
    NNKwargs,
    TailAffineMarginalTransform,
    flip,
    TailMarginalTransform,
    TailScaleShiftMarginalTransform,
    MaskedExtremeAutoregressiveTransform,
    CopulaMarginalTransform,
    RQSMarginalTransform,
    AffineMarginalTransform,
)
from tailnflows.models.base_distribution import TrainableStudentT, NormalStudentTJoint
from marginal_tail_adaptive_flows.utils.tail_permutation import (
    TailLU,
)
from tailnflows.models.comet_models import MarginalLayer


def _get_intial_permutation(degrees_of_freedom):
    # returns a permutation such that the light dimensions precede heavy ones
    # according to degrees_of_freedom argument
    num_light = int(sum(df == 0 for df in degrees_of_freedom))
    light_ix = 0
    heavy_ix = num_light
    perm_ix = torch.zeros(len(degrees_of_freedom), dtype=torch.int)
    for ix, df in enumerate(degrees_of_freedom):
        if df == 0:
            perm_ix[ix] = light_ix
            light_ix += 1
        else:
            perm_ix[ix] = heavy_ix
            heavy_ix += 1

    permutation = InverseTransform(Permutation(perm_ix))  # ie forward
    rearranged_dfs, _ = permutation(degrees_of_freedom.clone().reshape(1, -1))
    return permutation, rearranged_dfs.squeeze()


ModelUse = Literal["density_estimation", "variational_inference"]


class ModelKwargs(TypedDict, total=False):
    tail_bound: Optional[float]
    num_bins: Optional[int]
    tail_init: Optional[float]
    rotation: Optional[bool]
    fix_tails: Optional[bool]


# Utility transforms
class ConstraintTransform(Transform):
    """Transform that"""

    def __init__(self, dim, transforms, index_sets: list[set[int]]):

        assert (
            len(index_sets) == 1 or len(set.intersection(*index_sets)) == 0
        ), "Overlap in index sets!"
        print(len(index_sets), len(transforms))
        assert len(index_sets) == len(
            transforms
        ), "One index set required for each transform"
        super().__init__()
        self.transforms = transforms
        self.index_sets = [
            torch.tensor(list(index_set), dtype=torch.int) for index_set in index_sets
        ]
        identity_index = torch.tensor(
            list(set(range(dim)).difference(set.union(*index_sets))), dtype=torch.int
        )
        self.transforms.append(IdentityTransform())
        self.index_sets.append(identity_index)

    def forward(self, inputs, context=None):
        batch_size = inputs.size(0)
        outputs = torch.zeros_like(inputs)
        logabsdet = inputs.new_zeros(batch_size)

        for index_set, transform in zip(self.index_sets, self.transforms):
            trans_out, trans_lad = transform(inputs[:, index_set])
            outputs[:, index_set] = trans_out
            logabsdet += trans_lad  # accumulate across dims

        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        batch_size = inputs.size(0)
        outputs = torch.zeros_like(inputs)
        logabsdet = inputs.new_zeros(batch_size)

        for index_set, transform in zip(self.index_sets, self.transforms):
            trans_out, trans_lad = transform.inverse(inputs[:, index_set])
            outputs[:, index_set] = trans_out
            logabsdet += trans_lad  # accumulate across dims

        return outputs, logabsdet


class Softplus(Transform):
    """
    Softplus non-linearity
    """

    THRESHOLD = 20.0
    EPS = 1e-8  # a small value, to ensure that the inverse doesn't evaluate to 0.

    def __init__(self, offset=1e-3):
        super().__init__()
        self.offset = offset

    def inverse(self, z, context=None):
        # maps real z to postive real x, with log grad
        x = torch.zeros_like(z)
        above = z > self.THRESHOLD
        x[above] = z[above]
        x[~above] = torch.log1p(z[~above].exp())
        lad = logsigmoid(z)
        return self.EPS + x, lad.sum(axis=1)

    def forward(self, x, context=None):
        # if x = 0, little can be done
        if torch.min(x) <= 0:
            raise InputOutsideDomain()

        z = x + torch.log(-torch.expm1(-x))
        lad = x - torch.log(torch.expm1(x))

        return z, lad.sum(axis=1)


#######################
# base transforms

BaseTransform = Callable[[int], list[Transform]]


def base_nsf_transform(
    dim: int,
    *,
    condition_dim: Optional[int] = None,
    depth: int = 1,
    num_bins: int = 5,
    tail_bound: float = 3.0,
    linear_layer: bool = False,
    random_permute: bool = False,
    affine_autoreg_layer: bool = False,
    householder_rotatation_layer: bool = False,
    batch_norm: bool = False,
    nn_kwargs: NNKwargs = {},
) -> list[Transform]:
    """
    An autoregressive RQS transform of configurable depth.
    """
    transforms: list[Transform] = []
    if linear_layer:
        transforms.append(LULinear(features=dim))

    if affine_autoreg_layer:
        if householder_rotatation_layer:
            transforms.append(HouseholderSequence(dim, dim))

        transforms.append(
            MaskedAffineAutoregressiveTransform(
                features=dim,
                context_features=condition_dim,
                **nn_kwargs,
            )
        )

    nn_kwargs = configure_nn(nn_kwargs)
    for _ in range(depth):
        if batch_norm:
            transforms.append(BatchNorm(features=dim))

        if random_permute:
            transforms.append(RandomPermutation(dim))

        if householder_rotatation_layer:
            transforms.append(HouseholderSequence(dim, dim))

        transforms.append(
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=dim,
                context_features=condition_dim,
                num_bins=num_bins,
                tail_bound=tail_bound,
                tails="linear",
                **nn_kwargs,
            )
        )

    return transforms


def base_umn_transform(
    dim: int,
    *,
    condition_dim: Optional[int] = None,
    depth: int = 1,
    transformation_layers: list[int] = [20, 20, 20],
    linear_layer: bool = False,
    random_permute: bool = False,
    affine_autoreg_layer: bool = False,
    householder_rotatation_layer: bool = False,
    batch_norm: bool = False,
    nn_kwargs: NNKwargs = {},
) -> list[Transform]:
    """
    An autoregressive RQS transform of configurable depth.
    """
    transforms: list[Transform] = []
    if linear_layer:
        transforms.append(LULinear(features=dim))

    if affine_autoreg_layer:
        if householder_rotatation_layer:
            transforms.append(HouseholderSequence(dim, dim))

        transforms.append(
            MaskedAffineAutoregressiveTransform(
                features=dim,
                context_features=condition_dim,
                **nn_kwargs,
            )
        )

    nn_kwargs = configure_nn(nn_kwargs)
    for _ in range(depth):
        if batch_norm:
            transforms.append(BatchNorm(features=dim))

        if random_permute:
            transforms.append(RandomPermutation(dim))

        if householder_rotatation_layer:
            transforms.append(HouseholderSequence(dim, dim))

        transforms.append(
            MaskedUMNNAutoregressiveTransform(
                features=dim,
                integrand_net_layers=transformation_layers,
                **nn_kwargs,
            )
        )

    return transforms


def base_affine_transform(
    flow_depth,
):
    for _ in range(flow_depth):
        transforms.append(
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=dim,
                context_features=condition_dim,
                num_bins=num_bins,
                tails="linear",
                tail_bound=tail_bound,
                **nn_kwargs,
            )
        )


def base_linear_transform(
    flow_depth,
):
    for _ in range(flow_depth):
        # transforms.append(BatchNorm(features=dim))
        # transforms.append(LULinear(features=dim))
        transforms.append(
            MaskedAffineAutoregressiveTransform(
                features=dim,
                context_features=condition_dim,
                # num_bins=num_bins,
                # tails="linear",
                # tail_bound=tail_bound,
                **nn_kwargs,
            )
        )


#######################
# flow models
class ExperimentFlow(Flow):
    def __init__(
        self,
        use: ModelUse,
        base_distribution: Distribution,
        base_transformation_init: Optional[BaseTransform],
        final_transformation: Transform,
        constraint_transformation: Optional[Transform] = None,
    ):
        if base_transformation_init is None:
            base_transformations = []
        else:
            base_transformations = base_transformation_init(base_distribution._shape[0])

        # change direction of autoregression, note that this will also change
        # forward/inverse so care is needed if direction matters
        if use == "variational_inference":
            base_transformations = [
                InverseTransform(transformation)
                for transformation in base_transformations
            ]

        # transformation order is data->noise
        transformations = [final_transformation] + base_transformations

        if constraint_transformation is not None:
            transformations = [constraint_transformation] + transformations

        super().__init__(
            transform=CompositeTransform(transformations),
            distribution=base_distribution,
        )


def build_base_model(
    dim: int,
    use: ModelUse = "density_estimation",
    *,
    base_transformation_init: Optional[BaseTransform] = None,
    constraint_transformation: Optional[Transform] = None,
    model_kwargs: ModelKwargs = {},
    nn_kwargs: NNKwargs = {},
):
    # base distribution
    base_distribution = StandardNormal([dim])

    # final transformation
    final_transformation = AffineMarginalTransform(dim)

    return ExperimentFlow(
        use=use,
        base_distribution=base_distribution,
        base_transformation_init=base_transformation_init,
        final_transformation=final_transformation,
        constraint_transformation=constraint_transformation,
    )


def build_ttf_m(
    dim: int,
    use: ModelUse = "density_estimation",
    base_transformation_init: Optional[BaseTransform] = None,
    constraint_transformation: Optional[Transform] = None,
    model_kwargs: ModelKwargs = {},
    nn_kwargs: NNKwargs = {},
):
    # configure model specific settings
    pos_tail_init = model_kwargs.get("pos_tail_init", None)
    neg_tail_init = model_kwargs.get("neg_tail_init", None)
    fix_tails = model_kwargs.get("fix_tails", False)
    nn_kwargs = configure_nn(nn_kwargs)

    # base distribution
    base_distribution = StandardNormal([dim])

    # set up tail transform
    tail_transform = TailAffineMarginalTransform(
        features=dim, pos_tail_init=pos_tail_init, neg_tail_init=neg_tail_init
    )

    if fix_tails:
        assert (
            pos_tail_init is not None
        ), "Fixing tails, but no init provided for pos tails"
        assert (
            neg_tail_init is not None
        ), "Fixing tails, but no init provided for neg tails"
        tail_transform.fix_tails()

    # the tail transformation needs to be flipped this means data->noise is
    # a strictly lightening transformation
    tail_transform = flip(tail_transform)

    return ExperimentFlow(
        use=use,
        base_distribution=base_distribution,
        base_transformation_init=base_transformation_init,
        final_transformation=tail_transform,
        constraint_transformation=constraint_transformation,
    )


def build_gtaf(
    dim: int,
    use: ModelUse = "density_estimation",
    base_transformation_init: Optional[BaseTransform] = None,
    constraint_transformation: Optional[Transform] = None,
    model_kwargs: ModelKwargs = {},
    nn_kwargs: NNKwargs = {},
):
    # model specific settings
    tail_init = model_kwargs.get("tail_init", None)  # in df terms

    # base distribution
    base_distribution = TrainableStudentT(dim, init=tail_init)

    # final transformation
    final_transformation = AffineMarginalTransform(dim)

    return ExperimentFlow(
        use=use,
        base_distribution=base_distribution,
        base_transformation_init=base_transformation_init,
        final_transformation=final_transformation,
        constraint_transformation=constraint_transformation,
    )


def build_mtaf(
    dim: int,
    use: ModelUse = "density_estimation",
    base_transformation_init: Optional[BaseTransform] = None,
    constraint_transformation: Optional[Transform] = None,
    model_kwargs: ModelKwargs = {},
    nn_kwargs: NNKwargs = {},
):
    # model specific settings
    tail_init = model_kwargs.get("tail_init", None)  # in df terms
    fix_tails = model_kwargs.get("fix_tails", True)

    assert (
        "tail_init" in model_kwargs
    ), "mTAF requires the marginal tails at initialisation time!"
    assert model_kwargs["tail_init"].shape == torch.Size(
        [dim]
    ), "mTAF tail init must be 1 degree of freedom parameter per marginal!"

    # organise into heavy/light components
    num_light = int(sum(df == 0 for df in tail_init))
    num_heavy = int(dim - num_light)
    initial_permutation, permuted_degrees_of_freedom = _get_intial_permutation(
        tail_init
    )

    # base distribution
    base_distribution = NormalStudentTJoint(permuted_degrees_of_freedom)
    if fix_tails:
        for parameter in base_distribution.parameters():
            parameter.requires_grad = False

    # base transformation
    if base_transformations is not None and (tail_init > 0).sum() < dim:
        # if there are light components, rotation should be a
        # special LU rotation preserving groups of heavy/light
        base_transformations = [
            (
                transformation
                if not isinstance(transformation, LULinear)
                else TailLU(dim, int(num_heavy))
            )
            for transformation in base_transformations
        ]

    # final transformation shuffles into light/heavy
    final_transformation = CompositeTransform(
        [
            initial_permutation,
            AffineMarginalTransform(dim),
        ]
    )

    return ExperimentFlow(
        use=use,
        base_distribution=base_distribution,
        base_transformation_init=base_transformation_init,
        final_transformation=final_transformation,
        constraint_transformation=constraint_transformation,
    )


def build_comet(
    dim: int,
    use: ModelUse = "density_estimation",
    base_transformation_init: Optional[BaseTransform] = None,
    constraint_transformation: Optional[Transform] = None,
    model_kwargs: ModelKwargs = {},
    nn_kwargs: NNKwargs = {},
):
    # comet flow expects some data to estimate properties at init time
    # create some fake data if this isn't passed
    data = model_kwargs.get(
        "data", torch.distributions.Normal(0.0, 1.0).sample([1000, dim])
    )
    fix_tails = model_kwargs.get("fix_tails", False)
    tail_init = model_kwargs.get("tail_init", False)
    assert (
        use == "density_estimation"
    ), "COMET flows only defined for density estimation!"

    # base distribution
    base_distribution = StandardNormal([dim])

    # final transformation
    tail_transform = MarginalLayer(data)
    if fix_tails and tail_init is not None:
        assert len(tail_init) == dim
        for ix, tail_df in enumerate(tail_init):
            if tail_df == 0.0:
                tail_transform.tails[ix].lower_xi = torch.tensor(0.001)
                tail_transform.tails[ix].upper_xi = torch.tensor(0.001)
            else:
                tail_transform.tails[ix].lower_xi = 1 / tail_df
                tail_transform.tails[ix].upper_xi = 1 / tail_df

    final_transform = CompositeTransform([tail_transform, Logit()])

    return ExperimentFlow(
        use=use,
        base_distribution=base_distribution,
        base_transformation_init=base_transformation_init,
        final_transformation=final_transform,
        constraint_transformation=constraint_transformation,
    )
