from enum import Enum
from typing import TypedDict, Optional, Type, Callable, Literal, Union
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
    InterpMarginalTransform,
    MaskedExtremeAutoregressiveTransform,
    MaskedAutoregressiveTailAffineMarginalTransform,
    TailScaleShiftMarginalTransform,
    SmoothTailSwitchMarginalTransform,
    flip,
    AffineMarginalTransform,
    TailSwitchMarginalTransform,
    MaskedTailSwitchAffineTransform,
)

from tailnflows.models.base_distribution import (
    TrainableStudentT,
    NormalStudentTJoint,
    GeneralisedNormal,
    NormalMixture,
)
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
FinalRotation = Optional[Literal["householder", "lu"]]


class ModelKwargs(TypedDict, total=False):
    tail_bound: Optional[float]
    num_bins: Optional[int]
    tail_init: Optional[Union[list[float], float]]
    rotation: Optional[bool]
    fix_tails: Optional[bool]
    data: Optional[torch.Tensor]


#######################
# Utility transforms
class ConstraintTransform(Transform):
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
        return self.EPS + x, lad.sum(dim=1)

    def forward(self, x, context=None):
        # if x = 0, little can be done
        if torch.min(x) <= 0:
            raise Exception("Inputs <0 passed to Softplus transform")

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
    random_permute: bool = False,
    affine_autoreg_layer: bool = False,
    linear_layer: bool = False,
    nn_kwargs: NNKwargs = {},
) -> list[Transform]:
    """
    An autoregressive RQS transform of configurable depth.
    """
    transforms: list[Transform] = []
    specified_nn_kwargs = configure_nn(nn_kwargs)

    if affine_autoreg_layer:
        transforms.append(
            MaskedAffineAutoregressiveTransform(
                features=dim,
                context_features=condition_dim,
                **specified_nn_kwargs,
            )
        )

    for layer in range(depth):
        if layer > 0 and random_permute:
            transforms.append(RandomPermutation(dim))

        if layer > 0 and linear_layer:
            transforms.append(LULinear(dim, identity_init=True))

        transforms.append(
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=dim,
                context_features=condition_dim,
                num_bins=num_bins,
                tail_bound=tail_bound,
                tails="linear",
                **specified_nn_kwargs,
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
        transforms.append(
            MaskedAffineAutoregressiveTransform(
                features=dim,
                context_features=condition_dim,
                **nn_kwargs,
            )
        )

    specified_nn_kwargs = configure_nn(nn_kwargs)
    for _ in range(depth):
        if batch_norm:
            transforms.append(BatchNorm(features=dim))

        if random_permute:
            transforms.append(RandomPermutation(dim))

        transforms.append(
            MaskedUMNNAutoregressiveTransform(
                features=dim,
                integrand_net_layers=transformation_layers,
                **specified_nn_kwargs,
            )
        )

    return transforms


def base_affine_transform(
    dim,
    *,
    condition_dim: Optional[int] = None,
    depth: int = 1,
    random_permute: bool = False,
    nn_kwargs: NNKwargs = {},
):
    transforms: list[Transform] = []
    specified_nn_kwargs = configure_nn(nn_kwargs)
    for _ in range(depth):
        if random_permute:
            transforms.append(RandomPermutation(dim))

        transforms.append(
            MaskedAffineAutoregressiveTransform(
                features=dim,
                context_features=condition_dim,
                **specified_nn_kwargs,
            )
        )
    return transforms


#######################
# flow models
class ExperimentFlow(Flow):
    def __init__(
        self,
        use: ModelUse,
        base_distribution: Distribution,
        base_transformation_init: Optional[BaseTransform],
        final_transformation: Transform,
        final_rotation: FinalRotation,
        constraint_transformation: Optional[Transform] = None,
    ):
        dim = base_distribution._shape[0]
        if base_transformation_init is None:
            base_transformations = []
        else:
            base_transformations = base_transformation_init(dim)

        # change direction of autoregression, note that this will also change
        # forward/inverse so care is needed if direction matters
        if use == "variational_inference":
            base_transformations = [
                InverseTransform(transformation)
                for transformation in base_transformations
            ]

        if final_rotation is None:
            final_rotation_transformation = IdentityTransform()
        elif final_rotation == "householder":
            final_rotation_transformation = HouseholderSequence(dim, dim)
        elif final_rotation == "lu":
            final_rotation_transformation = LULinear(features=dim)

        if constraint_transformation is None:
            constraint_transformation = IdentityTransform()

        # transformation order is data (X) -> noise (Z)
        # samples are generated by X = transform.inverse(Z)
        transformations = [
            constraint_transformation,
            final_rotation_transformation,
            final_transformation,
            CompositeTransform(base_transformations),
        ]

        super().__init__(
            transform=CompositeTransform(transformations),
            distribution=base_distribution,
        )

    def get_constraint_transformation(self):
        return self._transform._transforms[0]

    def get_final_rotation(self):
        return self._transform._transforms[1]

    def set_final_rotation(self, transformation: Transform):
        self._transform._transforms[1] = transformation

    def get_final_transformation(self):
        return self._transform._transforms[2]

    def get_base_transformation(self):
        return self._transform._transforms[3]

    def add_base_transformation(self, transformation: Transform):
        # adds a transformation to the back of the flow
        current_base = [t for t in self._transform._transforms[3]._transforms]
        self._transform._transformations[3] = CompositeTransform(
            current_base + [transformation]
        )


def build_base_model(
    dim: int,
    use: ModelUse = "density_estimation",
    *,
    base_transformation_init: Optional[BaseTransform] = None,
    constraint_transformation: Optional[Transform] = None,
    final_rotation: FinalRotation = None,
    model_kwargs: ModelKwargs = {},
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
        final_rotation=final_rotation,
        constraint_transformation=constraint_transformation,
    )


def build_ttf_m(
    dim: int,
    use: ModelUse = "density_estimation",
    base_transformation_init: Optional[BaseTransform] = None,
    constraint_transformation: Optional[Transform] = None,
    final_rotation: FinalRotation = None,
    model_kwargs: ModelKwargs = {},
):
    # configure model specific settings
    pos_tail_init = model_kwargs.get("pos_tail_init", None)
    neg_tail_init = model_kwargs.get("neg_tail_init", None)
    fix_tails = model_kwargs.get("fix_tails", False)

    # base distribution
    base_distribution = StandardNormal([dim])

    # set up tail transform
    tail_transform = TailAffineMarginalTransform(
        features=dim,
        pos_tail_init=torch.tensor(pos_tail_init),
        neg_tail_init=torch.tensor(neg_tail_init),
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
        final_rotation=final_rotation,
        constraint_transformation=constraint_transformation,
    )


def build_ttf_interp(
    dim: int,
    use: ModelUse = "density_estimation",
    base_transformation_init: Optional[BaseTransform] = None,
    constraint_transformation: Optional[Transform] = None,
    final_rotation: FinalRotation = None,
    model_kwargs: ModelKwargs = {},
):
    # configure model specific settings
    pos_tail_init = model_kwargs.get("pos_tail_init", None)
    neg_tail_init = model_kwargs.get("neg_tail_init", None)
    fix_tails = model_kwargs.get("fix_tails", False)

    # base distribution
    base_distribution = StandardNormal([dim])

    # set up tail transform
    tail_transform = InterpMarginalTransform(
        features=dim,
        pos_tail_init=torch.tensor(pos_tail_init),
        neg_tail_init=torch.tensor(neg_tail_init),
    )

    if fix_tails:
        assert (
            pos_tail_init is not None
        ), "Fixing tails, but no init provided for pos tails"
        assert (
            neg_tail_init is not None
        ), "Fixing tails, but no init provided for neg tails"
        tail_transform.fix_tails()

    # by default the numerical inversion is in generation direction
    # if using for VI we want sampling to be quick
    if use == "variational_inference":
        tail_transform = flip(tail_transform)

    return ExperimentFlow(
        use=use,
        base_distribution=base_distribution,
        base_transformation_init=base_transformation_init,
        final_transformation=tail_transform,
        final_rotation=final_rotation,
        constraint_transformation=constraint_transformation,
    )


def build_ttf_switch_m(
    dim: int,
    use: ModelUse = "density_estimation",
    base_transformation_init: Optional[BaseTransform] = None,
    constraint_transformation: Optional[Transform] = None,
    final_rotation: FinalRotation = None,
    model_kwargs: ModelKwargs = {},
):
    # configure model specific settings
    pos_tail_init = model_kwargs.get("pos_tail_init", None)
    neg_tail_init = model_kwargs.get("neg_tail_init", None)
    fix_tails = model_kwargs.get("fix_tails", False)

    # base distribution
    base_distribution = StandardNormal([dim])

    # set up tail transform
    tail_transform = TailSwitchMarginalTransform(
        features=dim,
        pos_tail_init=torch.tensor(pos_tail_init),
        neg_tail_init=torch.tensor(neg_tail_init),
    )

    if fix_tails:
        assert (
            pos_tail_init is not None
        ), "Fixing tails, but no init provided for pos tails"
        assert (
            neg_tail_init is not None
        ), "Fixing tails, but no init provided for neg tails"
        tail_transform.fix_tails()

    return ExperimentFlow(
        use=use,
        base_distribution=base_distribution,
        base_transformation_init=base_transformation_init,
        final_transformation=tail_transform,
        final_rotation=final_rotation,
        constraint_transformation=constraint_transformation,
    )


def build_ttf_smooth_m(
    dim: int,
    use: ModelUse = "density_estimation",
    base_transformation_init: Optional[BaseTransform] = None,
    constraint_transformation: Optional[Transform] = None,
    final_rotation: FinalRotation = None,
    model_kwargs: ModelKwargs = {},
):
    # configure model specific settings
    pos_tail_init = model_kwargs.get("pos_tail_init", None)
    neg_tail_init = model_kwargs.get("neg_tail_init", None)
    fix_tails = model_kwargs.get("fix_tails", False)

    # base distribution
    base_distribution = StandardNormal([dim])

    # set up tail transform
    tail_transform = SmoothTailSwitchMarginalTransform(
        features=dim,
        pos_tail_init=torch.tensor(pos_tail_init),
        neg_tail_init=torch.tensor(neg_tail_init),
    )

    if fix_tails:
        assert (
            pos_tail_init is not None
        ), "Fixing tails, but no init provided for pos tails"
        assert (
            neg_tail_init is not None
        ), "Fixing tails, but no init provided for neg tails"
        tail_transform.fix_tails()

    return ExperimentFlow(
        use=use,
        base_distribution=base_distribution,
        base_transformation_init=base_transformation_init,
        final_transformation=tail_transform,
        final_rotation=final_rotation,
        constraint_transformation=constraint_transformation,
    )


def build_ttf_autoreg(
    dim: int,
    use: ModelUse = "density_estimation",
    base_transformation_init: Optional[BaseTransform] = None,
    constraint_transformation: Optional[Transform] = None,
    final_rotation: FinalRotation = None,
    model_kwargs: ModelKwargs = {},
    nn_kwargs: NNKwargs = {},
):
    # base distribution
    base_distribution = StandardNormal([dim])

    # set up tail transform
    tail_transform = MaskedExtremeAutoregressiveTransform(
        features=dim, nn_kwargs=configure_nn(nn_kwargs)
    )

    # the tail transformation needs to be flipped this means data->noise is
    # a strictly lightening transformation
    tail_transform = flip(tail_transform)

    return ExperimentFlow(
        use=use,
        base_distribution=base_distribution,
        base_transformation_init=base_transformation_init,
        final_transformation=tail_transform,
        final_rotation=final_rotation,
        constraint_transformation=constraint_transformation,
    )


def build_gtaf(
    dim: int,
    use: ModelUse = "density_estimation",
    base_transformation_init: Optional[BaseTransform] = None,
    constraint_transformation: Optional[Transform] = None,
    final_rotation: FinalRotation = None,
    model_kwargs: ModelKwargs = {},
):
    # model specific settings
    tail_init = model_kwargs.get("tail_init", None)  # in df terms
    if isinstance(tail_init, list):
        tail_init = torch.tensor(tail_init)

    # base distribution
    base_distribution = TrainableStudentT(dim, init=tail_init)

    # final transformation
    final_transformation = AffineMarginalTransform(dim)

    return ExperimentFlow(
        use=use,
        base_distribution=base_distribution,
        base_transformation_init=base_transformation_init,
        final_transformation=final_transformation,
        final_rotation=final_rotation,
        constraint_transformation=constraint_transformation,
    )


def build_mtaf(
    dim: int,
    use: ModelUse = "density_estimation",
    base_transformation_init: Optional[BaseTransform] = None,
    constraint_transformation: Optional[Transform] = None,
    final_rotation: FinalRotation = None,
    model_kwargs: ModelKwargs = {},
    nn_kwargs: NNKwargs = {},
):
    assert (
        "tail_init" in model_kwargs
        and model_kwargs["tail_init"] is not None
        and isinstance(model_kwargs["tail_init"], list)
    ), "mTAF requires the marginal tails at initialisation time!"
    assert (
        len(model_kwargs["tail_init"]) == dim
    ), "mTAF tail init must be 1 degree of freedom parameter per marginal!"
    assert final_rotation in [None, "lu"], "mTAF only supports LU rotation!"
    assert (
        "fix_tails" in model_kwargs and model_kwargs["fix_tails"]
    ), "mTAF must fix tails at init time!"

    # model specific settings
    tail_init = torch.tensor(model_kwargs["tail_init"])  # in df terms
    fix_tails = model_kwargs["fix_tails"]

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

    # final transformation shuffles into light/heavy
    final_transformation = CompositeTransform(
        [
            initial_permutation,
            AffineMarginalTransform(dim),
        ]
    )

    mtaf = ExperimentFlow(
        use=use,
        base_distribution=base_distribution,
        base_transformation_init=base_transformation_init,
        final_transformation=final_transformation,
        final_rotation=final_rotation,
        constraint_transformation=constraint_transformation,
    )

    # adjust the final rotation transformation
    if final_rotation is not None and (tail_init > 0).sum() < dim:
        mtaf.set_final_rotation(TailLU(dim, int(num_heavy)))

    # check for any rotations in the base transformation, these invalidate the
    # mtaf assumptions, we need to preserve groups of heavy/light
    base_transformations = mtaf.get_base_transformation()._transforms
    for transformation in base_transformations:
        if (
            isinstance(transformation, LULinear)
            or isinstance(transformation, HouseholderSequence)
            or isinstance(transformation, RandomPermutation)
        ):
            raise Exception("Non heavy/light preserving transformation in mtaf flow!")

    return mtaf


def build_comet(
    dim: int,
    use: ModelUse = "density_estimation",
    base_transformation_init: Optional[BaseTransform] = None,
    constraint_transformation: Optional[Transform] = None,
    final_rotation: FinalRotation = None,
    model_kwargs: ModelKwargs = {},
):
    # comet flow expects some data to estimate properties at init time
    # create some fake data if this isn't passed
    data = model_kwargs.get("data", torch.ones([1, dim]))
    fix_tails = model_kwargs.get("fix_tails", False)
    assert "tail_init" in model_kwargs and isinstance(model_kwargs["tail_init"], list)
    tail_init = model_kwargs["tail_init"]

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
                tail_transform.tails[ix].lower_xi = torch.tensor(1 / 1000.0)
                tail_transform.tails[ix].upper_xi = torch.tensor(1 / 1000.0)
            else:
                tail_transform.tails[ix].lower_xi = torch.tensor(1 / tail_df)
                tail_transform.tails[ix].upper_xi = torch.tensor(1 / tail_df)

    final_transform = CompositeTransform([tail_transform, Logit()])

    return ExperimentFlow(
        use=use,
        base_distribution=base_distribution,
        base_transformation_init=base_transformation_init,
        final_transformation=final_transform,
        final_rotation=final_rotation,
        constraint_transformation=constraint_transformation,
    )


def build_mix_normal(
    dim: int,
    use: ModelUse = "density_estimation",
    base_transformation_init: Optional[BaseTransform] = None,
    constraint_transformation: Optional[Transform] = None,
    final_rotation: FinalRotation = None,
    model_kwargs: ModelKwargs = {},
):
    # model specific settings
    n_component = model_kwargs.get("n_component", 10)

    # base distribution
    base_distribution = NormalMixture(dim, n_component)

    # final transformation
    final_transformation = AffineMarginalTransform(dim)

    return ExperimentFlow(
        use=use,
        base_distribution=base_distribution,
        base_transformation_init=base_transformation_init,
        final_transformation=final_transformation,
        final_rotation=final_rotation,
        constraint_transformation=constraint_transformation,
    )


def build_gen_normal(
    dim: int,
    use: ModelUse = "density_estimation",
    base_transformation_init: Optional[BaseTransform] = None,
    constraint_transformation: Optional[Transform] = None,
    final_rotation: FinalRotation = None,
    model_kwargs: ModelKwargs = {},
):
    # base distribution
    base_distribution = GeneralisedNormal(dim)

    # final transformation
    final_transformation = AffineMarginalTransform(dim)

    return ExperimentFlow(
        use=use,
        base_distribution=base_distribution,
        base_transformation_init=base_transformation_init,
        final_transformation=final_transformation,
        final_rotation=final_rotation,
        constraint_transformation=constraint_transformation,
    )
