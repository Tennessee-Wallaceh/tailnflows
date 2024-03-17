from enum import Enum
from typing import TypedDict, Optional, Type
import torch

# nflows dependencies
from nflows.flows import Flow
from nflows.distributions.uniform import BoxUniform
from nflows.distributions.normal import StandardNormal
from nflows.transforms.autoregressive import (
    MaskedAffineAutoregressiveTransform,
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
)
from nflows.transforms.lu import LULinear
from nflows.transforms.base import CompositeTransform, InverseTransform
from nflows.transforms.nonlinearities import Logit
from nflows.transforms import Permutation

# custom modules
from tailnflows.models.extreme_transformations import (
    configure_nn,
    NNKwargs,
    MaskedTailAutoregressiveTransform,
    TailAffineMarginalTransform,
    flip,
    TailMarginalTransform,
    CopulaMarginalTransform,
    HybridTailMarginalTransform,
)
from tailnflows.models.base_distribution import TrainableStudentT, NormalStudentTJoint
from marginal_tail_adaptive_flows.utils.tail_permutation import (
    TailLU,
)
from tailnflows.models.utils import Softplus
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


ModelUse = Enum("ModelUse", ["density_estimation", "variational_inference"])


class ModelKwargs(TypedDict, total=False):
    tail_bound: Optional[float]
    num_bins: Optional[int]
    tail_init: Optional[float]
    rotation: Optional[bool]
    fix_tails: Optional[bool]


Domain = Enum("Domain", ["positive"])

ModelName = Enum(
    "ModelName",
    [
        "TTF",
        "TTF_m",
        "TTF_m_hybrid",
        "RQS",
        "gTAF",
        "mTAF",
        "COMET",
        "Copula_m",
        "TailOnly",
    ],
)


# Define flow models
class TTF(Flow):
    def __init__(
        self,
        dim: int,
        use: ModelUse = ModelUse.density_estimation,
        model_kwargs: ModelKwargs = {},
        nn_kwargs: NNKwargs = {},
        domain: Optional[Domain] = None,
    ):
        tail_bound = model_kwargs.get("tail_bound", 2.5)
        num_bins = model_kwargs.get("num_bins", 8)
        rotation = model_kwargs.get("rotation", True)

        base_distribution = StandardNormal([dim])

        # autoregressive (fast) in forward (target -> latent) direction
        transforms = [
            MaskedTailAutoregressiveTransform(
                features=dim,
                nn_kwargs=nn_kwargs,
            )
        ]

        if use == ModelUse.density_estimation:
            # if using for density estimation, the tail transformation needs to be flipped
            # this keeps the autoregression in the data->noise direction, but means data->noise is
            # a strictly lightening transformation
            transforms[0] = flip(transforms[0])

        if rotation:
            transforms.append(LULinear(features=dim))

        transforms.append(
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=dim,
                num_bins=num_bins,
                tails="linear",
                tail_bound=tail_bound,
                nn_kwargs=nn_kwargs,
            )  # type: ignore
        )

        if use == ModelUse.variational_inference:
            transforms = [InverseTransform(transform) for transform in transforms]

        if domain == Domain.positive:
            transforms = [Softplus()] + transforms

        super().__init__(
            transform=CompositeTransform(transforms),
            distribution=base_distribution,
        )


class TTF_m(Flow):
    def __init__(
        self,
        dim: int,
        use: ModelUse = ModelUse.density_estimation,
        model_kwargs: ModelKwargs = {},
        nn_kwargs: NNKwargs = {},
    ):
        # configure default settings
        tail_bound = model_kwargs.get("tail_bound", 2.5)
        num_bins = model_kwargs.get("num_bins", 8)
        pos_tail_init = model_kwargs.get("pos_tail_init", None)
        neg_tail_init = model_kwargs.get("neg_tail_init", None)
        rotation = model_kwargs.get("rotation", True)
        fix_tails = model_kwargs.get("fix_tails", True)
        flow_depth = model_kwargs.get("flow_depth", 1)
        final_affine = model_kwargs.get("final_affine", False)
        nn_kwargs = configure_nn(nn_kwargs)

        # base distributin
        base_distribution = StandardNormal([dim])

        # set up tail transform
        if final_affine:
            tail_transform = TailAffineMarginalTransform(
                features=dim, pos_tail_init=pos_tail_init, neg_tail_init=neg_tail_init
            )
        else:
            tail_transform = TailMarginalTransform(
                features=dim, pos_tail_init=pos_tail_init, neg_tail_init=neg_tail_init
            )

        if fix_tails:
            assert (
                pos_tail_init is not None
            ), "Fixing tails, but no init provided for pos tails"
            assert (
                neg_tail_init is not None
            ), "Fixing tails, but no init provided for neg tails"
            for parameter in tail_transform.parameters():
                parameter.requires_grad = False

        if use == ModelUse.density_estimation:
            # if using for density estimation, the tail transformation needs to be flipped
            # this keeps the autoregression in the data->noise direction, but means data->noise is
            # a strictly lightening transformation
            tail_transform = flip(tail_transform)

        # add the rest of the transformations
        transforms = [tail_transform]
        if flow_depth = 0 and rotation: # still add rotation
            transforms.append(LULinear(features=dim))

        for _ in range(flow_depth):
            if rotation:
                transforms.append(LULinear(features=dim))

            transforms.append(
                MaskedAffineAutoregressiveTransform(features=dim, **nn_kwargs)
            )

            transforms.append(
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=dim,
                    num_bins=num_bins,
                    tails="linear",
                    tail_bound=tail_bound,
                    **nn_kwargs,
                )
            )

        if use == ModelUse.variational_inference:
            transforms = [InverseTransform(transform) for transform in transforms]

        super().__init__(
            transform=CompositeTransform(transforms),
            distribution=base_distribution,
        )


class TailOnly(Flow):
    def __init__(
        self,
        dim: int,
        use: ModelUse = ModelUse.density_estimation,
        model_kwargs: ModelKwargs = {},
        nn_kwargs: NNKwargs = {},
    ):
        # configure default settings
        pos_tail_init = model_kwargs.get("pos_tail_init", None)
        neg_tail_init = model_kwargs.get("neg_tail_init", None)
        rotation = model_kwargs.get("rotation", True)
        fix_tails = model_kwargs.get("fix_tails", True)
        final_affine = model_kwargs.get("final_affine", False)
        nn_kwargs = configure_nn(nn_kwargs)

        # base distributin
        base_distribution = StandardNormal([dim])

        # set up tail transform
        if final_affine:
            tail_transform = TailAffineMarginalTransform(
                features=dim, pos_tail_init=pos_tail_init, neg_tail_init=neg_tail_init
            )
        else:
            tail_transform = TailMarginalTransform(
                features=dim, pos_tail_init=pos_tail_init, neg_tail_init=neg_tail_init
            )

        if fix_tails:
            assert (
                pos_tail_init is not None
            ), "Fixing tails, but no init provided for pos tails"
            assert (
                neg_tail_init is not None
            ), "Fixing tails, but no init provided for neg tails"
            for parameter in tail_transform.parameters():
                parameter.requires_grad = False

        if use == ModelUse.density_estimation:
            # if using for density estimation, the tail transformation needs to be flipped
            # this keeps the autoregression in the data->noise direction, but means data->noise is
            # a strictly lightening transformation
            tail_transform = flip(tail_transform)

        # add the rest of the transformations
        transforms = [tail_transform]
        if rotation:
            transforms.append(LULinear(features=dim))

        if use == ModelUse.variational_inference:
            transforms = [InverseTransform(transform) for transform in transforms]

        super().__init__(
            transform=CompositeTransform(transforms),
            distribution=base_distribution,
        )


class TTF_m_hybrid(Flow):
    def __init__(
        self,
        dim: int,
        use: ModelUse = ModelUse.density_estimation,
        model_kwargs: ModelKwargs = {},
        nn_kwargs: NNKwargs = {},
    ):
        tail_bound = model_kwargs.get("tail_bound", 2.5)
        num_bins = model_kwargs.get("num_bins", 8)
        pos_tail_init = model_kwargs.get("pos_tail_init", None)
        neg_tail_init = model_kwargs.get("neg_tail_init", None)
        rotation = model_kwargs.get("rotation", True)
        fix_tails = model_kwargs.get("fix_tails", True)
        flow_depth = model_kwargs.get("flow_depth", 1)
        nn_kwargs = configure_nn(nn_kwargs)

        base_distribution = StandardNormal([dim])

        # set up tail transform
        tail_transform = HybridTailMarginalTransform(
            features=dim,
            pos_tail_init=pos_tail_init,
            neg_tail_init=neg_tail_init,
            nn_kwargs=nn_kwargs,
        )

        if fix_tails:
            tail_transform._unc_pos_tail.requires_grad = False
            tail_transform._unc_neg_tail.requires_grad = False

        if use == ModelUse.density_estimation:
            # if using for density estimation, the tail transformation needs to be flipped
            # this keeps the autoregression in the data->noise direction, but means data->noise is
            # a strictly lightening transformation
            tail_transform = flip(tail_transform)

        transforms = [tail_transform]

        for _ in range(flow_depth):
            if rotation:
                transforms.append(LULinear(features=dim))

            transforms.append(
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=dim,
                    num_bins=num_bins,
                    tails="linear",
                    tail_bound=tail_bound,
                    **nn_kwargs,
                )
            )

        if use == ModelUse.variational_inference:
            transforms = [InverseTransform(transform) for transform in transforms]

        super().__init__(
            transform=CompositeTransform(transforms),
            distribution=base_distribution,
        )


class Copula_m(Flow):
    def __init__(
        self,
        dim: int,
        use: ModelUse = ModelUse.density_estimation,
        model_kwargs: ModelKwargs = {},
    ):
        hidden_layer_size = model_kwargs.get("hidden_layer_size", dim * 2)
        num_hidden_layers = model_kwargs.get("num_hidden_layers", 2)
        num_bins = model_kwargs.get("num_bins", 8)
        tail_init = model_kwargs.get("tail_init", 1.0)

        if use == ModelUse.density_estimation:
            # if using for density estimation, the tail transformation needs to be flipped
            # this keeps the autoregression in the data->noise direction, but means data->noise is
            # a strictly lightening transformation
            transforms[0] = flip(transforms[0])

        transforms = [
            CopulaMarginalTransform(features=dim, init=tail_init),
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=dim,
                hidden_features=hidden_layer_size,
                num_blocks=num_hidden_layers,
                num_bins=num_bins,
                tails=None,
                tail_bound=1.0,
            ),
        ]

        if use == ModelUse.variational_inference:
            transforms = [InverseTransform(transform) for transform in transforms]

        super().__init__(
            transform=CompositeTransform(transforms),
            distribution=BoxUniform(low=-torch.ones(dim), high=torch.ones(dim)),
        )


class RQS(Flow):
    def __init__(
        self,
        dim: int,
        use: ModelUse = ModelUse.density_estimation,
        model_kwargs: ModelKwargs = {},
    ):
        hidden_layer_size = model_kwargs.get("hidden_layer_size", dim * 2)
        num_hidden_layers = model_kwargs.get("num_hidden_layers", 2)
        tail_bound = model_kwargs.get("tail_bound", 2.5)
        num_bins = model_kwargs.get("num_bins", 8)
        rotation = model_kwargs.get("rotation", True)

        transforms = [
            MaskedAffineAutoregressiveTransform(
                features=dim, hidden_features=dim * 2, num_blocks=2
            )
        ]

        if rotation:
            transforms.append(LULinear(features=dim))

        transforms.append(
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=dim,
                hidden_features=hidden_layer_size,
                num_blocks=num_hidden_layers,
                num_bins=num_bins,
                tails="linear",
                tail_bound=tail_bound,
            )
        )

        if use == ModelUse.variational_inference:
            transforms = [InverseTransform(transform) for transform in transforms]

        super().__init__(
            transform=CompositeTransform(transforms),
            distribution=StandardNormal([dim]),
        )


class gTAF(Flow):
    def __init__(
        self,
        dim: int,
        use: ModelUse = ModelUse.density_estimation,
        model_kwargs: ModelKwargs = {},
    ):
        hidden_layer_size = model_kwargs.get("hidden_layer_size", dim * 2)
        num_hidden_layers = model_kwargs.get("num_hidden_layers", 2)
        tail_bound = model_kwargs.get("tail_bound", 2.5)
        num_bins = model_kwargs.get("num_bins", 8)
        tail_init = model_kwargs.get("tail_init", None)
        rotation = model_kwargs.get("rotation", True)
        nn_activation = model_kwargs.get("nn_activation", torch.nn.functional.relu)

        base_dist = TrainableStudentT(dim, init=tail_init)

        if rotation:
            transforms = [LULinear(features=dim)]
        else:
            transforms = []

        transforms += [
            MaskedAffineAutoregressiveTransform(
                features=dim,
                hidden_features=hidden_layer_size,
                num_blocks=num_hidden_layers,
                activation=nn_activation,
            ),
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=dim,
                hidden_features=hidden_layer_size,
                num_blocks=num_hidden_layers,
                num_bins=num_bins,
                tails="linear",
                tail_bound=tail_bound,
                activation=nn_activation,
            ),
        ]

        # all transformations are reversible
        if use == ModelUse.variational_inference:
            transforms = [InverseTransform(transform) for transform in transforms]

        super().__init__(
            transform=CompositeTransform(transforms),
            distribution=base_dist,
        )


class mTAF(Flow):
    def __init__(
        self,
        dim: int,
        use: ModelUse = ModelUse.density_estimation,
        model_kwargs: ModelKwargs = {},
    ):
        hidden_layer_size = model_kwargs.get("hidden_layer_size", dim * 2)
        num_hidden_layers = model_kwargs.get("num_hidden_layers", 2)
        tail_bound = model_kwargs.get("tail_bound", 2.5)
        num_bins = model_kwargs.get("num_bins", 8)
        fix_tails = model_kwargs.get("fix_tails", True)
        rotation = model_kwargs.get("rotation", True)
        flow_depth = model_kwargs.get("flow_depth", 1)
        nn_activation = model_kwargs.get("nn_activation", torch.nn.functional.relu)

        assert (
            "tail_init" in model_kwargs
        ), "mTAF requires the marginal tails at initialisation time!"
        assert model_kwargs["tail_init"].shape == torch.Size(
            [dim]
        ), "mTAF tail init must be 1 degree of freedom parameter per marginal!"

        self.tail_init = model_kwargs["tail_init"]
        self.dim = dim

        # configure tails
        num_light = int(sum(df == 0 for df in self.tail_init))
        num_heavy = int(dim - num_light)
        initial_permutation, permuted_degrees_of_freedom = _get_intial_permutation(
            self.tail_init
        )

        # always perform the initial permutation
        transforms = [initial_permutation]

        for _ in range(flow_depth):
            if rotation:
                # special rotation within heavy/light groups
                transforms.append(TailLU(dim, int(num_heavy), device="cpu"))

            transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=dim,
                    hidden_features=num_hidden_layers,
                    num_blocks=num_hidden_layers,
                    activation=nn_activation,
                )
            )

            transforms.append(
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=dim,
                    hidden_features=hidden_layer_size,
                    num_blocks=num_hidden_layers,
                    num_bins=num_bins,
                    tails="linear",
                    tail_bound=tail_bound,
                    activation=nn_activation,
                )
            )

        if use == ModelUse.variational_inference:
            # avoid inverting the permutation
            transforms = [transforms[0]] + [
                InverseTransform(transform) for transform in transforms[1:]
            ]

        # fix the base distribution if required (usually true, since we estimate tail in separate procedure)
        base_distribution = NormalStudentTJoint(permuted_degrees_of_freedom)
        if fix_tails:
            for parameter in base_distribution.parameters():
                parameter.requires_grad = False

        super().__init__(
            distribution=base_distribution,
            transform=CompositeTransform(transforms),
        )


class COMET(Flow):
    def __init__(
        self,
        dim: int,
        data: torch.Tensor,
        use: ModelUse = ModelUse.density_estimation,
        model_kwargs: ModelKwargs = {},
    ):
        assert use == ModelUse.density_estimation, "Only valid for density estimation!"

        hidden_layer_size = model_kwargs.get("hidden_layer_size", dim * 2)
        num_hidden_layers = model_kwargs.get("num_hidden_layers", 2)
        tail_bound = model_kwargs.get("tail_bound", 2.5)
        num_bins = model_kwargs.get("num_bins", 8)
        tail_init = model_kwargs.get("tail_init", None)
        rotation = model_kwargs.get("rotation", False)
        fix_tails = model_kwargs.get("fix_tails", True)

        assert rotation == False, "Rotation not implemented for COMET flow"

        tail_transform = MarginalLayer(data)

        if fix_tails and tail_init is not None:
            assert len(tail_init) == dim
            for ix, tail_df in enumerate(tail_init):
                if tail_df == 0.0:
                    tail_transform.tails[ix].lower_xi = 0.0
                    tail_transform.tails[ix].upper_xi = 0.0
                else:
                    tail_transform.tails[ix].lower_xi = 1 / tail_df
                    tail_transform.tails[ix].upper_xi = 1 / tail_df

        transform = CompositeTransform(
            [
                tail_transform,
                Logit(),
                MaskedAffineAutoregressiveTransform(
                    features=dim,
                    hidden_features=hidden_layer_size,
                    num_blocks=num_hidden_layers,
                ),
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=dim,
                    hidden_features=hidden_layer_size,
                    num_blocks=num_hidden_layers,
                    num_bins=num_bins,
                    tails="linear",
                    tail_bound=tail_bound,
                ),
            ]
        )

        super().__init__(
            transform=transform,
            distribution=StandardNormal([dim]),
        )


# map model names to appropriate class
models: dict[ModelName, Type[Flow]] = {
    ModelName.TTF: TTF,
    ModelName.TTF_m: TTF_m,
    ModelName.RQS: RQS,
    ModelName.gTAF: gTAF,
    ModelName.mTAF: mTAF,
    ModelName.COMET: COMET,
    ModelName.Copula_m: Copula_m,
    ModelName.TTF_m_hybrid: TTF_m_hybrid,
    ModelName.TailOnly: TailOnly,
}


def get_model(
    model_name: ModelName, dim: int, cond_dim: int = 0, model_kwargs: ModelKwargs = {}
):
    assert (
        model_name in models.keys()
    ), f"Invalid model name {model_name} not one of {list(models.keys())}"
    return models[model_name](dim, cond_dim, model_kwargs)
