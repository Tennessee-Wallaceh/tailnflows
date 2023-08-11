import torch

# nflows dependencies
from nflows.flows import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.autoregressive import (
    MaskedAffineAutoregressiveTransform,
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
)
from nflows.transforms.lu import LULinear
from nflows.transforms.base import CompositeTransform, InverseTransform
from nflows.transforms.nonlinearities import Logit

# custom modules
from tailnflows.models.extreme_transformations import (
    MaskedTailAutoregressiveTransform,
    flip,
    MaskedExtremeAutoregressiveTransform,
    TailMarginalTransform,
    EXMarginalTransform,
    PowerMarginalTransform,
    SinhMarginalTransform,
)
from tailnflows.models.base_distribution import TrainableStudentT
from tailnflows.models.mtaf import mTAF

from enum import Enum
from typing import TypedDict, Optional, Type, Type


class ModelKwargs(TypedDict, total=False):
    hidden_layer_size: Optional[int]
    num_hidden_layers: Optional[int]
    tail_bound: Optional[float]
    num_bins: Optional[int]
    tail_init: Optional[float]
    rotation: Optional[bool]


ModelName = Enum(
    "ModelName",
    ["TTF", "TTF_dextreme", "TTF_m", "EXF", "EXF_m", "RQS", "gTAF", "mTAF", "COMET"],
)


def preconfigure_model(model, strategy, x_precon):
    if strategy == "mTAF":
        dfs = [
            estimate_df(x_precon[:, _dim], verbose=False)
            for _dim in range(x_precon.shape[1])
        ]
        model.configure_tails(dfs)

    elif strategy == "fix_dextreme":
        dfs = [estimate_df(x_precon[:, _dim]) for _dim in range(x_precon.shape[1])]
        marginal_tail_params = torch.tensor(
            [1 / torch.tensor(df) if df != 0 else torch.tensor(1e-5) for df in dfs]
        ).repeat_interleave(2)
        model._transform._transforms[0] = flip(
            EXMarginalTransform(x_precon.shape[1], init=marginal_tail_params)
        )
        # freeze
        for param in model._transform._transforms[0].parameters():
            param.requires_grad = False

    # elif strategy == 'train_marginal':

    return model


# Define flow models
class TTF(Flow):
    def __init__(self, dim: int, model_kwargs: ModelKwargs = {}):
        hidden_layer_size = model_kwargs.get("hidden_layer_size", dim * 2)
        num_hidden_layers = model_kwargs.get("num_hidden_layers", 2)
        tail_bound = model_kwargs.get("tail_bound", 2.5)
        num_bins = model_kwargs.get("num_bins", 8)
        tail_init = model_kwargs.get("tail_init", None)
        rotation = model_kwargs.get("rotation", True)

        base_distribution = StandardNormal([dim])

        # element wise fcn flip, so heavy->light becomes forward (to noise in nflows) transform
        transforms = [
            flip(
                MaskedTailAutoregressiveTransform(
                    features=dim, hidden_features=hidden_layer_size, num_blocks=2
                )
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
            )  # type: ignore
        )

        if tail_init is not None:
            for _dim in range(dim):
                torch.nn.init.constant_(
                    transform._transforms[0].autoregressive_net.final_layer.bias[
                        _dim * 4
                    ],
                    tail_init,
                )
                torch.nn.init.constant_(
                    transform._transforms[0].autoregressive_net.final_layer.bias[
                        _dim * 4 + 1
                    ],
                    tail_init,
                )

        super().__init__(
            transform=CompositeTransform(transforms),
            distribution=base_distribution,
        )

    @staticmethod
    def from_data(data, model_kwargs: ModelKwargs = {}):
        return TTF(data.shape[1], model_kwargs)


class TTF_dextreme(Flow):
    def __init__(self, dim: int, model_kwargs: ModelKwargs = {}):
        hidden_layer_size = model_kwargs.get("hidden_layer_size", dim * 2)
        num_hidden_layers = model_kwargs.get("num_hidden_layers", 2)
        tail_bound = model_kwargs.get("tail_bound", 1.0)
        num_bins = model_kwargs.get("num_bins", 8)
        rotation = model_kwargs.get("rotation", True)

        base_distribution = StandardNormal([dim])
        transforms = [
            InverseTransform(
                MaskedTailAutoregressiveTransform(
                    features=dim,
                    hidden_features=dim * 2,
                    num_blocks=2,
                )
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

        super().__init__(
            transform=CompositeTransform(transforms),
            distribution=base_distribution,
        )


class TTF_m(Flow):
    def __init__(self, dim: int, model_kwargs: ModelKwargs = {}):
        hidden_layer_size = model_kwargs.get("hidden_layer_size", dim * 2)
        num_hidden_layers = model_kwargs.get("num_hidden_layers", 2)
        tail_bound = model_kwargs.get("tail_bound", 2.5)
        num_bins = model_kwargs.get("num_bins", 8)
        tail_init = model_kwargs.get("tail_init", 0.0)
        rotation = model_kwargs.get("rotation", True)

        base_distribution = StandardNormal([dim])
        transforms = [
            flip(TailMarginalTransform(features=dim, init=tail_init)),
            MaskedAffineAutoregressiveTransform(
                features=dim, hidden_features=dim * 2, num_blocks=2
            ),
        ]

        if rotation:
            transforms.append(LULinear(features=dim))

        transforms += [
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=dim,
                hidden_features=hidden_layer_size,
                num_blocks=num_hidden_layers,
                num_bins=num_bins,
                tails="linear",
                tail_bound=tail_bound,
            )
        ]

        super().__init__(
            transform=CompositeTransform(transforms),
            distribution=base_distribution,
        )


class EXF(Flow):
    def __init__(self, dim: int, model_kwargs: ModelKwargs = {}):
        hidden_layer_size = model_kwargs.get("hidden_layer_size", dim * 2)
        num_hidden_layers = model_kwargs.get("num_hidden_layers", 2)
        tail_bound = model_kwargs.get("tail_bound", 2.5)
        num_bins = model_kwargs.get("num_bins", 8)
        rotation = model_kwargs.get("rotation", True)

        base_distribution = StandardNormal([dim])
        transforms = [
            # element wise fcn flip, so heavy->light becomes forward transform
            # always lightens, will push subgaussian onto RQS
            flip(
                MaskedExtremeAutoregressiveTransform(
                    features=dim, hidden_features=dim * 2, num_blocks=2
                )
            ),
        ]

        if rotation:
            transforms += [LULinear(features=dim)]

        transforms += [
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=dim,
                hidden_features=hidden_layer_size,
                num_blocks=num_hidden_layers,
                num_bins=num_bins,
                tails="linear",
                tail_bound=tail_bound,
            ),
        ]

        super().__init__(
            transform=CompositeTransform(transforms),
            distribution=base_distribution,
        )


class ExtremeMarginal(Flow):
    def __init__(self, kind="power", model_kwargs: ModelKwargs = {}):
        tail_bound = model_kwargs.get("tail_bound", 2.5)
        num_bins = model_kwargs.get("num_bins", 8)

        base_distribution = StandardNormal([1])
        transforms = [
            flip(EXMarginalTransform(features=1)),
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=1,
                hidden_features=0,
                num_blocks=0,
                num_bins=num_bins,
                tails="linear",
                tail_bound=tail_bound,
            ),
        ]

        if kind == "power":
            transforms.append(flip(PowerMarginalTransform(features=1)))
        else:
            transforms.append(flip(SinhMarginalTransform(features=1)))

        super().__init__(
            transform=CompositeTransform(transforms),
            distribution=base_distribution,
        )


class EXF_m(Flow):
    def __init__(self, dim: int, model_kwargs: ModelKwargs = {}):
        hidden_layer_size = model_kwargs.get("hidden_layer_size", dim * 2)
        num_hidden_layers = model_kwargs.get("num_hidden_layers", 2)
        tail_bound = model_kwargs.get("tail_bound", 2.5)
        num_bins = model_kwargs.get("num_bins", 8)
        rotation = model_kwargs.get("rotation", True)

        base_distribution = StandardNormal([dim])
        transforms = [
            flip(EXMarginalTransform(features=dim)),
            MaskedAffineAutoregressiveTransform(
                features=dim, hidden_features=dim * 2, num_blocks=2
            ),
        ]

        if rotation:
            transforms.append(LULinear(features=dim))

        transforms += [
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=dim,
                hidden_features=hidden_layer_size,
                num_blocks=num_hidden_layers,
                num_bins=num_bins,
                tails="linear",
                tail_bound=tail_bound,
            )
        ]

        super().__init__(
            transform=CompositeTransform(transforms),
            distribution=base_distribution,
        )


class RQS(Flow):
    def __init__(self, dim: int, model_kwargs: ModelKwargs = {}):
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

        super().__init__(
            transform=CompositeTransform(transforms),
            distribution=StandardNormal([dim]),
        )


class gTAF(Flow):
    def __init__(self, dim: int, model_kwargs: ModelKwargs = {}):
        hidden_layer_size = model_kwargs.get("hidden_layer_size", dim * 2)
        num_hidden_layers = model_kwargs.get("num_hidden_layers", 2)
        tail_bound = model_kwargs.get("tail_bound", 2.5)
        num_bins = model_kwargs.get("num_bins", 8)
        tail_init = model_kwargs.get("tail_init", 10.0)  # default init from FTVI code

        base_dist = TrainableStudentT(dim, init=tail_init)
        transform = CompositeTransform(
            [
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
                LULinear(features=dim),
            ]
        )

        super().__init__(
            transform=transform,
            distribution=base_dist,
        )


class COMET(Flow):
    def __init__(self, dim: int, model_kwargs: ModelKwargs = {}):
        hidden_layer_size = model_kwargs.get("hidden_layer_size", dim * 2)
        num_hidden_layers = model_kwargs.get("num_hidden_layers", 2)
        tail_bound = model_kwargs.get("tail_bound", 2.5)
        num_bins = model_kwargs.get("num_bins", 8)
        transform = CompositeTransform(
            [
                MarginalLayer(x_trn),
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
                LULinear(features=dim),
            ]
        )

        super().__init__(
            transform=transform,
            distribution=StandardNormal([dim]),
        )


# map model names to appropriate class
models: dict[ModelName, Type[Flow]] = {
    ModelName.TTF: TTF,
    ModelName.EXF: EXF,
    ModelName.TTF_m: TTF_m,
    ModelName.EXF_m: EXF_m,
    ModelName.TTF_dextreme: TTF_dextreme,
    ModelName.RQS: RQS,
    ModelName.gTAF: gTAF,
    ModelName.mTAF: mTAF,
    ModelName.COMET: COMET,
}


def get_model(model_name: ModelName, dim: int, model_kwargs: ModelKwargs = {}):
    assert (
        model_name in models.keys()
    ), f"Invalid model name {model_name} not one of {list(models.values())}"
    return models[model_name](dim, model_kwargs)
