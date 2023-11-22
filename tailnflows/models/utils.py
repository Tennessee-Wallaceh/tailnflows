import torch
import numpy as np
from nflows.transforms.base import Transform
from torch.nn.functional import softplus
from nflows.utils import torchutils


def inv_sftplus(x):
    return x + torch.log(-torch.expm1(-x))


class Marginal(Transform):
    def __init__(self, dim, transform):
        super().__init__()
        self.transform = transform
        self.params = [
            torch.nn.Parameter(
                torch.zeros(
                    dim * shape,
                )
            )
            for shape in transform.param_shape
        ]
        for ix, p in enumerate(self.params):
            self.register_parameter(f"p_{ix}", p)

    def forward(self, z, context=None):
        tiled_p = tuple(torch.tile(p, (z.shape[0], 1)) for p in self.params)
        x, logabsdet = self.transform.forward_and_lad(z, *tiled_p)
        return x, logabsdet

    def inverse(self, z, context=None):
        tiled_p = tuple(torch.tile(p, (z.shape[0], 1)) for p in self.params)
        z, lad = self.transform.inverse_and_lad(z, *tiled_p)
        return z, lad


class Softplus(Transform):
    def __init__(self, temperature=1.0, learn_temperature=False):
        super().__init__()
        if learn_temperature:
            self.temperature = nn.Parameter(torch.Tensor([temperature]))
        else:
            self.temperature = torch.Tensor([temperature])

    # data -> noise
    def forward(self, outputs, context=None):
        if torch.min(outputs) < 0:
            raise InputOutsideDomain()

        inputs = inv_sftplus(outputs)
        logabsdet = -torchutils.sum_except_batch(
            torch.where(
                inputs < 0,
                self.temperature * inputs
                - torch.log1p(torch.exp(self.temperature * inputs)),
                -torch.log1p(torch.exp(-self.temperature * inputs)),
            )
        )
        return inputs, logabsdet

    # noise -> data
    def inverse(self, inputs, context=None):
        outputs = softplus(inputs)
        logabsdet = torchutils.sum_except_batch(
            torch.where(
                inputs < 0,
                self.temperature * inputs
                - torch.log1p(torch.exp(self.temperature * inputs)),
                -torch.log1p(torch.exp(-self.temperature * inputs)),
            )
        )
        return outputs, logabsdet
