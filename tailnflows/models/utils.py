import torch
import numpy as np
from nflows.transforms.base import Transform

def inv_sftplus(x):
    return x + torch.log(-torch.expm1(-x))

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