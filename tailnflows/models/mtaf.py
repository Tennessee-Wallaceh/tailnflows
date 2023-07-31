# nflows dependencies
import torch
from nflows.flows import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.autoregressive import (
    MaskedAffineAutoregressiveTransform, 
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
)
from nflows.transforms import Permutation
from nflows.transforms.base import CompositeTransform

from tailnflows.models.base_distribution import TrainableStudentT, NormalStudentTJoint
from marginal_tail_adaptive_flows.utils.tail_permutation import TailRandomPermutation, TailLU

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

    return Permutation(perm_ix)

class mTAF(Flow):
    def __init__(self, dim, model_kwargs={}):
        self.hidden_layer_size = model_kwargs.get('hidden_layer_size', dim  * 2)
        self.num_hidden_layers = model_kwargs.get('num_hidden_layers', 2)
        self.tail_bound = model_kwargs.get('tail_bound', 2.5)
        self.num_bins = model_kwargs.get('num_bins', 8)
        self.num_layers = model_kwargs.get('num_layers', 1)
        self.dim = dim

        super().__init__(
            distribution=TrainableStudentT(dim, init=10.),
            transform=CompositeTransform([]),
        )

    def configure_tails(self, degrees_of_freedom):
        self.degrees_of_freedom = degrees_of_freedom
        self._distribution = NormalStudentTJoint(degrees_of_freedom)
        num_light = int(sum(df == 0 for df in degrees_of_freedom))
        num_heavy = int(self._distribution.dim - num_light)

        transforms = [_get_intial_permutation(degrees_of_freedom)]
        for _ in range(self.num_layers):
            transforms.append(TailRandomPermutation(num_light, num_heavy))
            transforms.append(TailLU(self._distribution.dim, int(num_heavy), device='cpu'))
            transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=self.dim,
                hidden_features=self.hidden_layer_size,
                num_blocks=self.num_hidden_layers,
                num_bins=self.num_bins,
                tail_bound = self.tail_bound,
                tails="linear",
                use_residual_blocks=True,
            ))
        
        transforms.append(TailRandomPermutation(num_light, num_heavy))
        transforms.append(TailLU(self._distribution.dim, int(num_heavy), device='cpu'))
        self._transform = CompositeTransform(transforms)

    def load_state_dict(self, state_dict):
        assert 'degrees_of_freedom' in state_dict, 'Loading an unconfigured mTAF'
        self.configure_tails(state_dict['degrees_of_freedom'])
        del state_dict['degrees_of_freedom']
        super().load_state_dict(state_dict)

    def state_dict(self):
        assert self.degrees_of_freedom is not None, 'Saving unconfigured mTAF'
        state_dict = super().state_dict()
        state_dict['degrees_of_freedom'] = self.degrees_of_freedom
        return state_dict