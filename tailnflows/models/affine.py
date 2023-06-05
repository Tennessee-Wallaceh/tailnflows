from torch.nn.functional import softplus
import torch

class ScaleTransform():
    MIN_SCALE = 1e-2
    def __init__(self):
        self.param_shape = (1,)
    
    # sampling direction
    def forward_and_lad(self, z, scale_param):
        scale_param = self.MIN_SCALE + softplus(scale_param).squeeze()
        x = z * scale_param
        return x, torch.log(scale_param).sum(axis=1)

    # density evaluation direction
    def inverse_and_lad(self, x, scale_param):
        # transform parameters
        scale_param = self.MIN_SCALE + softplus(scale_param).squeeze()
        z = x / scale_param
        return z, -torch.log(scale_param).sum(axis=1)
    

class AffineTransform():
    MIN_SCALE = 1e-2
    def __init__(self):
        self.param_shape = (1, 1)
    
    # sampling direction
    def forward_and_lad(self, z, scale_param, shift_param):
        shift = shift_param.squeeze()
        scale_param = self.MIN_SCALE + softplus(scale_param).squeeze()
        x = shift + z * scale_param
        return x, torch.log(scale_param).sum(axis=1)

    # density evaluation direction
    def inverse_and_lad(self, x, scale_param, shift_param):
        shift = shift_param.squeeze()
        scale_param = self.MIN_SCALE + softplus(scale_param).squeeze()
        z = (x - shift) / scale_param
        return z, -torch.log(scale_param).sum(axis=1)