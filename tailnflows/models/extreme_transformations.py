import torch
from torch.nn.functional import softplus

MAX_TAIL = 5.
SQRT_2 = torch.sqrt(torch.tensor(2.))
SQRT_PI = torch.sqrt(torch.tensor(torch.pi))

def _erfcinv(x):
    return -torch.special.ndtri(0.5 * x) / SQRT_2

def _shift_power_transform_and_lad(z, tail_param):
    transformed = (SQRT_2 / SQRT_PI) * (torch.pow(1 + z / tail_param, tail_param) - 1)
    lad = (tail_param - 1) * torch.log(1 + z / tail_param)
    lad += torch.log(SQRT_2 / SQRT_PI)
    return transformed, lad

def _shift_power_inverse_and_lad(x, tail_param):
    transformed = (SQRT_PI / SQRT_2) * tail_param * (torch.pow(1 + x, 1 / tail_param) - 1)
    lad = ((1 / tail_param) - 1) * torch.log(1 + x)
    lad -= torch.log(SQRT_2 / SQRT_PI)
    return transformed, lad

def _extreme_transform_and_lad(z, tail_param):
    g = torch.erfc(z / SQRT_2)
    x = (torch.pow(g, -tail_param) - 1) / tail_param
    
    lad = torch.log(g) * (-tail_param -1)
    lad -= 0.5 * torch.square(z) 
    lad += torch.log(SQRT_2 / SQRT_PI)

    return x, lad

def _extreme_inverse_and_lad(x, tail_param):
    inner = 1 + tail_param * x
    g = torch.pow(inner, -1 / tail_param)
    z = SQRT_2 * _erfcinv(g)

    lad = (-1 - 1 / tail_param) * torch.log(inner)
    lad += torch.log(0.5 * SQRT_2 * SQRT_PI)
    lad += torch.square(_erfcinv(g))

    return z, lad

class ExtremeScaleTransform():
    def __init__(self):
        self.param_shape = (1, 1)
    
    # sampling direction
    # first apply the tail, then the scale
    def forward_and_lad(self, z, unc_params):
        pos_param = softplus(unc_params)
        tail_param = pos_param[:,0::2]
        scale_param = pos_param[:,1::2]
        sign = torch.sign(z)
        x, lad = _extreme_transform_and_lad(torch.abs(z), tail_param)
        return sign * x * scale_param, lad.sum(axis=1) + torch.log(scale_param).sum(axis=1)
    
    def inverse_and_lad(self, x, unc_params):
        pos_param = softplus(unc_params)
        tail_param = pos_param[:,0::2]
        scale_param = pos_param[:,1::2]
    
        x = x / scale_param
        sign = torch.sign(x)
        z, lad = _extreme_inverse_and_lad(torch.abs(x), tail_param)

        return sign * z, lad.sum(axis=1) - torch.log(scale_param).sum(axis=1)


class SwitchScaleTransform():
    def __init__(self):
        self.param_shape = (1, 1)
    
    # sampling direction
    # first apply the tail, then the scale
    def forward_and_lad(self, z, tail_param, scale_param):
        tail_param = 1e-3 + softplus(tail_param).squeeze()
        scale_param = 1e-3 + softplus(scale_param).squeeze()

        sign = torch.sign(z)
        heavy_x, heavy_lad = _extreme_transform_and_lad(torch.abs(z), tail_param - 1)
        light_x, light_lad = _shift_power_transform_and_lad(torch.abs(z), tail_param + 1)
        x = torch.where(tail_param > 1., heavy_x, light_x)
        lad = torch.where(tail_param > 1., heavy_lad, light_lad)
        return sign * x * scale_param, lad.sum(axis=1) + torch.log(scale_param).sum(axis=1)
    
    def inverse_and_lad(self, x, tail_param, scale_param):
        tail_param = 1e-3 + softplus(tail_param).squeeze()
        scale_param = 1e-3 + softplus(scale_param).squeeze()

        # scale transform
        x = x / scale_param

        # tail transform
        sign = torch.sign(x)
        heavy_z, heavy_lad = _extreme_inverse_and_lad(torch.abs(x), tail_param - 1)
        light_z, light_lad = _shift_power_inverse_and_lad(torch.abs(x), tail_param + 1)
        z = torch.where(tail_param > 1., heavy_z, light_z)
        lad = torch.where(tail_param > 1., heavy_lad, light_lad)
        
        return sign * z, lad.sum(axis=1) - torch.log(scale_param).sum(axis=1)


class SwitchTransform():
    def __init__(self):
        self.param_shape = (1,)
    
    # sampling direction
    # first apply the tail, then the scale
    def forward_and_lad(self, z, tail_param):
        tail_param = 1e-3 + softplus(tail_param).squeeze()

        sign = torch.sign(z)
        heavy_x, heavy_lad = _extreme_transform_and_lad(torch.abs(z), tail_param - 1)
        light_x, light_lad = _shift_power_transform_and_lad(torch.abs(z), tail_param + 1)
        x = torch.where(tail_param > 1., heavy_x, light_x)
        lad = torch.where(tail_param > 1., heavy_lad, light_lad)
        return sign * x, lad.sum(axis=1)
    
    def inverse_and_lad(self, x, tail_param):
        tail_param = 1e-3 + softplus(tail_param).squeeze()

        # tail transform
        sign = torch.sign(x)
        heavy_z, heavy_lad = _extreme_inverse_and_lad(torch.abs(x), tail_param - 1)
        light_z, light_lad = _shift_power_inverse_and_lad(torch.abs(x), tail_param + 1)
        z = torch.where(tail_param > 1., heavy_z, light_z)
        lad = torch.where(tail_param > 1., heavy_lad, light_lad)
        
        return sign * z, lad.sum(axis=1)


class SwitchAffineTransform():
    def __init__(self):
        self.param_shape = (1, 1, 1)
    
    # sampling direction
    # first apply the tail, then the scale
    def forward_and_lad(self, z, tail_param, scale_param, loc_param):
        # tail_param = -1. + (MAX_TAIL + 1) * torch.sigmoid(tail_param).squeeze()
        tail_param = -1. + softplus(tail_param).squeeze()
        scale_param = 1e-3 + softplus(scale_param).squeeze()
        shift_param = loc_param.squeeze()

        sign = torch.sign(z)
        heavy_x, heavy_lad = _extreme_transform_and_lad(torch.abs(z), tail_param)
        light_x, light_lad = _shift_power_transform_and_lad(torch.abs(z), tail_param + 2.)
        
        x = torch.where(tail_param > 0., heavy_x, light_x)
        lad = torch.where(tail_param > 0., heavy_lad, light_lad)

        return sign * x * scale_param + shift_param, lad.sum(axis=1) + torch.log(scale_param).sum(axis=1)
    
    def inverse_and_lad(self, x, tail_param, scale_param, loc_param):
        # tail_param = -1. + (MAX_TAIL + 1) * torch.sigmoid(tail_param).squeeze()
        tail_param = -1. + softplus(tail_param).squeeze()
        scale_param = 1e-3 + softplus(scale_param).squeeze()
        shift_param = loc_param.squeeze()

        # affine transform
        x = (x - shift_param) / scale_param

        # tail transform
        sign = torch.sign(x)
        heavy_z, heavy_lad = _extreme_inverse_and_lad(torch.abs(x), tail_param)
        light_z, light_lad = _shift_power_inverse_and_lad(torch.abs(x), tail_param + 2)
        z = torch.where(tail_param > 0., heavy_z, light_z)
        lad = torch.where(tail_param > 0., heavy_lad, light_lad)
        
        return sign * z, lad.sum(axis=1) - torch.log(scale_param).sum(axis=1)
