import torch.nn as nn
import torch.nn.functional as F

from .grids import DenseGridGen

__all__ = ['BilinearWarpingScheme',
           'GaussianWarpingScheme']


class BilinearWarpingScheme(nn.Module):
    def __init__(self, padding_mode='zeros'):
        super(BilinearWarpingScheme, self).__init__()
        self.grid = DenseGridGen()
        self.padding_mode = padding_mode

    def forward(self, im, w):
        return F.grid_sample(im, self.grid(w), padding_mode=self.padding_mode, mode='bilinear')
    

class GaussianWarpingScheme(nn.Module):
    def __init__(self, padding_mode='zeros', F=3, std=0.25):
        super(GaussianWarpingScheme, self).__init__()
        self.grid = DenseGridGen()
        self.F = F
        self.std = std
        self.padding_mode = padding_mode

    def forward(self, im, w):
        return F.grid_sample(im, self.grid(w), padding_mode=self.padding_mode, mode='gaussian', F=self.F, std=self.std)