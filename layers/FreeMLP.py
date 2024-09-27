

import torch
import torch.nn as nn
from layers.CWT import cwt,icwt,get_scales_size
import numpy as np
from ssqueezepy import Wavelet
import matplotlib.pyplot as plt

import torch.utils.checkpoint as checkpoint



class FreeMLP(nn.Module):

    def __init__(self,scale,embed_size,sparsity_threshold,SFT = False):
        super(FreeMLP, self).__init__()

        self.embed_size = embed_size #embed_size linear in embed size
        self.scale = scale
        self.sparsity_threshold = sparsity_threshold
        assert SFT==False or type(SFT)==int , "SFT should be False or provide the amount of windows"
        if not SFT:
            self.r = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
            self.i = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
            self.rb = nn.Parameter(self.scale * torch.randn(self.embed_size))
            self.ib = nn.Parameter(self.scale * torch.randn(self.embed_size))
            self.dim_str = 'bijd,dd->bijd'
        else:
            print("using the channel dim")
            self.r = nn.Parameter(self.scale * torch.randn(self.embed_size,SFT, self.embed_size))
            self.i = nn.Parameter(self.scale * torch.randn(self.embed_size,SFT, self.embed_size))
            self.rb = nn.Parameter(self.scale * torch.randn(self.embed_size))
            self.ib = nn.Parameter(self.scale * torch.randn(self.embed_size))
            self.dim_str = 'bijkd,djd->bijkd'

    def forward(self,x):
        o1_real = torch.nn.functional.relu(
            torch.einsum(self.dim_str, x.real, self.r) - \
            torch.einsum(self.dim_str, x.imag, self.i) + \
            self.rb
        )

        o1_imag = torch.nn.functional.relu(
            torch.einsum(self.dim_str, x.imag, self.r) + \
            torch.einsum(self.dim_str, x.real, self.i) + \
            self.ib
        )

        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = torch.nn.functional.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)

        return y


