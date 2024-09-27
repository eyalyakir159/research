import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import List, Tuple
import math
from functools import partial
from torch import nn, einsum, diagonal
from math import log2, ceil
import pdb
from layers.MultiWaveletCorrelation import FourierCrossAttentionW
import logging
print = logging.info

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class YakirCorrelation(nn.Module):
    def __init__(self,c,d_model,nWaveletBlocks,nheads,filter_size = 8,filter_initialization='Random',M=16):
        super(YakirCorrelation, self).__init__()

        self.c = c #take to higher dim by c
        self.M = M
        self.d_model = d_model
        self.nWaveletBlocks = nWaveletBlocks
        self.filter_size = filter_size
        self.L1 = nn.Linear(d_model,int(d_model*c))
        self.L2 = nn.Linear(int(d_model*c) , d_model)
        self.nheads = nheads
        self.ich = (d_model//nheads)*nheads
        self.WaveletBlocks = nn.ModuleList([WaveletBlock(filter_initialization,filter_size,c,M,self.ich,d_model) for i in range(self.nWaveletBlocks)])

    def forward(self,queries, keys, values,attn_mask):
        """here the queries,keys,values will be the same, but in general
        values=keys and quries might be diffren"""
        B,N, H, E = values.shape #Batch, input len, number of heads, dmodel / num of heads

        values = values.view(B,N,-1) # turn it into normal B,N,dmodel
        values = self.L1(values).view(B,N,self.c,-1) #multiply the size by c to B,N,C,D


        for block in self.WaveletBlocks:
            values = F.relu(block(values)) #run the value into WaveletBlocks
        values = self.L2(values.view(B,N,-1)) #refactor by c
        return (values.contiguous(),None)

class YakirCorrelationCross(nn.Module):
    def __init__(self,d_model,c,nheads,filter_initialization='Random', filter_size=16,M=64):
        super(YakirCorrelationCross, self).__init__()

        self.hp_attn = FourierCrossAttentionW(in_channels=1, out_channels=1, seq_len_q=None,
                                            seq_len_kv=None)
        self.lp_attn = FourierCrossAttentionW(in_channels=1, out_channels=1, seq_len_q=None,
                                            seq_len_kv=None)
        self.c = c
        self.d_model = d_model
        self.L1k = nn.Linear(d_model,int(d_model*c))
        self.L1q = nn.Linear(d_model,int(d_model*c))
        self.L1v = nn.Linear(d_model,int(d_model*c))
        self.out = nn.Linear(int(d_model*c),d_model)
        self.ich = (d_model//nheads)*nheads
        self.DWTk = DWT(filter_initialization, filter_size,c,M,self.ich,d_model)
        self.DWTv = DWT(filter_initialization, filter_size,c,M,self.ich,d_model)
        self.DWTq = DWT(filter_initialization, filter_size,c,M,self.ich,d_model)

        self.IDWT_out = IDWT(filter_initialization, filter_size,c,self.ich)


    def forward(self, q, k, v,attn_mask):
        """ values=keys and quries might be diffren """
        B,N,H,E = q.shape #Batch, input len, dmodel//num of heads, num of heads
        _,S,_,_ = v.shape

        #make it the same size
        if N > S:
            zeros = torch.zeros_like(q[:, :(N - S), :]).float()
            v = torch.cat([v, zeros], dim=1)
            k = torch.cat([k, zeros], dim=1)
        else:
            v = v[:, :N, :, :]
            k = k[:, :N, :, :]

        #Pass through the linear layers
        v = self.L1v(v.reshape(B,N,-1)).reshape(B,N,self.c,self.d_model)
        q = self.L1q(q.reshape(B,N,-1)).reshape(B,N,self.c,self.d_model)
        k = self.L1k(k.reshape(B,N,-1)).reshape(B,N,self.c,self.d_model)

        cA_v,detail_coefficients_v = self.DWTv(v)
        cA_q,detail_coefficients_q = self.DWTq(q)
        cA_k,detail_coefficients_k = self.DWTk(k)

        cA = self.lp_attn(cA_q,cA_k,cA_v,None)[0]
        detail_coefficients = []
        for i in range(len(detail_coefficients_k)):
            detail_coefficients.append(self.hp_attn(detail_coefficients_q[i],detail_coefficients_k[i],detail_coefficients_v[i],None)[0])
        v = self.IDWT_out(cA,detail_coefficients)
        v = self.out(v[:, :N, :, :].contiguous().view(B, N, -1))
        return (v.contiguous(), None)

class WaveletBlock(nn.Module):
    def __init__(self,filter_initialization,filter_size,c,M,ich,d_model):
        super(WaveletBlock, self).__init__()
        self.M = M
        self.c = c
        self.DWT = DWT(filter_initialization,filter_size,c,M,ich,d_model)
        self.IDWT = IDWT(filter_initialization,filter_size,c,ich)


    def forward(self,x):
        cA,detail_coefficients = self.DWT(x)

        return self.IDWT(cA,detail_coefficients)/math.sqrt(self.c)

class DWT(nn.Module):
    def __init__(self, filter_initialization, filter_size,c,M,ich,d_model): #ich = dmodl//number of ehads
        super(DWT, self).__init__()
        assert filter_size%2==1, "filter size must be odd"

        self.filter_initialization = filter_initialization
        self.filter_size = filter_size
        self.M = M
        self.c = c
        self.d_model=d_model
        self.layer_norm = nn.LayerNorm([self.c, d_model])
        padding = (self.filter_size - 1) // 2

        # create the filters
        if filter_initialization in ['legendre', 'chebyshev']:
            pass  # inisialize the filters with the correct base
        else:
            self.ec_hp = nn.Conv1d(int(ich*c), int(ich*c),filter_size,stride=2,padding=padding)
            self.ec_lp = nn.Conv1d(int(ich*c), int(ich*c),filter_size,stride=2,padding=padding)

        self.hp_process_layer = sparseKernelFT1d(ich,M,c)
        self.lp_process_layer = sparseKernelFT1d(ich,M,c)
    # 
    def forward(self, x):
        #x = self.layer_norm(x)

        # x is of size B,N,C,D
        B, N, C, D = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, C * D, N)
        cA = x
        detail_coefficients = []

        # find the maximum amount of wavelet levels
        max_rank = math.floor(log2(N)) ## or max_rank = math.floor(log2(N)/self.filter_size)
        assert max_rank > 0, "Filter size should be bigger than or equal to the sequence length."
        # DWT
        for lv in range(max_rank):
            # Convolution with stride 2 for downsampling
            #exploastion here need to understnad why ....

            cD = self.hp_process_layer(self.ec_hp(cA.reshape(B,C*D,-1)).reshape(B,-1,C,D))
            cA = self.lp_process_layer(self.ec_lp(cA.reshape(B,C*D,-1)).reshape(B,-1,C,D))



            detail_coefficients.append(cD)
        return cA,detail_coefficients
class IDWT(nn.Module):
    def __init__(self, filter_initialization, filter_size,c,ich):
        super(IDWT, self).__init__()
        assert filter_size%2==1, "filter size must be odd"
        self.filter_initialization = filter_initialization
        self.filter_size = filter_size
        self.c = c
        self.ich = ich
        padding = (self.filter_size - 1) // 2


        # create the filters
        if filter_initialization in ['legendre', 'chebyshev']:
            pass  # inisialize the filters with the correct base
        else:

            self.rc_lp = nn.Conv1d(ich * c, ich * c, filter_size, padding=padding)
            self.rc_hp = nn.Conv1d(ich * c, ich * c, filter_size, padding=padding)


    def forward(self,cA,detail_coefficients):
        max_rank = len(detail_coefficients)
        B,_,C,D = detail_coefficients[-1].shape
        cA = cA.reshape(B,C*D,-1)

        #IDWT
        for lv in range(max_rank):
            #up sample by 2
            reshaped_coefficient = detail_coefficients[-lv-1].reshape(B,C*D,-1)


            cA = cA [:,:,:reshaped_coefficient.size(2)]

            cD = F.interpolate(reshaped_coefficient, scale_factor=2, mode='nearest')
            cA = F.interpolate(cA, scale_factor=2, mode='nearest')
            # Convolution to apply the filter
            cA = self.rc_lp(cA)
            cD = self.rc_hp(cD)
            cA = cA+cD
            cA = cA/torch.std(cA)
        return cA.reshape(B,-1,C,D)


class sparseKernelFT1d(nn.Module):
    def __init__(self,
                 k, alpha, c=1,
                 **kwargs):
        super(sparseKernelFT1d, self).__init__()

        self.modes1 = alpha
        self.scale = (1 / (c * k * c * k))
        self.weights1 = nn.Parameter(self.scale * torch.rand(c * k, c * k, self.modes1, dtype=torch.cfloat))
        self.weights1.requires_grad = True

        self.k = k

    def compl_mul1d(self, x, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", x, weights)

    def forward(self, x):
        B, N, c, k = x.shape  # (B, N, c, k)

        x = x.view(B, N, -1)
        x = x.permute(0, 2, 1)
        x_fft = torch.fft.rfft(x)
        # Multiply relevant Fourier modes
        l = min(self.modes1, N // 2 + 1)
        # l = N//2+1
        out_ft = torch.zeros(B, c * k, N // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :l] = self.compl_mul1d(x_fft[:, :, :l], self.weights1[:, :, :l])
        x = torch.fft.irfft(out_ft, n=N)
        x = x.permute(0, 2, 1).view(B, N, c, k)
        return (x-torch.mean(x))/torch.std(x)








