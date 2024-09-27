""" Full assembly of the parts to form the complete network """

from unet_parts import *
import torch
from layers.MultiWaveletCorrelation import FourierCrossAttentionW


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


class Unet_corr(nn.Module):
    def __init__(self, dmodel, n_classes, bilinear=False):
        self.unet = UNet(dmodel,n_classes,bilinear)
        self.attn = FourierCrossAttentionW(in_channels=1, out_channels=1, seq_len_q=None,
                                            seq_len_kv=None)
        self.lk = nn.Linear(dmodel,dmodel)
        self.lq = nn.Linear(dmodel,dmodel)
        self.lv = nn.Linear(dmodel,dmodel)
    def forward(self,k,v,q):

        k,v,q = self.lk(k),self.lv(v),self.lq(q)
        k = self.attn(k,v,q)
        k = self.unet(k)

        return k


#model = UNet(128,128)
data = torch.rand(64,128,96)
B,N,D = data.shape
data1 = torch.stft(data.permute(0, 2, 1).reshape(B * D, N),16,return_complex=True)

freq_bins = data1.shape[1]
time_steps = data1.shape[2]
data1 = data1.view(B, D, freq_bins, time_steps)

data2 = [torch.stft(data[:,:,x],16,return_complex=True) for x in range(data.shape[-1])]

#they are equal

data = torch.stft(data,N/8)
