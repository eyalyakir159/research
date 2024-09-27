import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.complex_mul_layer import ComplexMul
from layers.Window_Layer import WindowLayer

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.embed_size = configs.embed_size #embed_size
        self.hidden_size = configs.hidden_size #hidden_size
        self.pre_length = configs.pred_len
        self.feature_size = configs.enc_in #channels
        self.seq_length = configs.seq_len
        self.channel_independence = configs.channel_independence
        self.sparsity_threshold = 0.01
        self.scale = 0.02
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.TopM = configs.TopM

        self.hide = configs.hide
        assert self.hide in ['None' ,'Rinput' , 'Iinput' , 'RWeight' , 'IWeight' ,'MIXR','MIXI'] , f"hide should be in {['None' ,'Rinput' , 'Iinput' , 'RWeight' , 'IWeight','MIXR','MIXI']} "
        self.hop_length = configs.hop_length
        self.n_fft = configs.n_fft
        self.number_of_windows = self.seq_length//self.hop_length +1

        assert self.TopM <= (self.n_fft//2) +1, f"Top M should be lower or equal to the amount of frequnecys = {(self.n_fft//2) +1}"
        assert self.seq_length%configs.hop_length==0, "hop lengh should be a divodor of seq lenght"
        assert self.seq_length%configs.n_fft==0, "nfft should be a divodor of seq lenght"



        self.window_layers = torch.nn.ModuleList([WindowLayer(self.embed_size,self.sparsity_threshold,self.scale,self.hide) for i in range(self.number_of_windows)])





        self.fc = nn.Sequential(
            nn.Linear(self.seq_length * self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pre_length)
        )

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        B, T, N = x.shape

        x = self.tokenEmb(x)
        bias = x
        #x_f = torch.fft.rfft(x, dim=2, norm='ortho') # FFT on L dimension
        x = self.MLP_temporal(x)
        x = x + bias
        x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)
        return x

    # dimension extension
    def tokenEmb(self, x):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        # N*T*1 x 1*D = N*T*D
        y = self.embeddings
        return x * y

    # frequency temporal learner


    def MLP_temporal(self, x):
        x = self.SFT(x, n_fft=self.n_fft,hop_length=self.hop_length)
        original_size = x.shape

        if self.TopM!=0:
            x,indices = self.SelectMFrequencies(x)

        x_new = []
        for i in range(self.number_of_windows):
            currect_layer = self.window_layers[i]
            currect_window = x.select(2,i)
            prev_window = x.select(2, i - 1) if i != 0 else None
            next_layer = x.select(2, i + 1) if i < (self.number_of_windows-1) else None
            x_new.append(currect_layer(mid=currect_window, right=next_layer, left=prev_window))
        # preform caculations
        x = torch.stack(x_new,axis=2)

        if self.TopM!=0:
            x = self.pad(x,indices,original_size)

        x = self.ISFT(x,n_fft=self.n_fft,hop_length=self.hop_length)
        return x

    def SFT(self, x, n_fft=48, hop_length=32):

        # preform cwt by dim 2
        # B D N T
        B, D, N, T = x.shape
        x = x.permute(0, 1, 3, 2)
        # [B,D,T,N]
        x = x.reshape(-1, N)
        window = torch.hann_window(n_fft).to(x.device)

        x_in_batchs = torch.split(x, 50 * 1000, dim=0)
        stft_batchs = []
        for b in x_in_batchs:
            bsftf = torch.stft(b, n_fft=n_fft, return_complex=True,
                               hop_length=hop_length, onesided=True,
                               window=window)
            stft_batchs.append(bsftf)

        # Concatenate the results along dimension 0 instead of stacking
        x = torch.cat(stft_batchs, dim=0)

        # window options are ones, hann , hamming, blackman,bartlett,kaiser,gaussian,nuttal,parzen,bohman,triangular.


        x = x.reshape(B, D, T, x.shape[1], x.shape[2])
        # B D N C T
        x = x.transpose(2, 4)
        # B C N D T
        if self.hide in ['Rinput','MIXR']:
            x.real = 0
        if self.hide==['Iinput','MIXI']:
            x.imag = 0


        return x

    def SelectMFrequencies(self, x):
        # x is of size B, D, W, F, E

        # Get the absolute values of x (we only use this to find the top M frequencies)
        amps = torch.abs(x)

        # Get the indices of the Top M values along the frequency dimension (F)
        _, indices = torch.topk(amps, self.TopM, dim=3)  # Top M in the F dimension

        # Gather the original x values (not amplitudes) at the positions of the top M frequencies
        top_frequencies = torch.gather(x, dim=3, index=indices)

        return top_frequencies,indices

    def pad(self, top_frequencies, indices, original_size):
        """
        Restores the original tensor size by placing the top frequencies at their original positions
        and padding zeros in the non-top-M positions.

        Parameters:
        top_frequencies (torch.Tensor): Tensor containing only the top-M frequencies.
        indices (torch.Tensor): Indices of the top-M frequencies.
        original_size (torch.Size): The original size of x before reduction.

        Returns:
        torch.Tensor: Tensor restored to the original size with top-M frequencies in place and zeros elsewhere.
        """
        # Create an empty tensor of the original size, initialized to zeros
        padded_x = torch.zeros(original_size, device=top_frequencies.device, dtype=top_frequencies.dtype)

        # Expand the indices to match the embedding size (E dimension)

        # Scatter the top frequencies back into the padded tensor at the original positions
        padded_x.scatter_(dim=3, index=indices, src=top_frequencies)

        return padded_x

    def ISFT(self,wx,n_fft=48,hop_length=32):
        B, D, N, C, T = wx.shape
        # B D T C N
        wx = wx.transpose(2, 4)
        # B*D*T C N
        wx = wx.reshape(-1, C, N)
        x = torch.istft(wx,n_fft,hop_length=hop_length,window=torch.hann_window(n_fft, device=wx.device),onesided=True)
        x = x.reshape(B,D,T,-1)
        x = x.transpose(2,3)
        return x
    def get_needed_weights(self):
        weights = {f'window_{i}': self.window_layers[i] for i in range(len(self.window_layers))}
        return weights





class Configs(object):
    enc_in = 7
    dec_in = 7
    seq_len = 96
    pred_len = 96
    channel_independence = '1'
    embed_size = 128
    hop_length = 8
    hidden_size = 256
    n_fft = 24
    TopM = 6



if False:
    m = Model(Configs())
    # Define the dimensions
    B, D, N = 64, 7, 96
    random_tensor = torch.randn(B, N, D)
    q = m(random_tensor)
