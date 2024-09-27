import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.HyperComplexMul import OctMLP,SedMLP,QuatMLP

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
        self.TopM = configs.TopM
        self.HyperComplexMode = configs.HyperComplexMode
        assert self.HyperComplexMode in ["OctMLP","QuatMLP","SedMLP"], f"HyperComplexMode should be in {['OctMLP','QuatMLP','SedMLP']}"
        HyperComplexLayerDict = {"OctMLP":OctMLP,
                        "QuatMLP":QuatMLP,
                        "SedMLP":SedMLP
        }

        HyperComplexLayerClass = HyperComplexLayerDict[self.HyperComplexMode]
        self.HyperComplexLayer = HyperComplexLayerClass(self.scale,self.embed_size)

        if self.HyperComplexMode == "QuatMLP":
            self.number_of_windows = 2
        elif self.HyperComplexMode == "OctMLP":
            self.number_of_windows = 4
        elif self.HyperComplexMode == "SedMLP":
            self.number_of_windows = 8
        else:
            raise ValueError(f"Unknown HyperComplexMode: {self.HyperComplexMode}")

        window_size = 2 * self.seq_length // (self.number_of_windows + 1)  # Calculate window size based on P windows with 50% overlap
        loss = int(self.seq_length - (2*(window_size//2) + (window_size//2)*(self.number_of_windows-1)))
        print(f"loss in FFT {loss}")

        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))





        self.fc = nn.Sequential(
            nn.Linear((self.seq_length-loss) * self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pre_length)
        )

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
        x = self.SFT1(x)
        original_size = x.shape

        if self.TopM != 0:
            x, indices = self.SelectMFrequencies(x)

        # preform caculations
        x = torch.stack(self.HyperComplexLayer(x),axis=2)

        if self.TopM != 0:
            x = self.pad(x, indices, original_size)

        x = self.ISFT1(x)
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

    def split_sequence_along_axis(self,x, axis=-2):
        N = x.shape[axis]  # Size along the axis where you want to apply windowing
        window_size = 2 * N // (self.number_of_windows + 1)  # Calculate window size based on P windows with 50% overlap
        hop_length = window_size // 2  # 50% overlap

        # Create a list of slices for each axis
        idx = [slice(None)] * len(x.shape)

        # Compute the number of steps to get exactly P windows
        windows = []
        start_indices = range(0, (self.number_of_windows* hop_length), hop_length)  # Start indices for exactly P windows

        for start in start_indices:
            # Set the slicing for the specified axis (-2 in this case)
            idx[axis] = slice(start, start + window_size)

            # Apply the slice along the chosen axis
            window = x[tuple(idx)]
            windows.append(window)
        return windows

    def SFT1(self,x):
        B, D, N, T = x.shape
        windows = self.split_sequence_along_axis(x)
        fft_windows =torch.stack([torch.fft.rfft(w,axis=-2) for w in windows],axis=2)


        return fft_windows

    def ISFT1(self, x):
        B, D, C, N, T = x.shape

        # Perform inverse FFT
        ifft_windows = torch.fft.irfft(x, axis=-2)

        # Determine the size of each window
        window_size = ifft_windows.shape[-2]

        # Initialize a list to store the reconstructed signal
        rec_signal = []

        # Start with the first half of the first window
        rec_signal.append(ifft_windows[:, :, 0, :window_size // 2, :])

        # Iterate over all windows and average the overlapping regions
        for i in range(1, C):
            current_window = ifft_windows[:, :, i, :window_size // 2, :]  # First half of the current window
            prev_window = ifft_windows[:, :, i - 1, window_size // 2:, :]  # Second half of the previous window

            # Average the overlapping parts of the current and previous windows
            avg_windows = (current_window + prev_window) / 2

            # Append the averaged result and the non-overlapping part of the current window
            rec_signal.append(avg_windows)

        # Append the second half of the last window (not averaged with any other window)
        rec_signal.append(ifft_windows[:, :, -1, window_size // 2:, :])

        # Stack the list of reconstructed signal parts into a single tensor
        return torch.cat(rec_signal, dim=-2)

    def SFT(self, x):


        # preform cwt by dim 2
        # B D N T
        B, D, N, T = x.shape
        x = x.permute(0, 1, 3, 2)
        # [B,D,T,N]
        x = x.reshape(-1, N)
        window = torch.hann_window(self.nfft).to(x.device)

        x_in_batchs = torch.split(x, 50 * 1000, dim=0)
        stft_batchs = []
        for b in x_in_batchs:
            bsftf = torch.stft(b, n_fft=self.nfft, return_complex=True,
                               hop_length=self.hop_length, onesided=True,
                               window=window)
            stft_batchs.append(bsftf)

        # Concatenate the results along dimension 0 instead of stacking
        x = torch.cat(stft_batchs, dim=0)

        # window options are ones, hann , hamming, blackman,bartlett,kaiser,gaussian,nuttal,parzen,bohman,triangular.


        x = x.reshape(B, D, T, x.shape[1], x.shape[2])
        # B D N C T
        x = x.transpose(2, 4)
        # B C N D T

        return x

    def ISFT(self,wx):
        B, D, N, C, T = wx.shape
        # B D T C N
        wx = wx.transpose(2, 4)
        # B*D*T C N
        wx = wx.reshape(-1, C, N)
        x = torch.istft(wx,self.nfft,hop_length=self.hop_length,window=torch.hann_window(self.nfft, device=wx.device),onesided=True)
        x = x.reshape(B,D,T,-1)
        x = x.transpose(2,3)
        return x

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        B, T, N = x.shape

        x = self.tokenEmb(x)
        x = self.MLP_temporal(x)
        x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)
        return x







class Configs(object):
    enc_in = 7
    dec_in = 7
    seq_len = 96
    pred_len = 96
    channel_independence = '1'
    embed_size = 32
    hidden_size = 256
    TopM = 6
    HyperComplexMode = "SedMLP"


if False:
    m = Model(Configs())


    # Define the dimensions
    B, D, N = 64, 7, 96
    random_tensor = torch.randn(B, N, D)
    q = m(random_tensor)
    print("done and working")



