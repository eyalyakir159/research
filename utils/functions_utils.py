
import pywt
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class wavelet_filter:
    def __init__(self,high_pass,low_pass,inverse_low_pass,inverse_high_pass):
        self.high_pass=high_pass
        self.low_pass=low_pass
        self.inverse_low_pass=inverse_low_pass
        self.inverse_high_pass=inverse_high_pass

def get_wavelet_filters():
    # Get a list of all available wavelets
    wavelets = pywt.wavelist()
    waveletarray = []
    # Loop through each wavelet
    for i,wavelet_name in enumerate(wavelets):
        try:
            # Check if the wavelet is from the 'shan' family and needs parameters
            if wavelet_name.startswith("shan") and "-" not in wavelet_name:
                #print(f"Deprecated 'shan' wavelet format: {wavelet_name}")
                # You can choose to skip it or handle it with default parameters
                continue

            # Attempt to create a discrete wavelet
            wavelet = pywt.Wavelet(wavelet_name)

            # Decomposition low-pass and high-pass filters
            dec_low_pass = torch.tensor(wavelet.dec_lo).to(device)
            dec_high_pass = torch.tensor(wavelet.dec_hi).to(device)

            # Reconstruction low-pass and high-pass filters
            rec_low_pass = torch.tensor(wavelet.rec_lo).to(device)
            rec_high_pass = torch.tensor(wavelet.rec_hi).to(device)
            waveletarray.append(wavelet_filter(dec_low_pass,dec_high_pass,rec_low_pass,rec_high_pass))
            # Print the wavelet name and its filters
            #print(f"{i} - Discrete Wavelet: {wavelet_name} ,lenght: {len(dec_high_pass)} , {len(dec_low_pass)==len(dec_high_pass)}")

        except ValueError:
            pass

    return waveletarray

