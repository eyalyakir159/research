import torch
import torch.nn as nn
class ComplexMul(nn.Module):
    def __init__(self, embed_size=128,sparsity_threshold=0.01,scale=0.02,hide = False):
        super(ComplexMul, self).__init__()
        self.hide = hide
        self.embed_size = embed_size #embed_size
        self.sparsity_threshold = sparsity_threshold
        self.scale = scale
        self.first_pass = True

        if self.hide in ["RWeight","MIXR"]:
            self.yr = torch.zeros(embed_size, embed_size)
            self.yi = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))

        elif self.hide in ["IWeight","MIXI"]:
            self.yr = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
            self.yi = torch.zeros(embed_size, embed_size)
        else:
            self.yr = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
            self.yi = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))



    def forward(self, x,conj = False):
        if self.first_pass:
            self.yr = self.yr.to(x.device)
            self.yi = self.yi.to(x.device)
            self.first_pass = False
        x_R = x.real
        x_I = x.imag

        yr = self.yr
        yi = self.yi


        minus = 1 if not conj else -1

        # Perform complex multiplication using Einstein summation notation
        real_part = torch.einsum('...t,td->...t', x_R, yr) - torch.einsum('...t,td->...t', x_I,minus*yi)
        imag_part = torch.einsum('...t,td->...t', x_R, minus * yi) + torch.einsum('...t,td->...t', x_I, yr)

        y = torch.stack([real_part, imag_part], dim=-1)
        y = torch.nn.functional.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y
