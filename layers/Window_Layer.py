import torch
import torch.nn as nn
from layers.complex_mul_layer import ComplexMul


class WindowLayer(nn.Module):
    def __init__(self, embed_size,sparsity_threshold,scale,hide = 'None'):
        #hide in [None / RWeight / IWeight]
        super(WindowLayer, self).__init__()
        self.hide = hide
        self.embed_size = embed_size #embed_size
        self.sparsity_threshold = sparsity_threshold
        self.scale = scale
        self.first_pass = True

        self.SelfProjection = ComplexMul(self.embed_size,self.sparsity_threshold,self.scale,self.hide)
        self.RightProjection = ComplexMul(self.embed_size,self.sparsity_threshold,self.scale,self.hide)
        self.LeftProjection = ComplexMul(self.embed_size,self.sparsity_threshold,self.scale,self.hide)

        if self.hide in ["RWeight","MIXR"]:
            self.BiasR = torch.zeros(embed_size)
            self.BiasI = nn.Parameter(self.scale * torch.randn(self.embed_size))
        elif self.hide in ["IWeight","MIXI"]:
            self.BiasI = torch.zeros(embed_size)
            self.BiasR = nn.Parameter(self.scale * torch.randn(self.embed_size))
        else:
            self.BiasR = nn.Parameter(self.scale * torch.randn(self.embed_size))
            self.BiasI = nn.Parameter(self.scale * torch.randn(self.embed_size))

    def forward(self, mid, right=None, left=None):
        if self.first_pass:
            self.BiasR = self.BiasR.to(mid.device)
            self.BiasI = self.BiasI.to(mid.device)
            self.first_pass = False

        # Always apply SelfProjection to mid
        mid_projected = self.SelfProjection(mid, conj=False)

        # If right or left are None, set them to a tensor of zeros with the same shape as mid
        if right is not None:
            right_projected = self.RightProjection(right, conj=True)
        else:
            right_projected = torch.zeros_like(mid)

        if left is not None:
            left_projected = self.LeftProjection(left, conj=True)
        else:
            left_projected = torch.zeros_like(mid)

        # Sum the projections (assuming they are complex tensors)
        total_sum = mid_projected + right_projected + left_projected

        # Add the real and imaginary bias separately
        # Split the total sum into real and imaginary parts
        total_real = total_sum.real
        total_imag = total_sum.imag

        # Add real and imaginary biases
        total_real = total_real + self.BiasR.to(mid.device)
        total_imag = total_imag + self.BiasI.to(mid.device)

        # Combine the real and imaginary parts back into a complex tensor
        y = torch.stack([total_real, total_imag], dim=-1)
        y = torch.nn.functional.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y

