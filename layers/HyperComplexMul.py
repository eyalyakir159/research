import torch
import torch.nn as nn
from layers.complex_mul_layer import ComplexMul
import torch.nn.functional as F

sparsity_threshold = 0.01

def complexMul(x, y):
    """
    Multiplies two complex tensors `x` and `y`.

    Parameters:
    x (torch.Tensor): A complex tensor of shape (a, b, c, d), where the last dimension represents
                      the complex components (real and imaginary) of the tensor.
    y (torch.Tensor): A complex tensor or matrix of shape (d, d), where the last two dimensions
                      represent the complex components of the matrix.

    Returns:
    torch.Tensor: A complex tensor of the same shape as `x`, representing the product of the complex
                  numbers in the last dimension.
    """
    # Extract real and imaginary parts of x and y
    x_R = x.real
    x_I = x.imag
    y_R = y[0]
    y_I = y[1]

    # Perform complex multiplication using Einstein summation notation
    real_part = torch.einsum('...t,td->...t', x_R, y_R) - torch.einsum('...t,td->...t', x_I, y_I)
    imag_part = torch.einsum('...t,td->...t', x_R, y_I) + torch.einsum('...t,td->...t', x_I, y_R)
    real_part = F.softshrink(real_part, lambd=sparsity_threshold)
    imag_part = F.softshrink(imag_part, lambd=sparsity_threshold)

    y = torch.stack([real_part, imag_part], dim=-1)
    y = torch.view_as_complex(y)

    return y


def add_bias(x, biasR, biasI):
    total_real = x.real
    total_imag = x.imag

    # Add real and imaginary biases
    total_real = total_real + biasR
    total_imag = total_imag + biasI

    # Combine the real and imaginary parts back into a complex tensor
    y = torch.stack([total_real, total_imag], dim=-1)
    y = torch.nn.functional.softshrink(y, lambd=sparsity_threshold)
    y = torch.view_as_complex(y)
    return y



class OctMLP(nn.Module):
    def __init__(self,scale,embed_size):
        super(OctMLP, self).__init__()
        self.scale =  scale
        self.embed_size = embed_size

        self.y1r = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.y1i = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))

        self.BiasR1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.BiasI1 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        self.y2r = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.y2i = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))

        self.BiasR2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.BiasI2 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        self.y3r = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.y3i = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))

        self.BiasR3 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.BiasI3 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        self.y4r = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.y4i = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))

        self.BiasR4 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.BiasI4 = nn.Parameter(self.scale * torch.randn(self.embed_size))


    def forward(self,x):
        """
            Multiply two octonions represented by four complex numbers each.

            Parameters:
            x (torch.Tensor): A tensor of shape (..., 4, ...) where the last 4 dimensions represent
                              four complex numbers (each complex number is represented by two real numbers).
            y (torch.Tensor): A tensor of shape (..., 4, ...) where the last 4 dimensions represent
                              four complex numbers (each complex number is represented by two real numbers).

            Returns:
            torch.Tensor: A tensor of the same shape as the input tensors, representing the product of two octonions.

            Octonions are treated as (z1, z2, z3, z4), where each zi is a complex number, and the
            multiplication follows the Cayley-Dickson construction.
            """
        assert x.shape[2]==4 , "STFT output should have 4 context windows for OctMLP"
        # Extract the four complex components of x and y
        x1 = x.select(2, 0)  # z1 from x
        x2 = x.select(2, 1)  # z2 from x
        x3 = x.select(2, 2)  # z3 from x
        x4 = x.select(2, 3)  # z4 from x

        a1 = complexMul(x1, (self.y1r, self.y1i)) - complexMul(x2, (self.y2r, -self.y2i)) - complexMul(
            x3, (self.y3r, -self.y3i)) - complexMul(x4, (self.y4r, -self.y4i))

        a2 = complexMul(x2, (self.y1r, self.y1i)) + complexMul(x1, (self.y2r, self.y2i)) + complexMul(x4,
                                                                                                      (
                                                                                                          self.y3r,
                                                                                                          -self.y3i)) - complexMul(
            x3, (self.y4r, -self.y4i))

        a3 = complexMul(x3, (self.y1r, self.y1i)) - complexMul(x4, (self.y2r, -self.y2i)) + complexMul(
            x1, (self.y3r, self.y3i)) + complexMul(x2, (self.y4r, -self.y4i))

        a4 = complexMul(x4, (self.y1r, self.y1i)) + complexMul(x3, (self.y2r, -self.y2i)) - complexMul(
            x2, (self.y3r, -self.y3i)) + complexMul(x1, (self.y4r, self.y4i))

        return add_bias(a1, self.BiasR1, self.BiasI1), add_bias(a2, self.BiasR2, self.BiasI2), add_bias(
            a3
            , self.BiasR3
            , self.BiasI3), add_bias \
            (a4, self.BiasR4, self.BiasI4)




class QuatMLP(nn.Module):
    def __init__(self,scale,embed_size):
        super(QuatMLP, self).__init__()

        self.scale = scale
        self.embed_size = embed_size

        self.y1r = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.y1i = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))

        self.BiasR1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.BiasI1 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        self.y2r = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.y2i = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))

        self.BiasR2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.BiasI2 = nn.Parameter(self.scale * torch.randn(self.embed_size))


    def forward(self,x):
        """
            Multiply two quaternions represented by 2 complex numbers each.

            Parameters:
            x (torch.Tensor): A tensor of shape (..., 2, ...) where the last 4 dimensions represent
                              four complex numbers (each complex number is represented by two real numbers).
            y (torch.Tensor): A tensor of shape (..., 2, ...) where the last 4 dimensions represent
                              four complex numbers (each complex number is represented by two real numbers).

            Returns:
            torch.Tensor: A tensor of the same shape as the input tensors, representing the product of two quaternions.

            Octonions are treated as (z1, z2), where each zi is a complex number, and the
            multiplication follows the Cayley-Dickson construction.
            """
        assert x.shape[2]==2 , "STFT output should have 2 context windows for QuatMLP"
        # Extract the four complex components of x and y
        x1 = x.select(2, 0)  # z1 from x
        x2 = x.select(2, 1)  # z2 from x


        a1 = complexMul(x1,[self.y1r,self.y1i])-complexMul(x2.conj(),[self.y2r,self.y2i])
        a2 = complexMul(x2, [self.y1r, -self.y1i]) + complexMul(x1, [self.y2r, self.y2i])

        return add_bias(a1,self.BiasR1,self.BiasI1),add_bias(a2,self.BiasR2,self.BiasI2)


class SedMLP(nn.Module):
    def __init__(self,scale,embed_size):
        self.scale = scale
        self.embed_size = embed_size

        self.yr = [nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size)) for i in range(8)]
        self.yi = [nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size)) for i in range(8)]
        self.BiasI = [nn.Parameter(self.scale * torch.randn(self.embed_size)) for i in range(8)]
        self.BiasR = [nn.Parameter(self.scale * torch.randn(self.embed_size)) for i in range(8)]
        super(SedMLP, self).__init__()



    def forward(self,x):
        """
            Multiply two Sedenions represented by 8 complex numbers each.

            Parameters:
            x (torch.Tensor): A tensor of shape (..., 8, ...) where the last 4 dimensions represent
                              four complex numbers (each complex number is represented by two real numbers).
            y (torch.Tensor): A tensor of shape (..., 8, ...) where the last 4 dimensions represent
                              four complex numbers (each complex number is represented by two real numbers).

            Returns:
            torch.Tensor: A tensor of the same shape as the input tensors, representing the product of two Sedenions.

            Sedenions are treated as (z1, z2, z3, z4...z8), where each zi is a complex number, and the
            multiplication follows the Cayley-Dickson construction.
            """
        assert x.shape[2]==8 , "STFT output should have 4 context windows for OctMLP"
        x0 = x.select(2, 0)
        x1 = x.select(2, 1)
        x2 = x.select(2, 2)
        x3 = x.select(2, 3)
        x4 = x.select(2, 4)
        x5 = x.select(2, 5)
        x6 = x.select(2, 6)
        x7 = x.select(2, 7)

        # Compute the first row
        c1 = (complexMul(x0, [self.yr[0], self.yi[0]]) - complexMul(x1, [self.yr[1], -self.yi[1]])
              - complexMul(x2, [self.yr[2], -self.yi[2]]) - complexMul(x3, [self.yr[3], -self.yi[3]])
              - complexMul(x4, [self.yr[4], -self.yi[4]]) - complexMul(x5, [self.yr[5], -self.yi[5]])
              - complexMul(x6, [self.yr[6], -self.yi[6]]) - complexMul(x7, [self.yr[7], -self.yi[7]]))

        # Compute the second row
        c2 = (complexMul(x0, [self.yr[1], self.yi[1]]) + complexMul(x1, [self.yr[0], self.yi[0]])
              + complexMul(x2, [self.yr[3], -self.yi[3]]) - complexMul(x3, [self.yr[2], -self.yi[2]])
              - complexMul(x4, [self.yr[5], -self.yi[5]]) + complexMul(x5, [self.yr[4], self.yi[4]])
              + complexMul(x6, [self.yr[7], -self.yi[7]]) - complexMul(x7, [self.yr[6], -self.yi[6]]))

        # Compute the third row
        c3 = (complexMul(x0, [self.yr[2], self.yi[2]]) - complexMul(x1, [self.yr[3], -self.yi[3]])
              + complexMul(x2, [self.yr[0], self.yi[0]]) + complexMul(x3, [self.yr[1], self.yi[1]])
              - complexMul(x4, [self.yr[6], -self.yi[6]]) + complexMul(x5, [self.yr[7], -self.yi[7]])
              + complexMul(x6, [self.yr[4], self.yi[4]]) - complexMul(x7, [self.yr[5], -self.yi[5]]))

        # Compute the fourth row
        c4 = (complexMul(x0, [self.yr[3], self.yi[3]]) + complexMul(x1, [self.yr[2], -self.yi[2]])
              - complexMul(x2, [self.yr[1], -self.yi[1]]) + complexMul(x3, [self.yr[0], self.yi[0]])
              - complexMul(x4, [self.yr[7], -self.yi[7]]) + complexMul(x5, [self.yr[6], -self.yi[6]])
              - complexMul(x6, [self.yr[5], -self.yi[5]]) + complexMul(x7, [self.yr[4], self.yi[4]]))

        # Compute the fifth row
        c5 = (complexMul(x0, [self.yr[4], self.yi[4]]) - complexMul(x1, [self.yr[5], -self.yi[5]])
              - complexMul(x2, [self.yr[6], -self.yi[6]]) - complexMul(x3, [self.yr[7], -self.yi[7]])
              + complexMul(x4, [self.yr[0], self.yi[0]]) + complexMul(x5, [self.yr[1], self.yi[1]])
              + complexMul(x6, [self.yr[2], self.yi[2]]) + complexMul(x7, [self.yr[3], self.yi[3]]))

        # Compute the sixth row
        c6 = (complexMul(x0, [self.yr[5], self.yi[5]]) + complexMul(x1, [self.yr[4], self.yi[4]])
              - complexMul(x2, [self.yr[7], -self.yi[7]]) + complexMul(x3, [self.yr[6], -self.yi[6]])
              - complexMul(x4, [self.yr[1], -self.yi[1]]) + complexMul(x5, [self.yr[0], self.yi[0]])
              + complexMul(x6, [self.yr[3], self.yi[3]]) - complexMul(x7, [self.yr[2], -self.yi[2]]))

        # Compute the seventh row
        c7 = (complexMul(x0, [self.yr[6], self.yi[6]]) + complexMul(x1, [self.yr[7], -self.yi[7]])
              + complexMul(x2, [self.yr[4], self.yi[4]]) - complexMul(x3, [self.yr[5], -self.yi[5]])
              - complexMul(x4, [self.yr[2], -self.yi[2]]) + complexMul(x5, [self.yr[3], self.yi[3]])
              + complexMul(x6, [self.yr[0], self.yi[0]]) - complexMul(x7, [self.yr[1], -self.yi[1]]))

        # Compute the eighth row
        c8 = (complexMul(x0, [self.yr[7], self.yi[7]]) - complexMul(x1, [self.yr[6], -self.yi[6]])
              + complexMul(x2, [self.yr[5], self.yi[5]]) - complexMul(x3, [self.yr[4], -self.yi[4]])
              - complexMul(x4, [self.yr[3], -self.yi[3]]) + complexMul(x5, [self.yr[2], -self.yi[2]])
              - complexMul(x6, [self.yr[1], -self.yi[1]]) + complexMul(x7, [self.yr[0], self.yi[0]]))

        # Return the result as a tuple of all rows
        return add_bias(c1,self.BiasI[0],self.BiasR[0]) ,add_bias(c2,self.BiasI[1],self.BiasR[1]),add_bias(c3,self.BiasI[2],self.BiasR[2])\
            ,add_bias(c4,self.BiasI[3],self.BiasR[3]),add_bias(c5,self.BiasI[4],self.BiasR[4]),add_bias(c6,self.BiasI[5],self.BiasR[5]),\
            add_bias(c7,self.BiasI[6],self.BiasR[6]),add_bias(c8,self.BiasI[7],self.BiasR[7]),


