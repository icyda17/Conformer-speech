import torch.nn as nn
import torch.nn.init as init
from torch import Tensor

class Linear(nn.Module):
    """
    Wrapper class of torch.nn.Linear
        Weight initialize by xavier initialization and bias initialize to zeros.
    """    
    def __init__(self, in_channels:int, out_channels:int, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)
    
    def forward(self, input):
        output = self.linear(input)
        return output

class ResidualConnectionModule(nn.Module):
    """
    Residual Connection Module.
    outputs = (module(inputs) x module_factor + inputs x input_factor)
    """
    def __init__(self, module: nn.Module, module_factor: float = 1.0, input_factor: float = 1.0):
        super(ResidualConnectionModule, self).__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor

    def forward(self, inputs: Tensor) -> Tensor:
        return (self.module(inputs) * self.module_factor) + (inputs * self.input_factor)

class Transpose(nn.Module):
    """ Wrapper class of torch.transpose() for Sequential module. """
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)