import torch.nn as nn
import torch.nn.init as init

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
