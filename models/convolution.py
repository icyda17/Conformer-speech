import torch.nn as nn
from torch import Tensor
from typing import Tuple
import torch

class Conv2dSubSampling(nn.Module):
    """
    Convolutional 2D subsampling (to 1/4 length)
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
        Inputs: inputs
            - **inputs** (batch, time, dim): Tensor containing sequence of inputs
        Returns: outputs, output_lengths
            - **outputs** (batch, time, dim): Tensor produced by the convolution
            - **output_lengths** (batch): list of sequence output lengths
    """    
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        inputs = inputs.unsqueeze(1) # B * C(1) * T * D
        outputs = self.sequential(inputs) # B * C * T *D
        batch_size, channels, subsampled_lengths, sumsampled_dim = outputs.size()

        outputs = outputs.permute(0, 2, 1, 3) # B * T * C *D
        outputs = outputs.contiguous().view(batch_size, subsampled_lengths, channels * sumsampled_dim) # B * T * (C*D)

        output_lengths = input_lengths >> 2 # same as floor division by a power of 2 : input_lengths//2^2
        output_lengths -= 1

        return outputs, output_lengths

if __name__ == "__main__":
    test = Conv2dSubSampling(1, 3) 
    t = torch.randn(3,32,32) # B * T * Dim
    l = torch.randn(3) # B

    outputs, lengths = test(t, l)
    print(outputs.size()) # torch.Size([3, 7, 21])