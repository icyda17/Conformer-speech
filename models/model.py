import torch.nn as nn

class Conformer(nn.Module):
    """
     Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **input_lengths** (batch): list of sequence input lengths
    Returns: outputs, output_lengths
        - **outputs** (batch, out_channels, time): Tensor produces by conformer.
        - **output_lengths** (batch): list of sequence output lengths
    """
    def __init__(self):
        super().__init__()
        self.encoder = None
        self.decoder = None
        self.fc = None

    def forward(self):
        pass
    