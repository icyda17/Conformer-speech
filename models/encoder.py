import torch.nn as nn
from models.convolution import Conv2dSubSampling
from models.module import Linear, ResidualConnectionModule

class ConformerBlock(nn.Module):
    """
    Conformer block contains two Feed Forward modules sandwiching the Multi-Headed Self-Attention module
    and the Convolution module. This sandwich structure is inspired by Macaron-Net, which proposes replacing
    the original feed-forward layer in the Transformer block into two half-step feed-forward layers,
    one before the attention layer and one after.
    Args:
        encoder_dim (int, optional): Dimension of conformer encoder
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of conformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of conformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not
    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing input vector
    Returns: outputs
        - **outputs** (batch, time, dim): Tensor produces by conformer block.
    """    
    def __init__(self, half_step_residual: bool=True):
        super().__init__()
        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1  
        # TODO: continue https://github.com/sooftware/conformer/blob/aead2f267157726b088eb301207a64aa983b6cc2/conformer/encoder.py#L32
        self.sequential = nn.Sequential(
            ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            ),
            ResidualConnectionModule(
                module=MultiHeadedSelfAttentionModule(
                    d_model=encoder_dim,
                    num_heads=num_attention_heads,
                    dropout_p=attention_dropout_p,
                ),
            ),
            ResidualConnectionModule(
                module=ConformerConvModule(
                    in_channels=encoder_dim,
                    kernel_size=conv_kernel_size,
                    expansion_factor=conv_expansion_factor,
                    dropout_p=conv_dropout_p,
                ),
            ),
            ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            ),
            nn.LayerNorm(encoder_dim),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)

class ConformerEncoder(nn.Module):
    """
    Conformer encoder first processes the input with a convolution subsampling layer and then
    with a number of conformer blocks.
        Args:
            input_dim (int, optional): Dimension of input vector
            encoder_dim (int, optional): Dimension of conformer encoder
            num_layers (int, optional): Number of conformer blocks
            num_attention_heads (int, optional): Number of attention heads
            feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
            conv_expansion_factor (int, optional): Expansion factor of conformer convolution module
            feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
            attention_dropout_p (float, optional): Probability of attention module dropout
            conv_dropout_p (float, optional): Probability of conformer convolution module dropout
            conv_kernel_size (int or tuple, optional): Size of the convolving kernel
            half_step_residual (bool): Flag indication whether to use half step residual or not
        Inputs: inputs, input_lengths
            - **inputs** (batch, time, dim): Tensor containing input vector
            - **input_lengths** (batch): list of sequence input lengths
        Returns: outputs, output_lengths
            - **outputs** (batch, out_channels, time): Tensor produces by conformer encoder.
            - **output_lengths** (batch): list of sequence output lengths
    """

    def __init__(self,
                 input_dim: int = 80,
                 encoder_dim: int = 512,
                 input_dropout: float = 0.1):
        super().__init__()

        # Processing
        self.conv_subsample = Conv2dSubSampling(
            in_channels=1, out_channels=encoder_dim)
        self.linear = Linear(
            encoder_dim * (((input_dim - 1) // 2 - 1) // 2), encoder_dim)
        self.dropout = nn.Dropout(input_dropout)

        # encoder Module
        self.layers = None

    def forward(self, inputs):
        """
        Forward propagate a `inputs` for  encoder training.
            Args:
                inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                    `FloatTensor` of size ``(batch, seq_length, dimension)``.
                input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
            Returns:
                (Tensor, Tensor)
                * outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                    ``(batch, seq_length, dimension)``
                * output_lengths (torch.LongTensor): The length of output tensor. ``(batch)``
        """
        pass
