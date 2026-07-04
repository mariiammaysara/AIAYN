import torch
import torch.nn as nn

class PositionWiseFFN(nn.Module):
    """
    Implements the Position-Wise Feed-Forward Network (FFN).
    
    As described in the paper, this block consists of two linear transformations
    with a ReLU activation and dropout in between:
    FFN(x) = max(0, xW1 + b1)W2 + b2
    
    The network is applied to each position separately and identically.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Initializes the PositionWiseFFN.
        
        Args:
            d_model (int): Input and output dimension of the network.
            d_ff (int): Hidden layer dimension.
            dropout (float): Dropout probability for intermediate representations.
        """
        super().__init__()
        # First linear transformation projects d_model to a higher dimension d_ff
        self.w_1 = nn.Linear(d_model, d_ff)
        # Activation function
        self.activation = nn.ReLU()
        # Dropout applied to the intermediate representation
        self.dropout = nn.Dropout(p=dropout)
        # Second linear transformation projects back to d_model
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the PositionWiseFFN.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, seq_len, d_model).
            
        Returns:
            torch.Tensor: The output tensor with shape (batch_size, seq_len, d_model).
        """
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
