import torch
import torch.nn as nn
from typing import Callable

class LayerNorm(nn.Module):
    """
    Manually computes Layer Normalization over the last dimension (d_model).
    
    Formula:
        y = (x - mean) / sqrt(var + eps) * gamma + beta
    where gamma (scale) and beta (shift) are learnable parameters.
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Initializes the manual LayerNorm layer.
        
        Args:
            d_model (int): Dimension of the input vectors.
            eps (float): Small constant for numerical stability during normalization.
        """
        super().__init__()
        self.eps = eps
        # Learnable scale (gamma) and shift (beta) parameter vectors
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies layer normalization to the input tensor.
        """
        # Calculate mean and variance over the last dimension (dim=-1)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        
        # Standardize and rescale
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class AddNorm(nn.Module):
    """
    Implements the residual connection with Layer Normalization using the Pre-LN pattern.
    
    Pre-Norm vs. Post-Norm:
    - Original Paper (Post-LN): Applies normalization *after* the sublayer and addition:
      x = LayerNorm(x + Dropout(SubLayer(x)))
    - Pre-LN (This implementation): Applies normalization *before* passing to the sublayer:
      x = x + Dropout(SubLayer(LayerNorm(x)))
      
    Why Pre-LN trains more stably:
    In Post-LN, the gradient of the parameters at early layers passes through normalization layers
    which tend to scale down gradients recursively, leading to vanishing or unstable gradients.
    Pre-LN establishes an unobstructed identity mapping path from inputs to outputs (the residual shortcut).
    Gradients can flow directly through this shortcut to earlier layers, which vastly stabilizes optimization,
    allowing for higher learning rates and reducing the necessity of a warm-up training phase.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, eps: float = 1e-6):
        """
        Initializes the AddNorm block with manual LayerNorm.
        
        Args:
            d_model (int): Dimension of the input vectors.
            dropout (float): Dropout probability applied to the sublayer output.
            eps (float): Small constant for numerical stability during normalization.
        """
        super().__init__()
        self.norm = LayerNorm(d_model, eps)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        Performs the forward pass using the Pre-LN residual connection.
        
        Formula:
            y = x + Dropout(SubLayer(LayerNorm(x)))
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, seq_len, d_model).
            sublayer (Callable): A callable submodule/function (e.g. self-attention or FFN).
            
        Returns:
            torch.Tensor: The output tensor with shape (batch_size, seq_len, d_model).
        """
        return x + self.dropout(sublayer(self.norm(x)))
