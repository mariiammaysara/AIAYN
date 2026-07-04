import torch
import torch.nn as nn
from transformer_core.layers.attention import MultiHeadSelfAttention
from transformer_core.layers.feedforward import PositionWiseFFN
from transformer_core.layers.normalization import AddNorm

class TransformerEncoderLayer(nn.Module):
    """
    A single Encoder Layer of the Transformer.
    
    This layer corresponds to one block in the left stack ("Encoder") of Figure 1
    in the original Transformer paper ("Attention Is All You Need").
    
    It consists of two main sub-layers:
    1. Multi-Head Self-Attention
    2. Position-Wise Feed-Forward Network
    
    Each of these sub-layers is wrapped in an AddNorm block using the Pre-LN residual pattern.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initializes the TransformerEncoderLayer.
        
        Args:
            d_model (int): Dimension of the model.
            n_heads (int): Number of attention heads.
            d_ff (int): Dimension of the feed-forward network's hidden layer.
            dropout (float): Dropout rate.
        """
        super().__init__()
        # Self-attention module
        self.self_attn = MultiHeadSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        # Feed-forward network module
        self.ffn = PositionWiseFFN(d_model=d_model, d_ff=d_ff, dropout=dropout)
        
        # Two AddNorm blocks wrapping the self-attention and feed-forward modules
        self.attn_add_norm = AddNorm(d_model=d_model, dropout=dropout)
        self.ffn_add_norm = AddNorm(d_model=d_model, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Performs the forward pass of the TransformerEncoderLayer.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, seq_len, d_model).
            mask (torch.Tensor, optional): Encoder padding mask to prevent attention to padding tokens.
                                          Shape: (batch_size, 1, 1, seq_len) or broadcastable.
                                          
        Returns:
            torch.Tensor: The output tensor with shape (batch_size, seq_len, d_model).
        """
        # 1. Multi-Head Self-Attention sub-layer with Pre-LN AddNorm wrapper.
        # We pass a lambda to AddNorm because Self-Attention's forward signature takes (q, k, v, mask),
        # but AddNorm's forward method expects a callable of a single argument.
        x = self.attn_add_norm(x, lambda y: self.self_attn(query=y, key=y, value=y, mask=mask))
        
        # 2. Position-Wise Feed-Forward sub-layer with Pre-LN AddNorm wrapper.
        x = self.ffn_add_norm(x, self.ffn)
        
        return x
