import torch
import torch.nn as nn
from transformer_core.layers.attention import MultiHeadSelfAttention
from transformer_core.layers.feedforward import PositionWiseFFN
from transformer_core.layers.normalization import AddNorm

class TransformerDecoderLayer(nn.Module):
    """
    A single Decoder Layer of the Transformer.
    
    This layer corresponds to one block in the right stack ("Decoder") of Figure 1
    in the original Transformer paper ("Attention Is All You Need").
    
    It consists of three main sub-layers:
    1. Masked Multi-Head Self-Attention (prevents attending to future target positions)
    2. Multi-Head Cross-Attention (attends to encoder representation outputs)
    3. Position-Wise Feed-Forward Network
    
    Each sub-layer is wrapped in a Pre-LN AddNorm block.
    
    Causal Mask vs. Padding Mask Explanation for Interviews:
    - Padding Mask:
      * Purpose: Prevents the model from attending to meaningless [PAD] tokens added
        to equalize sequence lengths within a batch.
      * Used in: Encoder self-attention and Decoder cross-attention (where we prevent the
        decoder from attending to source sentence padding tokens).
      * Nature: Bi-directional. Active tokens can attend forward or backward.
    - Causal Mask:
      * Purpose: Prevents the model from "looking into the future" during auto-regressive
        generation. During training, we feed the entire target sequence at once for speed,
        but we must mask out future tokens so the model only relies on past predictions.
      * Used in: Decoder self-attention only.
      * Nature: Uni-directional (lower-triangular). Position (i, j) is masked if j > i.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initializes the TransformerDecoderLayer.
        
        Args:
            d_model (int): Dimension of the model.
            n_heads (int): Number of attention heads.
            d_ff (int): Dimension of the feed-forward network's hidden layer.
            dropout (float): Dropout rate.
        """
        super().__init__()
        # 1. Masked self-attention (target sequence)
        self.self_attn = MultiHeadSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        # 2. Encoder-decoder cross-attention
        self.cross_attn = MultiHeadSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        # 3. Position-wise feed-forward network
        self.ffn = PositionWiseFFN(d_model=d_model, d_ff=d_ff, dropout=dropout)
        
        # AddNorm wrappers around each of the three sublayers
        self.self_attn_add_norm = AddNorm(d_model=d_model, dropout=dropout)
        self.cross_attn_add_norm = AddNorm(d_model=d_model, dropout=dropout)
        self.ffn_add_norm = AddNorm(d_model=d_model, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Performs the forward pass of the TransformerDecoderLayer.
        
        Args:
            x (torch.Tensor): Input tensor from the decoder stack with shape (batch_size, tgt_len, d_model).
            enc_output (torch.Tensor): Output tensor from the encoder stack with shape (batch_size, src_len, d_model).
            src_mask (torch.Tensor, optional): Source mask to prevent attention to source padding,
                                              shape (batch_size, 1, 1, src_len) or broadcastable.
            tgt_mask (torch.Tensor, optional): Target mask to prevent attending to subsequent or padding tokens,
                                              shape (batch_size, 1, tgt_len, tgt_len) or broadcastable.
                                              
        Returns:
            torch.Tensor: The output tensor with shape (batch_size, tgt_len, d_model).
        """
        # Sub-layer 1: Masked Causal Self-Attention (Pre-LN)
        # Uses tgt_mask (causal mask) to hide future target tokens
        x = self.self_attn_add_norm(x, lambda y: self.self_attn(query=y, key=y, value=y, mask=tgt_mask))
        
        # Sub-layer 2: Encoder-Decoder Cross-Attention (Pre-LN)
        # Query comes from decoder sequence, Keys & Values come from encoder representations.
        # Uses src_mask (padding mask) to hide source padding tokens.
        x = self.cross_attn_add_norm(x, lambda y: self.cross_attn(query=y, key=enc_output, value=enc_output, mask=src_mask))
        
        # Sub-layer 3: Feed-Forward Network (Pre-LN)
        x = self.ffn_add_norm(x, self.ffn)
        
        return x
