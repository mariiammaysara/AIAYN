import math
import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    """
    Implements Multi-Head Attention as described in the Transformer paper.
    
    This layer projects queries, keys, and values into multiple subspaces,
    computes scaled dot-product attention in parallel, and concatenates the outputs.
    
    Scaling Factor Rationale (dividing by sqrt(d_k)):
    For large values of the projection dimension d_k, the dot product scores tend to
    grow large in magnitude. When passed to softmax, these large values lead to extremely
    small gradients (the vanishing gradient problem). Dividing the dot products by sqrt(d_k)
    rescales the variance to 1 (assuming components of Q and K are independent random
    variables with mean 0 and variance 1), preventing the softmax from saturating.
    
    Multi-Use Unified Architecture:
    The query, key, and value projection tensors are handled separately inside forward()
    which enables this single class to perform three distinct roles:
    1. Encoder Self-Attention:
       Q, K, V are all projected from the same encoder inputs (query=key=value=X).
       A source key padding mask is applied to prevent attending to PAD tokens.
    2. Decoder Masked Self-Attention:
       Q, K, V are all projected from the decoder inputs (query=key=value=Y).
       A combined target padding and causal mask is applied to prevent attending to future tokens.
    3. Encoder-Decoder Cross-Attention:
       Q is projected from decoder representations (query=Y), while K and V are projected
       from encoder outputs (key=value=encoder_outputs).
       A source padding mask is applied.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Initializes the MultiHeadSelfAttention.
        
        Args:
            d_model (int): Total dimension of the model. Must be divisible by n_heads.
            n_heads (int): Number of attention heads.
            dropout (float): Dropout probability for attention weights.
        """
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Projection layers for Queries, Keys, and Values
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Final output linear projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.attn_dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Performs the forward pass of the MultiHeadSelfAttention.
        
        Args:
            query (torch.Tensor): Query tensor with shape (batch_size, q_len, d_model).
            key (torch.Tensor): Key tensor with shape (batch_size, k_len, d_model).
            value (torch.Tensor): Value tensor with shape (batch_size, v_len, d_model).
            mask (torch.Tensor, optional): Mask tensor to prevent attention to certain positions.
                                          Shape must be broadcastable to (batch_size, n_heads, q_len, k_len).
                                          0 or False indicates positions to mask out.
                                          
        Returns:
            torch.Tensor: The output of the attention layer with shape (batch_size, q_len, d_model).
        """
        batch_size = query.size(0)
        
        # 1. Linear projections: (batch_size, seq_len, d_model)
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # 2. Reshape to split the d_model into n_heads * d_k, and transpose dimensions
        # Output shape: (batch_size, n_heads, seq_len, d_k)
        q = q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 3. Compute scaled dot-product attention scores
        # scores shape: (batch_size, n_heads, q_len, k_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 4. Apply additive mask if provided
        if mask is not None:
            # Mask out invalid positions by setting them to -1e9 before softmax
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # 5. Softmax along key length dimension (last dimension) to get probability weights
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 6. Compute weighted combination of values: (batch_size, n_heads, q_len, d_k)
        context = torch.matmul(attn_weights, v)
        
        # 7. Concatenate attention heads: (batch_size, q_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 8. Apply the output linear projection
        return self.out_proj(context)
