import math
import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    """
    Implements the Sinusoidal Positional Encoding to inject sequence order information
    into the token embeddings.
    
    Since the self-attention mechanism contains no recurrence or convolution,
    positional encodings are added to the input embeddings to provide positioning context.
    
    Sinusoidal vs. Learned Positional Encodings:
    1. Extrapolative capability: Fixed sinusoidal encodings use periodic functions (sin/cos)
       which allow the model to generalize/extrapolate to sequences longer than those seen during training.
       Learned embeddings cannot easily handle sequence lengths larger than the training maximum.
    2. Linear relative relationships: For any fixed offset k, the positional encoding at position pos + k
       can be represented as a linear function of the encoding at position pos. This mathematical property
       helps the attention mechanism learn patterns based on relative positions.
    3. Parameter efficiency: Precomputing fixed encodings does not add trainable parameters, keeping the
       model size smaller and faster to optimize.
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initializes the SinusoidalPositionalEncoding.
        
        Args:
            d_model (int): Dimension of the embedding vectors.
            max_len (int): Maximum sequence length to pre-calculate positions for.
            dropout (float): Dropout probability applied after adding encodings.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Precompute the positional encodings matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        # position indices: [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # div_term = 10000^(2i/d_model). We compute it in the log domain for numerical stability.
        # We step by 2 because we assign sine to even dimensions and cosine to odd dimensions.
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        
        # Set even indices (2i) to sine, odd indices (2i+1) to cosine
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Reshape to (1, max_len, d_model) to easily broadcast over batch size
        pe = pe.unsqueeze(0)
        
        # Register pe as a persistent buffer. It is part of the module's state dict but is NOT
        # a trainable parameter (requires_grad = False). When calling .to(device) on the model,
        # registered buffers are automatically moved to the target device.
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the SinusoidalPositionalEncoding.
        
        Args:
            x (torch.Tensor): Input token embeddings tensor with shape (batch_size, seq_len, d_model).
            
        Returns:
            torch.Tensor: Embeddings with positional encodings added and dropout applied, shape (batch_size, seq_len, d_model).
        """
        # Slice the precomputed positional encoding buffer to match input sequence length
        # pe shape becomes [1, seq_len, d_model], which broadcasts over batch_size
        x = x + self.pe[:, :x.size(1)]
        
        # Apply dropout to the sum as specified in section 5.4 of the paper
        return self.dropout(x)
