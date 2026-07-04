import torch
import torch.nn as nn
from transformer_core.layers.embedding import TokenEmbedding
from transformer_core.layers.positional import SinusoidalPositionalEncoding
from transformer_core.layers.normalization import LayerNorm
from transformer_core.blocks.encoder_layer import TransformerEncoderLayer

class EncoderStack(nn.Module):
    """
    The full Transformer Encoder Stack.
    
    This stack corresponds to the left stack ("Encoder") of Figure 1 in the original
    paper. It converts source token sequences to dense context-aware representations
    by applying:
    1. Input Token Embeddings (TokenEmbedding)
    2. Positional Encodings (SinusoidalPositionalEncoding)
    3. N sequential encoder layer blocks (TransformerEncoderLayer via nn.ModuleList)
    4. A final LayerNorm step (as required to scale final features in Pre-LN configurations)
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        """
        Initializes the EncoderStack.
        
        Args:
            vocab_size (int): Size of the source vocabulary.
            d_model (int): Dimension of the model embeddings.
            n_layers (int): Number of encoder layers to stack.
            n_heads (int): Number of attention heads.
            d_ff (int): Hidden dimension of FFN.
            max_len (int): Maximum sequence length for positional encoding.
            dropout (float): Dropout probability.
        """
        super().__init__()
        # Token Embedding layer
        self.embedding = TokenEmbedding(vocab_size=vocab_size, d_model=d_model)
        # Sinusoidal Positional Encoding
        self.positional_encoding = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)
        
        # Sequential list of encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        # In Pre-LN architectures, the final outputs are not normalized by any internal layer.
        # Thus, we apply a final LayerNorm step here before passing representations out of the stack.
        self.final_norm = LayerNorm(d_model=d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Performs the forward pass of the EncoderStack.
        
        Args:
            x (torch.Tensor): Input source token indices with shape (batch_size, seq_len).
            mask (torch.Tensor, optional): Source mask for padding, shape (batch_size, 1, 1, seq_len).
            
        Returns:
            torch.Tensor: Encoder representations with shape (batch_size, seq_len, d_model).
        """
        # 1. Token embeddings
        out = self.embedding(x)
        # 2. Add positional encoding
        out = self.positional_encoding(out)
        
        # 3. Process through stacked layers
        for layer in self.layers:
            out = layer(out, mask=mask)
            
        # 4. Final normalization
        return self.final_norm(out)
