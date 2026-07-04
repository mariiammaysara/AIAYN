import math
import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    """
    Token Embedding layer that converts token indices into dense vector representations.
    
    This layer scales the embedding weights by the square root of the embedding dimension
    as described in the original paper to match the scale of positional encodings.
    """
    def __init__(self, vocab_size: int, d_model: int):
        """
        Initializes the TokenEmbedding.
        
        Args:
            vocab_size (int): Size of the vocabulary.
            d_model (int): Dimension of the embedding vectors.
        """
        super().__init__()
        self.d_model = d_model
        # Standard PyTorch embedding layer to map sparse token indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the TokenEmbedding.
        
        Args:
            x (torch.Tensor): Input tensor of token indices with shape (batch_size, seq_len).
            
        Returns:
            torch.Tensor: Embedded tokens scaled by sqrt(d_model) with shape (batch_size, seq_len, d_model).
        """
        # The learned embedding weights are typically initialized such that their variance is ~1/d_model.
        # To prevent these learned semantic signals from being dominated/washed out by the fixed
        # sinusoidal positional encodings (which naturally range between -1 and 1), we scale the
        # embeddings by multiplying by sqrt(d_model), restoring them to unit variance scale.
        return self.embedding(x) * math.sqrt(self.d_model)
