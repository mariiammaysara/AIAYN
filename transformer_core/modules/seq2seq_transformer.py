import torch
import torch.nn as nn
from transformer_core.modules.encoder_stack import EncoderStack
from transformer_core.modules.decoder_stack import DecoderStack

class Seq2SeqTransformer(nn.Module):
    """
    Complete Sequence-to-Sequence Transformer model.
    
    Combines the EncoderStack, DecoderStack, and a final linear projection layer
    to predict target token log-probability distributions.
    """
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        """
        Initializes the Seq2SeqTransformer.
        
        Args:
            src_vocab_size (int): Size of the source vocabulary.
            tgt_vocab_size (int): Size of the target vocabulary.
            d_model (int): Dimension of the model's hidden states.
            n_layers (int): Number of layers in encoder and decoder.
            n_heads (int): Number of attention heads.
            d_ff (int): Dimension of the hidden layer in FFN.
            max_len (int): Maximum sequence length.
            dropout (float): Dropout probability.
        """
        super().__init__()
        # Encoder Stack (embeddings + positional encoding + N encoder layers + LayerNorm)
        self.encoder = EncoderStack(
            vocab_size=src_vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout
        )
        
        # Decoder Stack (embeddings + positional encoding + N decoder layers + LayerNorm)
        self.decoder = DecoderStack(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout
        )
        
        # Final output projection layer mapping d_model back to target vocab space
        self.projection = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Runs the source sequence through the encoder stack.
        
        Args:
            src (torch.Tensor): Source tokens, shape (batch_size, src_len).
            src_mask (torch.Tensor, optional): Source mask, shape (batch_size, 1, 1, src_len).
            
        Returns:
            torch.Tensor: Encoder outputs, shape (batch_size, src_len, d_model).
        """
        return self.encoder(src, mask=src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Runs the target sequence and encoder outputs through the decoder stack.
        
        Args:
            tgt (torch.Tensor): Target tokens, shape (batch_size, tgt_len).
            enc_output (torch.Tensor): Encoder outputs, shape (batch_size, src_len, d_model).
            src_mask (torch.Tensor, optional): Source mask, shape (batch_size, 1, 1, src_len).
            tgt_mask (torch.Tensor, optional): Target mask, shape (batch_size, 1, tgt_len, tgt_len).
            
        Returns:
            torch.Tensor: Decoder outputs, shape (batch_size, tgt_len, d_model).
        """
        return self.decoder(tgt, enc_output, src_mask=src_mask, tgt_mask=tgt_mask)

    def project(self, decoder_output: torch.Tensor) -> torch.Tensor:
        """
        Maps the decoder output representations to log-probabilities over the target vocabulary.
        
        Args:
            decoder_output (torch.Tensor): Decoder representation tensor, shape (batch_size, tgt_len, d_model).
            
        Returns:
            torch.Tensor: Log-probabilities over target vocabulary, shape (batch_size, tgt_len, tgt_vocab_size).
        """
        # Apply the final projection and log_softmax along the vocabulary dimension (dim=-1)
        return torch.log_softmax(self.projection(decoder_output), dim=-1)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Performs full forward pass of the Seq2Seq Transformer model.
        
        Used primarily during parallel training.
        
        Args:
            src (torch.Tensor): Source tokens with shape (batch_size, src_len).
            tgt (torch.Tensor): Target tokens with shape (batch_size, tgt_len).
            src_mask (torch.Tensor, optional): Source mask.
            tgt_mask (torch.Tensor, optional): Target mask.
            
        Returns:
            torch.Tensor: Log-probabilities over the target vocabulary, shape (batch_size, tgt_len, tgt_vocab_size).
        """
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        return self.project(dec_output)


def build_seq2seq_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    num_layers: int = 6,
    num_heads: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048
) -> Seq2SeqTransformer:
    """
    Factory function to construct a Seq2SeqTransformer model with initialized weights.
    
    Applies Xavier Uniform initialization to all multi-dimensional weights (dim > 1).
    Biases are set to zero.
    
    Args:
        src_vocab_size (int): Size of the source vocabulary.
        tgt_vocab_size (int): Size of the target vocabulary.
        src_seq_len (int): Maximum sequence length for the source language.
        tgt_seq_len (int): Maximum sequence length for the target language.
        d_model (int): Hidden dimension size.
        num_layers (int): Number of encoder/decoder layers.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
        d_ff (int): Hidden dimension size of the FFN.
        
    Returns:
        Seq2SeqTransformer: The fully initialized model.
    """
    max_len = max(src_seq_len, tgt_seq_len)
    
    # Instantiate the model
    model = Seq2SeqTransformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        n_layers=num_layers,
        n_heads=num_heads,
        d_ff=d_ff,
        max_len=max_len,
        dropout=dropout
    )
    
    # Xavier uniform weight initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return model
