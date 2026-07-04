import torch
from transformer_core.modules.seq2seq_transformer import build_seq2seq_transformer

def main():
    print("Running fast Transformer shape sanity check...")
    
    # 1. Hyperparameters
    batch_size = 2
    seq_len = 10
    vocab_size = 100
    d_model = 64
    num_layers = 2
    num_heads = 2
    d_ff = 256
    
    # 2. Build model via factory function
    model = build_seq2seq_transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        src_seq_len=seq_len,
        tgt_seq_len=seq_len,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=0.1,
        d_ff=d_ff
    )
    
    # 3. Create random dummy inputs
    src = torch.randint(0, vocab_size, (batch_size, seq_len))
    tgt = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Create masks
    src_mask = torch.ones(batch_size, 1, 1, seq_len, dtype=torch.bool)
    tgt_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)).unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1, 1)
    
    # 4. Run step-by-step pipeline
    # encode
    enc_out = model.encode(src, src_mask)
    # decode
    dec_out = model.decode(tgt, enc_out, src_mask, tgt_mask)
    # project
    logits = model.project(dec_out)
    
    # 5. Asserts and validation
    print(f"Logits output shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, vocab_size), f"Expected shape {(batch_size, seq_len, vocab_size)}, got {logits.shape}"
    
    print("OK")

if __name__ == "__main__":
    main()
