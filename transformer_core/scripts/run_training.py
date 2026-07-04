import argparse
from transformer_core.training.settings import TrainingSettings
from transformer_core.training.loop import train_model

def main():
    """
    Main entry point for training the Transformer model.
    Loads default training settings, parses optional command line arguments to override them,
    and calls loop.py's main training function.
    """
    parser = argparse.ArgumentParser(description="Train the modular Seq2Seq Transformer model.")
    
    # Optional arguments to override TrainingSettings defaults
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--d_model", type=int, default=None, help="Dimension of the model.")
    parser.add_argument("--num_layers", type=int, default=None, help="Number of encoder/decoder layers.")
    parser.add_argument("--num_heads", type=int, default=None, help="Number of attention heads.")
    parser.add_argument("--d_ff", type=int, default=None, help="Feed-forward network hidden size.")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout probability.")
    parser.add_argument("--seq_len", type=int, default=None, help="Sequence length limit.")
    parser.add_argument("--dataset", type=str, default=None, help="HuggingFace dataset name.")
    parser.add_argument("--src_lang", type=str, default=None, help="Source language identifier (e.g. en).")
    parser.add_argument("--tgt_lang", type=str, default=None, help="Target language identifier (e.g. fr).")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to save checkpoints.")
    parser.add_argument("--log_dir", type=str, default=None, help="Directory to write logs.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume from.")
    parser.add_argument("--device", type=str, default=None, help="Device to execute training on (e.g. cuda, cpu).")
    
    args = parser.parse_args()
    
    # Load settings config
    settings = TrainingSettings()
    
    # Override settings from CLI arguments if provided
    if args.epochs is not None:
        settings.num_epochs = args.epochs
    if args.batch_size is not None:
        settings.batch_size = args.batch_size
    if args.lr is not None:
        settings.lr = args.lr
    if args.d_model is not None:
        settings.d_model = args.d_model
    if args.num_layers is not None:
        settings.num_layers = args.num_layers
    if args.num_heads is not None:
        settings.num_heads = args.num_heads
    if args.d_ff is not None:
        settings.d_ff = args.d_ff
    if args.dropout is not None:
        settings.dropout = args.dropout
    if args.seq_len is not None:
        settings.seq_len = args.seq_len
    if args.dataset is not None:
        settings.dataset_name = args.dataset
    if args.src_lang is not None:
        settings.src_lang = args.src_lang
    if args.tgt_lang is not None:
        settings.tgt_lang = args.tgt_lang
    if args.checkpoint_dir is not None:
        settings.checkpoint_dir = args.checkpoint_dir
    if args.log_dir is not None:
        settings.log_dir = args.log_dir
    if args.device is not None:
        settings.device = args.device
        
    print(f"Loaded configuration settings: {settings}")
    
    # Call loop.py's main training orchestrator
    train_model(settings, resume_path=args.resume)

if __name__ == "__main__":
    main()
