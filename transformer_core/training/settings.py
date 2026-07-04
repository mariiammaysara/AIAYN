from dataclasses import dataclass

@dataclass
class TrainingSettings:
    """
    Hyperparameters and settings configuration for training the Transformer model.
    Scaled up config for stronger, presentable translation outputs.
    """
    # Model parameters (Scaled up configuration)
    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 8
    d_ff: int = 512
    dropout: float = 0.1
    seq_len: int = 40

    # Training parameters
    batch_size: int = 32
    num_epochs: int = 40
    lr: float = 5e-4  # Acts as the peak learning rate in the warmup schedule
    warmup_steps: int = 4000  # Number of steps for linear warmup

    # Dataset details (en-it translation pair)
    dataset_name: str = "Helsinki-NLP/opus_books"
    src_lang: str = "en"
    tgt_lang: str = "it"

    # Quick training helper
    # Sliced to 20,000 translation pairs for training a robust vocabulary and sentence alignments
    max_samples: int = 20000

    # Directory and infrastructure settings
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    device: str = "cpu"  # Note: Overridden by runtime auto-detection in loop.py
