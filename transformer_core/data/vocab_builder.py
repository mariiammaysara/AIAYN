import os
from typing import Iterable, List
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

def build_tokenizer_from_iterator(
    iterator: Iterable[str],
    save_path: str = None,
    vocab_size: int = 37000,
    special_tokens: List[str] = None
) -> Tokenizer:
    """
    Builds (trains) or loads a WordLevel tokenizer using the HuggingFace `tokenizers` library.
    
    If save_path is provided and the file exists, it loads the saved tokenizer.
    Otherwise, it trains the tokenizer on the text iterator and saves it.
    
    Args:
        iterator (Iterable[str]): An iterator yielding text strings for training.
        save_path (str, optional): Local file path to save/load the tokenizer.
        vocab_size (int): The target size of the vocabulary.
        special_tokens (List[str], optional): Special tokens to include in the vocabulary.
                                             Defaults to ["[UNK]", "[PAD]", "[SOS]", "[EOS]"].
                                             
    Returns:
        Tokenizer: A trained/loaded HuggingFace WordLevel Tokenizer.
    """
    if special_tokens is None:
        special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"]
        
    # Check if a saved tokenizer config already exists on disk
    if save_path is not None and os.path.exists(save_path):
        print(f"Loading existing tokenizer from {save_path}")
        return Tokenizer.from_file(save_path)
        
    print("Training new WordLevel tokenizer...")
    # Instantiate a WordLevel tokenizer model with a default unknown token placeholder
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    
    # Use standard whitespace-based pre-tokenization
    tokenizer.pre_tokenizer = Whitespace()
    
    # Setup the trainer with vocab limits and required special tokens
    trainer = WordLevelTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2
    )
    
    # Train the model from the text iterator
    tokenizer.train_from_iterator(iterator, trainer)
    
    # Save the tokenizer for future quick loading
    if save_path is not None:
        # Create directory if it doesn't exist
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        tokenizer.save(save_path)
        print(f"Saved tokenizer to {save_path}")
        
    return tokenizer
