import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from typing import Dict, List, Tuple, Any

class ParallelCorpusDataset(Dataset):
    """
    A PyTorch Dataset wrapper for parallel translation corpora (source-target text pairs).
    
    This dataset tokenizes source and target sentences, adds special tokens ([SOS] and [EOS]),
    aligns sequences by shifting target inputs for training, and constructs appropriate padding
    and causal masks.
    """
    def __init__(
        self,
        hf_dataset: Any,
        src_lang: str,
        tgt_lang: str,
        src_tokenizer: Tokenizer,
        tgt_tokenizer: Tokenizer,
        seq_len: int = 128
    ):
        """
        Initializes the ParallelCorpusDataset.
        
        Args:
            hf_dataset (Any): A HuggingFace datasets split containing target translation pairs.
            src_lang (str): Source language identifier key (e.g. 'en').
            tgt_lang (str): Target language identifier key (e.g. 'fr').
            src_tokenizer (Tokenizer): Tokenizer for the source language.
            tgt_tokenizer (Tokenizer): Tokenizer for the target language.
            seq_len (int): Total sequence length constraint for padding/truncating.
        """
        self.dataset = hf_dataset
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.seq_len = seq_len
        
        # Cache token IDs for special sequences
        self.src_sos_id = self.src_tokenizer.token_to_id("[SOS]")
        self.src_eos_id = self.src_tokenizer.token_to_id("[EOS]")
        self.src_pad_id = self.src_tokenizer.token_to_id("[PAD]")
        
        self.tgt_sos_id = self.tgt_tokenizer.token_to_id("[SOS]")
        self.tgt_eos_id = self.tgt_tokenizer.token_to_id("[EOS]")
        self.tgt_pad_id = self.tgt_tokenizer.token_to_id("[PAD]")
        
        # Verify tokenizer special tokens are set up
        assert self.src_sos_id is not None, "Source tokenizer lacks '[SOS]'"
        assert self.src_eos_id is not None, "Source tokenizer lacks '[EOS]'"
        assert self.src_pad_id is not None, "Source tokenizer lacks '[PAD]'"
        assert self.tgt_sos_id is not None, "Target tokenizer lacks '[SOS]'"
        assert self.tgt_eos_id is not None, "Target tokenizer lacks '[EOS]'"
        assert self.tgt_pad_id is not None, "Target tokenizer lacks '[PAD]'"

    def __len__(self) -> int:
        """Returns the number of sentence pairs in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Tokenizes and returns the source and target pair at index idx with proper labels and masks.
        
        Args:
            idx (int): Dataset element index.
            
        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - encoder_input: LongTensor shape (seq_len)
                - decoder_input: LongTensor shape (seq_len)
                - label: LongTensor shape (seq_len)
                - encoder_mask: BoolTensor shape (1, 1, seq_len)
                - decoder_mask: BoolTensor shape (1, seq_len, seq_len)
        """
        item = self.dataset[idx]
        
        # Handle standard HuggingFace translation structure (e.g. Opus Books) vs flat structure
        if 'translation' in item:
            src_text = item['translation'][self.src_lang]
            tgt_text = item['translation'][self.tgt_lang]
        else:
            src_text = item[self.src_lang]
            tgt_text = item[self.tgt_lang]
            
        # Perform tokenization to retrieve token IDs
        src_tokens = self.src_tokenizer.encode(src_text).ids
        tgt_tokens = self.tgt_tokenizer.encode(tgt_text).ids
        
        # Enforce sequence limits (need space for special start/end tokens)
        if len(src_tokens) > self.seq_len - 2:
            raise ValueError(
                f"Source text at index {idx} has length {len(src_tokens)} tokens, "
                f"which exceeds the maximum limit of seq_len - 2 ({self.seq_len - 2})."
            )
        if len(tgt_tokens) > self.seq_len - 2:
            raise ValueError(
                f"Target text at index {idx} has length {len(tgt_tokens)} tokens, "
                f"which exceeds the maximum limit of seq_len - 2 ({self.seq_len - 2})."
            )
            
        # 1. encoder_input: [SOS] + tokens + [EOS] + [PAD] * padding
        enc_pad_len = self.seq_len - len(src_tokens) - 2
        encoder_input = torch.cat([
            torch.tensor([self.src_sos_id], dtype=torch.long),
            torch.tensor(src_tokens, dtype=torch.long),
            torch.tensor([self.src_eos_id], dtype=torch.long),
            torch.tensor([self.src_pad_id] * enc_pad_len, dtype=torch.long)
        ])
        
        # 2. decoder_input: [SOS] + tokens + [PAD] * padding (shifted right)
        dec_pad_len = self.seq_len - len(tgt_tokens) - 1
        decoder_input = torch.cat([
            torch.tensor([self.tgt_sos_id], dtype=torch.long),
            torch.tensor(tgt_tokens, dtype=torch.long),
            torch.tensor([self.tgt_pad_id] * dec_pad_len, dtype=torch.long)
        ])
        
        # 3. label: tokens + [EOS] + [PAD] * padding (shifted left)
        label = torch.cat([
            torch.tensor(tgt_tokens, dtype=torch.long),
            torch.tensor([self.tgt_eos_id], dtype=torch.long),
            torch.tensor([self.tgt_pad_id] * dec_pad_len, dtype=torch.long)
        ])
        
        # 4. encoder_mask: shape (1, 1, seq_len)
        # Masks out padding token locations. True for valid positions, False for padding.
        encoder_mask = (encoder_input != self.src_pad_id).unsqueeze(0).unsqueeze(0)
        
        # 5. decoder_mask: combines padding mask and lower-triangular causal mask
        # pad_mask shape: (1, 1, seq_len)
        decoder_pad_mask = (decoder_input != self.tgt_pad_id).unsqueeze(0).unsqueeze(0)
        # causal_mask shape: (1, seq_len, seq_len)
        causal_mask = torch.tril(torch.ones(self.seq_len, self.seq_len, dtype=torch.bool)).unsqueeze(0)
        # Combined mask shape: (1, seq_len, seq_len).
        # Collation by DataLoader yields batch dimension: (batch, 1, seq_len, seq_len).
        decoder_mask = decoder_pad_mask & causal_mask
        
        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "label": label,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask
        }