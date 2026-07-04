import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datasets import load_dataset
from tokenizers import Tokenizer
from torchmetrics.text import BLEUScore, WordErrorRate, CharErrorRate
from typing import Tuple, Any

from transformer_core.training.settings import TrainingSettings
from transformer_core.modules.seq2seq_transformer import build_seq2seq_transformer
from transformer_core.data.vocab_builder import build_tokenizer_from_iterator
from transformer_core.data.parallel_corpus import ParallelCorpusDataset

@torch.no_grad()
def greedy_decode(
    model: nn.Module,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    sos_id: int,
    eos_id: int,
    device: torch.device
) -> torch.Tensor:
    """
    Translates a source sequence token-by-token using greedy decoding.
    
    Args:
        model (nn.Module): The Seq2SeqTransformer model.
        src (torch.Tensor): A single source sequence token list with shape (1, seq_len).
        src_mask (torch.Tensor): Source mask with shape (1, 1, seq_len).
        max_len (int): Maximum generation length.
        sos_id (int): Start of sequence token ID.
        eos_id (int): End of sequence token ID.
        device (torch.device): Device to run predictions on.
        
    Returns:
        torch.Tensor: Decoded token indices tensor with shape (seq_len).
    """
    model.eval()
    src = src.to(device)
    if src_mask is not None:
        src_mask = src_mask.to(device)

    # Encode source tokens once
    enc_output = model.encode(src, src_mask)
    
    # Initialize target tensor with SOS token
    decoder_input = torch.empty(1, 1, dtype=torch.long, device=device).fill_(sos_id)
    
    for _ in range(max_len - 1):
        # Create target causal mask for the current decoded length
        tgt_len = decoder_input.size(1)
        tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device)).unsqueeze(0).unsqueeze(1)
        
        # Decode next step representation
        decoder_output = model.decode(decoder_input, enc_output, src_mask=src_mask, tgt_mask=tgt_mask)
        
        # Map last time-step to logits
        logits = model.project(decoder_output[:, -1, :])
        
        # Greedy choice
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        
        # Append predicted token
        decoder_input = torch.cat([decoder_input, next_token], dim=-1)
        
        # Break if EOS token is predicted
        if next_token.item() == eos_id:
            break
            
    return decoder_input.squeeze(0)


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    global_step: int,
    scheduler: torch.optim.lr_scheduler.LambdaLR = None
) -> int:
    """
    Executes a single training epoch.
    
    Logs per-batch loss, learning rate, and updates the global step counter.
    """
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for batch in pbar:
        # Move tensors to the execution device
        encoder_input = batch["encoder_input"].to(device)
        decoder_input = batch["decoder_input"].to(device)
        label = batch["label"].to(device)
        encoder_mask = batch["encoder_mask"].to(device)
        decoder_mask = batch["decoder_mask"].to(device)
        
        optimizer.zero_grad()
        
        # Model forward pass
        output = model(encoder_input, decoder_input, src_mask=encoder_mask, tgt_mask=decoder_mask)
        
        # Calculate loss (shape flattened to match cross entropy target requirements)
        loss = criterion(output.view(-1, output.size(-1)), label.view(-1))
        
        loss.backward()
        
        # Clip gradient norms to stabilize training
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # Run learning rate schedule step (at batch/step level)
        if scheduler is not None:
            scheduler.step()
            
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Summary writer logging
        writer.add_scalar("Loss/train_batch", loss.item(), global_step)
        writer.add_scalar("lr", current_lr, global_step)
        global_step += 1
        
        total_loss += loss.item()
        pbar.set_postfix({"batch_loss": f"{loss.item():.4f}", "lr": f"{current_lr:.6f}"})
        
    avg_loss = total_loss / len(dataloader)
    writer.add_scalar("Loss/train_epoch", avg_loss, epoch)
    print(f"Epoch {epoch+1} average training loss: {avg_loss:.4f}")
    return global_step


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    src_tokenizer: Tokenizer,
    tgt_tokenizer: Tokenizer,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    max_len: int,
    num_print_examples: int = 5,
    num_metric_examples: int = 100
):
    """
    Runs a validation loop, prints translation comparisons, and evaluates corpus scores.
    """
    model.eval()
    
    # Initialize metrics
    bleu_metric = BLEUScore()
    wer_metric = WordErrorRate()
    cer_metric = CharErrorRate()
    
    references = []
    hypotheses = []
    
    sos_id = tgt_tokenizer.token_to_id("[SOS]")
    eos_id = tgt_tokenizer.token_to_id("[EOS]")
    pad_id = tgt_tokenizer.token_to_id("[PAD]")
    
    print(f"\n--- Running Validation (Greedy Decoding) for Epoch {epoch+1} ---")
    
    count = 0
    for batch in dataloader:
        encoder_input = batch["encoder_input"].to(device)
        encoder_mask = batch["encoder_mask"].to(device)
        labels = batch["label"]
        
        batch_size = encoder_input.size(0)
        
        for i in range(batch_size):
            if count >= num_metric_examples:
                break
                
            src_seq = encoder_input[i:i+1]
            src_mask = encoder_mask[i:i+1]
            
            # Predict sequences
            predicted_ids = greedy_decode(model, src_seq, src_mask, max_len, sos_id, eos_id, device)
            
            # Extract target labels, stripping end-of-sequence and padding marks
            label_list = labels[i].tolist()
            if eos_id in label_list:
                label_list = label_list[:label_list.index(eos_id)]
            label_list = [tid for tid in label_list if tid != pad_id]
            
            # Extract prediction list
            predicted_list = predicted_ids.tolist()
            if eos_id in predicted_list:
                predicted_list = predicted_list[:predicted_list.index(eos_id)]
            predicted_list = [tid for tid in predicted_list if tid != sos_id and tid != pad_id]
            
            # Convert token lists back to textual strings
            src_text = src_tokenizer.decode(src_seq[0].tolist(), skip_special_tokens=True)
            ref_text = tgt_tokenizer.decode(label_list, skip_special_tokens=True)
            hyp_text = tgt_tokenizer.decode(predicted_list, skip_special_tokens=True)
            
            references.append([ref_text])
            hypotheses.append(hyp_text)
            
            # Print side-by-side comparison for inspection
            if count < num_print_examples:
                print(f"Example {count+1}:")
                print(f"  Source:     {src_text}")
                print(f"  Reference:  {ref_text}")
                print(f"  Predicted:  {hyp_text}")
                print("-" * 50)
                
            count += 1
            
        if count >= num_metric_examples:
            break
            
    # Compute and report validation metrics
    if len(hypotheses) > 0:
        bleu = bleu_metric(hypotheses, references)
        wer = wer_metric(hypotheses, [ref[0] for ref in references])
        cer = cer_metric(hypotheses, [ref[0] for ref in references])
        
        print(f"Validation Metrics:")
        print(f"  BLEU: {bleu.item():.4f}")
        print(f"  WER:  {wer.item():.4f}")
        print(f"  CER:  {cer.item():.4f}\n")
        
        writer.add_scalar("Validation/BLEU", bleu.item(), epoch)
        writer.add_scalar("Validation/WER", wer.item(), epoch)
        writer.add_scalar("Validation/CER", cer.item(), epoch)
    else:
        print("No validation items processed.")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
    global_step: int,
    settings: TrainingSettings,
    filename: str = "checkpoint.pt"
):
    """
    Saves the state of the model, optimizer, scheduler, epoch, and global steps to disk.
    """
    os.makedirs(settings.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(settings.checkpoint_dir, filename)
    
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "settings": settings
    }, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    checkpoint_path: str
) -> Tuple[int, int]:
    """
    Loads saved state to resume training.
    """
    print(f"Resuming from checkpoint {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["epoch"], checkpoint["global_step"]


def load_and_prepare_data(settings: TrainingSettings) -> Tuple[DataLoader, DataLoader, Tokenizer, Tokenizer]:
    """
    Loads dataset, trains tokenizers, filters sequences, and creates DataLoaders.
    """
    print(f"Preparing dataset: {settings.dataset_name} for pair: {settings.src_lang}-{settings.tgt_lang}...")
    
    # Load dataset splits
    if settings.dataset_name in ["opus_books", "Helsinki-NLP/opus_books"]:
        raw_dataset = load_dataset(
            settings.dataset_name,
            f"{settings.src_lang}-{settings.tgt_lang}",
            split="train"
        )
        if settings.max_samples > 0:
            # Subset the dataset to a small number of samples for rapid local CPU execution
            subset_size = min(settings.max_samples, len(raw_dataset))
            raw_dataset = raw_dataset.select(range(subset_size))
        split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
    else:
        split_dataset = load_dataset(
            settings.dataset_name,
            f"{settings.src_lang}-{settings.tgt_lang}"
        )
        if settings.max_samples > 0:
            for split_name in split_dataset.keys():
                subset_size = min(settings.max_samples, len(split_dataset[split_name]))
                split_dataset[split_name] = split_dataset[split_name].select(range(subset_size))
        
    train_raw = split_dataset["train"]
    val_raw = split_dataset["test" if "test" in split_dataset else "validation"]
    
    # Generator to fetch sentences for tokenizer training
    def get_texts(dataset_split, lang):
        for item in dataset_split:
            if "translation" in item:
                yield item["translation"][lang]
            else:
                yield item[lang]
                
    os.makedirs(settings.checkpoint_dir, exist_ok=True)
    src_tokenizer_path = os.path.join(settings.checkpoint_dir, f"tokenizer_{settings.src_lang}.json")
    tgt_tokenizer_path = os.path.join(settings.checkpoint_dir, f"tokenizer_{settings.tgt_lang}.json")
    
    # Train source and target tokenizers
    src_tokenizer = build_tokenizer_from_iterator(
        get_texts(train_raw, settings.src_lang),
        save_path=src_tokenizer_path
    )
    tgt_tokenizer = build_tokenizer_from_iterator(
        get_texts(train_raw, settings.tgt_lang),
        save_path=tgt_tokenizer_path
    )
    
    # Filter out sentences that exceed seq_len - 2 tokens to avoid ValueError crashes in Dataset
    def filter_length(example):
        if "translation" in example:
            src_text = example["translation"][settings.src_lang]
            tgt_text = example["translation"][settings.tgt_lang]
        else:
            src_text = example[settings.src_lang]
            tgt_text = example[settings.tgt_lang]
            
        src_ids = src_tokenizer.encode(src_text).ids
        tgt_ids = tgt_tokenizer.encode(tgt_text).ids
        return len(src_ids) <= settings.seq_len - 2 and len(tgt_ids) <= settings.seq_len - 2

    print("Filtering sentences exceeding max sequence length...")
    train_filtered = train_raw.filter(filter_length)
    val_filtered = val_raw.filter(filter_length)
    print(f"Filtered Train size: {len(train_filtered)} (from {len(train_raw)})")
    print(f"Filtered Val size:   {len(val_filtered)} (from {len(val_raw)})")

    # Wrap in PyTorch Datasets
    train_dataset = ParallelCorpusDataset(
        hf_dataset=train_filtered,
        src_lang=settings.src_lang,
        tgt_lang=settings.tgt_lang,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        seq_len=settings.seq_len
    )
    
    val_dataset = ParallelCorpusDataset(
        hf_dataset=val_filtered,
        src_lang=settings.src_lang,
        tgt_lang=settings.tgt_lang,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        seq_len=settings.seq_len
    )
    
    # Instantiate DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=settings.batch_size, shuffle=False)
    
    return train_loader, val_loader, src_tokenizer, tgt_tokenizer


def train_model(settings: TrainingSettings, resume_path: str = None):
    """
    Main training orchestrator function.
    """
    # 1. Prepare data loaders and tokenizers
    train_loader, val_loader, src_tokenizer, tgt_tokenizer = load_and_prepare_data(settings)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using execution device: {device}")
    
    # 2. Build the Model
    model = build_seq2seq_transformer(
        src_vocab_size=src_tokenizer.get_vocab_size(),
        tgt_vocab_size=tgt_tokenizer.get_vocab_size(),
        src_seq_len=settings.seq_len,
        tgt_seq_len=settings.seq_len,
        d_model=settings.d_model,
        num_layers=settings.num_layers,
        num_heads=settings.num_heads,
        dropout=settings.dropout,
        d_ff=settings.d_ff
    )
    model.to(device)
    
    # 3. Setup optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.lr, eps=1e-9)
    
    tgt_pad_id = tgt_tokenizer.token_to_id("[PAD]")
    # CrossEntropyLoss with target padding ignored and label smoothing applied
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_id, label_smoothing=0.1)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=settings.log_dir)
    
    # Define learning rate multiplier calculation for step-level scheduling
    # linear warmup for the first settings.warmup_steps steps, then inverse square root decay
    def get_lr_multiplier(step: int) -> float:
        step = max(1, step)
        if step <= settings.warmup_steps:
            return step / settings.warmup_steps
        else:
            return (settings.warmup_steps / step) ** 0.5

    # Define LambdaLR scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_multiplier)
    
    # 4. Handle checkpoint recovery
    start_epoch = 0
    global_step = 0
    if resume_path is not None and os.path.exists(resume_path):
        start_epoch, global_step = load_checkpoint(model, optimizer, scheduler, resume_path)
        # Advance starting epoch to next
        start_epoch += 1
        
    # 5. Training loop
    for epoch in range(start_epoch, settings.num_epochs):
        # Train one epoch
        global_step = run_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            writer=writer,
            epoch=epoch,
            global_step=global_step,
            scheduler=scheduler
        )
        
        # Run validation
        validate_epoch(
            model=model,
            dataloader=val_loader,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            device=device,
            writer=writer,
            epoch=epoch,
            max_len=settings.seq_len
        )
        
        # Save checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            settings=settings,
            filename=f"checkpoint_epoch_{epoch+1}.pt"
        )
        
    writer.close()
    print("Training process finished successfully.")
