# Attention is all you need

A from-scratch, modular PyTorch reimplementation of the Transformer architecture introduced in [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017), implemented by **Mariam Maysara**.

## Motivation

Rather than treating the Transformer as a black box imported from a library, this repository reconstructs it component by component — attention, positional encoding, encoder/decoder stacks, masking — to build an implementation-level understanding of:

- how scaled dot-product attention distributes context across a sequence
- why positional encoding is necessary in a permutation-invariant architecture
- how masking enforces causality in autoregressive decoding
- how encoder and decoder stacks compose into a full sequence-to-sequence model

## Architecture

Implemented as a modular package rather than a single monolithic file, mirroring how attention-based components are typically organized in research codebases:

```
transformer_core/
├── layers/
│   ├── embedding.py          # TokenEmbedding
│   ├── positional.py         # SinusoidalPositionalEncoding
│   ├── attention.py          # MultiHeadSelfAttention
│   ├── feedforward.py        # PositionWiseFFN
│   └── normalization.py      # AddNorm (residual + layer norm)
├── blocks/
│   ├── encoder_layer.py      # TransformerEncoderLayer
│   └── decoder_layer.py      # TransformerDecoderLayer
├── modules/
│   ├── encoder_stack.py      # EncoderStack
│   ├── decoder_stack.py      # DecoderStack
│   └── seq2seq_transformer.py  # Seq2SeqTransformer (full model)
├── data/
│   ├── vocab_builder.py      # tokenizer training/loading
│   └── parallel_corpus.py    # dataset + padding/causal mask construction
├── training/
│   ├── settings.py           # hyperparameter configuration
│   └── loop.py                # training loop, validation, checkpointing
└── scripts/
    ├── run_training.py       # training entry point
    └── shape_check.py        # architecture smoke test
```

Each module maps directly to a specific section of the original paper (referenced in code comments), so the implementation can be read alongside the paper section by section.

## Setup

```bash
git clone https://github.com/<your-username>/attention-is-all-you-need.git
cd attention-is-all-you-need
pip install -r requirements.txt
```

## Sanity check

Before training, verify the architecture wiring with a fast shape test on a tiny model:

```bash
python -m transformer_core.scripts.shape_check
```

## Training

Trained on a small parallel-translation corpus ([Multi30k](https://github.com/multi30k/dataset) / [opus_books](https://huggingface.co/datasets/opus_books)) as a controlled task for validating that the architecture learns correctly — the goal of this repo is architectural understanding, not translation quality.

```bash
python -m transformer_core.scripts.run_training
```

Training logs (loss, CER/WER/BLEU) are tracked via TensorBoard:

```bash
tensorboard --logdir runs/
```

## Results

*(fill in after training: loss curve screenshot, final BLEU/CER/WER, a few example translations)*

| Metric | Value |
|---|---|
| Final training loss | — |
| Validation BLEU | — |
| Validation CER | — |
| Validation WER | — |

## Reference

```bibtex
@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017}
}
```
---
<div align="center">

Implemented by Mariam Maysara.

</div>
