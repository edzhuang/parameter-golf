## SwiGLU MLP + MTP + BigramHash 4k

Replaces the relu-squared MLP with SwiGLU at matched parameter count, enables multi-token prediction as a free auxiliary training signal, and doubles the bigram hash table size.

### Key changes from SOTA (2026-03-22 GPTQ-lite, 1.1233 bpb)

| | Previous SOTA | This run |
|---|---|---|
| MLP | relu² 2 matrices, 3x expansion (hidden=1536) | SwiGLU 3 matrices, 2x expansion (hidden=1024) |
| MLP params/layer | 2 * 512 * 1536 = 1,572,864 | 3 * 512 * 1024 = 1,572,864 |
| BigramHash buckets | 2048 | 4096 |
| MTP heads | 0 | 1 (weight=0.2, excluded from export) |

### Why SwiGLU

SwiGLU (Shazeer 2020) replaces the relu-squared activation with a gated linear unit using SiLU gating. At matched parameter count (3 matrices at 2x expansion = 2 matrices at 3x expansion), SwiGLU consistently outperforms other activation functions in language modeling benchmarks. It's used in LLaMA, Gemma, Mistral, and most modern architectures.

The gate-up-down factorization also provides a natural inductive bias: the gate learns what information to keep, while the up projection learns the transformation.

### Why MTP

Multi-token prediction adds a lightweight auxiliary head that predicts token t+2 from the hidden state at position t. This head is excluded from the exported artifact (0 extra bytes), but provides additional gradient signal during training that encourages the model to learn more predictive representations.

### Why BigramHash 4k

The Int5MLP submission demonstrated that larger bigram hash tables reduce hash collisions and improve bpb. Doubling from 2048 to 4096 adds ~0.25MB compressed but reduces collisions significantly.

### All other techniques preserved from SOTA

- 11 layers, 512-dim, 8 heads, 4 KV heads (GQA)
- GPTQ-lite int6 quantization with per-row clip percentile search
- EMA (decay=0.997) + SWA during warmdown
- Late QAT (STE int6 fake-quant when LR scale < 0.15)
- Partial RoPE (16/64 dims) + LN Scale (1/sqrt(layer_idx+1))
- XSA on last 4 layers
- SmearGate + U-Net skip connections
- Value Embeddings on layers 9, 10
- Muon WD=0.04, momentum 0.99, warmdown=3500
- FlashAttention 3, zstd-22 compression
- Sliding window eval stride=64

### Command

```bash
RUN_ID=swiglu_mtp_bigram4k \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Results

_To be filled after RunPod run._
