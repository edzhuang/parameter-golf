## Depth Recurrence

Uses depth recurrence (looped transformer blocks) to trade unique parameters for effective depth, freeing parameter budget for a wider model.

### Key changes from baseline

| | Baseline | This run |
|---|---|---|
| Architecture | 9 unique layers, dim=512 | 3 unique × 5 passes = 15 effective layers, dim=768 |
| MLP | relu² 2x (2 matrices) | SwiGLU hidden=1024 (3 matrices, same param count) |
| Heads | 8 heads, 4 KV | 12 heads, 4 KV (head_dim=64) |
| Skip connections | U-Net encoder/decoder | x0 residual mixing per virtual layer |
| Unique params | ~17M | ~12.6M |
| Effective depth | 9 | 15 |
| Compressed size | ~15.8MB | ~10MB (estimated) |

### Architecture details

Each of the 3 unique blocks contains shared heavy weights (Q/K/V/O projections, SwiGLU gate/up/down). The model loops through these blocks 5 times, using per-virtual-layer lightweight parameters (attn_scale, mlp_scale, resid_mix, q_gain) to differentiate each of the 15 virtual layers. This allows the model to apply the same transformations repeatedly while refining its representation.

### Optimizer adjustments for shared weights

- `matrix_lr` reduced from 0.04 to 0.018 (shared weights accumulate gradients from 5 passes)
- `grad_clip_norm` enabled at 1.0 (stability for deep gradient flow)
- `warmdown_iters` reduced to 600 (fewer total steps expected due to heavier model)
- `muon_momentum_warmup_steps` reduced to 250

### Command

```bash
RUN_ID=depth_recurrence \
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
