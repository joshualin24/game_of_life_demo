# Representation-vector similarity

Study of how similar the internal representation vectors of the trained
GoL CNN-Transformer models are — across grids, patterns, transformations,
and across the two model scales (V8, 291K params; V10, 1.5M params).

## Models under study

| Model | arch | d_model | heads | layers | dim_ff | params |
|-------|------|---------|-------|--------|--------|--------|
| V8  | CNNTransformerV4 | 64  | 4 | 4 | 256 | 291,888   |
| V10 | CNNTransformerV4 | 128 | 8 | 6 | 512 | 1,504,240 |

Both are the **same class** (`CNNTransformerV4`); they differ only in constructor args.
`dim_ff` is auto-set to `d_model * 4`, not a constructor arg.

### Checkpoints (weights)

Live in `nn/checkpoints/` — **gitignored**, so present only in the main checkout,
NOT in git worktrees. Bare `state_dict` saves (no optimizer/epoch/config inside).

| Model | best checkpoint | size | other files |
|-------|-----------------|------|-------------|
| V8  | `/Users/Hao-Yuan/game_of_life_demo/nn/checkpoints/task16_cnn_transformer_v8_best.pt`  | 1.1 MB | `_final.pt`, `_ep010/020/030.pt` |
| V10 | `/Users/Hao-Yuan/game_of_life_demo/nn/checkpoints/task18_cnn_transformer_v10_best.pt` | 5.8 MB | `_final.pt`, `_ep010`…`_ep100.pt` |

### Training summary

- **V8** (Task 16): warm-started from V7 best; +30 epochs with 3× high-density
  upsampled data; scheduled sampling K=3. val_loss=0.20064, F1=1.0000, prec=100%, rec=99.9%.
  "Best per-grid accuracy at 291K params."
- **V10** (Task 18): trained **from scratch** (d_model change blocks weight transfer
  from V8); scheduled sampling K=3, V8's dataset config (not V9's 5× data), 100 epochs.
  val_loss=0.19852, F1/prec/rec = 100%. **Zero errors on 2,500/2,500** failure-search grids.
- V7 (0.20114) is the stable single-step baseline both build on.
- V9 was a negative result (5× more high-density data → tail errors ~7× worse); abandoned.

## Code copied into this folder (self-contained reference)

| File | What | Source |
|------|------|--------|
| `model.py`       | `CNNTransformerV4` class, verbatim | `nn/models.py` (Task 12 section) |
| `load_models.py` | `load_v8()`, `load_v10()`, `DEVICE`, config + checkpoint-path dicts | `nn/train_cnn_transformer_v{8,10}.py` |

`model.py` is a copy — re-sync if `nn/models.py::CNNTransformerV4` changes.
Checkpoints are **not** copied (large binaries, gitignored); `load_models.py`
points at the main checkout by absolute path (override with `$GOL_CKPT_DIR`).

Quick check: `python load_models.py` — prints param counts for both models.

## Representation vectors available per grid

From `CNNTransformerV4.forward` (all for a single input grid `x: (B, 1, 40, 40)`):

- `feat`   — CNN encoder output, `(B, d_model, 40, 40)`
- `tokens` — per-patch tokens after `patch_proj` + `pos_embed`, `(B, 100, d_model)`
- `out`    — per-patch tokens after the transformer encoder, `(B, 100, d_model)`

No CLS token; the reconstruction head is per-patch. `n_patches = (40/4)^2 = 100`.

## Questions

- (fill in)

## Layout

- `data/`    — cached grids / extracted representation tensors
- `results/` — figures and numeric summaries
