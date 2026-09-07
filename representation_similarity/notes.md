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

- (fill in — user has a specific direction in mind, TBD)

## Layout

- `data/`    — cached grids / extracted representation tensors
- `results/` — figures and numeric summaries

---

# Build 1 — reusable scaffolding (inference-only, no training)

General-purpose tooling for directions 2 (cross-grid similarity) and 3
(layer-wise evolution). Built ahead of the real research question so it's on
hand later.

| File | Purpose |
|------|---------|
| `common.py`   | toroidal GoL, period detection, pooling, cosine RSM, linear CKA, participation ratio |
| `extract.py`  | `extract_all(model, x)` / `embeddings_for_grids(model, grids)` → per-stage acts `{cnn, tokens_in, tf_L1…tf_Lk}`, each `(N,100,d_model)`. Verified **bitwise-exact** vs `model.forward` (`tf_Lk == transformer(pre)`, recon logits == `model(x)`). |
| `stimuli.py`  | `build_stimuli()` → 246 grids: still lifes (6) + oscillators (48, incl. all 15 pentadecathlon phases) + spaceships (32) + random by density 0.1–0.8 (160). Each tagged category/name/period/phase/density. |
| `analysis_grid_similarity.py` | direction 2 — per-stage RSMs + probes A/B/C |
| `analysis_layerwise.py`       | direction 3 — spread / align-to-CNN / step-CKA / participation-ratio vs depth |

Run: `python analysis_grid_similarity.py` and `python analysis_layerwise.py`
(from this folder, `pytorch-env`). Outputs → `results/`:
`grid_rsm_{v8,v10}.{png,npz}`, `layerwise_metrics.{png,json}`.

**Caveat on these first numbers:** per-grid vectors are **mean-pooled over the
100 patches**, so a large shared "average-patch" component dominates and pushes
every cosine to ~0.99. Relative structure is still meaningful (differences
between stages / models / conditions), absolute values are not. The flattened
`100·d_model` form (already supported in `common.flatten_emb`) or
mean-centering per stage will be more sensitive — do that when the real
question is set.

**What Build-1 already shows (relative, not absolute):**
- **V10 carries content structure that V8 mostly discards.** Probe B (pattern
  clustering) separation: V10 within−between ≈ +0.014 and rising with depth;
  V8 ≈ +0.000. Probe C (density gradient): V8 stays r≈−0.81 at every stage
  (embedding ≈ raw density); V10's CNN is r≈−0.91 but the transformer
  *reduces* the raw-density correlation to r≈−0.52 — it's re-coding density
  into something less linear.
- **The transformer barely moves the representation after layer 1.** step-CKA
  ≈ 0.98–1.00 for every tf layer in both models; participation ratio creeps
  up slowly (V8 1.9→2.5, V10 1.6→2.2). Most of the transform is `tokens_in`
  (patch_proj + PE), which is near-orthogonal to the CNN code
  (align_to_cnn ≈ 0 for V8, ≈ −0.13 for V10).
- Probe A (doomed-cell pairs) is near-saturated under mean-pool (gap ~0.003),
  but the gap grows monotonically with depth for both models — consistent
  with deeper layers ignoring an about-to-die isolated cell.

Raw dumps below.

## Build-1 results — direction 2 (grid similarity)

```
### Probe A — same-future pairs  (cos: A~B doomed-cell pair | A~unrelated)
    V8 cnn        same_future=1.0000   unrelated=0.9998   gap=+0.0002
    V8 tokens_in  same_future=1.0000   unrelated=0.9991   gap=+0.0009
    V8 tf_L1      same_future=1.0000   unrelated=0.9983   gap=+0.0017
    V8 tf_L2      same_future=1.0000   unrelated=0.9982   gap=+0.0018
    V8 tf_L3      same_future=1.0000   unrelated=0.9976   gap=+0.0024
    V8 tf_L4      same_future=1.0000   unrelated=0.9974   gap=+0.0026
   V10 cnn        same_future=1.0000   unrelated=0.9996   gap=+0.0004
   V10 tokens_in  same_future=1.0000   unrelated=0.9978   gap=+0.0022
   V10 tf_L1      same_future=1.0000   unrelated=0.9972   gap=+0.0028
   V10 tf_L2      same_future=1.0000   unrelated=0.9963   gap=+0.0037
   V10 tf_L3      same_future=1.0000   unrelated=0.9958   gap=+0.0042
   V10 tf_L4      same_future=1.0000   unrelated=0.9956   gap=+0.0044
   V10 tf_L5      same_future=1.0000   unrelated=0.9955   gap=+0.0045
   V10 tf_L6      same_future=1.0000   unrelated=0.9953   gap=+0.0047

### Probe B — phase clustering  (within-pattern cos | between-pattern cos)
    V8 cnn        within=0.9990   between=0.9987   sep=+0.0003
    V8 tokens_in  within=0.9996   between=0.9996   sep=+0.0001
    V8 tf_L1      within=0.9992   between=0.9991   sep=+0.0001
    V8 tf_L2      within=0.9991   between=0.9990   sep=+0.0001
    V8 tf_L3      within=0.9987   between=0.9986   sep=+0.0000
    V8 tf_L4      within=0.9985   between=0.9985   sep=+0.0000
   V10 cnn        within=0.9966   between=0.9916   sep=+0.0050
   V10 tokens_in  within=0.9989   between=0.9884   sep=+0.0105
   V10 tf_L1      within=0.9989   between=0.9865   sep=+0.0124
   V10 tf_L2      within=0.9988   between=0.9855   sep=+0.0133
   V10 tf_L3      within=0.9988   between=0.9851   sep=+0.0138
   V10 tf_L4      within=0.9988   between=0.9849   sep=+0.0139
   V10 tf_L5      within=0.9988   between=0.9848   sep=+0.0140
   V10 tf_L6      within=0.9988   between=0.9847   sep=+0.0141

### Probe C — corr(|Δdensity|, cosine sim) over random grids  (want negative)
    V8 cnn        r=-0.8193
    V8 tokens_in  r=-0.8281
    V8 tf_L1      r=-0.8188
    V8 tf_L2      r=-0.8212
    V8 tf_L3      r=-0.8095
    V8 tf_L4      r=-0.8059
   V10 cnn        r=-0.9051
   V10 tokens_in  r=-0.6069
   V10 tf_L1      r=-0.5793
   V10 tf_L2      r=-0.5357
   V10 tf_L3      r=-0.5278
   V10 tf_L4      r=-0.5239
   V10 tf_L5      r=-0.5214
   V10 tf_L6      r=-0.5201
```


## Build-1 results — direction 3 (layer-wise)

```
  V8
    cnn        spread=+0.585  align_cnn=+1.000  step_cka=1.000  part_ratio=  1.07
    tokens_in  spread=+0.957  align_cnn=+0.020  step_cka=0.833  part_ratio=  1.87
    tf_L1      spread=+0.947  align_cnn=+0.011  step_cka=0.993  part_ratio=  2.15
    tf_L2      spread=+0.942  align_cnn=+0.009  step_cka=0.999  part_ratio=  2.21
    tf_L3      spread=+0.936  align_cnn=+0.017  step_cka=0.982  part_ratio=  2.41
    tf_L4      spread=+0.937  align_cnn=+0.016  step_cka=0.995  part_ratio=  2.51
  V10
    cnn        spread=+0.642  align_cnn=+1.000  step_cka=1.000  part_ratio=  1.38
    tokens_in  spread=+0.857  align_cnn=-0.137  step_cka=0.484  part_ratio=  1.58
    tf_L1      spread=+0.857  align_cnn=-0.136  step_cka=0.998  part_ratio=  1.67
    tf_L2      spread=+0.854  align_cnn=-0.130  step_cka=0.986  part_ratio=  1.95
    tf_L3      spread=+0.857  align_cnn=-0.123  step_cka=0.990  part_ratio=  2.16
    tf_L4      spread=+0.859  align_cnn=-0.122  step_cka=1.000  part_ratio=  2.18
    tf_L5      spread=+0.859  align_cnn=-0.122  step_cka=1.000  part_ratio=  2.19
    tf_L6      spread=+0.858  align_cnn=-0.122  step_cka=1.000  part_ratio=  2.20
```
