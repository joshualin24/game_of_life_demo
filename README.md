# Game of Life Demo

A demonstration of [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) — a classic cellular automaton — with a focus on **perturbation sensitivity analysis**.

## Rules

1. Any live cell with fewer than 2 live neighbours dies (underpopulation).
2. Any live cell with 2 or 3 live neighbours survives.
3. Any live cell with more than 3 live neighbours dies (overpopulation).
4. Any dead cell with exactly 3 live neighbours becomes alive (reproduction).

## Getting Started

```bash
pip install numpy matplotlib
```

Run the base simulation:
```bash
python simulate.py
```

Run the pattern taxonomy demo:
```bash
python pattern_taxonomy.py
```

---

## Perturbation Analysis

The core research in this repo studies how sensitive Game of Life trajectories are to small changes in the initial (or mid-run) state. Two perturbation methods are implemented.

### Method 1 — Cell Flip (`perturbation.py`, `perturbation_patterns.py`)

For each cell in the grid, flip its state (alive → dead or dead → alive) at `t=0`, then measure how much the resulting trajectory diverges from the unperturbed baseline over the simulation.

**Divergence** is measured as the number of cells that differ from the baseline at each step, summed (cumulative) or taken at the final step.

Run the random-grid flip analysis:
```bash
python perturbation.py
```

Run the named-pattern flip analysis (all 10 patterns):
```bash
python perturbation_patterns.py
```

Outputs → `figures/`

---

### Method 2 — Cell Move (`perturbation_move.py`)

For each **living** cell, move it to a new position by a displacement vector `(dr, dc)`, then measure trajectory divergence against the baseline. A move is only valid if the destination cell is empty, ensuring every perturbation is a genuine two-cell change (source disappears, destination appears).

**Perturbation order** is defined by the Manhattan distance of the displacement. For order `d`, all `4d` displacement vectors at that distance are tried per cell, and the maximum divergence across all valid directions is recorded.

| Order | Directions tried per cell | Example displacements |
|-------|--------------------------|----------------------|
| 1 | 4 | `(0,±1)`, `(±1,0)` |
| 2 | 8 | above + `(±1,±1)` |
| 3 | 12 | above + `(±1,±2)`, `(±2,±1)` |

The sweep supports **mid-run perturbation** via `t_perturb > 0` (inject the perturbation at any generation, not just `t=0`).

#### Running the move analysis

```bash
# Random initial grid, orders 1–3
python run_move_analysis.py

# Acorn methuselah, orders 1–3
python run_acorn_analysis.py

# All 9 named patterns, orders 1–3
python run_patterns_analysis.py
```

Outputs → `figures_move/<pattern>/order_<d>/`

---

## Examples

### Patterns covered

| Pattern | Category | Steps | Living cells |
|---------|----------|-------|-------------|
| block | Still life | 40 | 4 |
| beehive | Still life | 40 | 6 |
| blinker | Oscillator (p2) | 60 | 3 |
| toad | Oscillator (p2) | 60 | 6 |
| beacon | Oscillator (p2) | 60 | 6 |
| pulsar | Oscillator (p3) | 60 | 48 |
| glider | Spaceship | 80 | 5 |
| lwss | Spaceship | 80 | 9 |
| r_pentomino | Methuselah | 120 | 5 |
| acorn | Methuselah | 120 | 7 |
| random | Random (seed 42, density 35%) | 60 | ~560 |

### Output figures (per pattern per order)

| File | Description |
|------|-------------|
| `01_sensitivity_cumulative.png` | Heatmap of cumulative divergence per source cell |
| `02_sensitivity_final.png` | Heatmap of divergence at the final step |
| `03_divergence_over_time.png` | Divergence curves for top / mid / low impact cells |
| `04_impact_distribution.png` | Histogram of impact across all valid perturbations |
| `05_baseline_vs_top.png` | Baseline vs. highest-impact perturbation snapshots |
| `06_difference_maps.png` | Difference grids over time for the top perturbation |
| `07_sensitivity_gif.gif` | Animated cumulative divergence map |

### Folder structure

```
figures/                        ← flip-perturbation results
  <pattern>/                    ← per-pattern subfolder
  *.png / *.gif                 ← random-grid summary figures

figures_move/                   ← move-perturbation results
  <pattern>/
    order_1/
    order_2/
    order_3/
  random/
    order_1/
    order_2/
    order_3/
```

---

## Die-down Analysis

A perturbation **dies down** when the perturbed trajectory fully converges back to the baseline by the final step (final divergence = 0). The analysis uses two definitions:

- **Worst-case (max-cumulative direction):** only the direction that caused the largest total divergence per cell is checked. A cell dies down only if even its worst perturbation converges back.
- **Any-direction:** a cell dies down if *any* of its valid displacement directions converges back. This is the more natural definition and is reported below.

Running simulations at 5× and 20× the original step count produces identical die-down counts, confirming that convergence happens immediately or not at all.

### Results (any-direction die-down)

| Pattern | Order | Die-down pairs / valid pairs | Cells with ≥1 die-down dir | Notes |
|---------|-------|-----------------------------|-----------------------------|-------|
| block | 1 | 8 / 8 (100%) | 4 / 4 (100%) | Self-repairs in 1 step |
| block | 2 | 12 / 28 (43%) | 4 / 4 (100%) | Some directions recover |
| block | 3 | 24 / 48 (50%) | 4 / 4 (100%) | Half of directions recover |
| beehive | 1 | 2 / 20 (10%) | 2 / 6 (33%) | Two cells have a safe direction |
| beehive | 2–3 | 0 | 0 | — |
| glider | 1 | 2 / 14 (14%) | 2 / 5 (40%) | Two cells have a safe direction |
| glider | 2 | 1 / 32 (3%) | 1 / 5 (20%) | |
| glider | 3 | 0 | 0 | — |
| lwss | 1 | 4 / 26 (15%) | 2 / 9 (22%) | Spaceship is partially resilient |
| lwss | 2 | 4 / 58 (7%) | 1 / 9 (11%) | |
| lwss | 3 | 6 / 94 (6%) | 1 / 9 (11%) | |
| acorn | 1 | 1 / 22 (5%) | 1 / 7 (14%) | |
| acorn | 2 | 1 / 50 (2%) | 1 / 7 (14%) | |
| acorn | 3 | 0 | 0 | — |
| random (seed 42) | 1 | 22 / 1424 (1.5%) | 20 / 548 (3.7%) | |
| random | 2–3 | ≤4 / 2872+ (<0.2%) | ≤3 | — |
| blinker / toad / beacon / pulsar / r_pentomino | 1–3 | 0 | 0 | No recovery in any direction |
| **TOTAL** | | **93 / 10612 (0.88%)** | **48 / 1973 (2.43%)** | |

### Key observations

- **Still lifes (block, beehive)** are the most resilient: the block recovers in 100% of cells at every order because its symmetric structure can self-repair from many directions.
- **Spaceships (glider, lwss)** have a small but nonzero recovery rate — certain displacement directions preserve enough local structure for the spaceship to reform.
- **Oscillators (blinker, toad, beacon, pulsar)** have zero die-downs at all orders. Their periodic dynamics appear to be fragile to any displacement.
- **Methuselahs (r_pentomino)** also show zero die-downs, consistent with their explosive and chaotic growth phase.
- Overall, **97.6% of valid cells diverge permanently** under their best-recovery direction, confirming that Game of Life trajectories are overwhelmingly sensitive to move perturbations.

### Animated GIFs (worst-case die-downs)

The 8 cases where even the *worst-case* direction converges are in `figures_move/die_down/`. Each GIF shows three panels: baseline (left), perturbed (centre), difference map (right). Source cell marked cyan (★), destination yellow (●).

```
figures_move/die_down/
  block_order1_cell_r24_c24_dir-1+0.gif    random_order1_cell_r5_c29_dir+0-1.gif
  block_order1_cell_r24_c25_dir-1+0.gif    random_order1_cell_r7_c29_dir+0+1.gif
  block_order1_cell_r25_c24_dir+0-1.gif    random_order1_cell_r9_c2_dir+1+0.gif
  block_order1_cell_r25_c25_dir+0+1.gif    random_order1_cell_r19_c5_dir+0-1.gif
```

To regenerate:
```bash
python generate_diedown_gifs.py   # worst-case die-down GIFs (8 cases)
python collect_stats_any_dir.py   # full any-direction die-down table
```

---

## Neural Network Models (`nn/`)

A suite of neural network experiments applying deep learning to Game of Life dynamics on 40×40 toroidal grids. All models are defined in `nn/models.py` and trained with MPS (Apple Silicon) acceleration.

### Tasks 1–6: Convolutional baselines

| Task | Model | Objective |
|------|-------|-----------|
| 1 | `NextStatePredictor` | Predict t+1 from t (residual CNN) |
| 2 | `SensitivityUNet` | Predict per-cell sensitivity map (U-Net) |
| 3 | `ChaosPredictor` | Predict divergence score from initial grid + perturbation location |
| 4 | `NeuralCA` | Learn the GoL update rule as a tiny conv net |
| 5 | `RolloutPredictor` | Predict t+k directly (residual tower + step embedding) |
| 6 | `FateClassifier` | Classify attractor type: dies / still-life / oscillator / active |

### Task 7: Trajectory Transformer (embedding)

`TrajectoryTransformer` encodes a full T=60 step GoL trajectory into a single embedding vector via a ViT-style [CLS] token.

**Architecture**: CNN frame encoder (shared weights across frames) → sinusoidal positional encoding → 4-layer pre-norm transformer → CLS embedding (d_model=64 or 128). Trained on 5000 trajectories (40% random, 60% named patterns with D4 augmentation) with BCE reconstruction loss.

**Training**: `nn/train_trajectory_embedding.py` — trains d_model=64 and d_model=128 variants. Best val loss: 0.953 (d=64), 0.946 (d=128).

```bash
python -m nn.train_trajectory_embedding
```

Results (loss curves, embedding visualisations): `nn/results/traj_emb_d{64,128}_*.png`

### Task 8: Next-Step Transformer (ViT-style)

`NextStepTransformer` predicts the GoL state at t+1 given t, and can be unrolled autoregressively for any number of steps.

**Architecture**: Raw 40×40 grid divided into 100 non-overlapping 4×4 patches → linear patch embedding → learnable 2D positional encoding → 4-layer pre-norm transformer → per-patch linear head → 40×40 next-state logits. 208K parameters.

**Training**: `nn/train_next_step_transformer.py` — 200K (state_t, state_t+1) pairs from 2000 random trajectories, with D4 symmetry augmentation applied per batch (GoL is exactly D4-equivariant). 50 epochs, BCE loss. **Val accuracy: 92%**.

```bash
python -m nn.train_next_step_transformer
```

**Limitation**: The 4×4 patching cuts across the 3×3 GoL neighbourhood, so cells at patch boundaries must rely on global attention to see their neighbours. This causes stripe/checkerboard artifacts in long autoregressive rollouts.

Results: `nn/results/task8_next_step_transformer_*.png`

### Task 9: CNN-Transformer Hybrid

`CNNTransformer` addresses the patch-boundary problem by inserting a CNN local encoder before the transformer.

**Architecture**:
1. **CNN local encoder** (RF=5×5): `Conv(1→32, 3×3) → GroupNorm → GELU → Conv(32→64, 3×3) → GroupNorm → GELU` — every position already sees its full 3×3 GoL neighbourhood before tokenisation. GroupNorm (not BatchNorm) ensures train/eval consistency.
2. **Patch tokenisation**: spatial avg-pool over 4×4 windows → (B, 100, 64) tokens + learnable positional embeddings.
3. **Transformer encoder**: 4-layer pre-norm, global self-attention for long-range context.
4. **Per-patch head**: linear → 4×4 cell logits → reshape to 40×40.

226K parameters (comparable to Task 8 for a fair comparison).

**Training**: `nn/train_cnn_transformer.py` — same data and D4 augmentation as Task 8. 50 epochs. **Val accuracy: 89.4%**. Val loss was still declining at epoch 50; longer training expected to improve further.

```bash
python -m nn.train_cnn_transformer
```

**Vs. pure ViT**: Better at sustaining sparse out-of-distribution patterns (glider stays alive for 30+ steps vs. collapsing to a static dot); slightly lower flat accuracy on random grids (89% vs. 92%). Horizontal stripe artifacts remain in long rollouts — likely addressable with 2D sinusoidal positional embeddings.

Results: `nn/results/task9_cnn_transformer_*.png`

### Task 10: CNN-Transformer V2 (2D sinusoidal positional encoding)

`CNNTransformerV2` replaces the 1D learnable positional embedding of Task 9 with a fixed 2D sinusoidal encoding. Each patch (row i, col j) receives `[sinusoidal(i) | sinusoidal(j)]` — no learnable positional parameters, no row-major index bias.

**Architecture**: Identical to Task 9 except `pos_embed` (learnable, 6.4K params) is replaced by `_SinusoidalPE2D` (fixed buffer). 219K parameters.

**Training**: `nn/train_cnn_transformer_v2.py` — same data and augmentation as Tasks 8–9, 100 epochs. **Best val accuracy: 84%** (best checkpoint from epoch 1; training showed large val-loss oscillations throughout).

```bash
python -m nn.train_cnn_transformer_v2
```

**Findings**: The 2D sinusoidal PE did not improve over the learnable embedding — training was less stable and rollout quality was worse. The core bottleneck is **error compounding in autoregressive rollout**: at 92% single-step accuracy, ~40% of cells are wrong after just 6 steps. Architectural fixes (patch size, positional encoding) have limited impact; the next step is **scheduled sampling** during training to teach the model to recover from its own prediction errors.

Results: `nn/results/task10_cnn_transformer_2d_*.png`

### Task 11: CNN-Transformer V3 (lossless flatten+linear tokenization) ⭐

`CNNTransformerV3` fixes the key information bottleneck of V1/V2: the `avg_pool2d` that compressed each 4×4 patch of CNN features into a single vector is replaced by a **flatten + learned linear projection** (`Linear(d_model·p², d_model)` = `Linear(1024, 64)`). All CNN features within each patch are preserved — the transformer decides via learned weights which aspects to keep, rather than a fixed average that discards spatial detail.

**Why this matters for GoL**: a single alive cell in the corner of a patch contributes only 1/16 of its signal after avg_pool, but its full signal after flatten+project. GoL is a sparse binary rule — every cell counts.

**Architecture**: identical to V1 except Stage 2 tokenization. 291K parameters (65K extra for `patch_proj`).

**Training**: `nn/train_cnn_transformer_v3.py` — same 200K pairs and D4 augmentation. Converges in the very first epoch. **Val accuracy: 99.0%**, best val loss: 0.021.

```bash
python -m nn.train_cnn_transformer_v3
```

**Results**: First model to produce qualitatively correct long-run autoregressive rollouts — random grids evolve with realistic GoL dynamics for 10+ steps; glider stays alive and evolving for all 30 steps. The avg_pool was the primary bottleneck across all previous architectures.

| Model | Val accuracy | Random rollout | Glider rollout |
|---|---|---|---|
| Task 8 — Pure ViT | 92% | freezes to checkerboard | static dot |
| Task 9 — CNN + avg_pool | 89% | near-dead + edge artifact | dynamic blob |
| Task 10 — CNN + sinusoidal PE | 84% | 2 dots | static dot |
| **Task 11 — CNN + flatten+proj** | **99%** | **realistic GoL dynamics** | **stays alive + evolves** |

Results: `nn/results/task11_cnn_transformer_v3_*.png`

### Task 12: CNNTransformerV4 — circular (toroidal) padding

`CNNTransformerV4` fixes a boundary bug in V3: the CNN stage used zero-padding, so the ~156 cells along the grid border saw incorrect neighbour counts every step (their true GoL neighbourhood wraps around the torus, but zero-padding treats the edge as a hard wall). `padding_mode='circular'` makes the convolution respect the toroidal topology exactly.

**Training**: `nn/train_cnn_transformer_v4.py` — warmstart from V3, label smoothing ε=0.1, LR=1e-4, D4-only augmentation, 100 epochs. Best val_loss=0.22623, F1=0.970, prec=100%, rec=94.3%.

```bash
python -m nn.train_cnn_transformer_v4
```

Results: `nn/results/task12_cnn_transformer_v4_*.png`

### Task 13: CNNTransformerV5 — density-diverse training

Fixes V4's density bias (over-predicts births on sparse grids, over-predicts deaths on dense grids) by training on 8 densities × 200 trajectories + 13 named patterns × 60 placements + 1000 multi-pattern combos (93,600 pairs total).

**Training**: `nn/train_cnn_transformer_v5.py` — warmstart V4, LR=1e-4, 100 epochs. Best val_loss=0.20125, prec=99.4%, rec=99.7%, but **born accuracy only 93.8%** — a symptom of teacher-forcing exposure bias that becomes the focus of Tasks 14–15.

```bash
python -m nn.train_cnn_transformer_v5
```

Results: `nn/results/task13_cnn_transformer_v5_*.png`

### Task 14: CNNTransformerV6 — multi-step STE (failed)

Attempts to fix autoregressive rollout collapse (V5 unrolled on its own predictions drifts to all-dead on sparse grids) by unrolling K=3 steps during training with a straight-through estimator (STE): hard binarized predictions forward, sigmoid gradient backward.

**Result: diverges.** Best at epoch 5 (val=0.23421, prec=92.7%, born=99.5%), then precision collapses to ~20% by epoch 10 and oscillates chaotically through epoch 50. **Root cause**: STE gradients carry a systematic "predict more alive" bias — a missed birth at step *t* makes step *t+1*'s gradient push toward predicting more live cells, regardless of whether that's actually correct. Gradient clipping doesn't help because the problem is gradient *direction*, not magnitude.

```bash
python -m nn.train_cnn_transformer_v6
```

Results: `nn/results/task14_cnn_transformer_v6_*.png`

### Task 15: CNNTransformerV7 — scheduled sampling ⭐

Replaces STE with **scheduled sampling**: at each of K=3 unrolled steps, the next input is the model's own prediction with probability *p* (linearly increasing 0→1 over training) or the ground truth with probability 1−*p*. Gradients flow only through the current step's logits — no cascading bias.

**Training**: `nn/train_cnn_transformer_v7.py` — warmstart V5, LR=1e-5, 50 epochs. **Best val_loss=0.20114**, F1=0.9985, prec=99.7%, rec=100%, **born=100%, surv=100%, died=99.7%** — zero precision collapse across all 50 epochs, and the sparse-grid rollout-collapse artifact from V5/V6 disappears entirely.

```bash
python -m nn.train_cnn_transformer_v7
```

Results: `nn/results/task15_cnn_transformer_v7_*.png`

### Task 16: CNNTransformerV8 — high-density fine-tuning

A targeted failure search on V7 (random grids, densities 0.02–0.80, plus named patterns) finds essentially all remaining errors concentrated at density=0.80 (~100 false positives/grid). V8 continues training from V7 for 30 epochs on a dataset upsampling density 0.65/0.80 trajectories 3×, while keeping the full range of lower densities.

**Training**: `nn/train_cnn_transformer_v8.py` — warmstart V7, LR=5e-6, 30 epochs. Best val_loss=0.20064, prec=100%, rec=99.9%. False positives at d=0.80 drop from ~100/grid to ~4.6/grid (**21× improvement**).

```bash
python -m nn.train_cnn_transformer_v8
```

Results: `nn/results/task16_cnn_transformer_v8_*.png`, `nn/compare_v7_v8_failures.py` (side-by-side failure-case comparison)

### Task 17: CNNTransformerV9 — 5× data scale-up (regressed)

Tests whether scaling *all* training data 5× (≈546K samples, warmstart V8) improves further. Aggregate validation metrics look essentially unchanged (val_loss=0.20118, prec=99.7%, rec=99.6%), **but per-grid tail-case errors get an order of magnitude worse**: false positives at d=0.80 jump from V8's 2,296 (across 500 grids) to 22,190; false negatives at d=0.50 jump from 67 to 6,948.

**Lesson**: once a model has saturated its representational capacity, more data of the kind it already sees plenty of doesn't help — and can hurt. Aggregate metrics alone completely missed this regression; it only showed up under a density-stratified failure audit.

```bash
python -m nn.train_cnn_transformer_v9
```

Results: `nn/results/task17_cnn_transformer_v9_*.png`, `nn/compare_v8_v10_errors.py`

### Task 18: CNNTransformerV10 — 5.15× model scale-up ⭐

Scales the *model* instead of the data: `d_model` 64→128, heads 4→8, layers 4→6 (1.5M params vs. V8's 291K). Trained from scratch (different embedding width can't warmstart from V8) with the same scheduled-sampling recipe and V8's dataset.

**Result**: converges to 100% precision/recall/born/surv/died by epoch 10 of 100, val_loss=0.19852 (below every prior version's floor). Repeating V8/V9's exact failure-search protocol (500 grids × 5 densities), **V10 makes zero errors across all 2,500 evaluated grids**.

```bash
python -m nn.train_cnn_transformer_v10
```

Results: `nn/results/task18_cnn_transformer_v10_*.png`

### Task 19: CNNTransformerV11 — polynomial activation (negative result)

Motivated by Ahmed & Davis (2026), arXiv:2606.23587, which shows a learnable 2nd-degree polynomial activation lets a *tiny* CNN learn GoL where ReLU fails almost completely (see `poly_activation_verify/` below). V11 swaps the same polynomial activation into the CNN stage of V8's architecture (only +288 params over V8), trained from scratch with V10's scheduled-sampling recipe.

**Result: diverges.** Promising at epoch 5 (F1=0.89, born=86.8%), then regresses sharply by epoch 10 (val=0.49) and diverges further by epoch 15 (val=1.05), with per-epoch time also ballooning ~5×. Training loss stayed flat throughout, meaning the instability is specific to the autoregressive/scheduled-sampling evaluation distribution, not general optimization failure — likely the unbounded quadratic term (`w2·x²`) compounding across the multi-step rollout, a failure mode structurally similar to V6's STE divergence. Not resolved within this project; candidate fixes (bounding the quadratic term, separate LR for poly coefficients, single-step verification before adding rollout) are noted in the paper.

```bash
python -m nn.train_cnn_transformer_v11
```

Results: `nn/train_log_cnn_transformer_v11.txt`

### Version summary (Tasks 12–19)

| Ver. | Key change | Best val loss | Prec. | Rec. | Born | Status |
|---|---|---|---|---|---|---|
| V4 | Circular padding fix | 0.22623 | 100% | 94.3% | 94.3% | Boundary bug fixed |
| V5 | Density-diverse training | 0.20125 | 99.4% | 99.7% | 93.8% | Collapses under rollout |
| V6 | +STE multi-step | 0.23421¹ | 92.7%¹ | — | 99.5%¹ | **Failed** — diverges |
| V7 | STE → scheduled sampling | 0.20114 | 99.7% | 100% | 100% | Stable, strong |
| V8 | +3× high-density upsample | 0.20064 | 100% | 99.9% | 99.9% | Best per-grid accuracy (291K) |
| V9 | +5× all data | 0.20118 | 99.7% | 99.6% | 99.6% | **Regressed** per-grid errors |
| V10 | 5.15× model size, from scratch | 0.19852 | 100% | 100% | 100% | **Zero errors**, 2,500/2,500 grids |
| V11 | +Polynomial CNN activation | — | — | — | — | **Failed** — diverges |

¹V6's best epoch (5) before divergence.

### Embedding Analysis (`nn/embedding_analysis.py`)

A reusable toolkit (`EmbeddingExtractor`, `Transforms`, `compare()`, plotting helpers) for studying what the patch embeddings of V8/V10 encode, extracted both **pre-transformer** (patch content + learnable positional embedding) and **post-transformer** (after self-attention mixes all 100 patches).

- **`run_embedding_gallery.py`** — patch-norm heatmaps + pairwise cosine-similarity matrices for oscillators, gliders, and random configurations (`--model v8|v10|both`)
- **`run_embedding_examples.py`** — embedding comparison under rotation/flip/translation/cell-level perturbation, one figure per (transform, config) pair
- **`run_embedding_diff.py`** — Δ-norm heatmaps, cross-transformation similarity matrices, and PCA scatter plots of how embeddings move under each transform (`--model v8|v10|both`)
- **`compare_v2_v8_equivariance.py`** — quantifies translation-sensitivity of V2's fixed 2D sinusoidal positional encoding vs. V8's learned positional embedding. **Finding**: mean post-transformer cosine similarity under translation is 0.997 (V2, sinusoidal) vs. 0.890 (V8, learned) — sinusoidal PE's angle-addition identity gives it a genuine equivariant structure under translation that a learned per-slot positional vector doesn't have. The gap is largest on dense random grids (0.99 vs. 0.75–0.78 at density 0.35) and nearly invisible on sparse patterns (where most patches are empty either way).

**Key findings**: sparse configurations (gliders, oscillators) are nearly embedding-invariant under all transforms, since most of the 100 patches are empty and empty-patch embeddings collapse together regardless of position. Dense random grids are far more sensitive to rotation/reflection/translation (cosine similarity drops to ~0.75–0.92), because every patch carries genuine content and the *learned* absolute positional embedding is not translation- or rotation-equivariant — this is exactly why augmentation uses only the D4 group (rotation/reflection) and not translation.

Results: `nn/results/gallery_*.png`, `nn/results/embdiff_*.png`, `nn/results/embed_v8_*.png`, `nn/results/compare_v2_v8_translation_equivariance.png`

### Independent Verification: Polynomial Activations (`poly_activation_verify/`)

A self-contained reproduction (no dependency on the rest of this repo) of Ahmed & Davis (2026), arXiv:2606.23587 — "It's Much Easier for Neural Networks to Learn Game of Life Dynamics with the Right Activation Function: Polynomial Kolmogorov-Arnold Networks." Builds their minimal `L(1,m=1)` CNN (one 3×3 circular conv, one 1×1 conv, one 1×1 conv+sigmoid) and trains two variants — identical except the activation function — on our own 40×40 toroidal grid, 10 random seeds each.

| Activation | Parameters | Success rate (10 seeds) | Mean final accuracy |
|---|---|---|---|
| ReLU | 25 | 1/10 | 89.4% |
| Polynomial (2nd-degree, learnable) | 34 | **10/10** | **100.0%** |

Parameter counts (25/34) match the paper's reported minimal-network sizes exactly, confirming a faithful reproduction. ReLU converges to perfect accuracy on only 1 of 10 seeds; the polynomial activation converges on every seed, each run completing in 1–2 seconds on CPU.

```bash
python poly_activation_verify/train_compare.py
```

Results: `poly_activation_verify/results/relu_vs_poly_convergence.png`, `summary.json`

### Research Paper (`paper/`)

A LaTeX writeup (`paper/main.tex`, compiles with `latexmk -pdf main.tex`) documenting the full V5–V11 progression, the embedding analysis, and the polynomial-activation verification/negative-result pair, structured as: introduction, architecture, training dynamics (exposure bias → STE failure → scheduled sampling → data-vs-capacity), embedding analysis, activation-function inductive bias (reproduction + V11 negative result), and discussion.

```bash
cd paper && latexmk -pdf main.tex
```

---

## Notes

- The logistic-map cobweb simulation in `period_coexist.py` is a companion study of periodicity in a related 1D dynamical system.
- The move-perturbation pipeline (`perturbation_move.py`) is designed to be extended: subclass `SensitivitySweep` to plug in new perturbation types, or set `t_perturb > 0` to study mid-run sensitivity.
