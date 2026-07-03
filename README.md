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

---

## Notes

- The logistic-map cobweb simulation in `period_coexist.py` is a companion study of periodicity in a related 1D dynamical system.
- The move-perturbation pipeline (`perturbation_move.py`) is designed to be extended: subclass `SensitivitySweep` to plug in new perturbation types, or set `t_perturb > 0` to study mid-run sensitivity.
