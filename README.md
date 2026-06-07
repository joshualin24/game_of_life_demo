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

## Notes

- The logistic-map cobweb simulation in `period_coexist.py` is a companion study of periodicity in a related 1D dynamical system.
- The move-perturbation pipeline (`perturbation_move.py`) is designed to be extended: subclass `SensitivitySweep` to plug in new perturbation types, or set `t_perturb > 0` to study mid-run sensitivity.
