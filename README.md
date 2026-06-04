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

## Notes

- The logistic-map cobweb simulation in `period_coexist.py` is a companion study of periodicity in a related 1D dynamical system.
- The move-perturbation pipeline (`perturbation_move.py`) is designed to be extended: subclass `SensitivitySweep` to plug in new perturbation types, or set `t_perturb > 0` to study mid-run sensitivity.
