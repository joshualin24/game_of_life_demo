"""
collect_stats_extended.py
--------------------------
Re-runs all move-perturbation sweeps at 1×, 5×, and 20× the original step
counts to test whether more perturbations die down given longer simulation time.
"""

import time
import numpy as np
from perturbation_move import (
    SweepConfig, SensitivitySweep, manhattan_displacements
)
from perturbation_patterns import PATTERNS, GRID_SIZE

# ── helpers ───────────────────────────────────────────────────────────────────

def make_grid(pattern_cells: np.ndarray, size: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=np.uint8)
    ph, pw = pattern_cells.shape
    top  = (size - ph) // 2
    left = (size - pw) // 2
    grid[top:top+ph, left:left+pw] = pattern_cells
    return grid

ACORN_CELLS = [
    (0, 1), (1, 3),
    (2, 0), (2, 1), (2, 4), (2, 5), (2, 6),
]
def make_acorn_grid(size: int = 40) -> np.ndarray:
    grid = np.zeros((size, size), dtype=np.uint8)
    top  = (size - 3) // 2
    left = (size - 7) // 2
    for r, c in ACORN_CELLS:
        grid[top + r, left + c] = 1
    return grid

# ── build configuration list ──────────────────────────────────────────────────

BASE_CONFIGS = []

# random
BASE_CONFIGS.append(dict(
    label="random", base_steps=60, grid_size=40,
    cfg_kwargs=dict(seed=42, density=0.35), init=None,
))

# acorn
BASE_CONFIGS.append(dict(
    label="acorn", base_steps=60, grid_size=40,
    cfg_kwargs={}, init=make_acorn_grid(),
))

# named patterns (excluding acorn)
for name, spec in PATTERNS.items():
    if name == "acorn":
        continue
    BASE_CONFIGS.append(dict(
        label=name, base_steps=spec["steps"], grid_size=GRID_SIZE,
        cfg_kwargs={}, init=make_grid(spec["cells"], GRID_SIZE),
    ))

MULTIPLIERS = [1, 5, 20]
ORDERS      = [1, 2, 3]

# ── run ───────────────────────────────────────────────────────────────────────

# Results: label → order → multiplier → (n_valid, n_died)
results = {bc["label"]: {d: {} for d in ORDERS} for bc in BASE_CONFIGS}

for bc in BASE_CONFIGS:
    label = bc["label"]
    for d in ORDERS:
        disps = manhattan_displacements(d)
        for mult in MULTIPLIERS:
            steps = bc["base_steps"] * mult
            cfg = SweepConfig(
                grid_size=bc["grid_size"],
                steps=steps,
                t_perturb=0,
                out_dir="",
                **bc["cfg_kwargs"],
            )
            t0    = time.time()
            sweep = SensitivitySweep(cfg, displacements=disps, init=bc["init"])
            res   = sweep.run()
            elapsed = time.time() - t0

            valid  = res.valid_mask
            n_valid = int(valid.sum())
            n_died  = int((res.final_div[valid] == 0).sum())
            results[label][d][mult] = (n_valid, n_died)
            pct = 100.0 * n_died / n_valid if n_valid else float("nan")
            print(f"  {label:<14} ord={d}  steps={steps:>5} ({mult:>2}×)"
                  f"  valid={n_valid:>4}  died={n_died:>4}  {pct:>6.1f}%"
                  f"  [{elapsed:.1f}s]")

# ── summary table ─────────────────────────────────────────────────────────────

print()
print("=" * 90)
print(f"{'Pattern':<14} {'Ord':>3}  {'':>6}  "
      + "  ".join(f"{m:>2}× ({m*1:>4}st) died%" for m in [1, 5, 20]))
print("=" * 90)

for bc in BASE_CONFIGS:
    label = bc["label"]
    for d in ORDERS:
        row_parts = [f"{label:<14} {d:>3}  steps={bc['base_steps']:>3}  "]
        for mult in MULTIPLIERS:
            n_valid, n_died = results[label][d][mult]
            pct = 100.0 * n_died / n_valid if n_valid else float("nan")
            row_parts.append(f"{pct:>7.1f}%")
        print("  ".join(row_parts))
    print()

# ── totals ────────────────────────────────────────────────────────────────────

print("=" * 90)
print("TOTALS across all patterns & orders")
for mult in MULTIPLIERS:
    tv = sum(results[bc["label"]][d][mult][0]
             for bc in BASE_CONFIGS for d in ORDERS)
    td = sum(results[bc["label"]][d][mult][1]
             for bc in BASE_CONFIGS for d in ORDERS)
    pct = 100.0 * td / tv if tv else float("nan")
    print(f"  {mult:>2}×  valid={tv:>5}  died={td:>4}  {pct:.2f}%")
