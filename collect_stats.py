"""
collect_stats.py
----------------
Re-runs all move-perturbation sweeps and reports what fraction of valid
perturbations had final divergence == 0 (i.e., the perturbed trajectory
converged back to the baseline by the end of the simulation).
"""

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
    (0, 1),
    (1, 3),
    (2, 0), (2, 1), (2, 4), (2, 5), (2, 6),
]
def make_acorn_grid(size: int = 40) -> np.ndarray:
    grid = np.zeros((size, size), dtype=np.uint8)
    top  = (size - 3) // 2
    left = (size - 7) // 2
    for r, c in ACORN_CELLS:
        grid[top + r, left + c] = 1
    return grid

# ── configurations to sweep ───────────────────────────────────────────────────

configs = []

# random
for d in [1, 2, 3]:
    configs.append(dict(
        label=f"random/order_{d}",
        cfg=SweepConfig(grid_size=40, steps=60, seed=42, density=0.35,
                        t_perturb=0, out_dir=""),
        disps=manhattan_displacements(d),
        init=None,
    ))

# acorn
acorn_init = make_acorn_grid()
for d in [1, 2, 3]:
    configs.append(dict(
        label=f"acorn/order_{d}",
        cfg=SweepConfig(grid_size=40, steps=60, seed=42, density=0.35,
                        t_perturb=0, out_dir=""),
        disps=manhattan_displacements(d),
        init=acorn_init,
    ))

# named patterns (excluding acorn)
for name, spec in PATTERNS.items():
    if name == "acorn":
        continue
    init = make_grid(spec["cells"], GRID_SIZE)
    for d in [1, 2, 3]:
        configs.append(dict(
            label=f"{name}/order_{d}",
            cfg=SweepConfig(grid_size=GRID_SIZE, steps=spec["steps"],
                            t_perturb=0, out_dir=""),
            disps=manhattan_displacements(d),
            init=init,
        ))

# ── run and collect ───────────────────────────────────────────────────────────

print(f"{'Configuration':<30} {'valid':>6} {'died_down':>10} {'pct':>8}")
print("-" * 58)

total_valid = 0
total_died  = 0

for c in configs:
    sweep  = SensitivitySweep(c["cfg"], displacements=c["disps"], init=c["init"])
    result = sweep.run()

    # valid_mask: cells that had at least one valid perturbation
    # final_div:  divergence at the last step for that cell's best perturbation
    valid = result.valid_mask
    n_valid = int(valid.sum())

    # "died down" = final divergence is 0 among valid cells
    n_died  = int((result.final_div[valid] == 0).sum())
    pct     = 100.0 * n_died / n_valid if n_valid else float("nan")

    total_valid += n_valid
    total_died  += n_died

    print(f"{c['label']:<30} {n_valid:>6} {n_died:>10} {pct:>7.1f}%")

print("-" * 58)
overall = 100.0 * total_died / total_valid if total_valid else float("nan")
print(f"{'TOTAL':<30} {total_valid:>6} {total_died:>10} {overall:>7.1f}%")
