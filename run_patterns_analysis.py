"""
run_patterns_analysis.py
------------------------
Move-perturbation analysis (orders 1–3) for the 9 named patterns from
perturbation_patterns.py, using the same grid size (50) and per-pattern
step counts as the original flip-perturbation study.

Results are saved to:
  figures_move/<pattern_name>/order_1/
  figures_move/<pattern_name>/order_2/
  figures_move/<pattern_name>/order_3/
"""

import numpy as np
from perturbation_patterns import PATTERNS, GRID_SIZE
from perturbation_move import (
    SweepConfig, SensitivitySweep, SensitivityVisualizer, manhattan_displacements
)

SKIP = {"acorn"}   # already handled by run_acorn_analysis.py


def make_grid(pattern_cells: np.ndarray, size: int) -> np.ndarray:
    """Centre a pattern cell array on an empty grid."""
    grid = np.zeros((size, size), dtype=np.uint8)
    ph, pw = pattern_cells.shape
    top  = (size - ph) // 2
    left = (size - pw) // 2
    grid[top:top+ph, left:left+pw] = pattern_cells
    return grid


for name, spec in PATTERNS.items():
    if name in SKIP:
        continue

    init = make_grid(spec["cells"], GRID_SIZE)
    n_living = int(init.sum())

    for d in [1, 2, 3]:
        disps = manhattan_displacements(d)
        print(f"\n{'='*60}")
        print(f"  {name} ({spec['category']}) — Order {d}: "
              f"{len(disps)} directions, {n_living} living cells, "
              f"{spec['steps']} steps")
        print(f"{'='*60}")

        cfg = SweepConfig(
            grid_size=GRID_SIZE,
            steps=spec["steps"],
            t_perturb=0,
            out_dir=f"figures_move/{name}/order_{d}",
        )
        sweep  = SensitivitySweep(cfg, displacements=disps, init=init)
        result = sweep.run()
        SensitivityVisualizer(result).save_all()

print("\nAll patterns complete.")
