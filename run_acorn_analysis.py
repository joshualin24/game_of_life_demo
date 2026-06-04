"""
run_acorn_analysis.py
---------------------
Move-perturbation analysis using the acorn methuselah as the initial state,
for displacement orders 1, 2, and 3.

The acorn stabilises after ~5206 generations; with 60 steps we observe
the early explosive growth phase, which is the most sensitive period.

Results are saved to:
  figures_move/acorn/order_1/
  figures_move/acorn/order_2/
  figures_move/acorn/order_3/
"""

import numpy as np
from perturbation_move import (
    SweepConfig, SensitivitySweep, SensitivityVisualizer, manhattan_displacements
)

# ── Acorn pattern ──────────────────────────────────────────────────────────────
#
#   .#.....
#   ...#...
#   ##..###
#
ACORN = [
    (0, 1),
    (1, 3),
    (2, 0), (2, 1), (2, 4), (2, 5), (2, 6),
]

GRID_SIZE = 40


def make_acorn_grid(size: int = GRID_SIZE) -> np.ndarray:
    """Place the acorn at the centre of an empty grid."""
    grid = np.zeros((size, size), dtype=np.uint8)
    # bounding box of acorn: 3 rows × 7 cols — centre it
    top  = (size - 3) // 2
    left = (size - 7) // 2
    for r, c in ACORN:
        grid[top + r, left + c] = 1
    return grid


# ── Run orders 1–3 ────────────────────────────────────────────────────────────

acorn_init = make_acorn_grid()

for d in [1, 2, 3]:
    disps = manhattan_displacements(d)
    print(f"\n{'='*60}")
    print(f"  Acorn — Order {d}: {len(disps)} directions — {disps}")
    print(f"{'='*60}")
    cfg = SweepConfig(
        grid_size=GRID_SIZE,
        steps=60,
        seed=42,          # unused (custom init provided), kept for record
        density=0.35,     # unused (custom init provided), kept for record
        t_perturb=0,
        out_dir=f"figures_move/acorn/order_{d}",
    )
    sweep  = SensitivitySweep(cfg, displacements=disps, init=acorn_init)
    result = sweep.run()
    SensitivityVisualizer(result).save_all()

print("\nAll acorn orders complete.")
