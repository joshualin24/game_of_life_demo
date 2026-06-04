"""
run_move_analysis.py
--------------------
Run move-perturbation analysis for displacement orders 1, 2, and 3,
using the same grid parameters as the original perturbation.py.

Results are saved to:
  figures_move/random/order_1/
  figures_move/random/order_2/
  figures_move/random/order_3/
"""

from perturbation_move import (
    SweepConfig, SensitivitySweep, SensitivityVisualizer, manhattan_displacements
)

BASE_CFG = dict(
    grid_size=40,
    steps=60,
    seed=42,
    density=0.35,
    t_perturb=0,
)

for d in [1, 2, 3]:
    disps = manhattan_displacements(d)
    print(f"\n{'='*60}")
    print(f"  Order {d}: {len(disps)} directions — {disps}")
    print(f"{'='*60}")
    cfg    = SweepConfig(**BASE_CFG, out_dir=f"figures_move/random/order_{d}")
    sweep  = SensitivitySweep(cfg, displacements=disps)
    result = sweep.run()
    SensitivityVisualizer(result).save_all()

print("\nAll orders complete.")
