"""
collect_stats_any_dir.py
------------------------
For each valid (cell, direction) pair, independently check whether the
perturbation dies down (final divergence == 0).

This is different from collect_stats.py, which only checks the max-cumulative
direction per cell.  Here we count:
  - any_diedown_cells : cells where AT LEAST ONE direction dies down
  - total_diedown_pairs: all (cell, direction) pairs that die down

Prints a side-by-side comparison: "max-only" vs "any-direction".
"""

import numpy as np
from perturbation_move import (
    Engine, CellMovePerturbation, SweepConfig,
    manhattan_displacements
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


def full_sweep(label, grid_size, steps, disps, init=None,
               seed=42, density=0.35):
    """
    Iterate over every (cell, direction) pair and count die-downs independently
    for each pair, rather than only tracking the max-cumulative direction.
    """
    if init is None:
        rng = np.random.default_rng(seed)
        state = (rng.random((grid_size, grid_size)) < density).astype(np.uint8)
    else:
        state = init.astype(np.uint8).copy()

    baseline = Engine.run_trajectory(state, steps)

    n_valid_pairs  = 0   # valid (cell, dir) pairs
    n_died_pairs   = 0   # pairs with final_div == 0
    any_diedown    = np.zeros((grid_size, grid_size), dtype=bool)

    for r in range(grid_size):
        for c in range(grid_size):
            for dr, dc in disps:
                pert    = CellMovePerturbation(r, c, dr, dc, grid_size)
                p_state = pert.apply(state)
                if p_state is None:
                    continue
                p_traj = Engine.run_trajectory(p_state, steps)
                final  = int((p_traj[-1] != baseline[-1]).sum())
                n_valid_pairs += 1
                if final == 0:
                    n_died_pairs += 1
                    any_diedown[r, c] = True

    n_valid_cells = int((any_diedown | (state > 0)).sum())  # cells with ≥1 valid pair
    # recount: cells that had at least one valid pair
    valid_cell_mask = np.zeros((grid_size, grid_size), dtype=bool)
    for r in range(grid_size):
        for c in range(grid_size):
            for dr, dc in disps:
                pert = CellMovePerturbation(r, c, dr, dc, grid_size)
                if pert.apply(state) is not None:
                    valid_cell_mask[r, c] = True
                    break
    n_valid_cells     = int(valid_cell_mask.sum())
    n_any_diedown_cells = int(any_diedown.sum())

    return dict(
        label=label,
        n_valid_pairs=n_valid_pairs,
        n_died_pairs=n_died_pairs,
        n_valid_cells=n_valid_cells,
        n_any_diedown_cells=n_any_diedown_cells,
    )


# ── configurations ────────────────────────────────────────────────────────────

CONFIGS = []

CONFIGS.append(dict(label="random",  grid_size=40, steps=60,
                    init=None, seed=42, density=0.35))
CONFIGS.append(dict(label="acorn",   grid_size=40, steps=60,
                    init=make_acorn_grid()))

for name, spec in PATTERNS.items():
    if name == "acorn":
        continue
    CONFIGS.append(dict(label=name, grid_size=GRID_SIZE,
                        steps=spec["steps"],
                        init=make_grid(spec["cells"], GRID_SIZE)))

ORDERS = [1, 2, 3]

# ── run ───────────────────────────────────────────────────────────────────────

print(f"{'Config':<18} {'Ord':>3}  "
      f"{'valid_pairs':>12}  {'died_pairs':>10}  {'pair%':>7}  "
      f"{'valid_cells':>11}  {'any_down_cells':>14}  {'cell%':>7}")
print("-" * 100)

total_vp = total_dp = total_vc = total_ac = 0

for cfg in CONFIGS:
    for d in ORDERS:
        disps = manhattan_displacements(d)
        kw    = {k: v for k, v in cfg.items() if k != "label"}
        r     = full_sweep(cfg["label"], disps=disps, **kw)

        pp = 100.0 * r["n_died_pairs"]       / r["n_valid_pairs"]  if r["n_valid_pairs"]  else float("nan")
        cp = 100.0 * r["n_any_diedown_cells"] / r["n_valid_cells"] if r["n_valid_cells"] else float("nan")

        total_vp += r["n_valid_pairs"]
        total_dp += r["n_died_pairs"]
        total_vc += r["n_valid_cells"]
        total_ac += r["n_any_diedown_cells"]

        print(f"{cfg['label']:<18} {d:>3}  "
              f"{r['n_valid_pairs']:>12}  {r['n_died_pairs']:>10}  {pp:>7.2f}%  "
              f"{r['n_valid_cells']:>11}  {r['n_any_diedown_cells']:>14}  {cp:>7.2f}%")

print("-" * 100)
pp_tot = 100.0 * total_dp / total_vp if total_vp else float("nan")
cp_tot = 100.0 * total_ac / total_vc if total_vc else float("nan")
print(f"{'TOTAL':<18} {'':>3}  "
      f"{total_vp:>12}  {total_dp:>10}  {pp_tot:>7.2f}%  "
      f"{total_vc:>11}  {total_ac:>14}  {cp_tot:>7.2f}%")
