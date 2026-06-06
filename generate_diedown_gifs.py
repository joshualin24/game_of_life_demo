"""
generate_diedown_gifs.py
------------------------
Find all die-down perturbations (final divergence == 0) across the move-
perturbation analysis and generate an animated GIF for each one showing:
  Left   — baseline grid evolution
  Centre — perturbed grid evolution
  Right  — difference map (red = only in baseline, blue = only in perturbed)

Results are saved to:
  figures_move/die_down/<pattern>_cell_r<R>_c<C>_dir<DR><DC>.gif
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from perturbation_move import (
    Engine, CellMovePerturbation, SweepConfig,
    SensitivitySweep, manhattan_displacements
)
from perturbation_patterns import PATTERNS, GRID_SIZE

OUT_DIR = "figures_move/die_down"
os.makedirs(OUT_DIR, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def make_grid_from_pattern(pattern_cells: np.ndarray, size: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=np.uint8)
    ph, pw = pattern_cells.shape
    top  = (size - ph) // 2
    left = (size - pw) // 2
    grid[top:top+ph, left:left+pw] = pattern_cells
    return grid


def make_gif(label: str, baseline: np.ndarray, perturbed: np.ndarray,
             cell: tuple[int, int], dr: int, dc: int, steps: int):
    """
    Save a 3-panel GIF: baseline | perturbed | difference.
    baseline / perturbed: shape (steps+1, H, W)
    """
    dest_r = (cell[0] + dr) % baseline.shape[1]
    dest_c = (cell[1] + dc) % baseline.shape[2]
    src_marker  = dict(color="cyan",   marker="*", s=120, zorder=5)
    dest_marker = dict(color="yellow", marker="o", s=80,  zorder=5)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor("#111111")
    titles = ["Baseline", "Perturbed", "Difference"]
    ims = []
    for ax, title in zip(axes, titles):
        ax.set_facecolor("black")
        ax.axis("off")
        ax.set_title(title, color="white", fontsize=11)

    im0 = axes[0].imshow(baseline[0],   cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
    im1 = axes[1].imshow(perturbed[0],  cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
    diff0 = perturbed[0].astype(int) - baseline[0].astype(int)
    im2 = axes[2].imshow(diff0, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ims = [im0, im1, im2]

    # mark source cell (cyan star) on baseline; destination (yellow circle) on perturbed
    axes[0].scatter([cell[1]], [cell[0]],  **src_marker)
    axes[1].scatter([dest_c],  [dest_r],   **dest_marker)

    suptitle = fig.suptitle(
        f"{label}\ncell ({cell[0]},{cell[1]}) → ({dest_r},{dest_c})  "
        f"dir=({dr:+d},{dc:+d})   t=0",
        color="white", fontsize=10, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    def _update(t):
        ims[0].set_data(baseline[t])
        ims[1].set_data(perturbed[t])
        diff = perturbed[t].astype(int) - baseline[t].astype(int)
        ims[2].set_data(diff)
        n_diff = int(np.abs(diff).sum())
        suptitle.set_text(
            f"{label}\ncell ({cell[0]},{cell[1]}) → ({dest_r},{dest_c})  "
            f"dir=({dr:+d},{dc:+d})   t={t}  Δ={n_diff}"
        )
        return ims + [suptitle]

    ani = animation.FuncAnimation(fig, _update, frames=steps + 1, interval=300, blit=False)

    sign = lambda x: ("+" if x >= 0 else "") + str(x)
    fname = f"{label}_cell_r{cell[0]}_c{cell[1]}_dir{sign(dr)}{sign(dc)}.gif"
    path  = os.path.join(OUT_DIR, fname)
    ani.save(path, writer=animation.PillowWriter(fps=4))
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def find_and_gif(sweep_label: str, result, steps: int):
    """Find die-down cells in a SweepResult and generate a GIF for each."""
    valid = result.valid_mask
    diedown_cells = np.argwhere(valid & (result.final_div == 0))
    print(f"\n{sweep_label}: {len(diedown_cells)} die-down cell(s)")

    paths = []
    for (r, c) in diedown_cells:
        dr = int(result.best_dr[r, c])
        dc = int(result.best_dc[r, c])
        pert      = CellMovePerturbation(r, c, dr, dc, result.config.grid_size)
        p_state   = pert.apply(result.perturb_state)
        perturbed = Engine.run_trajectory(p_state, steps)
        path = make_gif(sweep_label, result.baseline, perturbed, (r, c), dr, dc, steps)
        paths.append(path)
    return paths


# ── run sweeps ────────────────────────────────────────────────────────────────

all_paths = []

# 1. random / order 1
print("=" * 60)
print("random / order 1")
cfg = SweepConfig(grid_size=40, steps=60, seed=42, density=0.35, t_perturb=0, out_dir="")
res = SensitivitySweep(cfg, displacements=manhattan_displacements(1)).run()
all_paths += find_and_gif("random_order1", res, cfg.steps)

# 2. block / order 1
print("=" * 60)
print("block / order 1")
init  = make_grid_from_pattern(PATTERNS["block"]["cells"], GRID_SIZE)
cfg   = SweepConfig(grid_size=GRID_SIZE, steps=PATTERNS["block"]["steps"],
                    t_perturb=0, out_dir="")
res   = SensitivitySweep(cfg, displacements=manhattan_displacements(1), init=init).run()
all_paths += find_and_gif("block_order1", res, cfg.steps)

print("\n" + "=" * 60)
print(f"Done — {len(all_paths)} GIF(s) saved to {OUT_DIR}/")
for p in all_paths:
    print(f"  {p}")
