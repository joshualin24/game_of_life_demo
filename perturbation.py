"""
Perturbation analysis of Conway's Game of Life
-----------------------------------------------
For every cell in the initial grid, flip it (on→off or off→on) and measure
how much the trajectory diverges from the unperturbed baseline.

Outputs (saved to figures/):
  01_sensitivity_cumulative.png  – heatmap: total divergence summed over all steps
  02_sensitivity_final.png       – heatmap: divergence at the final step only
  03_divergence_over_time.png    – divergence curves for top/mid/low impact cells
  04_impact_distribution.png     – histogram of all perturbation impacts
  05_baseline_vs_top.png         – side-by-side snapshots at 4 time-points
  06_difference_maps.png         – per-step difference grids for the top perturbation
  07_sensitivity_gif.gif         – animated divergence map as it accumulates over time
"""

import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

GRID_SIZE = 40
STEPS     = 60
SEED      = 42
DENSITY   = 0.35

# ── Game-of-Life engine ────────────────────────────────────────────────────────

def step(cells: np.ndarray) -> np.ndarray:
    neighbors = sum(
        np.roll(np.roll(cells, i, 0), j, 1)
        for i in (-1, 0, 1) for j in (-1, 0, 1)
        if (i, j) != (0, 0)
    )
    return ((neighbors == 3) | (cells & (neighbors == 2))).astype(np.uint8)


def run_trajectory(init: np.ndarray, steps: int) -> np.ndarray:
    """Return trajectory array of shape (steps+1, H, W)."""
    traj = np.empty((steps + 1, *init.shape), dtype=np.uint8)
    traj[0] = init
    for t in range(1, steps + 1):
        traj[t] = step(traj[t - 1])
    return traj


# ── Build baseline ─────────────────────────────────────────────────────────────

rng  = np.random.default_rng(SEED)
init = (rng.random((GRID_SIZE, GRID_SIZE)) < DENSITY).astype(np.uint8)

print("Running baseline …")
baseline = run_trajectory(init, STEPS)          # (STEPS+1, N, N)

# ── Perturbation sweep ─────────────────────────────────────────────────────────

print(f"Perturbing {GRID_SIZE}×{GRID_SIZE} = {GRID_SIZE**2} cells …")

# cumulative divergence summed over all steps (excluding t=0 which is always 1)
cumulative = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
final_div  = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

# store per-step divergence for every cell: shape (N, N, STEPS)
per_step = np.zeros((GRID_SIZE, GRID_SIZE, STEPS), dtype=np.float32)

for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        p_init       = init.copy()
        p_init[r, c] = 1 - p_init[r, c]          # flip
        p_traj       = run_trajectory(p_init, STEPS)
        diffs        = (p_traj[1:] != baseline[1:]).sum(axis=(1, 2))  # (STEPS,)
        per_step[r, c]   = diffs
        cumulative[r, c] = diffs.sum()
        final_div[r, c]  = diffs[-1]

    if (r + 1) % 10 == 0:
        print(f"  row {r+1}/{GRID_SIZE}")

print("Sweep done.")

# ── Identify representative cells ─────────────────────────────────────────────

flat_cum   = cumulative.ravel()
order      = np.argsort(flat_cum)[::-1]
top_idx    = np.unravel_index(order[0],  (GRID_SIZE, GRID_SIZE))
mid_idx    = np.unravel_index(order[len(order)//2], (GRID_SIZE, GRID_SIZE))
low_idx    = np.unravel_index(order[-1], (GRID_SIZE, GRID_SIZE))

print(f"  Top impact cell : {top_idx}  cumulative={cumulative[top_idx]:.0f}")
print(f"  Mid impact cell : {mid_idx}  cumulative={cumulative[mid_idx]:.0f}")
print(f"  Low impact cell : {low_idx}  cumulative={cumulative[low_idx]:.0f}")

# ── 1. Sensitivity heatmap – cumulative ───────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cumulative, cmap="hot", interpolation="nearest")
ax.scatter([top_idx[1]], [top_idx[0]], s=120, c="cyan", marker="*",
           label=f"top {top_idx}", zorder=5)
ax.scatter([mid_idx[1]], [mid_idx[0]], s=80,  c="lime", marker="o",
           label=f"mid {mid_idx}", zorder=5)
ax.scatter([low_idx[1]], [low_idx[0]], s=80,  c="blue", marker="s",
           label=f"low {low_idx}", zorder=5)
plt.colorbar(im, ax=ax, label="Cumulative divergence (cells differing, summed)")
ax.set_title(f"Perturbation Sensitivity — Cumulative over {STEPS} steps",
             fontweight="bold")
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/01_sensitivity_cumulative.png", dpi=150)
plt.close(fig)
print("[1] Cumulative sensitivity heatmap saved.")

# ── 2. Sensitivity heatmap – final step ───────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(final_div, cmap="plasma", interpolation="nearest")
plt.colorbar(im, ax=ax, label=f"Divergence at step {STEPS}")
ax.set_title(f"Perturbation Sensitivity — Final state (step {STEPS})",
             fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/02_sensitivity_final.png", dpi=150)
plt.close(fig)
print("[2] Final-step sensitivity heatmap saved.")

# ── 3. Divergence-over-time curves ────────────────────────────────────────────

# pick top-5, mid-5 and low-5 for spread
top5  = [np.unravel_index(order[i],   (GRID_SIZE, GRID_SIZE)) for i in range(5)]
mid5  = [np.unravel_index(order[len(order)//2 + i - 2], (GRID_SIZE, GRID_SIZE)) for i in range(5)]
low5  = [np.unravel_index(order[-i-1], (GRID_SIZE, GRID_SIZE)) for i in range(5)]

fig, ax = plt.subplots(figsize=(10, 5))
ts = np.arange(1, STEPS + 1)

for idx in top5:
    ax.plot(ts, per_step[idx], color="crimson", alpha=0.7, linewidth=1.2)
for idx in mid5:
    ax.plot(ts, per_step[idx], color="goldenrod", alpha=0.7, linewidth=1.2)
for idx in low5:
    ax.plot(ts, per_step[idx], color="steelblue", alpha=0.7, linewidth=1.2)

# legend proxies
from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([0], [0], color="crimson",   lw=2, label="Top-5 impact"),
    Line2D([0], [0], color="goldenrod", lw=2, label="Mid-5 impact"),
    Line2D([0], [0], color="steelblue", lw=2, label="Low-5 impact"),
])
ax.set_xlabel("Step")
ax.set_ylabel("Cells differing from baseline")
ax.set_title("Divergence Over Time by Perturbation Impact Class", fontweight="bold")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/03_divergence_over_time.png", dpi=150)
plt.close(fig)
print("[3] Divergence-over-time saved.")

# ── 4. Impact distribution ────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(flat_cum, bins=50, color="#C44E52", edgecolor="white", linewidth=0.4)
axes[0].set_xlabel("Cumulative divergence")
axes[0].set_ylabel("Number of cells")
axes[0].set_title("Distribution of Perturbation Impact (cumulative)", fontweight="bold")
axes[0].axvline(flat_cum.mean(), color="black", linestyle="--",
                label=f"mean = {flat_cum.mean():.0f}")
axes[0].legend()

axes[1].hist(final_div.ravel(), bins=50, color="#4C72B0", edgecolor="white", linewidth=0.4)
axes[1].set_xlabel(f"Divergence at step {STEPS}")
axes[1].set_ylabel("Number of cells")
axes[1].set_title(f"Distribution of Perturbation Impact (step {STEPS})", fontweight="bold")
axes[1].axvline(final_div.mean(), color="black", linestyle="--",
                label=f"mean = {final_div.mean():.0f}")
axes[1].legend()

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/04_impact_distribution.png", dpi=150)
plt.close(fig)
print("[4] Impact distribution saved.")

# ── 5. Baseline vs top perturbation snapshots ─────────────────────────────────

snap_steps = [0, 10, 30, STEPS]
p_init_top = init.copy()
p_init_top[top_idx] = 1 - p_init_top[top_idx]
top_traj = run_trajectory(p_init_top, STEPS)

fig, axes = plt.subplots(3, len(snap_steps), figsize=(14, 9))
cmap_state = "inferno"
cmap_diff  = "RdBu_r"

for col, t in enumerate(snap_steps):
    base_frame  = baseline[t]
    top_frame   = top_traj[t]
    diff_frame  = base_frame.astype(int) - top_frame.astype(int)

    axes[0, col].imshow(base_frame, cmap=cmap_state, vmin=0, vmax=1, interpolation="nearest")
    axes[0, col].set_title(f"Step {t}", fontsize=10)
    axes[0, col].axis("off")

    axes[1, col].imshow(top_frame, cmap=cmap_state, vmin=0, vmax=1, interpolation="nearest")
    axes[1, col].axis("off")

    axes[2, col].imshow(diff_frame, cmap=cmap_diff, vmin=-1, vmax=1, interpolation="nearest")
    axes[2, col].axis("off")

axes[0, 0].set_ylabel("Baseline", fontsize=11)
for ax, label in zip(axes[:, 0], ["Baseline", f"Perturbed {top_idx}", "Difference"]):
    ax.set_ylabel(label, fontsize=10)

# mark the perturbed cell
for row in range(2):
    axes[row, 0].scatter([top_idx[1]], [top_idx[0]], s=80, c="cyan", marker="*", zorder=5)

fig.suptitle(f"Baseline vs Highest-Impact Perturbation at cell {top_idx}",
             fontweight="bold", fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/05_baseline_vs_top.png", dpi=150)
plt.close(fig)
print("[5] Baseline vs top perturbation snapshots saved.")

# ── 6. Difference maps over time (top perturbation) ───────────────────────────

show_steps = list(range(0, STEPS + 1, STEPS // 12))[:12]
ncols = 6
nrows = (len(show_steps) + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))
axes = axes.ravel()

diff_traj = top_traj.astype(int) - baseline.astype(int)   # -1, 0, +1

for i, t in enumerate(show_steps):
    axes[i].imshow(diff_traj[t], cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    axes[i].set_title(f"t={t}", fontsize=9)
    axes[i].axis("off")
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

fig.suptitle(f"Difference Maps: Perturbed {top_idx} vs Baseline", fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/06_difference_maps.png", dpi=150)
plt.close(fig)
print("[6] Difference maps saved.")

# ── 7. Animated cumulative divergence map ─────────────────────────────────────

# build cumulative sensitivity up to each step t
print("Building sensitivity animation …")
cum_over_time = np.cumsum(per_step, axis=2)   # (N, N, STEPS)

fig7, ax7 = plt.subplots(figsize=(6, 6))
fig7.patch.set_facecolor("black")
ax7.set_facecolor("black")
ax7.axis("off")

vmax = cum_over_time[:, :, -1].max()
im7  = ax7.imshow(cum_over_time[:, :, 0], cmap="hot", vmin=0, vmax=vmax,
                  interpolation="nearest")
ttl  = ax7.set_title("Cumulative divergence  step 1", color="white", fontsize=11)
plt.colorbar(im7, ax=ax7).ax.yaxis.set_tick_params(color="white", labelcolor="white")

def _update7(frame):
    im7.set_data(cum_over_time[:, :, frame])
    ttl.set_text(f"Cumulative divergence  step {frame + 1}")
    return im7, ttl

ani7 = animation.FuncAnimation(fig7, _update7, frames=STEPS, interval=120, blit=True)
ani7.save(f"{OUT_DIR}/07_sensitivity_gif.gif",
          writer=animation.PillowWriter(fps=10))
plt.close(fig7)
print("[7] Sensitivity animation saved.")

# ── Summary ────────────────────────────────────────────────────────────────────

total_cells = GRID_SIZE * GRID_SIZE
zero_impact = (flat_cum == 0).sum()

print("\n── Perturbation Summary ──────────────────────────────────────")
print(f"  Grid size      : {GRID_SIZE}×{GRID_SIZE}  ({total_cells} cells)")
print(f"  Steps simulated: {STEPS}")
print(f"  Cells with zero impact   : {zero_impact}  ({zero_impact/total_cells*100:.1f}%)")
print(f"  Mean cumulative divergence: {flat_cum.mean():.1f}")
print(f"  Max  cumulative divergence: {flat_cum.max():.0f}  at cell {top_idx}")
print(f"  Min  cumulative divergence: {flat_cum.min():.0f}  at cell {low_idx}")
print(f"\nAll figures saved to ./{OUT_DIR}/")
