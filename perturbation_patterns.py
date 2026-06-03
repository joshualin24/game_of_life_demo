"""
Perturbation analysis across named Game-of-Life initial conditions
------------------------------------------------------------------
For each well-known pattern (placed at the centre of a clean grid) we flip
every cell one at a time and measure trajectory divergence.

Patterns covered
  Still lifes  : block, beehive
  Oscillators  : blinker (p2), toad (p2), beacon (p2), pulsar (p3)
  Spaceships   : glider, lwss
  Methuselahs  : r_pentomino, acorn

Outputs per pattern  →  figures/<name>/
  01_initial.png               – initial grid with pattern highlighted
  02_sensitivity_cumulative.png
  03_sensitivity_final.png
  04_divergence_over_time.png
  05_impact_distribution.png
  06_baseline_vs_top.png
  07_difference_maps.png
  08_sensitivity_gif.gif
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D

GRID_SIZE = 50
BASE_FIGURES = "figures"

# ── Pattern library ────────────────────────────────────────────────────────────

PATTERNS = {
    # ── Still lifes ──────────────────────────────────────────
    "block": {
        "cells": np.array([[1,1],
                           [1,1]]),
        "steps": 40,
        "category": "Still life",
    },
    "beehive": {
        "cells": np.array([[0,1,1,0],
                           [1,0,0,1],
                           [0,1,1,0]]),
        "steps": 40,
        "category": "Still life",
    },
    # ── Oscillators ──────────────────────────────────────────
    "blinker": {
        "cells": np.array([[1,1,1]]),
        "steps": 60,
        "category": "Oscillator (p2)",
    },
    "toad": {
        "cells": np.array([[0,1,1,1],
                           [1,1,1,0]]),
        "steps": 60,
        "category": "Oscillator (p2)",
    },
    "beacon": {
        "cells": np.array([[1,1,0,0],
                           [1,0,0,0],
                           [0,0,0,1],
                           [0,0,1,1]]),
        "steps": 60,
        "category": "Oscillator (p2)",
    },
    "pulsar": {
        "cells": np.array([
            [0,0,1,1,1,0,0,0,1,1,1,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0],
            [1,0,0,0,0,1,0,1,0,0,0,0,1],
            [1,0,0,0,0,1,0,1,0,0,0,0,1],
            [1,0,0,0,0,1,0,1,0,0,0,0,1],
            [0,0,1,1,1,0,0,0,1,1,1,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,1,1,1,0,0,0,1,1,1,0,0],
            [1,0,0,0,0,1,0,1,0,0,0,0,1],
            [1,0,0,0,0,1,0,1,0,0,0,0,1],
            [1,0,0,0,0,1,0,1,0,0,0,0,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,1,1,1,0,0,0,1,1,1,0,0],
        ]),
        "steps": 60,
        "category": "Oscillator (p3)",
    },
    # ── Spaceships ───────────────────────────────────────────
    "glider": {
        "cells": np.array([[0,1,0],
                           [0,0,1],
                           [1,1,1]]),
        "steps": 80,
        "category": "Spaceship",
    },
    "lwss": {
        "cells": np.array([[0,1,0,0,1],
                           [1,0,0,0,0],
                           [1,0,0,0,1],
                           [1,1,1,1,0]]),
        "steps": 80,
        "category": "Spaceship",
    },
    # ── Methuselahs ──────────────────────────────────────────
    "r_pentomino": {
        "cells": np.array([[0,1,1],
                           [1,1,0],
                           [0,1,0]]),
        "steps": 120,
        "category": "Methuselah",
    },
    "acorn": {
        "cells": np.array([[0,1,0,0,0,0,0],
                           [0,0,0,1,0,0,0],
                           [1,1,0,0,1,1,1]]),
        "steps": 120,
        "category": "Methuselah",
    },
}

# ── Engine ─────────────────────────────────────────────────────────────────────

def gol_step(cells: np.ndarray) -> np.ndarray:
    neighbors = sum(
        np.roll(np.roll(cells, i, 0), j, 1)
        for i in (-1, 0, 1) for j in (-1, 0, 1)
        if (i, j) != (0, 0)
    )
    return ((neighbors == 3) | (cells & (neighbors == 2))).astype(np.uint8)


def run_trajectory(init: np.ndarray, steps: int) -> np.ndarray:
    traj = np.empty((steps + 1, *init.shape), dtype=np.uint8)
    traj[0] = init
    for t in range(1, steps + 1):
        traj[t] = gol_step(traj[t - 1])
    return traj


def place_at_center(pattern: np.ndarray, grid_size: int) -> np.ndarray:
    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    h, w = pattern.shape
    r0 = (grid_size - h) // 2
    c0 = (grid_size - w) // 2
    grid[r0:r0 + h, c0:c0 + w] = pattern
    return grid


# ── Per-pattern analysis ───────────────────────────────────────────────────────

def analyse_pattern(name: str, cfg: dict):
    STEPS     = cfg["steps"]
    category  = cfg["category"]
    out_dir   = os.path.join(BASE_FIGURES, name)
    os.makedirs(out_dir, exist_ok=True)

    init     = place_at_center(cfg["cells"], GRID_SIZE)
    baseline = run_trajectory(init, STEPS)

    print(f"\n{'='*55}")
    print(f"  {name.upper()}  [{category}]  grid={GRID_SIZE}  steps={STEPS}")
    print(f"{'='*55}")

    # ── Perturbation sweep ────────────────────────────────────────────────────
    cumulative = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    final_div  = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    per_step   = np.zeros((GRID_SIZE, GRID_SIZE, STEPS), dtype=np.float32)

    print(f"  Sweeping {GRID_SIZE**2} cells …", end="", flush=True)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            p = init.copy()
            p[r, c] = 1 - p[r, c]
            pt = run_trajectory(p, STEPS)
            diffs = (pt[1:] != baseline[1:]).sum(axis=(1, 2))
            per_step[r, c]   = diffs
            cumulative[r, c] = diffs.sum()
            final_div[r, c]  = diffs[-1]
        if (r + 1) % 10 == 0:
            print(f" {r+1}", end="", flush=True)
    print(" done.")

    flat_cum = cumulative.ravel()
    order    = np.argsort(flat_cum)[::-1]
    top_idx  = np.unravel_index(order[0],             (GRID_SIZE, GRID_SIZE))
    mid_idx  = np.unravel_index(order[len(order)//2], (GRID_SIZE, GRID_SIZE))
    low_idx  = np.unravel_index(order[-1],             (GRID_SIZE, GRID_SIZE))

    print(f"  top={top_idx} cum={cumulative[top_idx]:.0f}  "
          f"mid={mid_idx} cum={cumulative[mid_idx]:.0f}  "
          f"low={low_idx} cum={cumulative[low_idx]:.0f}")

    # helper: overlay pattern outline on an axis
    def _overlay_pattern(ax):
        mask = init.astype(bool)
        ys, xs = np.where(mask)
        ax.scatter(xs, ys, s=12, c="cyan", marker="s", alpha=0.6, zorder=4,
                   label="initial pattern")

    # ── Fig 01: initial grid ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    snap_t = [0, STEPS // 2, STEPS]
    for ax, t in zip(axes, snap_t):
        ax.imshow(baseline[t], cmap="inferno", vmin=0, vmax=1,
                  interpolation="nearest")
        if t == 0:
            _overlay_pattern(ax)
            ax.legend(fontsize=7, loc="upper right")
        ax.set_title(f"t = {t}", fontsize=10)
        ax.axis("off")
    fig.suptitle(f"{name}  [{category}] — baseline evolution", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/01_initial.png", dpi=150)
    plt.close(fig)

    # ── Fig 02: sensitivity heatmap – cumulative ──────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cumulative, cmap="hot", interpolation="nearest")
    _overlay_pattern(ax)
    ax.scatter([top_idx[1]], [top_idx[0]], s=140, c="cyan",  marker="*", zorder=6,
               label=f"top {top_idx}")
    ax.scatter([mid_idx[1]], [mid_idx[0]], s=80,  c="lime",  marker="o", zorder=6,
               label=f"mid {mid_idx}")
    ax.scatter([low_idx[1]], [low_idx[0]], s=80,  c="dodgerblue", marker="s",
               zorder=6, label=f"low {low_idx}")
    plt.colorbar(im, ax=ax, label="Cumulative divergence (sum over steps)")
    ax.set_title(f"{name} — Sensitivity (cumulative, {STEPS} steps)",
                 fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/02_sensitivity_cumulative.png", dpi=150)
    plt.close(fig)

    # ── Fig 03: sensitivity heatmap – final step ──────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(final_div, cmap="plasma", interpolation="nearest")
    _overlay_pattern(ax)
    plt.colorbar(im, ax=ax, label=f"Divergence at step {STEPS}")
    ax.set_title(f"{name} — Sensitivity (final state, step {STEPS})",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/03_sensitivity_final.png", dpi=150)
    plt.close(fig)

    # ── Fig 04: divergence over time ──────────────────────────────────────────
    n_show = min(5, len(order))
    top5 = [np.unravel_index(order[i], (GRID_SIZE, GRID_SIZE))
            for i in range(n_show)]
    mid5 = [np.unravel_index(order[len(order)//2 + i - n_show//2],
                              (GRID_SIZE, GRID_SIZE))
            for i in range(n_show)]
    low5 = [np.unravel_index(order[-(i+1)], (GRID_SIZE, GRID_SIZE))
            for i in range(n_show)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ts = np.arange(1, STEPS + 1)
    for idx in top5:
        ax.plot(ts, per_step[idx], color="crimson",   alpha=0.75, lw=1.3)
    for idx in mid5:
        ax.plot(ts, per_step[idx], color="goldenrod", alpha=0.75, lw=1.3)
    for idx in low5:
        ax.plot(ts, per_step[idx], color="steelblue", alpha=0.75, lw=1.3)
    ax.legend(handles=[
        Line2D([0],[0], color="crimson",   lw=2, label="Top-5 impact"),
        Line2D([0],[0], color="goldenrod", lw=2, label="Mid-5 impact"),
        Line2D([0],[0], color="steelblue", lw=2, label="Low-5 impact"),
    ])
    ax.set_xlabel("Step")
    ax.set_ylabel("Cells differing from baseline")
    ax.set_title(f"{name} — Divergence over time", fontweight="bold")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/04_divergence_over_time.png", dpi=150)
    plt.close(fig)

    # ── Fig 05: impact distribution ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(flat_cum, bins=40, color="#C44E52", edgecolor="white", lw=0.4)
    axes[0].axvline(flat_cum.mean(), color="black", ls="--",
                    label=f"mean={flat_cum.mean():.0f}")
    axes[0].set(xlabel="Cumulative divergence", ylabel="Cells",
                title="Impact distribution (cumulative)")
    axes[0].legend()
    axes[1].hist(final_div.ravel(), bins=40, color="#4C72B0",
                 edgecolor="white", lw=0.4)
    axes[1].axvline(final_div.mean(), color="black", ls="--",
                    label=f"mean={final_div.mean():.0f}")
    axes[1].set(xlabel=f"Divergence at step {STEPS}", ylabel="Cells",
                title=f"Impact distribution (step {STEPS})")
    axes[1].legend()
    fig.suptitle(f"{name} [{category}]", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/05_impact_distribution.png", dpi=150)
    plt.close(fig)

    # ── Fig 06: baseline vs top perturbation snapshots ────────────────────────
    p_init_top = init.copy()
    p_init_top[top_idx] = 1 - p_init_top[top_idx]
    top_traj = run_trajectory(p_init_top, STEPS)
    snap_steps = [0, STEPS // 4, STEPS // 2, STEPS]

    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    for col, t in enumerate(snap_steps):
        axes[0, col].imshow(baseline[t], cmap="inferno", vmin=0, vmax=1,
                            interpolation="nearest")
        axes[0, col].set_title(f"t={t}", fontsize=10)
        axes[1, col].imshow(top_traj[t], cmap="inferno", vmin=0, vmax=1,
                            interpolation="nearest")
        diff = baseline[t].astype(int) - top_traj[t].astype(int)
        axes[2, col].imshow(diff, cmap="RdBu_r", vmin=-1, vmax=1,
                            interpolation="nearest")
        for row in range(3):
            axes[row, col].axis("off")
    for ax, lbl in zip(axes[:, 0],
                       ["Baseline", f"Perturbed {top_idx}", "Difference"]):
        ax.set_ylabel(lbl, fontsize=10)
    axes[0, 0].scatter([top_idx[1]], [top_idx[0]], s=100, c="cyan",
                       marker="*", zorder=5)
    axes[1, 0].scatter([top_idx[1]], [top_idx[0]], s=100, c="cyan",
                       marker="*", zorder=5)
    fig.suptitle(f"{name} — Baseline vs highest-impact perturbation {top_idx}",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/06_baseline_vs_top.png", dpi=150)
    plt.close(fig)

    # ── Fig 07: difference maps grid ─────────────────────────────────────────
    n_panels   = 12
    show_steps = np.linspace(0, STEPS, n_panels, dtype=int)
    diff_traj  = top_traj.astype(int) - baseline.astype(int)
    ncols, nrows = 6, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))
    axes = axes.ravel()
    for i, t in enumerate(show_steps):
        axes[i].imshow(diff_traj[t], cmap="RdBu_r", vmin=-1, vmax=1,
                       interpolation="nearest")
        axes[i].set_title(f"t={t}", fontsize=9)
        axes[i].axis("off")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"{name} — Difference maps (perturbed {top_idx} vs baseline)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/07_difference_maps.png", dpi=150)
    plt.close(fig)

    # ── Fig 08: animated cumulative sensitivity ───────────────────────────────
    cum_over_time = np.cumsum(per_step, axis=2)   # (N, N, STEPS)
    vmax = max(cum_over_time[:, :, -1].max(), 1.0)

    fig8, ax8 = plt.subplots(figsize=(6, 6))
    fig8.patch.set_facecolor("black")
    ax8.set_facecolor("black")
    ax8.axis("off")
    im8  = ax8.imshow(cum_over_time[:, :, 0], cmap="hot", vmin=0, vmax=vmax,
                      interpolation="nearest")
    ttl8 = ax8.set_title(f"{name}  cumulative sensitivity  t=1",
                          color="white", fontsize=10)
    plt.colorbar(im8, ax=ax8).ax.yaxis.set_tick_params(
        color="white", labelcolor="white")

    def _upd8(frame):
        im8.set_data(cum_over_time[:, :, frame])
        ttl8.set_text(f"{name}  cumulative sensitivity  t={frame+1}")
        return im8, ttl8

    ani8 = animation.FuncAnimation(fig8, _upd8, frames=STEPS,
                                    interval=120, blit=True)
    ani8.save(f"{out_dir}/08_sensitivity_gif.gif",
              writer=animation.PillowWriter(fps=10))
    plt.close(fig8)

    print(f"  Saved 8 figures → {out_dir}/")

    return {
        "name":      name,
        "category":  category,
        "steps":     STEPS,
        "n_cells":   GRID_SIZE ** 2,
        "zero_pct":  (flat_cum == 0).mean() * 100,
        "mean_cum":  flat_cum.mean(),
        "max_cum":   flat_cum.max(),
        "top_cell":  top_idx,
    }


# ── Summary comparison plot ────────────────────────────────────────────────────

def plot_summary(results: list[dict]):
    names    = [r["name"]     for r in results]
    means    = [r["mean_cum"] for r in results]
    maxes    = [r["max_cum"]  for r in results]
    zero_pct = [r["zero_pct"] for r in results]

    x    = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # mean cumulative divergence
    axes[0].bar(x, means, color="#4C72B0", edgecolor="white")
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=35, ha="right")
    axes[0].set_ylabel("Mean cumulative divergence")
    axes[0].set_title("Average perturbation impact", fontweight="bold")

    # max cumulative divergence
    axes[1].bar(x, maxes, color="#C44E52", edgecolor="white")
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=35, ha="right")
    axes[1].set_ylabel("Max cumulative divergence")
    axes[1].set_title("Worst-case perturbation impact", fontweight="bold")

    # fraction of zero-impact cells
    axes[2].bar(x, zero_pct, color="#55A868", edgecolor="white")
    axes[2].set_xticks(x); axes[2].set_xticklabels(names, rotation=35, ha="right")
    axes[2].set_ylabel("Zero-impact cells (%)")
    axes[2].set_title("Fraction of inert cells", fontweight="bold")

    fig.suptitle("Perturbation Impact Across Initial Conditions",
                 fontweight="bold", fontsize=14)
    fig.tight_layout()
    out = os.path.join(BASE_FIGURES, "00_summary_comparison.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSummary comparison saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = []
    for name, cfg in PATTERNS.items():
        r = analyse_pattern(name, cfg)
        results.append(r)

    plot_summary(results)

    print("\n── Final Summary ─────────────────────────────────────────────")
    print(f"  {'Pattern':<14} {'Category':<20} {'Steps':>5}  "
          f"{'Mean':>8}  {'Max':>8}  {'Zero%':>6}")
    print("  " + "-"*65)
    for r in results:
        print(f"  {r['name']:<14} {r['category']:<20} {r['steps']:>5}  "
              f"{r['mean_cum']:>8.0f}  {r['max_cum']:>8.0f}  {r['zero_pct']:>5.1f}%")
    print()
