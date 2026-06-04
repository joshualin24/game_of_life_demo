"""
perturbation_move.py
--------------------
Move-perturbation sensitivity analysis for Conway's Game of Life.

For a given set of displacement vectors, each living cell in the state at
t_perturb is moved to every valid new position (toroidal wrap-around). The
resulting trajectory is compared against an unperturbed baseline. Per cell,
the maximum cumulative divergence across all tried directions is reported.

Use `manhattan_displacements(order)` to get all displacement vectors of a
given Manhattan distance (e.g. order=1 → 4 directions, order=2 → 8, etc.).

Outputs (saved to out_dir):
  01_sensitivity_cumulative.png  – heatmap: max cumulative divergence over steps
  02_sensitivity_final.png       – heatmap: divergence at the final step
  03_divergence_over_time.png    – curves for top / mid / low impact cells
  04_impact_distribution.png     – histogram of impacts (living cells only)
  05_baseline_vs_top.png         – side-by-side snapshots at 4 time-points
  06_difference_maps.png         – per-step difference grids for top perturbation
  07_sensitivity_gif.gif         – animated cumulative divergence map
"""

from __future__ import annotations

import abc
import functools
import math
import os
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D


# ── Displacement helpers ───────────────────────────────────────────────────────

def manhattan_displacements(order: int) -> list[tuple[int, int]]:
    """
    All integer (dr, dc) vectors with Manhattan distance exactly equal to order.

    order=1 → 4 vectors  (cardinal directions)
    order=2 → 8 vectors  (cardinal + diagonal)
    order=3 → 12 vectors
    """
    vectors = []
    for dr in range(-order, order + 1):
        dc_abs = order - abs(dr)
        if dc_abs == 0:
            vectors.append((dr, 0))
        else:
            vectors.append((dr,  dc_abs))
            vectors.append((dr, -dc_abs))
    return vectors


# ── Game-of-Life engine ────────────────────────────────────────────────────────

class Engine:
    """Pure-static Conway's Game of Life stepper (toroidal boundary)."""

    @staticmethod
    def step(cells: np.ndarray) -> np.ndarray:
        neighbors = sum(
            np.roll(np.roll(cells, i, 0), j, 1)
            for i in (-1, 0, 1) for j in (-1, 0, 1)
            if (i, j) != (0, 0)
        )
        return ((neighbors == 3) | (cells & (neighbors == 2))).astype(np.uint8)

    @staticmethod
    def run_trajectory(init: np.ndarray, steps: int) -> np.ndarray:
        """Return array of shape (steps+1, H, W)."""
        traj = np.empty((steps + 1, *init.shape), dtype=np.uint8)
        traj[0] = init
        for t in range(1, steps + 1):
            traj[t] = Engine.step(traj[t - 1])
        return traj


# ── Perturbation protocol ──────────────────────────────────────────────────────

class Perturbation(abc.ABC):
    """Encapsulates a single perturbation applied to a grid state."""

    @abc.abstractmethod
    def apply(self, cells: np.ndarray) -> np.ndarray | None:
        """
        Return a new perturbed grid, or None if the perturbation is invalid
        (e.g. source is dead or destination is already occupied).
        """

    @property
    @abc.abstractmethod
    def source(self) -> tuple[int, int]:
        """Grid coordinate used to index results into the heatmap."""

    @property
    @abc.abstractmethod
    def label(self) -> str:
        """Human-readable description for figure titles."""


class CellMovePerturbation(Perturbation):
    """
    Move the living cell at (row, col) by displacement (dr, dc),
    wrapping around the toroidal boundary.

    Returns None if the source is dead or the destination is already alive,
    ensuring every valid perturbation is a genuine two-cell change.
    """

    def __init__(self, row: int, col: int, dr: int, dc: int, grid_size: int):
        self._row, self._col = row, col
        self._dr, self._dc = dr, dc
        self._size = grid_size

    @property
    def source(self) -> tuple[int, int]:
        return (self._row, self._col)

    @property
    def displacement(self) -> float:
        return math.sqrt(self._dr ** 2 + self._dc ** 2)

    @property
    def label(self) -> str:
        return f"move ({self._row},{self._col}) by ({self._dr:+d},{self._dc:+d})"

    def apply(self, cells: np.ndarray) -> np.ndarray | None:
        if not cells[self._row, self._col]:
            return None  # source is dead — nothing to move
        nr = (self._row + self._dr) % self._size
        nc = (self._col + self._dc) % self._size
        if cells[nr, nc]:
            return None  # destination already alive — not a true move
        result = cells.copy()
        result[self._row, self._col] = 0
        result[nr, nc] = 1
        return result


# ── Configuration & results ────────────────────────────────────────────────────

@dataclass
class SweepConfig:
    """All parameters that define a sensitivity sweep."""
    grid_size: int = 40
    steps: int = 60          # steps simulated after the perturbation is applied
    seed: int = 42
    density: float = 0.35
    t_perturb: int = 0       # generation at which the perturbation is injected
    out_dir: str = "figures_move"


@dataclass
class SweepResult:
    """Collected outputs from a completed sensitivity sweep."""
    config: SweepConfig
    init: np.ndarray              # t=0 grid
    baseline: np.ndarray          # (steps+1, H, W) trajectory from t_perturb onward
    perturb_state: np.ndarray     # grid at t_perturb (= init when t_perturb=0)
    cumulative: np.ndarray        # (H, W) max cumulative divergence across all directions
    final_div: np.ndarray         # (H, W) final-step divergence for the best direction
    per_step: np.ndarray          # (H, W, steps) per-step divergence for the best direction
    valid_mask: np.ndarray        # (H, W) bool — True for cells with at least one valid move
    best_dr: np.ndarray           # (H, W) int — dr of the direction that achieved max cumulative
    best_dc: np.ndarray           # (H, W) int — dc of the direction that achieved max cumulative
    displacements: list           # all displacement vectors tried
    top_idx: tuple[int, int]      # highest-impact cell among valid cells
    mid_idx: tuple[int, int]      # median-impact cell among valid cells
    low_idx: tuple[int, int]      # lowest-impact cell among valid cells


# ── Sweep runner ───────────────────────────────────────────────────────────────

class SensitivitySweep:
    """
    Run a move-perturbation sensitivity sweep over all given displacement vectors.

    For each living cell, every displacement in `displacements` is tried.
    The heatmap records the MAX cumulative divergence across all valid
    directions, and the per-step trajectory for that best direction.

    Subclass and override `_perturbations(state)` to plug in a different
    perturbation type while reusing the rest of the pipeline.

    Parameters
    ----------
    config       : SweepConfig
    displacements: list of (dr, dc) tuples to try per cell
    init         : optional custom initial grid (overrides random generation)
    """

    def __init__(self, config: SweepConfig, displacements: list[tuple[int, int]],
                 init: np.ndarray | None = None):
        self.cfg          = config
        self.displacements = displacements
        self._custom_init = init

    def _make_init(self) -> np.ndarray:
        if self._custom_init is not None:
            return self._custom_init.astype(np.uint8).copy()
        rng = np.random.default_rng(self.cfg.seed)
        return (rng.random((self.cfg.grid_size, self.cfg.grid_size)) < self.cfg.density).astype(np.uint8)

    def _perturbations(self, state: np.ndarray):
        """Yield one Perturbation per (cell, displacement) pair."""
        N = self.cfg.grid_size
        for r in range(N):
            for c in range(N):
                for dr, dc in self.displacements:
                    yield CellMovePerturbation(r, c, dr, dc, N)

    def run(self) -> SweepResult:
        cfg = self.cfg
        N, S = cfg.grid_size, cfg.steps

        init = self._make_init()

        pre_traj      = Engine.run_trajectory(init, cfg.t_perturb)
        perturb_state = pre_traj[-1]

        print(f"Running baseline (t_perturb={cfg.t_perturb}, steps={S}) …")
        baseline = Engine.run_trajectory(perturb_state, S)

        # Track the MAX cumulative divergence across all directions per cell.
        # per_step and final_div are recorded for the best direction.
        cumulative = np.full((N, N), -1.0, dtype=np.float32)
        final_div  = np.zeros((N, N), dtype=np.float32)
        per_step   = np.zeros((N, N, S), dtype=np.float32)
        valid_mask = np.zeros((N, N), dtype=bool)
        best_dr    = np.zeros((N, N), dtype=np.int32)
        best_dc    = np.zeros((N, N), dtype=np.int32)

        n_dirs  = len(self.displacements)
        n_total = N * N * n_dirs
        print(
            f"Sweeping {N}×{N} cells × {n_dirs} direction(s) = {n_total} perturbations …"
        )

        for i, pert in enumerate(self._perturbations(perturb_state)):
            r, c    = pert.source
            p_state = pert.apply(perturb_state)
            if p_state is None:
                continue
            valid_mask[r, c] = True
            p_traj = Engine.run_trajectory(p_state, S)
            diffs  = (p_traj[1:] != baseline[1:]).sum(axis=(1, 2))
            cum    = float(diffs.sum())
            if cum > cumulative[r, c]:
                cumulative[r, c] = cum
                per_step[r, c]   = diffs
                final_div[r, c]  = diffs[-1]
                best_dr[r, c]    = pert._dr
                best_dc[r, c]    = pert._dc

            if (i + 1) % max(1, n_total // 10) == 0:
                print(f"  {i + 1}/{n_total}")

        # Cells with no valid move in any direction get cumulative = 0
        cumulative = np.where(valid_mask, cumulative, 0.0).astype(np.float32)

        print("Sweep done.")

        flat_valid = np.where(valid_mask.ravel(), cumulative.ravel(), -1.0)
        order_arr  = [i for i in np.argsort(flat_valid)[::-1] if flat_valid[i] >= 0]
        n          = len(order_arr)

        top_idx = np.unravel_index(order_arr[0],       (N, N))
        mid_idx = np.unravel_index(order_arr[n // 2],  (N, N))
        low_idx = np.unravel_index(order_arr[-1],      (N, N))

        print(f"  Directions tried per cell : {n_dirs}")
        print(f"  Valid cells (≥1 valid move): {valid_mask.sum()}")
        print(f"  Top: {top_idx}  cumulative={cumulative[top_idx]:.0f}"
              f"  best dir=({int(best_dr[top_idx]):+d},{int(best_dc[top_idx]):+d})")
        print(f"  Mid: {mid_idx}  cumulative={cumulative[mid_idx]:.0f}")
        print(f"  Low: {low_idx}  cumulative={cumulative[low_idx]:.0f}")

        return SweepResult(
            config=cfg, init=init, baseline=baseline,
            perturb_state=perturb_state,
            cumulative=cumulative, final_div=final_div, per_step=per_step,
            valid_mask=valid_mask, best_dr=best_dr, best_dc=best_dc,
            displacements=list(self.displacements),
            top_idx=top_idx, mid_idx=mid_idx, low_idx=low_idx,
        )


# ── Visualizer ─────────────────────────────────────────────────────────────────

class SensitivityVisualizer:
    """
    Produce the standard 7-figure analysis suite from a SweepResult.
    All figures are saved to result.config.out_dir.
    """

    def __init__(self, result: SweepResult):
        self.r   = result
        self.cfg = result.config
        os.makedirs(self.cfg.out_dir, exist_ok=True)

    # -- cached helpers ----------------------------------------------------

    @functools.cached_property
    def _top_traj(self) -> np.ndarray:
        """Trajectory for the top-impact cell's best direction (computed once)."""
        dr = int(self.r.best_dr[self.r.top_idx])
        dc = int(self.r.best_dc[self.r.top_idx])
        pert    = CellMovePerturbation(*self.r.top_idx, dr, dc, self.cfg.grid_size)
        p_state = pert.apply(self.r.perturb_state)
        return Engine.run_trajectory(p_state, self.cfg.steps)

    def _cum_masked(self) -> np.ndarray:
        """Cumulative heatmap with dead-cell positions set to NaN."""
        out = self.r.cumulative.astype(float).copy()
        out[~self.r.valid_mask] = np.nan
        return out

    def _disp_str(self) -> str:
        n   = len(self.r.displacements)
        dr0, dc0 = self.r.best_dr[self.r.top_idx], self.r.best_dc[self.r.top_idx]
        return (f"{n} direction(s)  |  "
                f"top-cell best dir=({int(dr0):+d},{int(dc0):+d})")

    def _path(self, fname: str) -> str:
        return os.path.join(self.cfg.out_dir, fname)

    def _representative_indices(self, n_each: int = 5):
        N  = self.cfg.grid_size
        fv = np.where(self.r.valid_mask.ravel(), self.r.cumulative.ravel(), -1.0)
        order = [i for i in np.argsort(fv)[::-1] if fv[i] >= 0]
        n = len(order)

        def pick(positions):
            return [np.unravel_index(order[p], (N, N))
                    for p in positions if 0 <= p < n]

        mid = n // 2
        return (
            pick(range(n_each)),
            pick(range(max(0, mid - n_each // 2), mid + n_each // 2 + 1)),
            pick(range(max(0, n - n_each), n)),
        )

    # -- 7 figures ---------------------------------------------------------

    def _save_01(self):
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(self._cum_masked(), cmap="hot", interpolation="nearest")
        ax.scatter([self.r.top_idx[1]], [self.r.top_idx[0]], s=120, c="cyan",
                   marker="*", label=f"top {self.r.top_idx}", zorder=5)
        ax.scatter([self.r.mid_idx[1]], [self.r.mid_idx[0]], s=80,  c="lime",
                   marker="o", label=f"mid {self.r.mid_idx}", zorder=5)
        ax.scatter([self.r.low_idx[1]], [self.r.low_idx[0]], s=80,  c="blue",
                   marker="s", label=f"low {self.r.low_idx}", zorder=5)
        plt.colorbar(im, ax=ax, label="Max cumulative divergence across directions")
        ax.set_title(
            f"Move-Perturbation Sensitivity — Cumulative over {self.cfg.steps} steps\n"
            f"{self._disp_str()}  |  t_perturb={self.cfg.t_perturb}",
            fontweight="bold", fontsize=10,
        )
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        fig.savefig(self._path("01_sensitivity_cumulative.png"), dpi=150)
        plt.close(fig)
        print("[1] Cumulative sensitivity heatmap saved.")

    def _save_02(self):
        fd = self.r.final_div.astype(float).copy()
        fd[~self.r.valid_mask] = np.nan
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(fd, cmap="plasma", interpolation="nearest")
        plt.colorbar(im, ax=ax, label=f"Divergence at step {self.cfg.steps}")
        ax.set_title(
            f"Move-Perturbation Sensitivity — Final State (step {self.cfg.steps})\n"
            f"{self._disp_str()}  |  t_perturb={self.cfg.t_perturb}",
            fontweight="bold", fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(self._path("02_sensitivity_final.png"), dpi=150)
        plt.close(fig)
        print("[2] Final-step sensitivity heatmap saved.")

    def _save_03(self):
        top5, mid5, low5 = self._representative_indices()
        ts = np.arange(1, self.cfg.steps + 1)
        fig, ax = plt.subplots(figsize=(10, 5))
        for idx in top5:
            ax.plot(ts, self.r.per_step[idx], color="crimson",   alpha=0.7, linewidth=1.2)
        for idx in mid5:
            ax.plot(ts, self.r.per_step[idx], color="goldenrod", alpha=0.7, linewidth=1.2)
        for idx in low5:
            ax.plot(ts, self.r.per_step[idx], color="steelblue", alpha=0.7, linewidth=1.2)
        ax.legend(handles=[
            Line2D([0], [0], color="crimson",   lw=2, label="Top-5 impact"),
            Line2D([0], [0], color="goldenrod", lw=2, label="Mid-5 impact"),
            Line2D([0], [0], color="steelblue", lw=2, label="Low-5 impact"),
        ])
        ax.set_xlabel("Step")
        ax.set_ylabel("Cells differing from baseline")
        ax.set_title(
            f"Divergence Over Time — {self._disp_str()}\n"
            f"t_perturb={self.cfg.t_perturb}",
            fontweight="bold",
        )
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(self._path("03_divergence_over_time.png"), dpi=150)
        plt.close(fig)
        print("[3] Divergence-over-time saved.")

    def _save_04(self):
        valid_cum   = self.r.cumulative[self.r.valid_mask].ravel()
        valid_final = self.r.final_div[self.r.valid_mask].ravel()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(valid_cum, bins=50, color="#C44E52", edgecolor="white", linewidth=0.4)
        axes[0].axvline(valid_cum.mean(), color="black", linestyle="--",
                        label=f"mean={valid_cum.mean():.0f}")
        axes[0].set_xlabel("Max cumulative divergence")
        axes[0].set_ylabel("Living cells")
        axes[0].set_title("Distribution of Move-Perturbation Impact (cumulative)", fontweight="bold")
        axes[0].legend()
        axes[1].hist(valid_final, bins=50, color="#4C72B0", edgecolor="white", linewidth=0.4)
        axes[1].axvline(valid_final.mean(), color="black", linestyle="--",
                        label=f"mean={valid_final.mean():.0f}")
        axes[1].set_xlabel(f"Divergence at step {self.cfg.steps}")
        axes[1].set_ylabel("Living cells")
        axes[1].set_title(
            f"Distribution of Move-Perturbation Impact (step {self.cfg.steps})", fontweight="bold"
        )
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(self._path("04_impact_distribution.png"), dpi=150)
        plt.close(fig)
        print("[4] Impact distribution saved.")

    def _save_05(self):
        S          = self.cfg.steps
        snap_steps = sorted({0, S // 6, S // 2, S})
        fig, axes  = plt.subplots(3, len(snap_steps), figsize=(14, 9))
        dr = int(self.r.best_dr[self.r.top_idx])
        dc = int(self.r.best_dc[self.r.top_idx])
        for col, t in enumerate(snap_steps):
            base_frame = self.r.baseline[t]
            top_frame  = self._top_traj[t]
            diff_frame = base_frame.astype(int) - top_frame.astype(int)
            axes[0, col].imshow(base_frame, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
            axes[0, col].set_title(f"Step {t}", fontsize=10)
            axes[0, col].axis("off")
            axes[1, col].imshow(top_frame, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
            axes[1, col].axis("off")
            axes[2, col].imshow(diff_frame, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
            axes[2, col].axis("off")
        for ax, lbl in zip(axes[:, 0],
                           ["Baseline", f"Perturbed {self.r.top_idx}", "Difference"]):
            ax.set_ylabel(lbl, fontsize=10)
        for row in range(2):
            axes[row, 0].scatter(
                [self.r.top_idx[1]], [self.r.top_idx[0]], s=80, c="cyan", marker="*", zorder=5
            )
        fig.suptitle(
            f"Baseline vs Highest-Impact Move Perturbation\n"
            f"cell {self.r.top_idx}, best dir=({dr:+d},{dc:+d}), "
            f"t_perturb={self.cfg.t_perturb}",
            fontweight="bold", fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(self._path("05_baseline_vs_top.png"), dpi=150)
        plt.close(fig)
        print("[5] Baseline vs top perturbation snapshots saved.")

    def _save_06(self):
        S          = self.cfg.steps
        show_steps = list(range(0, S + 1, max(1, S // 12)))[:12]
        diff_traj  = self._top_traj.astype(int) - self.r.baseline.astype(int)
        ncols      = 6
        nrows      = (len(show_steps) + ncols - 1) // ncols
        fig, axes  = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))
        axes       = axes.ravel()
        for i, t in enumerate(show_steps):
            axes[i].imshow(diff_traj[t], cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
            axes[i].set_title(f"t={t}", fontsize=9)
            axes[i].axis("off")
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        dr = int(self.r.best_dr[self.r.top_idx])
        dc = int(self.r.best_dc[self.r.top_idx])
        fig.suptitle(
            f"Difference Maps: Perturbed {self.r.top_idx} vs Baseline\n"
            f"best dir=({dr:+d},{dc:+d}), t_perturb={self.cfg.t_perturb}",
            fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(self._path("06_difference_maps.png"), dpi=150)
        plt.close(fig)
        print("[6] Difference maps saved.")

    def _save_07(self):
        cum_over_time = np.cumsum(self.r.per_step, axis=2)
        vmax          = cum_over_time[:, :, -1].max()

        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
        ax.axis("off")
        im  = ax.imshow(cum_over_time[:, :, 0], cmap="hot", vmin=0, vmax=vmax,
                        interpolation="nearest")
        ttl = ax.set_title("Cumulative divergence  step 1", color="white", fontsize=11)
        cb  = plt.colorbar(im, ax=ax)
        cb.ax.yaxis.set_tick_params(color="white", labelcolor="white")

        def _update(frame):
            im.set_data(cum_over_time[:, :, frame])
            ttl.set_text(f"Cumulative divergence  step {frame + 1}")
            return im, ttl

        ani = animation.FuncAnimation(fig, _update, frames=self.cfg.steps, interval=120, blit=True)
        ani.save(self._path("07_sensitivity_gif.gif"), writer=animation.PillowWriter(fps=10))
        plt.close(fig)
        print("[7] Sensitivity animation saved.")

    def save_all(self):
        self._save_01()
        self._save_02()
        self._save_03()
        self._save_04()
        self._save_05()
        self._save_06()
        self._save_07()
        n_valid = int(self.r.valid_mask.sum())
        n_total = self.cfg.grid_size ** 2
        n_dirs  = len(self.r.displacements)
        print(
            f"\n── Summary ──────────────────────────────────────────────\n"
            f"  Grid              : {self.cfg.grid_size}×{self.cfg.grid_size} ({n_total} cells)\n"
            f"  Living at t_perturb={self.cfg.t_perturb}: {n_valid}\n"
            f"  Directions tried  : {n_dirs}\n"
            f"  Steps             : {self.cfg.steps}\n"
            f"  Mean max-cumulative divergence : {self.r.cumulative[self.r.valid_mask].mean():.1f}\n"
            f"  Max  max-cumulative divergence : {self.r.cumulative[self.r.valid_mask].max():.0f}"
            f"  at {self.r.top_idx}\n"
            f"\nAll figures saved to ./{self.cfg.out_dir}/"
        )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = SweepConfig(
        grid_size=40,
        steps=60,
        seed=42,
        density=0.35,
        t_perturb=0,
        out_dir="figures_move",
    )
    sweep  = SensitivitySweep(cfg, displacements=manhattan_displacements(1))
    result = sweep.run()
    SensitivityVisualizer(result).save_all()
