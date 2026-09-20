"""Visual demos of the trained Model 2: autoregressive rollout accuracy,
D4 equivariance (using the patterns that were Model 1's problem cases), and
-- new, specific to Model 2's contribution -- a direct visualization of
feat_can itself proving it's now pixel-identical across position AND
orientation for a symmetric pattern.

Run: python visualize.py
Outputs -> results/gifs/*.gif, results/gifs/*.png
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "v1"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
from net import ToyCNNModel2  # noqa: E402
from adapter import gol_step_torch, torch_d4, torch_translate, D4_NAMES  # noqa: E402
from stimuli import _PATTERNS, _bitmap  # noqa: E402

GRID = 40
OUT = os.path.join(_HERE, "results", "gifs")
os.makedirs(OUT, exist_ok=True)


def load_model(ckpt_name: str = "best.pt"):
    model = ToyCNNModel2(m=4)
    ckpt = torch.load(os.path.join(_HERE, "checkpoints", ckpt_name),
                       map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def place(name: str, top: int, left: int, r_idx: int = 0) -> np.ndarray:
    bmp = _bitmap(_PATTERNS[name][1])
    if r_idx:
        from symmetry import d4_ops
        bmp = d4_ops()[D4_NAMES[r_idx]](bmp[None])[0]
    g = np.zeros((GRID, GRID), dtype=np.float32)
    h, w = bmp.shape
    g[top:top + h, left:left + w] = bmp
    return g


@torch.no_grad()
def rollout(model, g0: np.ndarray, steps: int):
    x_true = torch.from_numpy(g0).unsqueeze(0).unsqueeze(0)
    x_model = x_true.clone()
    truth, pred = [x_true[0, 0].numpy().copy()], [x_model[0, 0].numpy().copy()]
    for _ in range(steps):
        x_true = gol_step_torch(x_true)
        x_model = model.step(x_model)
        truth.append(x_true[0, 0].numpy().copy())
        pred.append(x_model[0, 0].numpy().copy())
    return truth, pred


def save_truth_vs_model_gif(name: str, g0: np.ndarray, steps: int, fps: int = 6):
    model = load_model()
    truth, pred = rollout(model, g0, steps)
    n_diff = [int((t != p).sum()) for t, p in zip(truth, pred)]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))
    titles = ["ground truth", "model (autoregressive)", "diff (should be blank)"]
    ims = []
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        im = ax.imshow(np.zeros((GRID, GRID)), cmap="Greys", vmin=0, vmax=1)
        ims.append(im)
    fig.suptitle(f"Model 2 rollout: {name}  (max cell diff over {steps} steps: {max(n_diff)})")

    def update(t):
        ims[0].set_data(truth[t])
        ims[1].set_data(pred[t])
        ims[2].set_data((truth[t] != pred[t]).astype(float))
        fig.axes[0].set_xlabel(f"t={t}")
        return ims

    ani = animation.FuncAnimation(fig, update, frames=len(truth), blit=False)
    path = os.path.join(OUT, f"rollout_{name}.gif")
    ani.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"{name}: max diff over rollout = {max(n_diff)} cells  -> {path}")
    return path


def save_equivariance_gif(name: str, top: int, left: int, steps: int, r_idx: int = 1, fps: int = 6):
    """Side-by-side: pattern vs the same pattern rotated by D4 element r_idx
    at t=0, both rolled out with the model. Uses block/pulsar -- Model 1's
    broken representation cases -- to tie this demo to Model 2's actual
    contribution, even though functional equivariance was already exact in
    Model 1 too (only the internal representation was the problem)."""
    model = load_model()
    g0 = place(name, top, left)
    g0_rot = torch_d4(torch.from_numpy(g0).unsqueeze(0).unsqueeze(0), r_idx)[0, 0].numpy()

    _, pred_a = rollout(model, g0, steps)
    _, pred_b = rollout(model, g0_rot, steps)

    max_err = 0
    for a, b in zip(pred_a, pred_b):
        expected = torch_d4(torch.from_numpy(a).unsqueeze(0).unsqueeze(0), r_idx)[0, 0].numpy()
        max_err = max(max_err, int(np.abs(expected - b).max()))

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.4))
    axes[0].set_title(f"{name}", fontsize=10)
    axes[1].set_title(f"{name}, {D4_NAMES[r_idx]}'d at t=0", fontsize=10)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    im0 = axes[0].imshow(pred_a[0], cmap="Greys", vmin=0, vmax=1)
    im1 = axes[1].imshow(pred_b[0], cmap="Greys", vmin=0, vmax=1)
    fig.suptitle(f"Model 2 D4 equivariance: both are pure model rollouts "
                 f"(max exact-{D4_NAMES[r_idx]} mismatch over {steps} steps: {max_err} cells)")

    def update(t):
        im0.set_data(pred_a[t])
        im1.set_data(pred_b[t])
        return im0, im1

    ani = animation.FuncAnimation(fig, update, frames=len(pred_a), blit=False)
    path = os.path.join(OUT, f"equivariance_{name}_{D4_NAMES[r_idx]}.gif")
    ani.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"{name} vs {D4_NAMES[r_idx]}(...): max exact mismatch = {max_err} cells -> {path}")
    return path


@torch.no_grad()
def save_feat_can_comparison_png(name: str, configs: list[tuple[int, int, int]]):
    """The demo specific to Model 2: visualize feat_can itself (summed
    across channels) for the SAME symmetric pattern at several different
    (top, left, r_idx) placements/orientations. In Model 1 these would look
    visibly shifted/misaligned for a symmetric pattern like block/pulsar;
    in Model 2 they should be pixel-identical."""
    model = load_model()
    maps, diffs = [], []
    ref = None
    for top, left, r_idx in configs:
        g = place(name, top, left, r_idx)
        x = torch.from_numpy(g).unsqueeze(0).unsqueeze(0)
        _, aux = model(x, return_features=True)
        fc = aux["feat_can"][0].sum(dim=0).numpy()  # (H,W), summed over channels
        maps.append(fc)
        if ref is None:
            ref = fc
        diffs.append(float(np.abs(fc - ref).max()))

    n = len(configs)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4))
    vmax = max(np.abs(m).max() for m in maps)
    for ax, (top, left, r_idx), fc, d in zip(axes, configs, maps, diffs):
        ax.set_title(f"({top},{left}), {D4_NAMES[r_idx]}\nmax|diff vs 1st| = {d:.2e}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(fc, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.suptitle(f"Model 2: feat_can (last layer, summed over channels) for '{name}'\n"
                 f"same symmetric pattern, different position AND orientation each time")
    fig.tight_layout()
    path = os.path.join(OUT, f"feat_can_invariance_{name}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"{name}: feat_can max diff across {n} (position,orientation) configs = "
          f"{max(diffs):.2e}  -> {path}")
    return path


if __name__ == "__main__":
    save_truth_vs_model_gif("glider", place("glider", 15, 15), steps=40)
    save_truth_vs_model_gif("pulsar", place("pulsar", 13, 13), steps=15)
    save_truth_vs_model_gif("lwss", place("lwss", 15, 10), steps=25)
    save_equivariance_gif("block", 8, 8, steps=10, r_idx=1)     # r90 -- Model 1's near-worst case
    save_equivariance_gif("pulsar", 5, 20, steps=15, r_idx=1)   # r90 -- Model 1's worst case (0.835)

    # Model 2's actual contribution, shown directly: same pattern, 4 very
    # different placements+orientations -> identical feat_can.
    save_feat_can_comparison_png("pulsar", [
        (13, 13, 0), (5, 20, 1), (22, 3, 2), (2, 2, 5),
    ])
    save_feat_can_comparison_png("block", [
        (18, 18, 0), (3, 30, 1), (30, 5, 3), (10, 25, 6),
    ])
