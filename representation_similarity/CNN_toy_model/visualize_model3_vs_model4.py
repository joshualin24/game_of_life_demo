"""Compare Model 3 (quadrupole/cubic moments) and Model 4 (bounding-box
extremes, Model 3's shift mechanism) side by side: a rollout-equivalence
demo, and a direct, minimal example of the perturbation-sensitivity
difference analysis_perturbation.py quantified (see each model's
notes.md). Lives at the CNN_toy_model/ level since it spans two model
folders, loaded via importlib (both define a module literally named
`net.py`, which would collide under a plain sys.path import).

Run: python visualize_model3_vs_model4.py
Outputs -> results_model3_vs_model4/*.gif, *.png
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "v1"))
sys.path.insert(0, _HERE)
from adapter import gol_step_torch, torch_d4, D4_NAMES  # noqa: E402
from stimuli import _PATTERNS, _bitmap  # noqa: E402

GRID = 40
OUT = os.path.join(_HERE, "results_model3_vs_model4")
os.makedirs(OUT, exist_ok=True)


def _load_module(rel_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, rel_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


net3 = _load_module("model3/net.py", "net3_cmp")
net4 = _load_module("model4/net.py", "net4_cmp")


def _load_model(net_mod, cls_name, ckpt_rel):
    model = getattr(net_mod, cls_name)(m=4)
    ckpt = torch.load(os.path.join(_HERE, ckpt_rel), map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


M3 = _load_model(net3, "ToyCNNModel3", "model3/checkpoints/best.pt")
M4 = _load_model(net4, "ToyCNNModel4", "model4/checkpoints/best.pt")


def place(name: str, top: int, left: int) -> np.ndarray:
    bmp = _bitmap(_PATTERNS[name][1])
    g = np.zeros((GRID, GRID), dtype=np.float32)
    h, w = bmp.shape
    g[top:top + h, left:left + w] = bmp
    return g


def random_grid(density: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((GRID, GRID)) < density).astype(np.float32)


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten(), b.flatten()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


# ── 1) rollout comparison: both models, same input, autoregressive ────────
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


def save_rollout_comparison_gif(name: str, g0: np.ndarray, steps: int, fps: int = 6):
    truth, pred3 = rollout(M3, g0, steps)
    _, pred4 = rollout(M4, g0, steps)
    diff3 = [int((t != p).sum()) for t, p in zip(truth, pred3)]
    diff4 = [int((t != p).sum()) for t, p in zip(truth, pred4)]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))
    titles = ["ground truth", "Model 3 (quadrupole)", "Model 4 (extremes)"]
    ims = []
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        im = ax.imshow(np.zeros((GRID, GRID)), cmap="Greys", vmin=0, vmax=1)
        ims.append(im)
    fig.suptitle(f"Model 3 vs Model 4 rollout: {name}  "
                 f"(max diff -- M3: {max(diff3)}, M4: {max(diff4)} cells)")

    def update(t):
        ims[0].set_data(truth[t])
        ims[1].set_data(pred3[t])
        ims[2].set_data(pred4[t])
        axes[0].set_xlabel(f"t={t}")
        return ims

    ani = animation.FuncAnimation(fig, update, frames=len(truth), blit=False)
    path = os.path.join(OUT, f"rollout_{name}_m3_vs_m4.gif")
    ani.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"{name}: max diff M3={max(diff3)} M4={max(diff4)} cells -> {path}")


# ── 2) the actual difference, minimal example ──────────────────────────────
@torch.no_grad()
def save_perturbation_difference_gif(g0: np.ndarray, r: int, c: int, fps: int = 1):
    """Same base grid, same single-cell flip at (r,c). Shows: the grid
    (toggling base/perturbed, flipped cell marked), and each model's
    feat_can (toggling before/after), with the canonicalization choice
    (gidx) and cosine similarity annotated -- the direct, minimal picture
    of what analysis_perturbation.py's aggregate numbers are measuring."""
    x0 = torch.from_numpy(g0).unsqueeze(0).unsqueeze(0)
    xp = x0.clone()
    xp[0, 0, r, c] = 1 - xp[0, 0, r, c]

    _, aux0_3 = M3(x0, return_features=True)
    _, auxp_3 = M3(xp, return_features=True)
    _, aux0_4 = M4(x0, return_features=True)
    _, auxp_4 = M4(xp, return_features=True)

    c3 = cos(aux0_3["feat_can"], auxp_3["feat_can"])
    c4 = cos(aux0_4["feat_can"], auxp_4["feat_can"])
    g3_before, g3_after = D4_NAMES[aux0_3["gidx"].item()], D4_NAMES[auxp_3["gidx"].item()]
    g4_before, g4_after = D4_NAMES[aux0_4["gidx"].item()], D4_NAMES[auxp_4["gidx"].item()]

    fc3 = [aux0_3["feat_can"][0].sum(0).numpy(), auxp_3["feat_can"][0].sum(0).numpy()]
    fc4 = [aux0_4["feat_can"][0].sum(0).numpy(), auxp_4["feat_can"][0].sum(0).numpy()]
    vmax = max(np.abs(fc3[0]).max(), np.abs(fc3[1]).max(), np.abs(fc4[0]).max(), np.abs(fc4[1]).max())
    grids = [g0, xp[0, 0].numpy()]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    axes[0].set_title("input grid\n(circle = flipped cell)", fontsize=10)
    axes[1].set_title("Model 3 feat_can\n(quadrupole scoring)", fontsize=10)
    axes[2].set_title("Model 4 feat_can\n(extreme scoring)", fontsize=10)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    im0 = axes[0].imshow(grids[0], cmap="Greys", vmin=0, vmax=1)
    circ = plt.Circle((c, r), 1.6, fill=False, edgecolor="crimson", linewidth=2)
    axes[0].add_patch(circ)
    im1 = axes[1].imshow(fc3[0], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    im2 = axes[2].imshow(fc4[0], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    txt1 = axes[1].set_xlabel(f"gidx={g3_before}", fontsize=9)
    txt2 = axes[2].set_xlabel(f"gidx={g4_before}", fontsize=9)
    fig.suptitle(f"Same single-cell flip at (row={r}, col={c}):\n"
                 f"Model 3 cos={c3:.4f} ({g3_before}->{g3_after})   "
                 f"Model 4 cos={c4:.4f} ({g4_before}->{g4_after})",
                 fontsize=11, y=0.99)
    fig.tight_layout(rect=[0, 0.02, 1, 0.82])

    def update(t):
        im0.set_data(grids[t])
        im1.set_data(fc3[t])
        im2.set_data(fc4[t])
        txt1.set_text(f"gidx={[g3_before, g3_after][t]}  (t={'before' if t == 0 else 'after'})")
        txt2.set_text(f"gidx={[g4_before, g4_after][t]}  (t={'before' if t == 0 else 'after'})")
        return im0, im1, im2

    ani = animation.FuncAnimation(fig, update, frames=2, interval=900, blit=False)
    path = os.path.join(OUT, "perturbation_difference_m3_vs_m4.gif")
    ani.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"perturbation difference example: M3 cos={c3:.4f} ({g3_before}->{g3_after})  "
          f"M4 cos={c4:.4f} ({g4_before}->{g4_after})  -> {path}")


if __name__ == "__main__":
    save_rollout_comparison_gif("glider", place("glider", 15, 15), steps=30)
    save_rollout_comparison_gif("pulsar", place("pulsar", 13, 13), steps=15)

    g0 = random_grid(0.1, 1)
    save_perturbation_difference_gif(g0, r=0, c=9)
