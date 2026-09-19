"""Out-of-distribution check: the model was trained ONLY on single-pattern
grids (data.py), never on random dense fills. This tests whether it learned
GoL's actual local rule (which would generalize to any density) or just
memorized sparse-pattern behavior (which would likely diverge on dense
grids, since GoL is chaotic and any single wrong cell compounds).

Run: python visualize_random.py
Outputs -> results/gifs/random_d*.gif, prints per-step disagreement counts.
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
from net import ToyCNNModel1  # noqa: E402
from adapter import gol_step_torch  # noqa: E402

GRID = 40
OUT = os.path.join(_HERE, "results", "gifs")
os.makedirs(OUT, exist_ok=True)


def load_model(ckpt_name: str = "best_v2.pt"):
    model = ToyCNNModel1(m=4)
    ckpt = torch.load(os.path.join(_HERE, "checkpoints", ckpt_name),
                       map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


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


def save_gif(density: float, seed: int, steps: int, fps: int = 6):
    rng = np.random.default_rng(seed)
    g0 = (rng.random((GRID, GRID)) < density).astype(np.float32)
    model = load_model()
    truth, pred = rollout(model, g0, steps)
    n_diff = [int((t != p).sum()) for t, p in zip(truth, pred)]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))
    titles = ["ground truth", "model (autoregressive)", "diff"]
    ims = []
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        im = ax.imshow(np.zeros((GRID, GRID)), cmap="Greys", vmin=0, vmax=1)
        ims.append(im)
    fig.suptitle(f"random grid, density={density}  (t=0 alive: {int(g0.sum())} cells)")

    def update(t):
        ims[0].set_data(truth[t])
        ims[1].set_data(pred[t])
        ims[2].set_data((truth[t] != pred[t]).astype(float))
        axes[2].set_xlabel(f"t={t}  diff={n_diff[t]} cells")
        return ims

    ani = animation.FuncAnimation(fig, update, frames=len(truth), blit=False)
    path = os.path.join(OUT, f"random_d{density:.2f}.gif")
    ani.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"density={density:.2f}  alive@t0={int(g0.sum())}  "
          f"diff per step: {n_diff}")
    return path, n_diff


if __name__ == "__main__":
    for d, seed in [(0.10, 1), (0.30, 2), (0.50, 3), (0.80, 4)]:
        save_gif(d, seed, steps=30)
