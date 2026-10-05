"""Compare Model 5v3 (shared trunk) against Model 4 (hand-coded teacher)
and Model 5v2 (fully separate Part A/Part B, ../model5/) side by side: a
rollout-equivalence demo across all three, and a direct, minimal example
of the perturbation-sensitivity difference analysis_perturbation.py
quantifies for each learned-canonicalization model (own predicted pose,
no teacher). Lives at the CNN_toy_model/ level since it spans three model
folders, loaded via importlib (each defines a module literally named
`net.py`, which would collide under a plain sys.path import).

Run: python visualize_model5v3_vs_others.py
Outputs -> results_model5v3_vs_others/*.gif, *.png
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
OUT = os.path.join(_HERE, "results_model5v3_vs_others")
os.makedirs(OUT, exist_ok=True)


def _load_module(rel_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, rel_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


net4 = _load_module("model4/net.py", "net4_v3cmp")
net5 = _load_module("model5/net.py", "net5_v3cmp")
net5v3 = _load_module("model5v3/net.py", "net5v3_v3cmp")


def _load_model(net_mod, cls_name, ckpt_rel):
    model = getattr(net_mod, cls_name)(m=4)
    ckpt = torch.load(os.path.join(_HERE, ckpt_rel), map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


M4 = _load_model(net4, "ToyCNNModel4", "model4/checkpoints/best.pt")
M5 = _load_model(net5, "ToyCNNModel5", "model5/checkpoints/best.pt")
M5V3 = _load_model(net5v3, "ToyCNNModel5v3", "model5v3/checkpoints/best.pt")


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


# ── 1) rollout comparison: ground truth + 3 models, same input, autoregressive ──
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
    truth, pred4 = rollout(M4, g0, steps)
    _, pred5 = rollout(M5, g0, steps)
    _, pred5v3 = rollout(M5V3, g0, steps)
    diff4 = [int((t != p).sum()) for t, p in zip(truth, pred4)]
    diff5 = [int((t != p).sum()) for t, p in zip(truth, pred5)]
    diff5v3 = [int((t != p).sum()) for t, p in zip(truth, pred5v3)]

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    titles = ["ground truth", "Model 4\n(hand-coded)", "Model 5v2\n(isolated parts)", "Model 5v3\n(shared trunk)"]
    ims = []
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        im = ax.imshow(np.zeros((GRID, GRID)), cmap="Greys", vmin=0, vmax=1)
        ims.append(im)
    fig.suptitle(f"Rollout: {name}  (max diff -- M4: {max(diff4)}, M5v2: {max(diff5)}, "
                 f"M5v3: {max(diff5v3)} cells)", fontsize=10)

    def update(t):
        ims[0].set_data(truth[t])
        ims[1].set_data(pred4[t])
        ims[2].set_data(pred5[t])
        ims[3].set_data(pred5v3[t])
        axes[0].set_xlabel(f"t={t}")
        return ims

    ani = animation.FuncAnimation(fig, update, frames=len(truth), blit=False)
    path = os.path.join(OUT, f"rollout_{name}_v3_vs_others.gif")
    ani.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"{name}: max diff M4={max(diff4)} M5v2={max(diff5)} M5v3={max(diff5v3)} cells -> {path}")


# ── 2) representation under the pure D4 orbit (own predicted pose, no teacher) ──
@torch.no_grad()
def save_d4_orbit_gif(name: str, g0: np.ndarray, fps: int = 1):
    """Same content, all 8 D4-transformed copies. Shows feat_can (model's
    OWN pose) cycling through the orbit for Model 5v2 vs Model 5v3 --
    Model 4 would be pixel-identical at every step (exact guarantee);
    these two need not be."""
    x0 = torch.from_numpy(g0).unsqueeze(0).unsqueeze(0)
    frames5, frames5v3, gidx5_list, gidx5v3_list, cos5_list, cos5v3_list = [], [], [], [], [], []
    _, aux0_5 = M5(x0, return_aux=True)
    _, aux0_5v3 = M5V3(x0, return_aux=True)
    fc0_5, fc0_5v3 = aux0_5["feat_can"], aux0_5v3["feat_can"]
    for g in range(8):
        xg = torch_d4(x0, g)
        _, auxg_5 = M5(xg, return_aux=True)
        _, auxg_5v3 = M5V3(xg, return_aux=True)
        frames5.append(auxg_5["feat_can"][0].sum(0).numpy())
        frames5v3.append(auxg_5v3["feat_can"][0].sum(0).numpy())
        gidx5_list.append(D4_NAMES[auxg_5["gidx_used"].item()])
        gidx5v3_list.append(D4_NAMES[auxg_5v3["gidx_used"].item()])
        cos5_list.append(cos(fc0_5, auxg_5["feat_can"]))
        cos5v3_list.append(cos(fc0_5v3, auxg_5v3["feat_can"]))

    vmax = max(np.abs(frames5).max(), np.abs(frames5v3).max())
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.2))
    axes[0].set_title("Model 5v2 feat_can", fontsize=10)
    axes[1].set_title("Model 5v3 feat_can", fontsize=10)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    im0 = axes[0].imshow(frames5[0], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    im1 = axes[1].imshow(frames5v3[0], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    txt0 = axes[0].set_xlabel("", fontsize=9)
    txt1 = axes[1].set_xlabel("", fontsize=9)
    suptitle = fig.suptitle("", fontsize=11, y=0.98)
    fig.tight_layout(rect=[0, 0.02, 1, 0.9])

    def update(t):
        im0.set_data(frames5[t])
        im1.set_data(frames5v3[t])
        txt0.set_text(f"gidx_used={gidx5_list[t]}  cos(vs g=e)={cos5_list[t]:.4f}")
        txt1.set_text(f"gidx_used={gidx5v3_list[t]}  cos(vs g=e)={cos5v3_list[t]:.4f}")
        suptitle.set_text(f"{name}: all 8 D4-equivalent inputs, same content -- "
                           f"applied g = {D4_NAMES[t]}\n(Model 4 would stay pixel-identical here by construction)")
        return im0, im1

    ani = animation.FuncAnimation(fig, update, frames=8, interval=900, blit=False)
    path = os.path.join(OUT, f"d4_orbit_{name}_v2_vs_v3.gif")
    ani.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"{name}: D4-orbit cos (vs g=e) -- M5v2 mean={np.mean(cos5_list):.4f}  "
          f"M5v3 mean={np.mean(cos5v3_list):.4f} -> {path}")


# ── 3) perturbation difference, minimal example: auto-pick an interesting cell ──
@torch.no_grad()
def _gidx_used(model, x):
    _, aux = model(x, return_aux=True)
    return aux["gidx_used"].item(), aux["feat_can"]


@torch.no_grad()
def find_branch_change_example(g0: np.ndarray):
    """Scan single-cell flips for one where EITHER model's gidx_used
    changes -- a more informative, honest pick than a fixed cell chosen
    before knowing how v3 actually trained."""
    x0 = torch.from_numpy(g0).unsqueeze(0).unsqueeze(0)
    gidx5_0, _ = _gidx_used(M5, x0)
    gidx5v3_0, _ = _gidx_used(M5V3, x0)
    H, W = GRID, GRID
    for r in range(H):
        for c in range(W):
            xp = x0.clone()
            xp[0, 0, r, c] = 1 - xp[0, 0, r, c]
            gidx5_p, _ = _gidx_used(M5, xp)
            gidx5v3_p, _ = _gidx_used(M5V3, xp)
            if gidx5_p != gidx5_0 or gidx5v3_p != gidx5v3_0:
                return r, c
    return 0, 9  # fallback: no branch change found anywhere, use an arbitrary cell


@torch.no_grad()
def save_perturbation_difference_gif(g0: np.ndarray, r: int, c: int, fps: int = 1):
    """Same base grid, same single-cell flip at (r,c). Shows: the grid
    (flipped cell marked), and each model's feat_can (toggling before/
    after), with the canonicalization choice (gidx_used) and cosine
    similarity annotated -- own predicted pose end to end, no teacher."""
    x0 = torch.from_numpy(g0).unsqueeze(0).unsqueeze(0)
    xp = x0.clone()
    xp[0, 0, r, c] = 1 - xp[0, 0, r, c]

    _, aux0_5 = M5(x0, return_aux=True)
    _, auxp_5 = M5(xp, return_aux=True)
    _, aux0_5v3 = M5V3(x0, return_aux=True)
    _, auxp_5v3 = M5V3(xp, return_aux=True)

    c5 = cos(aux0_5["feat_can"], auxp_5["feat_can"])
    c5v3 = cos(aux0_5v3["feat_can"], auxp_5v3["feat_can"])
    g5_before, g5_after = D4_NAMES[aux0_5["gidx_used"].item()], D4_NAMES[auxp_5["gidx_used"].item()]
    g5v3_before, g5v3_after = D4_NAMES[aux0_5v3["gidx_used"].item()], D4_NAMES[auxp_5v3["gidx_used"].item()]

    fc5 = [aux0_5["feat_can"][0].sum(0).numpy(), auxp_5["feat_can"][0].sum(0).numpy()]
    fc5v3 = [aux0_5v3["feat_can"][0].sum(0).numpy(), auxp_5v3["feat_can"][0].sum(0).numpy()]
    vmax = max(np.abs(fc5[0]).max(), np.abs(fc5[1]).max(), np.abs(fc5v3[0]).max(), np.abs(fc5v3[1]).max())
    grids = [g0, xp[0, 0].numpy()]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    axes[0].set_title("input grid\n(circle = flipped cell)", fontsize=10)
    axes[1].set_title("Model 5v2 feat_can\n(isolated parts)", fontsize=10)
    axes[2].set_title("Model 5v3 feat_can\n(shared trunk)", fontsize=10)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    im0 = axes[0].imshow(grids[0], cmap="Greys", vmin=0, vmax=1)
    circ = plt.Circle((c, r), 1.6, fill=False, edgecolor="crimson", linewidth=2)
    axes[0].add_patch(circ)
    im1 = axes[1].imshow(fc5[0], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    im2 = axes[2].imshow(fc5v3[0], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    txt1 = axes[1].set_xlabel(f"gidx_used={g5_before}", fontsize=9)
    txt2 = axes[2].set_xlabel(f"gidx_used={g5v3_before}", fontsize=9)
    fig.suptitle(f"Same single-cell flip at (row={r}, col={c}):\n"
                 f"Model 5v2 cos={c5:.4f} ({g5_before}->{g5_after})   "
                 f"Model 5v3 cos={c5v3:.4f} ({g5v3_before}->{g5v3_after})",
                 fontsize=11, y=0.99)
    fig.tight_layout(rect=[0, 0.02, 1, 0.82])

    def update(t):
        im0.set_data(grids[t])
        im1.set_data(fc5[t])
        im2.set_data(fc5v3[t])
        txt1.set_text(f"gidx_used={[g5_before, g5_after][t]}  (t={'before' if t == 0 else 'after'})")
        txt2.set_text(f"gidx_used={[g5v3_before, g5v3_after][t]}  (t={'before' if t == 0 else 'after'})")
        return im0, im1, im2

    ani = animation.FuncAnimation(fig, update, frames=2, interval=900, blit=False)
    path = os.path.join(OUT, "perturbation_difference_v2_vs_v3.gif")
    ani.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"perturbation difference example at ({r},{c}): M5v2 cos={c5:.4f} ({g5_before}->{g5_after})  "
          f"M5v3 cos={c5v3:.4f} ({g5v3_before}->{g5v3_after})  -> {path}")


if __name__ == "__main__":
    save_rollout_comparison_gif("glider", place("glider", 15, 15), steps=30)
    save_rollout_comparison_gif("pulsar", place("pulsar", 13, 13), steps=15)

    save_d4_orbit_gif("pulsar", place("pulsar", 13, 13))
    save_d4_orbit_gif("random_d0.1", random_grid(0.1, 1))

    g0 = random_grid(0.1, 1)
    r, c = find_branch_change_example(g0)
    save_perturbation_difference_gif(g0, r=r, c=c)
