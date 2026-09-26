"""Perturbation sensitivity of Model 2's last-layer representation (feat_can).

Adapts this repo's classic perturbation-sensitivity study (../../perturbation.py:
flip one cell, measure how much the OUTPUT trajectory diverges) to the
INTERNAL REPRESENTATION instead: flip one cell, measure how much feat_can
changes. Every cell in the grid is flipped, one at a time, producing a full
(H,W) sensitivity map per base distribution.

Specific hypothesis this is built to check: canonical_transform picks a
D4 element AND a shift via argmax over 8 discrete candidates (see net.py).
Argmax selection is a decision boundary -- a small input perturbation that
flips which candidate wins should cause a DISCONTINUOUS jump in feat_can,
even though the input changed by only one cell. This script directly
measures whether that happens, and whether it's exactly explained by the
canonicalization choice (gidx, shift) changing.

Run: python analysis_perturbation.py
Outputs -> results/perturbation_maps_*.png, results/perturbation_report.txt
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "v1"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
from net import ToyCNNModel2, canonical_transform  # noqa: E402
from data import _BITMAPS, GRID  # noqa: E402

RESULTS = os.path.join(_HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

CHUNK = 200  # batch size for the 1600 single-cell-flip forward passes


def load_model(ckpt_name: str = "best.pt"):
    model = ToyCNNModel2(m=4)
    ckpt = torch.load(os.path.join(_HERE, "checkpoints", ckpt_name),
                       map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def place(name: str, top: int, left: int) -> torch.Tensor:
    bmp = _BITMAPS[name]
    g = np.zeros((GRID, GRID), dtype=np.float32)
    h, w = bmp.shape
    g[top:top + h, left:left + w] = bmp
    return torch.from_numpy(g).unsqueeze(0).unsqueeze(0)


def random_grid(density: float, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    g = (rng.random((GRID, GRID)) < density).astype(np.float32)
    return torch.from_numpy(g).unsqueeze(0).unsqueeze(0)


def all_single_flips(x0: torch.Tensor) -> torch.Tensor:
    """x0: (1,1,H,W). Returns (H*W,1,H,W): x0 with cell i flipped, for every
    i, in row-major order."""
    H, W = x0.shape[-2:]
    flips = x0.repeat(H * W, 1, 1, 1).clone()
    idx = torch.arange(H * W)
    flat = flips.view(H * W, -1)
    flat[idx, idx] = 1.0 - flat[idx, idx]
    return flips


@torch.no_grad()
def sensitivity_maps(model, x0: torch.Tensor):
    """Returns dict of (H,W) numpy maps: cos_sim, l2_dist, branch_changed
    (bool: did canonical_transform pick a different gidx for the perturbed
    grid?), plus feat_can0 / gidx0 / shift0 for reference."""
    H, W = x0.shape[-2:]
    _, aux0 = model(x0, return_features=True)
    fc0 = aux0["feat_can"].flatten()
    gidx0 = aux0["gidx"].item()

    perturbed = all_single_flips(x0)  # (H*W, 1, H, W)
    n = perturbed.shape[0]
    cos_sim = np.zeros(n)
    l2_dist = np.zeros(n)
    branch_changed = np.zeros(n, dtype=bool)

    for s in range(0, n, CHUNK):
        chunk = perturbed[s:s + CHUNK]
        _, aux = model(chunk, return_features=True)
        fc = aux["feat_can"].reshape(chunk.shape[0], -1)
        gidx = aux["gidx"]
        num = (fc * fc0.unsqueeze(0)).sum(dim=1)
        den = fc.norm(dim=1) * fc0.norm() + 1e-12
        cos_sim[s:s + chunk.shape[0]] = (num / den).numpy()
        l2_dist[s:s + chunk.shape[0]] = (fc - fc0.unsqueeze(0)).norm(dim=1).numpy()
        branch_changed[s:s + chunk.shape[0]] = (gidx != gidx0).numpy()

    return dict(
        cos_sim=cos_sim.reshape(H, W),
        l2_dist=l2_dist.reshape(H, W),
        branch_changed=branch_changed.reshape(H, W),
        gidx0=gidx0,
    )


def report_and_plot(name: str, x0: torch.Tensor, maps: dict, lines: list[str]):
    cos_sim, l2_dist, branch = maps["cos_sim"], maps["l2_dist"], maps["branch_changed"]
    alive = x0[0, 0].numpy() > 0.5

    n_branch = int(branch.sum())
    cos_on_branch_change = cos_sim[branch]
    cos_off_branch_change = cos_sim[~branch]

    lines.append(f"\n{name}  (alive cells at t=0: {int(alive.sum())})")
    lines.append(f"  cos_sim over all 1600 single-cell flips: "
                 f"mean={cos_sim.mean():.4f} min={cos_sim.min():.4f} max={cos_sim.max():.4f}")
    lines.append(f"  canonicalization branch (gidx) changed for {n_branch}/1600 flips")
    if n_branch:
        lines.append(f"    cos_sim WHEN branch changed:     mean={cos_on_branch_change.mean():.4f} "
                     f"min={cos_on_branch_change.min():.4f}")
    lines.append(f"    cos_sim when branch did NOT change: mean={cos_off_branch_change.mean():.4f} "
                 f"min={cos_off_branch_change.min():.4f}")
    lines.append(f"  cos_sim on alive cells (flip = kill):    mean={cos_sim[alive].mean():.4f}")
    lines.append(f"  cos_sim on dead cells (flip = birth):    mean={cos_sim[~alive].mean():.4f}")

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    axes[0].imshow(x0[0, 0].numpy(), cmap="Greys", vmin=0, vmax=1)
    axes[0].set_title("base grid (t=0)", fontsize=10)
    im1 = axes[1].imshow(1 - cos_sim, cmap="inferno", vmin=0, vmax=max(1e-6, (1 - cos_sim).max()))
    axes[1].set_title("1 - cos_sim per flipped cell\n(0=no repr. change)", fontsize=10)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    im2 = axes[2].imshow(l2_dist, cmap="inferno")
    axes[2].set_title("L2 distance per flipped cell", fontsize=10)
    fig.colorbar(im2, ax=axes[2], fraction=0.046)
    axes[3].imshow(branch.astype(float), cmap="Greys", vmin=0, vmax=1)
    axes[3].set_title(f"canonicalization branch changed\n({n_branch}/1600 cells)", fontsize=10)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Model 2 representation-perturbation sensitivity: {name}")
    fig.tight_layout()
    path = os.path.join(RESULTS, f"perturbation_maps_{name}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"{name}: mean cos_sim={cos_sim.mean():.4f}  branch changes={n_branch}/1600  -> {path}")


def main():
    model = load_model()
    lines = ["Perturbation sensitivity of feat_can (single-cell flips), Model 2",
             "=" * 68]

    stimuli = [
        ("glider", place("glider", 15, 15)),
        ("pulsar", place("pulsar", 13, 13)),
        ("block", place("block", 18, 18)),
        ("random_d0.1", random_grid(0.1, 1)),
        ("random_d0.3", random_grid(0.3, 1)),
        ("random_d0.5", random_grid(0.5, 1)),
    ]
    for name, x0 in stimuli:
        maps = sensitivity_maps(model, x0)
        report_and_plot(name, x0, maps, lines)

    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(RESULTS, "perturbation_report.txt"), "w") as f:
        f.write(report + "\n")
    print(f"\n-> {RESULTS}/perturbation_report.txt")


if __name__ == "__main__":
    main()
