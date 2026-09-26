"""Does a transformed input produce the same last-layer representation?

"Last layer" = feat_can, the canonicalized (m,H,W) feature map right before
the final 1x1 head -- the analog of V8/V10's tf_L4 in ../../notes.md. Tested
on two kinds of stimuli, per the actual question asked:

  - well-known patterns (glider, lwss, pulsar, block, beehive, blinker,
    beacon, toad, pentadecathlon)
  - random-density distributions (several densities, several seeds each)

For each base grid x and each of the 8 D4 elements g, compares vec(x) vs
vec(g.x) two ways:

  - mean-pooled:  (m,)      global average over the 40x40 spatial map
  - flattened:    (m*40*40,) every value, position included

Mean-pooling is included ONLY as a cautionary baseline -- ../../notes.md's
own "Caveat on these first numbers" flags that a global average is nearly
invariant to ANY spatial permutation (not just D4) once you're comparing a
whole-grid map to itself, so it can look "invariant" even for a
representation that has moved. Flattened is the real test: it only matches
if the map is IDENTICAL, position by position.

Contrasts `feat` (pre-canonicalization, raw backbone output) against
`feat_can` (post-canonicalization) throughout, since the interesting claim
is specifically that canonicalization is what buys the invariance --
`feat` alone should NOT be flatten-invariant (it moves with the input, even
though the backbone is exactly equivariant), while `feat_can` should be.

Run: python analysis_representation.py
Outputs -> results/representation_report.txt
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "v1"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
from net import ToyCNNModel1  # noqa: E402
from data import _BITMAPS, GRID, MARGIN  # noqa: E402
from adapter import D4_NAMES, N_D4, torch_d4  # noqa: E402

RESULTS = os.path.join(_HERE, "results")
os.makedirs(RESULTS, exist_ok=True)


def load_model(ckpt_name: str = "best_v2.pt"):
    model = ToyCNNModel1(m=4)
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


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten(), b.flatten()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


@torch.no_grad()
def analyze_one(model, x: torch.Tensor, label: str, lines: list[str]):
    _, aux0 = model(x, return_features=True)
    feat0, featcan0 = aux0["feat"], aux0["feat_can"]
    pool0, poolcan0 = feat0.mean(dim=(2, 3)), featcan0.mean(dim=(2, 3))

    lines.append(f"\n{label}")
    lines.append(f"{'op':<9} {'feat pool cos':>14} {'feat flat cos':>14}  |  "
                  f"{'feat_can pool cos':>18} {'feat_can flat cos':>18}")
    for g in range(N_D4):
        xg = torch_d4(x, g)
        _, auxg = model(xg, return_features=True)
        featg, featcang = auxg["feat"], auxg["feat_can"]
        poolg, poolcang = featg.mean(dim=(2, 3)), featcang.mean(dim=(2, 3))
        lines.append(
            f"{D4_NAMES[g]:<9} {cos(pool0, poolg):>14.4f} {cos(feat0, featg):>14.4f}  |  "
            f"{cos(poolcan0, poolcang):>18.4f} {cos(featcan0, featcang):>18.4f}"
        )


@torch.no_grad()
def discriminability_check(model, grids: list[torch.Tensor], lines: list[str]):
    """Sanity: unrelated grids should NOT collapse to the same feat_can --
    invariance under transformation is only meaningful if different content
    still gives different vectors."""
    feats_can_flat, feats_can_pool = [], []
    for x in grids:
        _, aux = model(x, return_features=True)
        feats_can_flat.append(aux["feat_can"].flatten())
        feats_can_pool.append(aux["feat_can"].mean(dim=(2, 3)).flatten())
    n = len(grids)
    flat_sims = [cos(feats_can_flat[i], feats_can_flat[j])
                 for i in range(n) for j in range(i + 1, n)]
    pool_sims = [cos(feats_can_pool[i], feats_can_pool[j])
                 for i in range(n) for j in range(i + 1, n)]
    lines.append(f"\nDiscriminability across {n} UNRELATED grids (feat_can, same-orientation each):")
    lines.append(f"  flattened cos: mean={np.mean(flat_sims):.4f}  min={min(flat_sims):.4f}  max={max(flat_sims):.4f}")
    lines.append(f"  pooled    cos: mean={np.mean(pool_sims):.4f}  min={min(pool_sims):.4f}  max={max(pool_sims):.4f}")


def main():
    model = load_model()
    lines = ["Last-layer (feat_can) invariance under D4 transforms, Model 1 v2",
             "=" * 65,
             "cos(vec(x), vec(g.x)) for each of the 8 D4 elements g.",
             "1.0000 = exactly the same vector. feat = pre-canonicalization,",
             "feat_can = post-canonicalization (the actual 'last layer' claim)."]

    lines.append("\n" + "#" * 65)
    lines.append("# Well-known patterns")
    lines.append("#" * 65)
    for name, (top, left) in [
        ("glider", (15, 15)), ("lwss", (15, 10)), ("pulsar", (13, 13)),
        ("block", (18, 18)), ("beehive", (18, 18)), ("blinker", (18, 18)),
        ("beacon", (15, 15)), ("toad", (18, 15)), ("pentadecathlon", (15, 12)),
    ]:
        analyze_one(model, place(name, top, left), f"pattern: {name}", lines)

    lines.append("\n" + "#" * 65)
    lines.append("# Random-density distributions")
    lines.append("#" * 65)
    for density in (0.1, 0.3, 0.5, 0.8):
        for seed in (1, 2):
            analyze_one(model, random_grid(density, seed),
                        f"random density={density:.1f} seed={seed}", lines)

    unrelated = [place("glider", 15, 15), place("pulsar", 13, 13),
                 random_grid(0.3, 10), random_grid(0.5, 20), place("lwss", 4, 4)]
    discriminability_check(model, unrelated, lines)

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(RESULTS, "representation_report.txt"), "w") as f:
        f.write(report + "\n")
    print(f"\n-> {RESULTS}/representation_report.txt")


if __name__ == "__main__":
    main()
