"""Does a transformed input produce the same last-layer representation?
Model 2 version -- extends ../model1/analysis_representation.py with a
TRANSLATION test, since that's the whole point of Model 2 (Model 1 never
canonicalized position, so its feat_can was never invariant to translation
at all -- only to D4, and only for asymmetric patterns).

"Last layer" = feat_can, the jointly D4+translation-canonicalized (m,H,W)
feature map right before the final 1x1 head.

Tested on well-known patterns and random-density distributions, comparing
vec(x) vs vec(g.x) / vec(shift(x,d)) both mean-pooled and flattened (see
../model1/analysis_representation.py's docstring for why mean-pooling alone
is a misleading baseline -- included only as a cautionary contrast).

Run: python analysis_representation.py
Outputs -> results/representation_report.txt
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "v1"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
from net import ToyCNNModel2  # noqa: E402
from data import _BITMAPS, GRID  # noqa: E402
from adapter import D4_NAMES, N_D4, torch_d4, torch_translate  # noqa: E402

RESULTS = os.path.join(_HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

TRANSLATIONS = [(0, 0), (1, 0), (0, 1), (3, -2), (-5, 7), (10, 15), (-12, -18)]


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


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten(), b.flatten()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


@torch.no_grad()
def analyze_one(model, x: torch.Tensor, label: str, lines: list[str]):
    _, aux0 = model(x, return_features=True)
    feat0, featcan0 = aux0["feat"], aux0["feat_can"]
    pool0, poolcan0 = feat0.mean(dim=(2, 3)), featcan0.mean(dim=(2, 3))

    lines.append(f"\n{label}  -- D4")
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

    lines.append(f"{label}  -- translation (NEW in Model 2; Model 1 never canonicalized this)")
    lines.append(f"{'shift':<9} {'feat pool cos':>14} {'feat flat cos':>14}  |  "
                  f"{'feat_can pool cos':>18} {'feat_can flat cos':>18}")
    for dr, dc in TRANSLATIONS:
        xt = torch_translate(x, dr, dc)
        _, auxt = model(xt, return_features=True)
        featt, featcant = auxt["feat"], auxt["feat_can"]
        poolt, poolcant = featt.mean(dim=(2, 3)), featcant.mean(dim=(2, 3))
        lines.append(
            f"{f'({dr},{dc})':<9} {cos(pool0, poolt):>14.4f} {cos(feat0, featt):>14.4f}  |  "
            f"{cos(poolcan0, poolcant):>18.4f} {cos(featcan0, featcant):>18.4f}"
        )


@torch.no_grad()
def discriminability_check(model, grids: list[torch.Tensor], lines: list[str]):
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
    lines.append(f"\nDiscriminability across {n} UNRELATED grids (feat_can, same-placement each):")
    lines.append(f"  flattened cos: mean={np.mean(flat_sims):.4f}  min={min(flat_sims):.4f}  max={max(flat_sims):.4f}")
    lines.append(f"  pooled    cos: mean={np.mean(pool_sims):.4f}  min={min(pool_sims):.4f}  max={max(pool_sims):.4f}")


def main():
    model = load_model()
    lines = ["Last-layer (feat_can) invariance under D4 AND translation, Model 2",
             "=" * 68,
             "cos(vec(x), vec(g.x)) / cos(vec(x), vec(shift(x,d))).",
             "1.0000 = exactly the same vector. feat = pre-canonicalization,",
             "feat_can = post-canonicalization (the actual 'last layer' claim).",
             "Unlike Model 1, translation is now explicitly canonicalized too."]

    lines.append("\n" + "#" * 68)
    lines.append("# Well-known patterns (including the ones broken in Model 1)")
    lines.append("#" * 68)
    for name, (top, left) in [
        ("glider", (15, 15)), ("lwss", (15, 10)), ("pulsar", (13, 13)),
        ("block", (18, 18)), ("beehive", (18, 18)), ("blinker", (18, 18)),
        ("beacon", (15, 15)), ("toad", (18, 15)), ("pentadecathlon", (15, 12)),
    ]:
        analyze_one(model, place(name, top, left), f"pattern: {name}", lines)

    lines.append("\n" + "#" * 68)
    lines.append("# Random-density distributions")
    lines.append("#" * 68)
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
