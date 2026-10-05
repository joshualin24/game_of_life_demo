"""Does a transformed input produce the same last-layer representation?
Model 5v3 version, analogous to ../model2/analysis_representation.py, but
using the model's OWN learned pose end to end (no teacher, no hand-coded
canonicalization) -- this is the most direct test of "did the model learn
Model 4's representation" the user asked for: not just task accuracy
(which ../model5/notes.md shows is structurally uninformative -- see
"headline finding" there), but whether semantically-equivalent inputs
(the same content, D4-transformed or translated) actually produce the
SAME feat_can, the way Model 4's hand-coded canonicalization guarantees
exactly and this model only approximates.

"Last layer" = feat_can, the (m,H,W) feature map right before the final
1x1 head, canonicalized by the model's OWN predicted (shift, gidx).

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
from net import ToyCNNModel5v3  # noqa: E402
from data import _BITMAPS, GRID  # noqa: E402
from adapter import D4_NAMES, N_D4, torch_d4, torch_translate  # noqa: E402

RESULTS = os.path.join(_HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

TRANSLATIONS = [(0, 0), (1, 0), (0, 1), (3, -2), (-5, 7), (10, 15), (-12, -18)]


def load_model(ckpt_name: str = "best.pt"):
    model = ToyCNNModel5v3(m=4)
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
    _, aux0 = model(x, return_aux=True)
    feat0, featcan0 = aux0["feat"], aux0["feat_can"]
    pool0, poolcan0 = feat0.mean(dim=(2, 3)), featcan0.mean(dim=(2, 3))

    lines.append(f"\n{label}  -- D4")
    lines.append(f"{'op':<9} {'feat pool cos':>14} {'feat flat cos':>14}  |  "
                  f"{'feat_can pool cos':>18} {'feat_can flat cos':>18}  {'gidx_used':>10}")
    for g in range(N_D4):
        xg = torch_d4(x, g)
        _, auxg = model(xg, return_aux=True)
        featg, featcang = auxg["feat"], auxg["feat_can"]
        poolg, poolcang = featg.mean(dim=(2, 3)), featcang.mean(dim=(2, 3))
        lines.append(
            f"{D4_NAMES[g]:<9} {cos(pool0, poolg):>14.4f} {cos(feat0, featg):>14.4f}  |  "
            f"{cos(poolcan0, poolcang):>18.4f} {cos(featcan0, featcang):>18.4f}  "
            f"{D4_NAMES[auxg['gidx_used'].item()]:>10}"
        )

    lines.append(f"{label}  -- translation (model's own predicted shift, not hand-coded)")
    lines.append(f"{'shift':<9} {'feat pool cos':>14} {'feat flat cos':>14}  |  "
                  f"{'feat_can pool cos':>18} {'feat_can flat cos':>18}  {'shift_used':>12}")
    for dr, dc in TRANSLATIONS:
        xt = torch_translate(x, dr, dc)
        _, auxt = model(xt, return_aux=True)
        featt, featcant = auxt["feat"], auxt["feat_can"]
        poolt, poolcant = featt.mean(dim=(2, 3)), featcant.mean(dim=(2, 3))
        su = auxt["shift_used"][0].tolist()
        lines.append(
            f"{f'({dr},{dc})':<9} {cos(pool0, poolt):>14.4f} {cos(feat0, featt):>14.4f}  |  "
            f"{cos(poolcan0, poolcant):>18.4f} {cos(featcan0, featcant):>18.4f}  {str(su):>12}"
        )


@torch.no_grad()
def discriminability_check(model, grids: list[torch.Tensor], lines: list[str]):
    feats_can_flat, feats_can_pool = [], []
    for x in grids:
        _, aux = model(x, return_aux=True)
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
    lines = ["Last-layer (feat_can) invariance under D4 AND translation, Model 5v3",
             "=" * 70,
             "cos(vec(x), vec(g.x)) / cos(vec(x), vec(shift(x,d))), using the",
             "model's OWN predicted pose end to end (no teacher, no hand-coded",
             "canonicalization -- unlike Models 1-4, there is no guarantee this",
             "is 1.0000; that's exactly what this script measures).",
             "feat = pre-canonicalization, feat_can = post-canonicalization."]

    lines.append("\n" + "#" * 70)
    lines.append("# Well-known patterns (including the ones broken in Model 1)")
    lines.append("#" * 70)
    for name, (top, left) in [
        ("glider", (15, 15)), ("lwss", (15, 10)), ("pulsar", (13, 13)),
        ("block", (18, 18)), ("beehive", (18, 18)), ("blinker", (18, 18)),
        ("beacon", (15, 15)), ("toad", (18, 15)), ("pentadecathlon", (15, 12)),
    ]:
        analyze_one(model, place(name, top, left), f"pattern: {name}", lines)

    lines.append("\n" + "#" * 70)
    lines.append("# Random-density distributions")
    lines.append("#" * 70)
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
