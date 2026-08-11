"""
Compare translation-sensitivity of patch embeddings between:
  V2 — fixed 2D sinusoidal positional encoding (no learned position params)
  V8 — learned absolute positional embedding (CNNTransformerV4 architecture)

Motivation: sinusoidal positional encodings have a classical closed-form
property (via the angle-addition formulas) that shifting position by a
fixed offset corresponds to a FIXED linear operator applied uniformly in
each frequency pair -- i.e. embedding(shift(x)) ~ embedding(x) + (a
consistent transformation), closer to a true equivariant structure. A
learned absolute positional embedding has no such guarantee -- each patch
slot's position vector is independently learned with no relationship to
its neighbors.

This script measures, for both models, mean cosine similarity between
original and translated patch embeddings (pre- and post-transformer),
across several translation offsets and base configurations, to test
whether V2's sinusoidal PE produces embeddings that are empirically closer
to translation-equivariant than V8's learned PE.

CAVEAT: V2's CNN stage uses zero-padding (not circular), unlike V8's
circular padding which exactly respects the toroidal topology. This means
V2 has its own boundary-artifact confound (documented in V4's docstring in
models.py) independent of the positional-encoding difference under study.
Results near grid boundaries should be interpreted with this in mind.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from nn.models import CNNTransformerV2
from nn.embedding_analysis import (
    EmbeddingExtractor, Transforms, compare, load_v8,
    GRID_SIZE, N_PATCHES_1D,
)
from nn.utils import CKPT_DIR, DEVICE, RESULTS_DIR

N_PATCHES = N_PATCHES_1D ** 2


def load_v2():
    m = CNNTransformerV2(grid_size=GRID_SIZE, patch_size=4,
                         d_model=64, nhead=4, num_layers=4).to(DEVICE)
    m.load_state_dict(torch.load(
        os.path.join(CKPT_DIR, "task10_cnn_transformer_2d_best.pt"),
        map_location=DEVICE, weights_only=True,
    ))
    m.eval()
    return m


class V2EmbeddingExtractor:
    """Same interface as EmbeddingExtractor, but for CNNTransformerV2's
    avg-pool + fixed sinusoidal PE tokenization."""

    def __init__(self, model: CNNTransformerV2):
        self.model = model
        self.model.eval()

    def __call__(self, grid: np.ndarray):
        x = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            m = self.model
            feat = m.cnn(x)
            tokens = F.avg_pool2d(feat, kernel_size=m.patch_size, stride=m.patch_size)
            tokens = tokens.flatten(2).transpose(1, 2)
            pre = m.pos_enc(tokens)
            post = m.transformer(pre)
        return pre.squeeze(0).cpu().numpy(), post.squeeze(0).cpu().numpy()


# ── Test configurations (same as prior embedding analysis) ────────────────────

def make_glider(r0=10, c0=10):
    g = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    pat = np.array([[0,1,0],[0,0,1],[1,1,1]], dtype=np.uint8)
    g[r0:r0+3, c0:c0+3] = pat
    return g

def make_random(density, seed=42):
    rng = np.random.default_rng(seed)
    return (rng.random((GRID_SIZE, GRID_SIZE)) < density).astype(np.uint8)


def run_translation_sweep(extractor, name, grids):
    """Returns dict: {config_name: {shift_label: (pre_cos, post_cos)}}"""
    shifts = [(0, 1, "right1"), (0, 4, "right4"), (4, 0, "down4"), (4, 4, "diag4")]
    results = {}
    for cfg_name, grid in grids.items():
        results[cfg_name] = {}
        pre1, post1 = extractor(grid)
        for dr, dc, label in shifts:
            grid_tf = Transforms.translate(grid, dr=dr, dc=dc)
            pre2, post2 = extractor(grid_tf)
            pre_s = compare(pre1, pre2)
            post_s = compare(post1, post2)
            results[cfg_name][label] = (pre_s["mean_cos"], post_s["mean_cos"])
    return results


def main():
    print("Loading V2 (sinusoidal PE) and V8 (learned PE) ...")
    v2 = load_v2()
    v8 = load_v8()
    ext_v2 = V2EmbeddingExtractor(v2)
    ext_v8 = EmbeddingExtractor(v8)

    grids = {
        "glider":      make_glider(),
        "random_d35":  make_random(0.35),
        "random_d65":  make_random(0.65),
    }

    print("Running translation sweep ...")
    res_v2 = run_translation_sweep(ext_v2, "V2", grids)
    res_v8 = run_translation_sweep(ext_v8, "V8", grids)

    shifts = ["right1", "right4", "down4", "diag4"]
    print(f"\n{'='*100}")
    print("  Translation sensitivity: mean cosine similarity (pre-transformer / post-transformer)")
    print(f"{'='*100}")
    header = f"  {'Config':<14} {'Shift':<8} {'V2 pre':>8} {'V2 post':>8}   {'V8 pre':>8} {'V8 post':>8}   {'Δpost (V2-V8)':>14}"
    print(header)
    print("  " + "-"*96)
    for cfg_name in grids:
        for shift in shifts:
            v2_pre, v2_post = res_v2[cfg_name][shift]
            v8_pre, v8_post = res_v8[cfg_name][shift]
            delta = v2_post - v8_post
            print(f"  {cfg_name:<14} {shift:<8} {v2_pre:>8.4f} {v2_post:>8.4f}   "
                  f"{v8_pre:>8.4f} {v8_post:>8.4f}   {delta:>+14.4f}")

    # Aggregate summary
    print(f"\n  {'-'*96}")
    all_v2_post = [res_v2[c][s][1] for c in grids for s in shifts]
    all_v8_post = [res_v8[c][s][1] for c in grids for s in shifts]
    print(f"  Mean post-transformer cosine similarity across all configs/shifts:")
    print(f"    V2 (sinusoidal PE): {np.mean(all_v2_post):.4f}")
    print(f"    V8 (learned PE):    {np.mean(all_v8_post):.4f}")
    print()

    # ── Grouped bar chart ───────────────────────────────────────────────────
    labels = [f"{c}\n{s}" for c in grids for s in shifts]
    v2_vals = all_v2_post
    v8_vals = all_v8_post
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width/2, v2_vals, width, label="V2 (sinusoidal PE)", color="#4C72B0")
    ax.bar(x + width/2, v8_vals, width, label="V8 (learned PE)",   color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Post-transformer cosine similarity\n(original vs. translated)")
    ax.set_ylim(0.6, 1.02)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend()
    ax.set_title("Translation sensitivity: sinusoidal PE (V2) vs. learned PE (V8)\n"
                 "Higher = embedding changed less under translation (more equivariant)",
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "compare_v2_v8_translation_equivariance.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
