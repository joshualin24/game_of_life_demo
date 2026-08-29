"""
Compare V8 and V10 (both stable, well-converged CNNTransformerV4 models,
differing only in size/training regime) on how much their post-Transformer
patch embeddings change under specific geometric transforms, broken down by
individual case: two base patterns (glider, random) each under four
translation directions, three rotation angles, and two reflection axes.

For each (pattern, transform) case we extract post-Transformer patch
embeddings for the original and transformed grid, and measure mean per-patch
cosine similarity. Higher similarity = the embedding changed less under the
transform, i.e. is closer to invariant/equivariant under it.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from nn.embedding_analysis import EmbeddingExtractor, Transforms, compare, load_v8, load_v10
from nn.utils import RESULTS_DIR


def make_glider(r0=10, c0=10):
    g = np.zeros((40, 40), dtype=np.uint8)
    pat = np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.uint8)
    g[r0:r0 + 3, c0:c0 + 3] = pat
    return g


def make_random(density=0.5, seed=42):
    rng = np.random.default_rng(seed)
    return (rng.random((40, 40)) < density).astype(np.uint8)


PATTERNS = {
    "Glider": make_glider(),
    "Random": make_random(),
}

CASES = [
    ("Left 4",   lambda g: Transforms.translate(g, dr=0,  dc=-4)),
    ("Right 4",  lambda g: Transforms.translate(g, dr=0,  dc=4)),
    ("Up 4",     lambda g: Transforms.translate(g, dr=-4, dc=0)),
    ("Down 4",   lambda g: Transforms.translate(g, dr=4,  dc=0)),
    ("Rot 90",   lambda g: Transforms.rotate(g, k=1)),
    ("Rot 180",  lambda g: Transforms.rotate(g, k=2)),
    ("Rot 270",  lambda g: Transforms.rotate(g, k=3)),
    ("Flip H",   Transforms.flip_h),
    ("Flip V",   Transforms.flip_v),
]


def main():
    print("Loading V8 and V10 ...")
    extractors = {"V8": EmbeddingExtractor(load_v8()), "V10": EmbeddingExtractor(load_v10())}

    # results[pattern][case][model] = mean_cos
    results = {p: {c: {} for c, _ in CASES} for p in PATTERNS}

    print(f"\n{'='*80}")
    print("  Post-Transformer embedding cosine similarity, per (pattern, transform) case")
    print(f"{'='*80}")
    header = f"  {'Pattern':<8} {'Case':<10} {'V8':>8} {'V10':>8}"
    print(header)
    print("  " + "-"*40)
    for pattern_name, grid in PATTERNS.items():
        for case_name, fn in CASES:
            grid_tf = fn(grid)
            for model_name, ext in extractors.items():
                _, post1 = ext(grid)
                _, post2 = ext(grid_tf)
                sim = compare(post1, post2)["mean_cos"]
                results[pattern_name][case_name][model_name] = sim
            v8 = results[pattern_name][case_name]["V8"]
            v10 = results[pattern_name][case_name]["V10"]
            print(f"  {pattern_name:<8} {case_name:<10} {v8:>8.4f} {v10:>8.4f}")

    # ── Grouped bar chart: two panels (Glider, Random), 9 cases each ────────
    case_names = [c for c, _ in CASES]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    x = np.arange(len(case_names))
    width = 0.38

    for ax, pattern_name in zip(axes, PATTERNS.keys()):
        v8_vals  = [results[pattern_name][c]["V8"]  for c in case_names]
        v10_vals = [results[pattern_name][c]["V10"] for c in case_names]
        ax.bar(x - width/2, v8_vals,  width, label="V8 (291K params)",  color="#4C72B0")
        ax.bar(x + width/2, v10_vals, width, label="V10 (1.5M params)", color="#C44E52")
        ax.set_xticks(x)
        ax.set_xticklabels(case_names, rotation=40, ha="right", fontsize=8)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_title(pattern_name, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0.0, 1.05)

    axes[0].set_ylabel("Post-transformer cosine similarity\n(original vs. transformed)")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Embedding sensitivity to specific transforms: V8 vs. V10\n"
                 "Higher = embedding changed less under the transform (more invariant)",
                 fontweight="bold")
    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "v8_v10_transform_equivariance_detailed.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
