"""
Compare V8 and V10 (both stable, well-converged CNNTransformerV4 models,
differing only in size/training regime) on how much their post-Transformer
patch embeddings change under geometric transforms: rotation, reflection,
and translation.

For each of three base configurations (glider, random grid at density 0.35,
random grid at density 0.65) we apply each transform, extract post-Transformer
patch embeddings for the original and transformed grid, and measure mean
per-patch cosine similarity. Higher similarity = the embedding changed less
under the transform, i.e. is closer to invariant/equivariant under it.
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


def make_random(density, seed=42):
    rng = np.random.default_rng(seed)
    return (rng.random((40, 40)) < density).astype(np.uint8)


GRIDS = {
    "glider":     make_glider(),
    "random_d35": make_random(0.35),
    "random_d65": make_random(0.65),
}

TRANSFORM_GROUPS = {
    "Rotation\n(90/180/270deg)": [
        ("rot90",  lambda g: Transforms.rotate(g, k=1)),
        ("rot180", lambda g: Transforms.rotate(g, k=2)),
        ("rot270", lambda g: Transforms.rotate(g, k=3)),
    ],
    "Reflection\n(H/V)": [
        ("flip_h", Transforms.flip_h),
        ("flip_v", Transforms.flip_v),
    ],
    "Translation\n(1-4 cells)": [
        ("right1", lambda g: Transforms.translate(g, dr=0, dc=1)),
        ("right4", lambda g: Transforms.translate(g, dr=0, dc=4)),
        ("down4",  lambda g: Transforms.translate(g, dr=4, dc=0)),
        ("diag4",  lambda g: Transforms.translate(g, dr=4, dc=4)),
    ],
}


def group_similarity(extractor, group_transforms):
    vals = []
    for cfg_name, grid in GRIDS.items():
        _, post1 = extractor(grid)
        for _, fn in group_transforms:
            grid_tf = fn(grid)
            _, post2 = extractor(grid_tf)
            vals.append(compare(post1, post2)["mean_cos"])
    return float(np.mean(vals)), float(np.std(vals))


def main():
    print("Loading V8 and V10 ...")
    extractors = {"V8": EmbeddingExtractor(load_v8()), "V10": EmbeddingExtractor(load_v10())}

    results = {name: {} for name in extractors}
    print(f"\n{'='*80}")
    print("  Post-Transformer embedding cosine similarity under transform (V8 vs V10)")
    print(f"{'='*80}")
    header = f"  {'Transform group':<26} {'V8 mean':>10} {'V8 std':>8}   {'V10 mean':>10} {'V10 std':>8}"
    print(header)
    print("  " + "-"*76)
    for group_name, transforms in TRANSFORM_GROUPS.items():
        row = group_name.replace('\n', ' ')
        for model_name, ext in extractors.items():
            mean, std = group_similarity(ext, transforms)
            results[model_name][group_name] = (mean, std)
        m8, s8 = results["V8"][group_name]
        m10, s10 = results["V10"][group_name]
        print(f"  {row:<26} {m8:>10.4f} {s8:>8.4f}   {m10:>10.4f} {s10:>8.4f}")

    # ── Grouped bar chart ───────────────────────────────────────────────────
    group_names = list(TRANSFORM_GROUPS.keys())
    v8_vals  = [results["V8"][g][0] for g in group_names]
    v10_vals = [results["V10"][g][0] for g in group_names]
    v8_err   = [results["V8"][g][1] for g in group_names]
    v10_err  = [results["V10"][g][1] for g in group_names]

    x = np.arange(len(group_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, v8_vals, width, yerr=v8_err, capsize=3, label="V8 (291K params)", color="#4C72B0")
    ax.bar(x + width/2, v10_vals, width, yerr=v10_err, capsize=3, label="V10 (1.5M params)", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(group_names)
    ax.set_ylabel("Post-transformer cosine similarity\n(original vs. transformed)")
    ax.set_ylim(0.0, 1.05)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend()
    ax.set_title("Embedding sensitivity to geometric transforms: V8 vs. V10\n"
                 "Higher = embedding changed less under the transform (more invariant)",
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "v8_v10_transform_equivariance.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
