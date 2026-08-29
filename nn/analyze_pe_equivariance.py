"""
Characterize translation-sensitivity of the learned positional embedding in
our stable, well-trained CNN-Transformer models (V8, V10) -- WITHOUT using V2
as a comparison point.

Background: an earlier version of this analysis compared V8 (learned PE)
against V2 (fixed sinusoidal PE) to test whether sinusoidal encodings are
more translation-equivariant. That comparison was dropped: V2's "best"
checkpoint is from epoch 1 of an unstable run that never improved afterward
(see README Task 10), and a follow-up discriminability check
(check_v2_v8_discriminability.py) showed V2 could barely distinguish two
different random grids of different density (cosine sim 0.958) -- almost as
similar as it rated the SAME grid before/after translation (0.997). That
means V2's apparent "equivariance" advantage, on exactly the configurations
where it was largest (dense/random grids), is largely explained by V2 being
undertrained and undiscriminative, not by sinusoidal PE's structure.

This script instead asks a narrower, defensible question using only stable
models: how much does a learned positional embedding actually change patch
embeddings under translation, relative to the embedding's OWN baseline
sensitivity to unrelated content? For each model we report:
  - translation similarity : cosine sim between a grid's embedding and a
                              translated copy's embedding (same content, shifted)
  - content baseline       : cosine sim between embeddings of DIFFERENT,
                              untranslated grids (glider / random_d35 / random_d65)
A model with a real (partial) equivariance signal should show translation
similarity clearly above its own content baseline, without the content
baseline itself being suspiciously close to 1.0 (which would indicate
collapse rather than discrimination).
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
SHIFTS = [(0, 1, "right1"), (0, 4, "right4"), (4, 0, "down4"), (4, 4, "diag4")]


def translation_similarity(extractor):
    vals = []
    for name, grid in GRIDS.items():
        _, post1 = extractor(grid)
        for dr, dc, _ in SHIFTS:
            grid_tf = Transforms.translate(grid, dr=dr, dc=dc)
            _, post2 = extractor(grid_tf)
            vals.append(compare(post1, post2)["mean_cos"])
    return float(np.mean(vals))


def content_baseline(extractor):
    names = list(GRIDS.keys())
    embs = {n: extractor(g)[1] for n, g in GRIDS.items()}
    vals = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            vals.append(compare(embs[a], embs[b])["mean_cos"])
    return float(np.mean(vals))


def main():
    print("Loading V8 and V10 ...")
    models = {"V8": load_v8(), "V10": load_v10()}

    results = {}
    for name, model in models.items():
        ext = EmbeddingExtractor(model)
        trans = translation_similarity(ext)
        base = content_baseline(ext)
        results[name] = dict(translation=trans, baseline=base, gap=trans - base)
        print(f"  {name}: translation-sim={trans:.4f}  content-baseline={base:.4f}  gap={trans-base:+.4f}")

    # ── Bar chart ───────────────────────────────────────────────────────────
    names = list(results.keys())
    trans_vals = [results[n]["translation"] for n in names]
    base_vals  = [results[n]["baseline"] for n in names]
    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(x - width/2, trans_vals, width, label="Translation similarity\n(same grid, shifted)", color="#4C72B0")
    ax.bar(x + width/2, base_vals,  width, label="Content baseline\n(different, untranslated grids)", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Post-transformer cosine similarity")
    ax.set_ylim(0.6, 1.02)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Learned positional embedding: translation similarity\nvs. its own content-discrimination baseline",
                 fontweight="bold", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    for i, n in enumerate(names):
        ax.annotate(f"gap={results[n]['gap']:+.3f}", xy=(i, max(trans_vals[i], base_vals[i]) + 0.02),
                    ha="center", fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "pe_equivariance_v8_v10.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
