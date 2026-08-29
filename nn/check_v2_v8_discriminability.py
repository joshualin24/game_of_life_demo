"""
Sanity check for the V2-vs-V8 translation-equivariance comparison
(compare_v2_v8_equivariance.py).

Motivation: V2's "best" checkpoint is from epoch 1 of an unstable training
run (val loss oscillated for the rest of training and never beat the
epoch-1 checkpoint -- see README Task 10). A barely-trained model's
embeddings could look "translation-invariant" not because sinusoidal PE is
doing anything useful, but because the embeddings are collapsed / close to
initialization and don't encode much content-dependent structure at all --
a trivial invariance-by-collapse, not genuine equivariance.

This script distinguishes the two explanations by measuring cosine
similarity between embeddings of DIFFERENT, UNTRANSLATED grids (glider vs.
random_d35 vs. random_d65) for both V2 and V8. If a model is discriminative,
different content should produce dissimilar embeddings (low cosine sim). If
V2 shows high similarity here too, its high translation-similarity score is
not special -- everything looks similar to it, translated or not.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from nn.embedding_analysis import EmbeddingExtractor, compare, load_v8
from nn.compare_v2_v8_equivariance import load_v2, V2EmbeddingExtractor, make_glider, make_random


def main():
    print("Loading V2 (sinusoidal PE) and V8 (learned PE) ...")
    v2 = load_v2()
    v8 = load_v8()
    ext_v2 = V2EmbeddingExtractor(v2)
    ext_v8 = EmbeddingExtractor(v8)

    grids = {
        "glider":     make_glider(),
        "random_d35": make_random(0.35),
        "random_d65": make_random(0.65),
    }
    names = list(grids.keys())

    # Precompute embeddings once per grid.
    emb_v2 = {n: ext_v2(g) for n, g in grids.items()}
    emb_v8 = {n: ext_v8(g) for n, g in grids.items()}

    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]

    print(f"\n{'='*90}")
    print("  Cross-configuration similarity (DIFFERENT untranslated grids) -- discriminability check")
    print(f"{'='*90}")
    header = f"  {'Pair':<28} {'V2 pre':>8} {'V2 post':>8}   {'V8 pre':>8} {'V8 post':>8}"
    print(header)
    print("  " + "-"*86)

    v2_post_vals, v8_post_vals = [], []
    for a, b in pairs:
        pre_v2, post_v2 = compare(emb_v2[a][0], emb_v2[b][0])["mean_cos"], compare(emb_v2[a][1], emb_v2[b][1])["mean_cos"]
        pre_v8, post_v8 = compare(emb_v8[a][0], emb_v8[b][0])["mean_cos"], compare(emb_v8[a][1], emb_v8[b][1])["mean_cos"]
        v2_post_vals.append(post_v2)
        v8_post_vals.append(post_v8)
        print(f"  {a+' vs '+b:<28} {pre_v2:>8.4f} {post_v2:>8.4f}   {pre_v8:>8.4f} {post_v8:>8.4f}")

    print("  " + "-"*86)
    print(f"  Mean post-transformer cosine similarity across DIFFERENT grids:")
    print(f"    V2 (sinusoidal PE): {np.mean(v2_post_vals):.4f}")
    print(f"    V8 (learned PE):    {np.mean(v8_post_vals):.4f}")

    print(f"\n{'='*90}")
    print("  Recap: post-transformer cosine similarity under TRANSLATION (same grid, shifted)")
    print("  (from compare_v2_v8_equivariance.py -- reported previously)")
    print(f"{'='*90}")
    print(f"    V2 (sinusoidal PE): 0.9968")
    print(f"    V8 (learned PE):    0.8901")

    print(f"\n{'='*90}")
    print("  Interpretation")
    print(f"{'='*90}")
    v2_gap = 0.9968 - np.mean(v2_post_vals)
    v8_gap = 0.8901 - np.mean(v8_post_vals)
    print(f"  V2: translation-sim ({0.9968:.4f}) - different-content-sim ({np.mean(v2_post_vals):.4f}) = {v2_gap:+.4f}")
    print(f"  V8: translation-sim ({0.8901:.4f}) - different-content-sim ({np.mean(v8_post_vals):.4f}) = {v8_gap:+.4f}")
    if np.mean(v2_post_vals) > 0.95:
        print("  --> V2 shows high similarity even for UNRELATED content: embeddings look collapsed/")
        print("      undiscriminative. The high translation-similarity score is NOT specific to")
        print("      translation and does not support a genuine equivariance claim.")
    elif v2_gap > 2 * v8_gap:
        print("  --> V2 is discriminative between different content AND selectively stable under")
        print("      translation specifically -- consistent with genuine (near-)equivariant structure.")
    else:
        print("  --> Ambiguous: V2's gap between translation-similarity and content-discrimination is")
        print("      not clearly larger than V8's; the equivariance claim is not well separated from")
        print("      a general-similarity confound.")


if __name__ == "__main__":
    main()
