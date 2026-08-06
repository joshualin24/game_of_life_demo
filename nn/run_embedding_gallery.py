"""
Embedding gallery: 3 figures (oscillators, gliders/combos, random),
each showing 3 configurations with:
  Row 0 — grid configuration
  Row 1 — post-transformer patch embedding L2 norm (10×10 spatial heatmap)
  Row 2 — pairwise cosine similarity matrix between all 100 patch embeddings (100×100)
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from nn.embedding_analysis import EmbeddingExtractor, load_v8, load_v10, GRID_SIZE, PATCH_SIZE, N_PATCHES_1D
from nn.utils import RESULTS_DIR

N_PATCHES = N_PATCHES_1D ** 2   # 100

# ── Pattern helpers ────────────────────────────────────────────────────────────

def blank():
    return np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

def place(grid, pat, r0, c0):
    pat = np.array(pat, dtype=np.uint8)
    ph, pw = pat.shape
    for dr in range(ph):
        for dc in range(pw):
            grid[(r0 + dr) % GRID_SIZE, (c0 + dc) % GRID_SIZE] |= pat[dr, dc]

GLIDER  = [[0,1,0],[0,0,1],[1,1,1]]
BLINKER = [[1,1,1]]
TOAD    = [[0,1,1,1],[1,1,1,0]]
BEACON  = [[1,1,0,0],[1,1,0,0],[0,0,1,1],[0,0,1,1]]
PULSAR_SEED = [
    [0,0,1,1,1,0,0,0,0,1,1,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,0,1,0,0,1,0,0,0,0,1],
    [1,0,0,0,0,1,0,0,1,0,0,0,0,1],
    [1,0,0,0,0,1,0,0,1,0,0,0,0,1],
    [0,0,1,1,1,0,0,0,0,1,1,1,0,0],
]

def make_blinker(r=20, c=18):
    g = blank(); place(g, BLINKER, r, c); return g

def make_toad(r=18, c=17):
    g = blank(); place(g, TOAD, r, c); return g

def make_beacon(r=17, c=17):
    g = blank(); place(g, BEACON, r, c); return g

def make_glider(r=5, c=5):
    g = blank(); place(g, GLIDER, r, c); return g

def make_two_gliders():
    g = blank()
    place(g, GLIDER, 5, 5)
    # second glider mirrored horizontally, heading left
    place(g, [[0,1,0],[1,0,0],[1,1,1]], 5, 28)
    return g

def make_glider_blinker():
    g = blank()
    place(g, GLIDER, 5, 5)
    place(g, BLINKER, 20, 18)
    return g

def make_random(density, seed=42):
    rng = np.random.default_rng(seed)
    return (rng.random((GRID_SIZE, GRID_SIZE)) < density).astype(np.uint8)

def make_pulsar():
    g = blank(); place(g, PULSAR_SEED, 13, 13); return g


# ── Embedding metrics ──────────────────────────────────────────────────────────

def patch_norms(post: np.ndarray) -> np.ndarray:
    """post: (100, d) → (10, 10) L2 norm heatmap."""
    return np.linalg.norm(post, axis=1).reshape(N_PATCHES_1D, N_PATCHES_1D)

def pairwise_cosine(post: np.ndarray) -> np.ndarray:
    """post: (100, d) → (100, 100) cosine similarity matrix."""
    n = np.linalg.norm(post, axis=1, keepdims=True) + 1e-8
    normed = post / n
    return normed @ normed.T


# ── Figure builder ─────────────────────────────────────────────────────────────

def make_gallery(configs: list[tuple[str, np.ndarray]],
                 extractor: EmbeddingExtractor,
                 fig_title: str,
                 out_path: str):
    """
    configs: list of (label, grid) — exactly 3 entries.
    Produces a 3-row × 3-column figure.
    """
    assert len(configs) == 3

    fig = plt.figure(figsize=(15, 12))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    for col, (label, grid) in enumerate(configs):
        _, post = extractor(grid)
        norms   = patch_norms(post)          # (10, 10)
        cosmat  = pairwise_cosine(post)      # (100, 100)

        # ── Row 0: grid ────────────────────────────────────────────────────────
        ax0 = fig.add_subplot(gs[0, col])
        ax0.imshow(grid, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
        ax0.set_title(label, fontsize=10, fontweight="bold")
        ax0.axis("off")
        alive = int(grid.sum())
        ax0.set_xlabel(f"{alive} alive cells", fontsize=8)

        # Draw patch grid lines
        for k in range(0, GRID_SIZE + 1, PATCH_SIZE):
            ax0.axhline(k - 0.5, color="white", lw=0.3, alpha=0.4)
            ax0.axvline(k - 0.5, color="white", lw=0.3, alpha=0.4)

        # ── Row 1: patch norm heatmap ──────────────────────────────────────────
        ax1 = fig.add_subplot(gs[1, col])
        im1 = ax1.imshow(norms, cmap="viridis", interpolation="nearest")
        ax1.set_title("Post-transformer\npatch norms", fontsize=9)
        ax1.set_xticks(range(N_PATCHES_1D))
        ax1.set_yticks(range(N_PATCHES_1D))
        ax1.tick_params(labelsize=6)
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # ── Row 2: pairwise cosine similarity matrix ───────────────────────────
        ax2 = fig.add_subplot(gs[2, col])
        im2 = ax2.imshow(cosmat, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        ax2.set_title("Pairwise patch\ncosine similarity", fontsize=9)
        ax2.set_xlabel("Patch index (row-major)", fontsize=7)
        ax2.set_ylabel("Patch index (row-major)", fontsize=7)
        ax2.tick_params(labelsize=6)
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle(fig_title, fontsize=13, fontweight="bold", y=1.01)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def run_for_model(tag: str, extractor: EmbeddingExtractor):
    make_gallery(
        configs=[
            ("Blinker\n(period 2, 3 cells)",  make_blinker()),
            ("Toad\n(period 2, 6 cells)",      make_toad()),
            ("Beacon\n(period 2, 8 cells)",    make_beacon()),
        ],
        extractor=extractor,
        fig_title=f"{tag.upper()} Embeddings — Oscillators",
        out_path=os.path.join(RESULTS_DIR, f"gallery_{tag}_oscillators.png"),
    )
    make_gallery(
        configs=[
            ("Single glider\n(5 cells)",       make_glider()),
            ("Two gliders\n(5+5 cells)",        make_two_gliders()),
            ("Glider + Blinker\n(5+3 cells)",   make_glider_blinker()),
        ],
        extractor=extractor,
        fig_title=f"{tag.upper()} Embeddings — Gliders & Combinations",
        out_path=os.path.join(RESULTS_DIR, f"gallery_{tag}_gliders.png"),
    )
    make_gallery(
        configs=[
            ("Random  d=0.20\n(sparse)",       make_random(0.20)),
            ("Random  d=0.50\n(medium)",       make_random(0.50)),
            ("Random  d=0.80\n(dense)",        make_random(0.80)),
        ],
        extractor=extractor,
        fig_title=f"{tag.upper()} Embeddings — Random Configurations",
        out_path=os.path.join(RESULTS_DIR, f"gallery_{tag}_random.png"),
    )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["v8", "v10", "both"], default="v8")
    args = parser.parse_args()

    if args.model in ("v8", "both"):
        print("Loading V8 …")
        run_for_model("v8", EmbeddingExtractor(load_v8()))

    if args.model in ("v10", "both"):
        print("Loading V10 …")
        run_for_model("v10", EmbeddingExtractor(load_v10()))

    print("Done.")


if __name__ == "__main__":
    main()
