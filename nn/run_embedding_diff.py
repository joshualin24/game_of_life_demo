"""
Direct comparison of embedding difference vectors under various transformations.

Three figures per configuration (glider, blinker):
  1. Δ-norm heatmaps   — spatial map of ‖post_tf − post_orig‖ per patch for each transform
  2. Cross-similarity  — cosine similarity between flattened Δpost vectors (which transforms
                         cause "similar" embedding changes?)
  3. PCA scatter       — all 100 patch embeddings of original + every transform in 2D PCA
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA

from nn.embedding_analysis import (
    EmbeddingExtractor, Transforms, load_v8, load_v10,
    GRID_SIZE, PATCH_SIZE, N_PATCHES_1D,
)
from nn.utils import RESULTS_DIR

N_PATCHES = N_PATCHES_1D ** 2

# ── Configurations ─────────────────────────────────────────────────────────────

def blank(): return np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

def place(g, pat, r0, c0):
    pat = np.array(pat, dtype=np.uint8)
    for dr in range(pat.shape[0]):
        for dc in range(pat.shape[1]):
            g[(r0+dr)%GRID_SIZE, (c0+dc)%GRID_SIZE] |= pat[dr, dc]

GLIDER_PAT  = [[0,1,0],[0,0,1],[1,1,1]]
BLINKER_PAT = [[1,1,1]]
BEACON_PAT  = [[1,1,0,0],[1,1,0,0],[0,0,1,1],[0,0,1,1]]

def make_glider():
    g = blank(); place(g, GLIDER_PAT, 10, 10); return g

def make_blinker():
    g = blank(); place(g, BLINKER_PAT, 20, 18); return g

def make_beacon():
    g = blank(); place(g, BEACON_PAT, 18, 18); return g


# ── Build transformation list ──────────────────────────────────────────────────

def get_transforms(grid):
    """
    Returns list of (label, transformed_grid) for a given base grid.
    Cell-flip and cell-move pick the first live cell found.
    """
    live = list(zip(*np.where(grid == 1)))
    dead = list(zip(*np.where(grid == 0)))
    r0, c0 = live[0]   # first live cell
    rd, cd = dead[0]   # first dead cell

    tfs = [
        ("rot 90°",           Transforms.rotate(grid, 1)),
        ("rot 180°",          Transforms.rotate(grid, 2)),
        ("rot 270°",          Transforms.rotate(grid, 3)),
        ("flip H",            Transforms.flip_h(grid)),
        ("flip V",            Transforms.flip_v(grid)),
        ("trans +4 cols",     Transforms.translate(grid,  0,  4)),
        ("trans +4 rows",     Transforms.translate(grid,  4,  0)),
        ("trans +1 col",      Transforms.translate(grid,  0,  1)),
        ("trans +1 row",      Transforms.translate(grid,  1,  0)),
        ("cell flip\nalive→dead", Transforms.cell_flip(grid, r0, c0)),
        ("cell flip\ndead→alive", Transforms.cell_flip(grid, rd, cd)),
    ]

    # cell move: need a free destination
    for dr, dc, lbl in [(0,1,"right1"),(1,0,"down1"),(1,1,"diag1")]:
        nr, nc = (r0+dr)%GRID_SIZE, (c0+dc)%GRID_SIZE
        if grid[nr, nc] == 0:
            tfs.append((f"cell move\n{lbl}", Transforms.cell_move(grid, r0, c0, dr, dc)))
            break

    return tfs


# ── Figure 1: Δ-norm heatmaps ─────────────────────────────────────────────────

def fig_delta_norms(name, grid, transforms, ext, out_path):
    """
    One row per transformation.
    Columns: [original | transformed | Δ pre-norm (10×10) | Δ post-norm (10×10)]
    """
    n_tf = len(transforms)
    fig, axes = plt.subplots(n_tf, 4, figsize=(13, 2.1 * n_tf))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)

    pre_orig, post_orig = ext(grid)

    for row, (label, grid_tf) in enumerate(transforms):
        pre_tf, post_tf = ext(grid_tf)

        delta_pre  = np.linalg.norm(pre_tf  - pre_orig,  axis=1).reshape(N_PATCHES_1D, N_PATCHES_1D)
        delta_post = np.linalg.norm(post_tf - post_orig, axis=1).reshape(N_PATCHES_1D, N_PATCHES_1D)

        ax = axes[row]

        ax[0].imshow(grid,    cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
        ax[0].axis("off")
        if row == 0:
            ax[0].set_title("Original", fontsize=8, fontweight="bold")

        ax[1].imshow(grid_tf, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
        ax[1].axis("off")
        ax[1].set_ylabel(label, fontsize=7, rotation=0, labelpad=60, va="center")
        if row == 0:
            ax[1].set_title("Transformed", fontsize=8, fontweight="bold")

        vmax_pre  = delta_pre.max()  if delta_pre.max()  > 0 else 1
        vmax_post = delta_post.max() if delta_post.max() > 0 else 1
        # use shared scale across pre/post for fair comparison
        vmax = max(vmax_pre, vmax_post)

        im2 = ax[2].imshow(delta_pre,  cmap="hot", vmin=0, vmax=vmax, interpolation="nearest")
        ax[2].set_xticks(range(N_PATCHES_1D)); ax[2].set_yticks(range(N_PATCHES_1D))
        ax[2].tick_params(labelsize=4)
        if row == 0:
            ax[2].set_title("‖Δpre‖ per patch", fontsize=8, fontweight="bold")

        im3 = ax[3].imshow(delta_post, cmap="hot", vmin=0, vmax=vmax, interpolation="nearest")
        ax[3].set_xticks(range(N_PATCHES_1D)); ax[3].set_yticks(range(N_PATCHES_1D))
        ax[3].tick_params(labelsize=4)
        if row == 0:
            ax[3].set_title("‖Δpost‖ per patch", fontsize=8, fontweight="bold")

    fig.suptitle(f"Embedding differences — {name}", fontsize=11, fontweight="bold", y=1.005)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── Figure 2: Cross-similarity of Δpost vectors ───────────────────────────────

def fig_cross_similarity(name, grid, transforms, ext, out_path):
    """
    Cosine similarity between flattened Δpost vectors for each pair of transforms.
    Also includes the all-zeros Δ (original vs itself) as a reference row/col.
    """
    pre_orig, post_orig = ext(grid)

    labels = ["(identity)"] + [lbl.replace("\n"," ") for lbl, _ in transforms]
    deltas = [np.zeros_like(post_orig.flatten())]   # identity → zero delta

    for _, grid_tf in transforms:
        _, post_tf = ext(grid_tf)
        deltas.append((post_tf - post_orig).flatten())

    deltas = np.array(deltas)   # (n_tf+1, N_patches * d_model)

    # cosine similarity matrix
    norms = np.linalg.norm(deltas, axis=1, keepdims=True) + 1e-10
    normed = deltas / norms
    cosmat = normed @ normed.T

    n = len(labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cosmat, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=8)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cosmat[i,j]:.2f}", ha="center", va="center",
                    fontsize=6, color="black" if abs(cosmat[i,j]) < 0.6 else "white")

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title(f"Cross-similarity of Δpost vectors — {name}\n"
                 f"(cosine similarity between flattened embedding change vectors)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── Figure 3: PCA scatter of patch embeddings ─────────────────────────────────

def fig_pca_scatter(name, grid, transforms, ext, out_path):
    """
    Project all 100 patch embeddings (post-transformer) for original + each transform
    into the top-2 PCA components of the original. Plot as a scatter, one colour per
    transform, to see how transformations move points in embedding space.
    """
    _, post_orig = ext(grid)

    # fit PCA on original embeddings
    pca = PCA(n_components=2)
    pca.fit(post_orig)

    n_tf   = len(transforms)
    colors = plt.cm.tab20(np.linspace(0, 1, n_tf + 1))

    fig, ax = plt.subplots(figsize=(9, 7))

    # plot original
    proj = pca.transform(post_orig)   # (100, 2)
    ax.scatter(proj[:, 0], proj[:, 1], c=[colors[0]], s=40, label="original",
               zorder=5, edgecolors="black", linewidths=0.4)

    for idx, (label, grid_tf) in enumerate(transforms):
        _, post_tf = ext(grid_tf)
        proj_tf = pca.transform(post_tf)
        lbl = label.replace("\n", " ")
        ax.scatter(proj_tf[:, 0], proj_tf[:, 1], c=[colors[idx + 1]], s=18,
                   label=lbl, alpha=0.7)

    # draw arrows from original to each transform for the 10 most-changed patches
    for idx, (_, grid_tf) in enumerate(transforms):
        _, post_tf = ext(grid_tf)
        delta_norms = np.linalg.norm(post_tf - post_orig, axis=1)
        top_patches = np.argsort(delta_norms)[-5:]   # 5 most-changed
        proj    = pca.transform(post_orig)
        proj_tf = pca.transform(post_tf)
        for p in top_patches:
            ax.annotate("",
                xy=(proj_tf[p, 0], proj_tf[p, 1]),
                xytext=(proj[p, 0], proj[p, 1]),
                arrowprops=dict(arrowstyle="->", color=colors[idx+1], lw=0.8, alpha=0.6),
            )

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)", fontsize=9)
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)", fontsize=9)
    ax.set_title(f"PCA of post-transformer patch embeddings — {name}\n"
                 f"Arrows: 5 most-displaced patches per transform",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right", ncol=2, framealpha=0.85)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def run_for_model(tag: str, ext: EmbeddingExtractor):
    configs = [
        ("glider",  make_glider()),
        ("blinker", make_blinker()),
        ("beacon",  make_beacon()),
    ]
    for name, grid in configs:
        print(f"\n── {name} ──")
        transforms = get_transforms(grid)
        fig_delta_norms(
            name, grid, transforms, ext,
            os.path.join(RESULTS_DIR, f"embdiff_{tag}_deltanorms_{name}.png"),
        )
        fig_cross_similarity(
            name, grid, transforms, ext,
            os.path.join(RESULTS_DIR, f"embdiff_{tag}_crosssim_{name}.png"),
        )
        fig_pca_scatter(
            name, grid, transforms, ext,
            os.path.join(RESULTS_DIR, f"embdiff_{tag}_pca_{name}.png"),
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

    print("\nDone.")


if __name__ == "__main__":
    main()
