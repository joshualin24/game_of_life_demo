"""
Reusable embedding extraction and comparison utilities for CNNTransformerV4 models.

Usage:
    from nn.embedding_analysis import EmbeddingExtractor, Transforms, compare, plot_comparison
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors

from nn.models import CNNTransformerV4
from nn.utils  import CKPT_DIR, DEVICE

GRID_SIZE  = 40
PATCH_SIZE = 4
N_PATCHES_1D = GRID_SIZE // PATCH_SIZE   # 10
N_PATCHES    = N_PATCHES_1D ** 2          # 100

# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(ckpt_name, d_model=64, nhead=4, num_layers=4):
    """Load a CNNTransformerV4 checkpoint."""
    m = CNNTransformerV4(
        grid_size=GRID_SIZE, patch_size=PATCH_SIZE,
        d_model=d_model, nhead=nhead, num_layers=num_layers,
    ).to(DEVICE)
    m.load_state_dict(torch.load(
        os.path.join(CKPT_DIR, ckpt_name),
        map_location=DEVICE, weights_only=True,
    ))
    m.eval()
    return m

def load_v8():
    return load_model("task16_cnn_transformer_v8_best.pt", d_model=64,  nhead=4, num_layers=4)

def load_v10():
    return load_model("task18_cnn_transformer_v10_best.pt", d_model=128, nhead=8, num_layers=6)


# ── Embedding extractor ────────────────────────────────────────────────────────

class EmbeddingExtractor:
    """
    Wraps a CNNTransformerV4 and exposes intermediate embeddings.

    For a 40×40 grid with 4×4 patches there are 100 patches arranged in a 10×10 grid.

    Returns two arrays, both shape (100, d_model):
        pre  — patch_embed(patches) + pos_embed  (content + absolute position)
        post — transformer(pre)                  (attention-mixed context)
    """

    def __init__(self, model: CNNTransformerV4):
        self.model = model
        self.model.eval()

    def _forward_stages(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Replicate CNNTransformerV4.forward up to (but not including) the head.
        x: (B, 1, H, W) → pre (B, n_patches, d_model), post (B, n_patches, d_model)

        CNNTransformerV4 architecture:
          Stage 1 — cnn(x)                         : (B, d_model, H, W)
          Stage 2 — patch_proj(feat_patches) + pos_embed : (B, n_patches, d_model)  ← pre
          Stage 3 — transformer(pre)               : (B, n_patches, d_model)        ← post
        """
        m = self.model
        p  = m.patch_size
        B  = x.shape[0]

        feat = m.cnn(x)                                     # (B, C, H, W)
        C, H, W = feat.shape[1], feat.shape[2], feat.shape[3]
        h = w = H // p
        feat = feat.reshape(B, C, h, p, w, p)
        feat = feat.permute(0, 2, 4, 1, 3, 5)              # (B, h, w, C, p, p)
        feat = feat.reshape(B, m.n_patches, C * p * p)     # (B, n_patches, C·p²)

        pre  = m.patch_proj(feat) + m.pos_embed             # (B, n_patches, d_model)
        post = m.transformer(pre)                           # (B, n_patches, d_model)
        return pre, post

    def __call__(self, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """grid: (H, W) binary uint8/float → (pre, post) each (n_patches, d_model)."""
        x = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pre, post = self._forward_stages(x)
        return pre.squeeze(0).cpu().numpy(), post.squeeze(0).cpu().numpy()

    def batch(self, grids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """grids: (N, H, W) → (N, n_patches, d_model) each."""
        x = torch.tensor(grids, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        with torch.no_grad():
            pre, post = self._forward_stages(x)
        return pre.cpu().numpy(), post.cpu().numpy()


# ── Grid transforms ────────────────────────────────────────────────────────────

class Transforms:
    """All transforms preserve the (H, W) shape and use toroidal (periodic) boundary."""

    @staticmethod
    def rotate(grid: np.ndarray, k: int = 1) -> np.ndarray:
        """Rotate 90° * k counter-clockwise."""
        return np.rot90(grid, k=k).copy()

    @staticmethod
    def flip_h(grid: np.ndarray) -> np.ndarray:
        """Flip horizontally (left-right)."""
        return np.fliplr(grid).copy()

    @staticmethod
    def flip_v(grid: np.ndarray) -> np.ndarray:
        """Flip vertically (up-down)."""
        return np.flipud(grid).copy()

    @staticmethod
    def translate(grid: np.ndarray, dr: int, dc: int) -> np.ndarray:
        """Translate by (dr, dc) rows/cols with toroidal wrapping."""
        return np.roll(np.roll(grid, dr, axis=0), dc, axis=1).copy()

    @staticmethod
    def cell_flip(grid: np.ndarray, r: int, c: int) -> np.ndarray:
        """Flip the state of cell (r, c): 0↔1."""
        g = grid.copy()
        g[r, c] = 1 - g[r, c]
        return g

    @staticmethod
    def cell_move(grid: np.ndarray, r: int, c: int, dr: int, dc: int) -> np.ndarray:
        """
        Move living cell at (r, c) to (r+dr, c+dc) with toroidal wrapping.
        Source cell is cleared; destination is set alive (OR'd so existing alive cells stay).
        """
        assert grid[r, c] == 1, f"No live cell at ({r}, {c})"
        g = grid.copy()
        g[r, c] = 0
        g[(r + dr) % GRID_SIZE, (c + dc) % GRID_SIZE] = 1
        return g


# ── Comparison metrics ─────────────────────────────────────────────────────────

def compare(emb1: np.ndarray, emb2: np.ndarray) -> dict:
    """
    Compare two embedding arrays of shape (N_patches, d_model).

    Returns dict with:
        cos_sim   : (N_patches,) cosine similarity in [-1, 1]
        l2        : (N_patches,) L2 distance
        cos_map   : (10, 10) spatial heatmap of cos_sim
        l2_map    : (10, 10) spatial heatmap of l2
        mean_cos  : scalar global mean cosine similarity
        mean_l2   : scalar global mean L2 distance
    """
    # cosine similarity
    n1 = np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8
    n2 = np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8
    cos = (emb1 / n1 * emb2 / n2).sum(axis=1)   # (N_patches,)

    l2  = np.linalg.norm(emb1 - emb2, axis=1)   # (N_patches,)

    return dict(
        cos_sim  = cos,
        l2       = l2,
        cos_map  = cos.reshape(N_PATCHES_1D, N_PATCHES_1D),
        l2_map   = l2.reshape(N_PATCHES_1D, N_PATCHES_1D),
        mean_cos = float(cos.mean()),
        mean_l2  = float(l2.mean()),
    )


# ── Visualisation ──────────────────────────────────────────────────────────────

def plot_comparison(
    grid_orig: np.ndarray,
    grid_tf:   np.ndarray,
    pre_stats:  dict,
    post_stats: dict,
    title:      str,
    out_path:   str | None = None,
    label_orig: str = "Original",
    label_tf:   str = "Transformed",
) -> plt.Figure:
    """
    5-column figure:
      [original grid | transformed grid | grid diff |
       pre-transformer cos-sim heatmap | post-transformer cos-sim heatmap]

    Also annotates with mean cosine similarities.
    Returns the Figure object.
    """
    diff = grid_orig.astype(int) - grid_tf.astype(int)   # -1, 0, +1

    fig = plt.figure(figsize=(16, 4))
    gs  = gridspec.GridSpec(1, 5, figure=fig, wspace=0.35)

    def _grid_ax(ax, data, cmap, title_, vmin=0, vmax=1):
        ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title_, fontsize=9)
        ax.axis("off")

    # column 0 — original
    _grid_ax(fig.add_subplot(gs[0, 0]), grid_orig, "inferno", label_orig)

    # column 1 — transformed
    _grid_ax(fig.add_subplot(gs[0, 1]), grid_tf, "inferno", label_tf)

    # column 2 — grid diff
    diff_cmap = mcolors.ListedColormap(["#3366cc", "#111111", "#cc3333"])
    ax_d = fig.add_subplot(gs[0, 2])
    ax_d.imshow(diff, cmap=diff_cmap, vmin=-1, vmax=1, interpolation="nearest")
    ax_d.set_title("Grid diff\n(blue=removed, red=added)", fontsize=8)
    ax_d.axis("off")
    n_changed = int((diff != 0).sum())
    ax_d.set_xlabel(f"{n_changed} cells changed", fontsize=8)

    # column 3 — pre-transformer cosine similarity
    ax_pre = fig.add_subplot(gs[0, 3])
    im_pre = ax_pre.imshow(pre_stats["cos_map"], cmap="RdYlGn", vmin=-1, vmax=1,
                           interpolation="nearest")
    ax_pre.set_title(f"Pre-transformer\ncos sim (mean={pre_stats['mean_cos']:.3f})", fontsize=8)
    ax_pre.set_xticks(range(N_PATCHES_1D)); ax_pre.set_yticks(range(N_PATCHES_1D))
    ax_pre.tick_params(labelsize=6)
    plt.colorbar(im_pre, ax=ax_pre, fraction=0.046, pad=0.04)

    # column 4 — post-transformer cosine similarity
    ax_post = fig.add_subplot(gs[0, 4])
    im_post = ax_post.imshow(post_stats["cos_map"], cmap="RdYlGn", vmin=-1, vmax=1,
                             interpolation="nearest")
    ax_post.set_title(f"Post-transformer\ncos sim (mean={post_stats['mean_cos']:.3f})", fontsize=8)
    ax_post.set_xticks(range(N_PATCHES_1D)); ax_post.set_yticks(range(N_PATCHES_1D))
    ax_post.tick_params(labelsize=6)
    plt.colorbar(im_post, ax=ax_post, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=10, fontweight="bold", y=1.02)

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved {out_path}")

    return fig


def plot_perturbation_spread(
    grid_orig: np.ndarray,
    perturbed_cells: list[tuple[int,int]],   # list of (r, c) perturbed
    pre_stats:  dict,
    post_stats: dict,
    title:      str,
    out_path:   str | None = None,
) -> plt.Figure:
    """
    Like plot_comparison but highlights the perturbed cells on the grid.
    perturbed_cells: list of (row, col) coordinates that changed.
    """
    # Build an overlay marking which patches contain a perturbed cell
    patch_hit = np.zeros((N_PATCHES_1D, N_PATCHES_1D), dtype=bool)
    for r, c in perturbed_cells:
        patch_hit[r // PATCH_SIZE, c // PATCH_SIZE] = True

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.subplots_adjust(wspace=0.4)

    # original grid with perturbed cells marked
    ax = axes[0]
    ax.imshow(grid_orig, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
    for (r, c) in perturbed_cells:
        rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                              linewidth=1.5, edgecolor="cyan", facecolor="none")
        ax.add_patch(rect)
    ax.set_title("Grid (cyan = perturbed cells)", fontsize=9)
    ax.axis("off")

    # patch grid with hit patches marked
    ax2 = axes[1]
    ax2.imshow(np.zeros((N_PATCHES_1D, N_PATCHES_1D)), cmap="Greys", vmin=0, vmax=1,
               interpolation="nearest")
    for ri in range(N_PATCHES_1D):
        for ci in range(N_PATCHES_1D):
            if patch_hit[ri, ci]:
                rect = plt.Rectangle((ci - 0.5, ri - 0.5), 1, 1,
                                     linewidth=2, edgecolor="cyan", facecolor="cyan", alpha=0.5)
                ax2.add_patch(rect)
    ax2.set_title("Affected patches\n(cyan)", fontsize=9)
    ax2.set_xticks(range(N_PATCHES_1D)); ax2.set_yticks(range(N_PATCHES_1D))
    ax2.tick_params(labelsize=6)

    # pre-transformer Δ = 1 - cos_sim  (sensitivity)
    ax3 = axes[2]
    sens_pre = 1 - pre_stats["cos_map"]
    im3 = ax3.imshow(sens_pre, cmap="hot", vmin=0, vmax=2, interpolation="nearest")
    ax3.set_title(f"Pre-transformer sensitivity\n(1-cos, mean cos={pre_stats['mean_cos']:.4f})", fontsize=8)
    ax3.set_xticks(range(N_PATCHES_1D)); ax3.set_yticks(range(N_PATCHES_1D))
    ax3.tick_params(labelsize=6)
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # post-transformer Δ = 1 - cos_sim
    ax4 = axes[3]
    sens_post = 1 - post_stats["cos_map"]
    im4 = ax4.imshow(sens_post, cmap="hot", vmin=0, vmax=2, interpolation="nearest")
    ax4.set_title(f"Post-transformer sensitivity\n(1-cos, mean cos={post_stats['mean_cos']:.4f})", fontsize=8)
    ax4.set_xticks(range(N_PATCHES_1D)); ax4.set_yticks(range(N_PATCHES_1D))
    ax4.tick_params(labelsize=6)
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=10, fontweight="bold")

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved {out_path}")

    return fig
