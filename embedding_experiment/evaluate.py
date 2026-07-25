"""
Evaluate and visualize the trained trajectory-invariant encoder.

Produces:
  results/eval_umap.png          — UMAP of all embeddings coloured by fate
  results/eval_intra_dist.png    — intra-trajectory embedding spread (histogram)
  results/eval_loss_curve.png    — training/val loss curve
  results/eval_traj_paths.png    — PCA paths of several individual trajectories
  results/eval_similarity_mat.png— mean cosine similarity: within vs across traj

Usage:
    python evaluate.py [--n-vis 5000] [--n-traj-paths 8]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from model import TrajectoryEncoder

HERE  = os.path.dirname(os.path.abspath(__file__))
DATA  = os.path.join(HERE, "data", "trajectories.npz")
RES   = os.path.join(HERE, "results")

BG = "#0f1117"
TEXT_C = "white"
PALETTE = ["#3a7bd5", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]


# ── Helpers ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def embed_all(model, frames, device, batch=512):
    """frames: (N, T, H, W) uint8 → embeddings: (N, T, D)"""
    model.eval()
    N, T, H, W = frames.shape
    out = np.zeros((N, T, model.encoder.latent_dim), np.float32)
    x = torch.from_numpy(frames).float()  # (N, T, H, W)
    for i in range(0, N, batch):
        xb = x[i:i+batch]                        # (b, T, H, W)
        b = xb.size(0)
        xb = xb.view(b * T, 1, H, W).to(device)  # (b*T, 1, H, W)
        h = model.encode(xb).cpu().numpy()        # (b*T, D)
        out[i:i+b] = h.reshape(b, T, -1)
    return out


def styled_ax(ax, title=""):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT_C, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    if title:
        ax.set_title(title, color=TEXT_C, fontsize=10, fontweight="bold")


# ── Plot 1: Loss curve ────────────────────────────────────────────────────────

def plot_loss_curve(history, out):
    epochs = [r["epoch"] for r in history]
    tr     = [r["train_loss"] for r in history]
    va     = [r["val_loss"] for r in history]

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor(BG)
    styled_ax(ax, "NT-Xent Training & Validation Loss")
    ax.plot(epochs, tr, color="#3a7bd5", lw=2, label="train")
    ax.plot(epochs, va, color="#e74c3c", lw=2, label="val", linestyle="--")
    ax.set_xlabel("Epoch", color=TEXT_C, fontsize=9)
    ax.set_ylabel("Loss", color=TEXT_C, fontsize=9)
    ax.legend(facecolor="#1c1f26", edgecolor="#444", labelcolor=TEXT_C, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"  Saved → {out}")


# ── Plot 2: PCA of trajectory paths ──────────────────────────────────────────

def plot_traj_paths(emb, fate, fate_names, n_traj, out):
    """Show PCA paths of individual trajectories (colour = fate)."""
    N, T, D = emb.shape
    flat = emb.reshape(N * T, D)
    pca = PCA(n_components=2, random_state=0)
    flat2d = pca.fit_transform(flat).reshape(N, T, 2)
    ev = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(BG)
    styled_ax(ax, f"PCA Trajectory Paths  (var: {ev[0]:.1%}, {ev[1]:.1%})")

    # background: all endpoints
    ax.scatter(flat2d[:, -1, 0], flat2d[:, -1, 1],
               s=3, c="#333", alpha=0.3, zorder=1)

    rng = np.random.default_rng(42)
    chosen = rng.choice(N, size=min(n_traj, N), replace=False)
    fate_colors = {i: PALETTE[i % len(PALETTE)] for i in np.unique(fate)}

    for idx in chosen:
        path = flat2d[idx]           # (T, 2)
        color = fate_colors[fate[idx]]
        ax.plot(path[:, 0], path[:, 1], color=color, lw=1.2, alpha=0.8, zorder=2)
        ax.scatter(path[0, 0], path[0, 1], color="white", s=30, zorder=3)
        ax.scatter(path[-1, 0], path[-1, 1], color=color, s=50,
                   edgecolors="white", lw=0.5, zorder=4)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=fate_colors[i], lw=2,
                      label=fate_names[i] if i < len(fate_names) else str(i))
               for i in sorted(fate_colors)]
    handles += [Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
                       markersize=6, label="start"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#aaa",
                       markersize=6, label="end")]
    ax.legend(handles=handles, facecolor="#1c1f26", edgecolor="#444",
              labelcolor=TEXT_C, fontsize=8)
    ax.set_xlabel("PC1", color=TEXT_C, fontsize=9)
    ax.set_ylabel("PC2", color=TEXT_C, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"  Saved → {out}")


# ── Plot 3: Intra-trajectory spread ──────────────────────────────────────────

def plot_intra_dist(emb, fate, fate_names, out):
    """
    For each trajectory, compute the std of embeddings across time steps.
    A trajectory-invariant encoder should have low intra-traj spread.
    """
    # intra std: mean over dims, then mean over time → scalar per trajectory
    intra_std = emb.std(axis=1).mean(axis=1)   # (N,)

    # inter: std of trajectory means across trajectories
    traj_means = emb.mean(axis=1)              # (N, D)
    inter_std = traj_means.std(axis=0).mean()  # scalar

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(BG)
    styled_ax(ax, "Intra-Trajectory Embedding Spread (lower = more invariant)")

    unique_fates = np.unique(fate)
    for fi in unique_fates:
        mask = fate == fi
        label = fate_names[fi] if fi < len(fate_names) else str(fi)
        ax.hist(intra_std[mask], bins=40, alpha=0.7,
                color=PALETTE[fi % len(PALETTE)], label=label, density=True)

    ax.axvline(inter_std, color="white", lw=1.5, linestyle="--",
               label=f"inter-traj std={inter_std:.3f}")
    ax.set_xlabel("Mean intra-trajectory std", color=TEXT_C, fontsize=9)
    ax.set_ylabel("Density", color=TEXT_C, fontsize=9)
    ax.legend(facecolor="#1c1f26", edgecolor="#444", labelcolor=TEXT_C, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"  Saved → {out}")
    print(f"    mean intra-traj std = {intra_std.mean():.4f}")
    print(f"    inter-traj std      = {inter_std:.4f}")
    print(f"    invariance ratio    = {intra_std.mean() / inter_std:.4f}  (lower is better)")


# ── Plot 4: Within vs across trajectory cosine similarity ─────────────────────

def plot_similarity_matrix(emb, fate, fate_names, out, n_sample=500):
    """
    Sample n_sample trajectories, compute mean pairwise cosine similarity
    of their mean embeddings within and across fate categories.
    """
    rng = np.random.default_rng(0)
    idx = rng.choice(len(emb), size=min(n_sample, len(emb)), replace=False)
    means = emb[idx].mean(axis=1)           # (n, D)
    norms = means / (np.linalg.norm(means, axis=1, keepdims=True) + 1e-8)
    sim = norms @ norms.T                   # (n, n)

    fates_sub = fate[idx]
    unique = np.unique(fates_sub)
    n_cat = len(unique)
    cat_sim = np.zeros((n_cat, n_cat))
    for a, fi in enumerate(unique):
        for b, fj in enumerate(unique):
            ma = fates_sub == fi
            mb = fates_sub == fj
            if a == b:
                # within: exclude diagonal
                vals = sim[np.ix_(ma, mb)]
                np.fill_diagonal(vals, np.nan)
                cat_sim[a, b] = np.nanmean(vals)
            else:
                cat_sim[a, b] = sim[np.ix_(ma, mb)].mean()

    labels = [fate_names[fi] if fi < len(fate_names) else str(fi) for fi in unique]

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(BG)
    styled_ax(ax, "Mean Cosine Similarity Between Fate Categories")
    im = ax.imshow(cat_sim, cmap="RdYlGn", vmin=-0.1, vmax=1.0)
    ax.set_xticks(range(n_cat)); ax.set_xticklabels(labels, rotation=30,
                                                     ha="right", color=TEXT_C, fontsize=8)
    ax.set_yticks(range(n_cat)); ax.set_yticklabels(labels, color=TEXT_C, fontsize=8)
    for i in range(n_cat):
        for j in range(n_cat):
            ax.text(j, i, f"{cat_sim[i, j]:.2f}", ha="center", va="center",
                    color="black", fontsize=8, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(colors=TEXT_C)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"  Saved → {out}")


# ── Plot 5: UMAP ──────────────────────────────────────────────────────────────

def plot_umap(emb, fate, fate_names, out, n_vis):
    try:
        import umap
    except ImportError:
        print("  umap-learn not installed, skipping UMAP. Use PCA instead.")
        return _plot_pca_2d(emb, fate, fate_names, out, n_vis)

    rng = np.random.default_rng(0)
    N = len(emb)
    idx = rng.choice(N, size=min(n_vis, N), replace=False)
    means = emb[idx].mean(axis=1)   # use mean embedding per trajectory
    fates_sub = fate[idx]

    reducer = umap.UMAP(n_components=2, random_state=0, n_neighbors=30, min_dist=0.1)
    coords = reducer.fit_transform(means)

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(BG)
    styled_ax(ax, "UMAP of Trajectory Mean Embeddings (coloured by fate)")
    for fi in np.unique(fates_sub):
        mask = fates_sub == fi
        label = fate_names[fi] if fi < len(fate_names) else str(fi)
        ax.scatter(coords[mask, 0], coords[mask, 1], s=6, alpha=0.6,
                   color=PALETTE[fi % len(PALETTE)], label=label)
    ax.legend(facecolor="#1c1f26", edgecolor="#444", labelcolor=TEXT_C, fontsize=9,
              markerscale=2)
    ax.set_xlabel("UMAP-1", color=TEXT_C, fontsize=9)
    ax.set_ylabel("UMAP-2", color=TEXT_C, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"  Saved → {out}")


def _plot_pca_2d(emb, fate, fate_names, out, n_vis):
    rng = np.random.default_rng(0)
    idx = rng.choice(len(emb), size=min(n_vis, len(emb)), replace=False)
    means = emb[idx].mean(axis=1)
    fates_sub = fate[idx]
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(means)
    ev = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(BG)
    styled_ax(ax, f"PCA of Trajectory Mean Embeddings  (var: {ev[0]:.1%}, {ev[1]:.1%})")
    for fi in np.unique(fates_sub):
        mask = fates_sub == fi
        label = fate_names[fi] if fi < len(fate_names) else str(fi)
        ax.scatter(coords[mask, 0], coords[mask, 1], s=6, alpha=0.6,
                   color=PALETTE[fi % len(PALETTE)], label=label)
    ax.legend(facecolor="#1c1f26", edgecolor="#444", labelcolor=TEXT_C, fontsize=9,
              markerscale=2)
    ax.set_xlabel("PC1", color=TEXT_C, fontsize=9)
    ax.set_ylabel("PC2", color=TEXT_C, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"  Saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--ckpt", default=os.path.join(RES, "encoder.pt"))
    ap.add_argument("--config", default=os.path.join(RES, "train_config.json"))
    ap.add_argument("--history", default=os.path.join(RES, "train_history.json"))
    ap.add_argument("--n-vis", type=int, default=5000)
    ap.add_argument("--n-traj-paths", type=int, default=40)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.config) as f:
        cfg = json.load(f)

    model = TrajectoryEncoder(cfg["latent_dim"], cfg["proj_dim"]).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    print(f"Loaded encoder (latent={cfg['latent_dim']}, proj={cfg['proj_dim']})")

    d = np.load(args.data)
    frames     = d["frames"]       # (N, T, H, W)
    fate       = d["fate"]
    fate_names = list(d["fate_names"])
    print(f"Embedding {len(frames):,} trajectories …")
    emb = embed_all(model, frames, device)   # (N, T, D)
    print(f"  embeddings shape: {emb.shape}")

    os.makedirs(RES, exist_ok=True)

    with open(args.history) as f:
        history = json.load(f)

    print("Generating plots …")
    plot_loss_curve(history, os.path.join(RES, "eval_loss_curve.png"))
    plot_traj_paths(emb, fate, fate_names, args.n_traj_paths,
                    os.path.join(RES, "eval_traj_paths.png"))
    plot_intra_dist(emb, fate, fate_names,
                    os.path.join(RES, "eval_intra_dist.png"))
    plot_similarity_matrix(emb, fate, fate_names,
                           os.path.join(RES, "eval_similarity_mat.png"))
    plot_umap(emb, fate, fate_names,
              os.path.join(RES, "eval_umap.png"), args.n_vis)

    print("\nDone.")


if __name__ == "__main__":
    main()
