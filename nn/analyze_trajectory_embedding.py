"""
Trajectory Embedding Analysis
------------------------------
Loads the two trained TrajectoryTransformer checkpoints (d_model=64 and 128),
extracts embeddings for the full dataset, and produces:

  1. UMAP scatter plots — coloured by:
       a) grid type   (random / single / dual / triple)
       b) pattern category (random / still_life / oscillator / spaceship / methuselah / mixed)
       c) fate class  (0=dies / 1=still life / 2=oscillator / 3=active)
       d) population  (mean number of live cells across the trajectory)

  2. Embedding distance vs divergence — for each (baseline, perturbed) pair from
     the perturbation analysis, compare ||emb(baseline) - emb(perturbed)|| to
     final divergence and plot the correlation.

  3. t-SNE alternative if requested.

Outputs → nn/results/traj_emb_d<N>_umap_*.png
          nn/results/traj_emb_d<N>_dist_vs_div.png
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("[warn] umap-learn not installed — falling back to PCA for 2-D projection")

from nn.trajectory_data  import TrajectoryDataset, run_trajectory, PATTERN_CATEGORY
from nn.models            import TrajectoryTransformer
from nn.data_gen          import classify_fate
from nn.utils             import CKPT_DIR, RESULTS_DIR, DEVICE

# ── Config ─────────────────────────────────────────────────────────────────────

GRID_SIZE  = 40
T          = 60
N_SAMPLES  = 5000
SEED       = 42
D_MODELS   = [64, 128]
BATCH_SIZE = 32

FATE_NAMES     = {0: "dies", 1: "still_life", 2: "oscillator", 3: "active"}
FATE_COLORS    = {0: "#888888", 1: "#2ecc71", 2: "#3498db", 3: "#e74c3c"}
TYPE_COLORS    = {"random": "#e67e22", "single": "#9b59b6",
                  "dual": "#3498db",   "triple": "#2ecc71"}
CAT_COLORS     = {
    "random":      "#e67e22",
    "still_life":  "#2ecc71",
    "oscillator":  "#3498db",
    "spaceship":   "#e74c3c",
    "methuselah":  "#9b59b6",
    "mixed":       "#95a5a6",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _embed_all(model: TrajectoryTransformer,
               inits: list[np.ndarray]) -> np.ndarray:
    """Extract CLS embeddings for all initial grids."""
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(inits), BATCH_SIZE):
            batch_inits = inits[i:i + BATCH_SIZE]
            trajs = np.stack([run_trajectory(g, T) for g in batch_inits])  # (B, T+1, H, W)
            x   = torch.from_numpy(trajs).float().to(DEVICE)
            emb = model.encode(x)                                           # (B, d)
            embeddings.append(emb.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def _reduce_2d(embeddings: np.ndarray, method: str = "umap") -> np.ndarray:
    if method == "umap" and HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=SEED,
                            n_neighbors=30, min_dist=0.1)
        return reducer.fit_transform(embeddings)
    else:
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=SEED).fit_transform(embeddings)


def _scatter(ax, xy: np.ndarray, colors: list[str], title: str,
             legend_handles=None, alpha: float = 0.5, s: int = 12):
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=s, alpha=alpha, linewidths=0)
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=7, loc="best",
                  markerscale=1.5, framealpha=0.7)


def _patch(label, color):
    return mpatches.Patch(color=color, label=label)


# ── UMAP panel plots ───────────────────────────────────────────────────────────

def plot_umap_panels(d_model: int, xy: np.ndarray, meta: list[dict],
                     fate_labels: np.ndarray, populations: np.ndarray):
    method = "UMAP" if HAS_UMAP else "PCA"
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"TrajectoryTransformer d_model={d_model} — {method} of embeddings",
                 fontweight="bold", fontsize=13)

    # (a) grid type
    type_c = [TYPE_COLORS[m["type"]] for m in meta]
    handles = [_patch(k, v) for k, v in TYPE_COLORS.items()]
    _scatter(axes[0, 0], xy, type_c, "(a) Grid type", handles)

    # (b) pattern category
    cat_c = [CAT_COLORS.get(m["category"], "#95a5a6") for m in meta]
    handles_cat = [_patch(k, v) for k, v in CAT_COLORS.items()]
    _scatter(axes[0, 1], xy, cat_c, "(b) Pattern category", handles_cat)

    # (c) fate class
    fate_c = [FATE_COLORS[int(f)] for f in fate_labels]
    handles_fate = [_patch(FATE_NAMES[k], v) for k, v in FATE_COLORS.items()]
    _scatter(axes[1, 0], xy, fate_c, "(c) Fate class", handles_fate)

    # (d) mean population (continuous colour)
    sc = axes[1, 1].scatter(xy[:, 0], xy[:, 1], c=populations,
                             cmap="viridis", s=12, alpha=0.5, linewidths=0)
    plt.colorbar(sc, ax=axes[1, 1], label="Mean live cells")
    axes[1, 1].set_title("(d) Mean population", fontweight="bold", fontsize=10)
    axes[1, 1].set_xticks([]); axes[1, 1].set_yticks([])

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"traj_emb_d{d_model}_umap.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  UMAP panel  →  {out}")


# ── Embedding distance vs perturbation divergence ──────────────────────────────

def plot_distance_vs_divergence(d_model: int, model: TrajectoryTransformer,
                                n_pairs: int = 500, rng_seed: int = 0):
    """
    Sample `n_pairs` random GoL initial states, apply a random single-cell flip,
    compare embedding distance to final-step divergence.
    """
    from nn.data_gen import gol_step as _step

    rng  = np.random.default_rng(rng_seed)
    emb_dists, final_divs = [], []

    model.eval()
    with torch.no_grad():
        for _ in range(n_pairs):
            init = (rng.random((GRID_SIZE, GRID_SIZE)) < 0.35).astype(np.uint8)
            r, c = int(rng.integers(0, GRID_SIZE)), int(rng.integers(0, GRID_SIZE))
            pert = init.copy(); pert[r, c] ^= 1

            def _traj(g):
                t = run_trajectory(g, T)
                return torch.from_numpy(t).float().unsqueeze(0).to(DEVICE)

            emb_b = model.encode(_traj(init))[0].cpu().numpy()
            emb_p = model.encode(_traj(pert))[0].cpu().numpy()
            emb_dists.append(float(np.linalg.norm(emb_b - emb_p)))

            traj_b = run_trajectory(init, T)
            traj_p = run_trajectory(pert, T)
            final_divs.append(int((traj_b[-1] != traj_p[-1]).sum()))

    emb_dists  = np.array(emb_dists)
    final_divs = np.array(final_divs)
    r_val, p_val = pearsonr(emb_dists, final_divs)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(emb_dists, final_divs, s=8, alpha=0.4, color="#3498db")
    ax.set_xlabel("Embedding distance  ||emb(baseline) − emb(perturbed)||")
    ax.set_ylabel("Final divergence (cells differing at t=T)")
    ax.set_title(
        f"d_model={d_model} — Embedding distance vs divergence\n"
        f"Pearson r = {r_val:.3f}  (p = {p_val:.2e})",
        fontweight="bold",
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"traj_emb_d{d_model}_dist_vs_div.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Dist-vs-div →  {out}  (r={r_val:.3f})")
    return r_val


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("[data] Building dataset …")
    ds = TrajectoryDataset(N_SAMPLES, GRID_SIZE, T, random_frac=0.4, seed=SEED)
    meta  = ds.meta
    inits = ds.inits

    print("[data] Computing fate labels (this may take a minute) …")
    fate_labels  = np.array([classify_fate(g, steps=100) for g in inits])
    populations  = np.array([
        run_trajectory(g, T).mean(axis=(1, 2)).mean()
        for g in inits
    ])
    print(f"  Fate distribution: { {k: int((fate_labels==k).sum()) for k in range(4)} }")

    for d_model in D_MODELS:
        ckpt = os.path.join(CKPT_DIR, f"traj_emb_d{d_model}_best.pt")
        if not os.path.exists(ckpt):
            print(f"[skip] Checkpoint not found: {ckpt} — run train_trajectory_embedding.py first")
            continue

        print(f"\n── d_model={d_model} ──────────────────────────────────────")
        model = TrajectoryTransformer(
            d_model=d_model, nhead=4, num_layers=4,
            grid_size=GRID_SIZE, T=T,
        ).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))

        print("  Extracting embeddings …")
        embeddings = _embed_all(model, inits)    # (N, d_model)
        print(f"  Embeddings shape: {embeddings.shape}")

        print("  Running UMAP / PCA …")
        xy = _reduce_2d(embeddings)

        plot_umap_panels(d_model, xy, meta, fate_labels, populations)
        plot_distance_vs_divergence(d_model, model)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
