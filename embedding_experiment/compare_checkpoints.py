"""
compare_checkpoints.py — Side-by-side GIF: barely_trained vs best_val.

For each seed, generates a 3-panel GIF:
  Left   — GoL grid
  Center — embedding visualization from checkpoint A (barely_trained)
  Right  — embedding visualization from checkpoint B (best_val)

Two sets:
  comparison_pca_<seed>.gif     — PCA trajectory side by side
  comparison_raw_<seed>.gif     — raw 8x8 heatmap side by side

Plus combined versions with all seeds stacked.

Usage:
    python compare_checkpoints.py [--steps 80] [--fps 10]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
from sklearn.decomposition import PCA

from model import TrajectoryEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
BG = "#0f1117"
TEXT_C = "white"
SEED_COLORS = ["#3a7bd5", "#2ecc71", "#e74c3c", "#f39c12",
               "#9b59b6", "#1abc9c", "#e67e22"]


# ── GoL ───────────────────────────────────────────────────────────────────────

def gol_step(cells):
    n = sum(np.roll(np.roll(cells, i, 0), j, 1)
            for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0))
    return ((n == 3) | (cells & (n == 2))).astype(np.uint8)


# ── Seeds ─────────────────────────────────────────────────────────────────────

def _place(size, pat, r, c):
    g = np.zeros((size, size), np.uint8)
    g[r:r+pat.shape[0], c:c+pat.shape[1]] = pat
    return g

def _make_blinkers(size=64):
    g = np.zeros((size, size), np.uint8)
    for r, c in [(10,10),(10,40),(40,10),(40,40)]:
        g[r, c:c+3] = 1
    return g

GLIDER = np.array([[0,1,0],[0,0,1],[1,1,1]], np.uint8)
LWSS   = np.array([[0,1,0,0,1],[1,0,0,0,0],[1,0,0,0,1],[1,1,1,1,0]], np.uint8)
PULSAR = np.array([
    [0,0,1,1,1,0,0,0,1,1,1,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,0,1,0,1,0,0,0,0,1],[1,0,0,0,0,1,0,1,0,0,0,0,1],
    [1,0,0,0,0,1,0,1,0,0,0,0,1],[0,0,1,1,1,0,0,0,1,1,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,1,1,1,0,0,0,1,1,1,0,0],
    [1,0,0,0,0,1,0,1,0,0,0,0,1],[1,0,0,0,0,1,0,1,0,0,0,0,1],
    [1,0,0,0,0,1,0,1,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,1,0,0,0,1,1,1,0,0],
], np.uint8)
R_PENT = np.array([[0,1,1],[1,1,0],[0,1,0]], np.uint8)

def make_seeds(size=64):
    rng  = np.random.default_rng(7)
    s1 = (rng.random((size,size)) < 0.30).astype(np.uint8)
    for _ in range(5): s1 = gol_step(s1)
    rng2 = np.random.default_rng(99)
    s2 = (rng2.random((size,size)) < 0.40).astype(np.uint8)
    for _ in range(3): s2 = gol_step(s2)
    return {
        "glider":      _place(size, GLIDER, 2, 2),
        "lwss":        _place(size, LWSS, 4, 2),
        "blinker×4":   _make_blinkers(size),
        "pulsar":      _place(size, PULSAR, 25, 25),
        "r_pentomino": _place(size, R_PENT, 30, 30),
        "soup_A":      s1,
        "soup_B":      s2,
    }


# ── Encode ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_frame(model, grid, device):
    x = torch.from_numpy(grid).float().unsqueeze(0).unsqueeze(0).to(device)
    return model.encode(x).squeeze(0).detach().cpu().numpy()

@torch.no_grad()
def encode_bg(model, data_path, device, n=6000, batch=512):
    d = np.load(data_path)
    frames = d["frames"]
    idx = np.random.default_rng(0).choice(len(frames), min(n, len(frames)), replace=False)
    sample = frames[idx, -1]
    out = []
    x = torch.from_numpy(sample).float().unsqueeze(1)
    for i in range(0, len(x), batch):
        out.append(model.encode(x[i:i+batch].to(device)).detach().cpu().numpy())
    return np.concatenate(out)

def precompute(model, init_grid, device, steps):
    grids, embs = [], []
    g = init_grid.copy()
    for _ in range(steps):
        grids.append(g.copy())
        embs.append(encode_frame(model, g, device))
        g = gol_step(g)
    return grids, np.stack(embs)


# ── PCA comparison GIF ────────────────────────────────────────────────────────

def make_pca_comparison(name, init_grid, model_a, model_b, pca_a, pca_b,
                         bg_a, bg_b, device, steps, fps, out_path, color):
    grids, embs_a = precompute(model_a, init_grid, device, steps)
    _,     embs_b = precompute(model_b, init_grid, device, steps)
    traj_a = pca_a.transform(embs_a)
    traj_b = pca_b.transform(embs_b)
    cmap_colors = [plt.cm.plasma(t / max(steps-1,1)) for t in range(steps)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT_C, labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#333")

    ax_gol, ax_a, ax_b = axes

    # GoL
    ax_gol.axis("off")
    im_gol = ax_gol.imshow(grids[0], cmap="binary", vmin=0, vmax=1,
                            interpolation="nearest")
    title_gol = ax_gol.set_title(f"{name}  t=0", color=TEXT_C,
                                   fontsize=10, fontweight="bold")

    # Embedding A
    pad_a = (bg_a.max(0) - bg_a.min(0)) * 0.05
    ax_a.scatter(bg_a[:,0], bg_a[:,1], s=1.5, c="#333", alpha=0.4,
                 rasterized=True, zorder=1)
    ax_a.set_xlim(bg_a[:,0].min()-pad_a[0], bg_a[:,0].max()+pad_a[0])
    ax_a.set_ylim(bg_a[:,1].min()-pad_a[1], bg_a[:,1].max()+pad_a[1])
    ax_a.set_title("Barely Trained (epoch 1)", color="#f39c12",
                    fontsize=9, fontweight="bold")
    ax_a.set_xlabel("PC1", color=TEXT_C, fontsize=8)
    ax_a.set_ylabel("PC2", color=TEXT_C, fontsize=8)
    line_a, = ax_a.plot([], [], lw=1.5, alpha=0.8, color=color, zorder=2)
    dot_a = ax_a.scatter([], [], s=80, color="white", zorder=4,
                          edgecolors=color, linewidths=1.5)

    # Embedding B
    pad_b = (bg_b.max(0) - bg_b.min(0)) * 0.05
    ax_b.scatter(bg_b[:,0], bg_b[:,1], s=1.5, c="#333", alpha=0.4,
                 rasterized=True, zorder=1)
    ax_b.set_xlim(bg_b[:,0].min()-pad_b[0], bg_b[:,0].max()+pad_b[0])
    ax_b.set_ylim(bg_b[:,1].min()-pad_b[1], bg_b[:,1].max()+pad_b[1])
    ax_b.set_title("Best Val (epoch 22)", color="#2ecc71",
                    fontsize=9, fontweight="bold")
    ax_b.set_xlabel("PC1", color=TEXT_C, fontsize=8)
    ax_b.set_ylabel("PC2", color=TEXT_C, fontsize=8)
    line_b, = ax_b.plot([], [], lw=1.5, alpha=0.8, color=color, zorder=2)
    dot_b = ax_b.scatter([], [], s=80, color="white", zorder=4,
                          edgecolors=color, linewidths=1.5)

    fig.tight_layout(pad=1.0)

    def update(t):
        im_gol.set_data(grids[t])
        title_gol.set_text(f"{name}  t={t}")
        line_a.set_data(traj_a[:t+1,0], traj_a[:t+1,1])
        line_a.set_color(cmap_colors[t])
        dot_a.set_offsets(traj_a[t])
        line_b.set_data(traj_b[:t+1,0], traj_b[:t+1,1])
        line_b.set_color(cmap_colors[t])
        dot_b.set_offsets(traj_b[t])
        return im_gol, line_a, dot_a, line_b, dot_b, title_gol

    ani = animation.FuncAnimation(fig, update, frames=steps,
                                  interval=1000//fps, blit=True)
    ani.save(out_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"  Saved → {os.path.basename(out_path)}")


# ── Raw embedding comparison GIF ──────────────────────────────────────────────

def make_raw_comparison(name, init_grid, model_a, model_b, device,
                         steps, fps, out_path, vmin_a, vmax_a, vmin_b, vmax_b):
    grids, embs_a = precompute(model_a, init_grid, device, steps)
    _,     embs_b = precompute(model_b, init_grid, device, steps)
    side = 8

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(BG)

    ax_gol, ax_a, ax_b = axes
    ax_gol.axis("off")
    ax_a.axis("off")
    ax_b.axis("off")

    im_gol = ax_gol.imshow(grids[0], cmap="binary", vmin=0, vmax=1,
                            interpolation="nearest")
    title_gol = ax_gol.set_title(f"{name}  t=0", color=TEXT_C,
                                   fontsize=10, fontweight="bold")

    im_a = ax_a.imshow(embs_a[0].reshape(side, side), cmap="RdBu_r",
                        vmin=vmin_a, vmax=vmax_a, interpolation="nearest",
                        aspect="equal")
    ax_a.set_title("Barely Trained (epoch 1)", color="#f39c12",
                    fontsize=9, fontweight="bold")
    cb_a = fig.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04)
    cb_a.ax.tick_params(colors=TEXT_C, labelsize=7)

    im_b = ax_b.imshow(embs_b[0].reshape(side, side), cmap="RdBu_r",
                        vmin=vmin_b, vmax=vmax_b, interpolation="nearest",
                        aspect="equal")
    ax_b.set_title("Best Val (epoch 22)", color="#2ecc71",
                    fontsize=9, fontweight="bold")
    cb_b = fig.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
    cb_b.ax.tick_params(colors=TEXT_C, labelsize=7)

    # dim labels
    for ax, in zip([ax_a, ax_b],):
        for i in range(side):
            for j in range(side):
                ax.text(j, i, str(i*side+j), ha="center", va="center",
                        color="white", fontsize=5, alpha=0.45)

    fig.tight_layout(pad=1.0)

    def update(t):
        im_gol.set_data(grids[t])
        title_gol.set_text(f"{name}  t={t}")
        im_a.set_data(embs_a[t].reshape(side, side))
        im_b.set_data(embs_b[t].reshape(side, side))
        return im_gol, im_a, im_b, title_gol

    ani = animation.FuncAnimation(fig, update, frames=steps,
                                  interval=1000//fps, blit=True)
    ani.save(out_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"  Saved → {os.path.basename(out_path)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-a", default=os.path.join(HERE, "results", "encoder_epoch001.pt"))
    ap.add_argument("--ckpt-b", default=os.path.join(HERE, "results", "encoder_best_val.pt"))
    ap.add_argument("--steps",  type=int, default=80)
    ap.add_argument("--fps",    type=int, default=10)
    ap.add_argument("--data",   default=os.path.join(HERE, "data", "trajectories.npz"))
    ap.add_argument("--config", default=os.path.join(HERE, "results", "train_config.json"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results", "comparison"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    with open(args.config) as f:
        cfg = json.load(f)

    model_a = TrajectoryEncoder(cfg["latent_dim"], cfg["proj_dim"]).to(device)
    model_a.load_state_dict(torch.load(args.ckpt_a, map_location=device))
    model_a.eval()

    model_b = TrajectoryEncoder(cfg["latent_dim"], cfg["proj_dim"]).to(device)
    model_b.load_state_dict(torch.load(args.ckpt_b, map_location=device))
    model_b.eval()

    seeds  = make_seeds()
    names  = list(seeds.keys())
    grids  = [seeds[n] for n in names]
    colors = SEED_COLORS[:len(names)]

    print("Building PCA backgrounds …")
    bg_emb_a = encode_bg(model_a, args.data, device)
    bg_emb_b = encode_bg(model_b, args.data, device)
    pca_a = PCA(n_components=2, random_state=0); bg_a = pca_a.fit_transform(bg_emb_a)
    pca_b = PCA(n_components=2, random_state=0); bg_b = pca_b.fit_transform(bg_emb_b)

    print("Computing embedding value ranges …")
    vals_a, vals_b = [], []
    for init in grids:
        g = init.copy()
        for _ in range(args.steps):
            vals_a.append(encode_frame(model_a, g, device))
            vals_b.append(encode_frame(model_b, g, device))
            g = gol_step(g)
    vabs_a = np.abs(np.stack(vals_a)).max()
    vabs_b = np.abs(np.stack(vals_b)).max()
    vmin_a, vmax_a = -vabs_a, vabs_a
    vmin_b, vmax_b = -vabs_b, vabs_b
    print(f"  barely_trained range: ±{vabs_a:.2f}")
    print(f"  best_val range:       ±{vabs_b:.2f}")

    print("\nPCA comparison GIFs …")
    for name, grid, color in zip(names, grids, colors):
        out = os.path.join(args.out_dir, f"comparison_pca_{name.replace(' ','_')}.gif")
        make_pca_comparison(name, grid, model_a, model_b, pca_a, pca_b,
                             bg_a, bg_b, device, args.steps, args.fps, out, color)

    print("\nRaw embedding comparison GIFs …")
    for name, grid in zip(names, grids):
        out = os.path.join(args.out_dir, f"comparison_raw_{name.replace(' ','_')}.gif")
        make_raw_comparison(name, grid, model_a, model_b, device,
                             args.steps, args.fps, out,
                             vmin_a, vmax_a, vmin_b, vmax_b)

    print(f"\nAll saved to {args.out_dir}")


if __name__ == "__main__":
    main()
