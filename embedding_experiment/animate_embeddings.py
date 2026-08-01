"""
animate_embeddings.py — Multi-panel GIF animations of GoL evolution + embedding trajectory.

For each seed pattern, evolves the GoL grid for N steps, encodes every frame
through the trained contrastive encoder, projects to 2D via PCA (fitted on a
background sample from the training data), and saves a GIF with:

  Left panel   — GoL grid at current step
  Right panel  — PCA embedding space:
                   • background: all training trajectory endpoints (gray)
                   • current seed's path coloured by time (plasma)
                   • current position: bright dot

One combined "all seeds" GIF is also saved showing all seed trajectories
animated together on a shared embedding canvas.

Usage:
    python animate_embeddings.py [--steps 80] [--fps 10] [--out-dir results/]
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


# ── GoL engine ────────────────────────────────────────────────────────────────

def gol_step(cells):
    n = sum(
        np.roll(np.roll(cells, i, 0), j, 1)
        for i in (-1, 0, 1) for j in (-1, 0, 1)
        if (i, j) != (0, 0)
    )
    return ((n == 3) | (cells & (n == 2))).astype(np.uint8)


# ── Initial conditions ────────────────────────────────────────────────────────

def _place(size, pattern, r, c):
    g = np.zeros((size, size), np.uint8)
    h, w = pattern.shape
    g[r:r+h, c:c+w] = pattern
    return g


GLIDER = np.array([[0,1,0],[0,0,1],[1,1,1]], np.uint8)
LWSS   = np.array([[0,1,0,0,1],[1,0,0,0,0],[1,0,0,0,1],[1,1,1,1,0]], np.uint8)
BLINKER = np.array([[1,1,1]], np.uint8)
PULSAR = np.array([
    [0,0,1,1,1,0,0,0,1,1,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,0,1,0,1,0,0,0,0,1],
    [1,0,0,0,0,1,0,1,0,0,0,0,1],
    [1,0,0,0,0,1,0,1,0,0,0,0,1],
    [0,0,1,1,1,0,0,0,1,1,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,1,0,0,0,1,1,1,0,0],
    [1,0,0,0,0,1,0,1,0,0,0,0,1],
    [1,0,0,0,0,1,0,1,0,0,0,0,1],
    [1,0,0,0,0,1,0,1,0,0,0,0,1],
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,1,0,0,0,1,1,1,0,0],
], np.uint8)
R_PENT = np.array([[0,1,1],[1,1,0],[0,1,0]], np.uint8)


def _make_blinkers(size=64):
    g = np.zeros((size, size), np.uint8)
    for r, c in [(10, 10), (10, 40), (40, 10), (40, 40)]:
        g[r, c:c+3] = 1
    return g


def make_seeds(size=64):
    rng = np.random.default_rng(7)
    soup = (rng.random((size, size)) < 0.30).astype(np.uint8)
    for _ in range(5):
        soup = gol_step(soup)

    rng2 = np.random.default_rng(99)
    soup2 = (rng2.random((size, size)) < 0.40).astype(np.uint8)
    for _ in range(3):
        soup2 = gol_step(soup2)

    seeds = {
        "glider":      _place(size, GLIDER, 2, 2),
        "lwss":        _place(size, LWSS, 4, 2),
        "blinker×4":   _make_blinkers(size),
        "pulsar":      _place(size, PULSAR, 25, 25),
        "r_pentomino": _place(size, R_PENT, 30, 30),
        "soup_A":      soup,
        "soup_B":      soup2,
    }
    return seeds


# ── Encoding ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_sequence(model, grids, device):
    """grids: list of (H,W) uint8 → (T, D) float32"""
    x = torch.from_numpy(np.stack(grids)).float().unsqueeze(1).to(device)
    return model.encode(x).cpu().numpy()


@torch.no_grad()
def encode_dataset_sample(model, data_path, device, n=8000, batch=512):
    """Encode n random frames from the training dataset for PCA background."""
    d = np.load(data_path)
    frames = d["frames"]                   # (N, T, H, W)
    N, T, H, W = frames.shape
    rng = np.random.default_rng(0)
    traj_idx = rng.choice(N, size=min(n, N), replace=False)
    # use the last frame of each sampled trajectory as background point
    sample = frames[traj_idx, -1]          # (n, H, W)
    out = []
    x = torch.from_numpy(sample).float().unsqueeze(1)
    for i in range(0, len(x), batch):
        out.append(model.encode(x[i:i+batch].to(device)).cpu().numpy())
    return np.concatenate(out)             # (n, D)


# ── Single-seed GIF ───────────────────────────────────────────────────────────

def make_seed_gif(name, init_grid, model, pca, bg_2d, device,
                  steps, fps, out_path, seed_color):
    grids, embs = [], []
    g = init_grid.copy()
    for _ in range(steps):
        grids.append(g.copy())
        embs.append(model.encode(
            torch.from_numpy(g).float().unsqueeze(0).unsqueeze(0).to(device)
        ).squeeze(0).detach().cpu().numpy())
        g = gol_step(g)

    traj = pca.transform(np.stack(embs))  # (T, 2)
    cmap = plt.cm.plasma
    colors = [cmap(t / max(steps - 1, 1)) for t in range(steps)]

    fig, (ax_gol, ax_emb) = plt.subplots(1, 2, figsize=(10, 5))
    fig.patch.set_facecolor(BG)
    for ax in (ax_gol, ax_emb):
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT_C, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

    # background
    ax_emb.scatter(bg_2d[:, 0], bg_2d[:, 1], s=1.5, c="#333", alpha=0.4,
                   rasterized=True, zorder=1)
    ax_emb.set_xlabel("PC1", color=TEXT_C, fontsize=8)
    ax_emb.set_ylabel("PC2", color=TEXT_C, fontsize=8)

    im = ax_gol.imshow(grids[0], cmap="binary", vmin=0, vmax=1,
                       interpolation="nearest")
    ax_gol.axis("off")
    title_gol = ax_gol.set_title(f"{name}  t=0", color=TEXT_C, fontsize=10,
                                  fontweight="bold")
    ax_emb.set_title("Contrastive Embedding (PCA)", color=TEXT_C, fontsize=10,
                      fontweight="bold")

    line, = ax_emb.plot([], [], lw=1.5, alpha=0.8, color=seed_color, zorder=2)
    dot = ax_emb.scatter([], [], s=80, color="white", zorder=4,
                         edgecolors=seed_color, linewidths=1.5)

    # fix axes limits
    pad = (bg_2d.max(0) - bg_2d.min(0)) * 0.05
    ax_emb.set_xlim(bg_2d[:, 0].min() - pad[0], bg_2d[:, 0].max() + pad[0])
    ax_emb.set_ylim(bg_2d[:, 1].min() - pad[1], bg_2d[:, 1].max() + pad[1])

    fig.tight_layout(pad=0.8)

    def update(t):
        im.set_data(grids[t])
        title_gol.set_text(f"{name}  t={t}")
        line.set_data(traj[:t+1, 0], traj[:t+1, 1])
        line.set_color(colors[t])
        dot.set_offsets(traj[t])
        return im, line, dot, title_gol

    ani = animation.FuncAnimation(fig, update, frames=steps,
                                  interval=1000 // fps, blit=True)
    ani.save(out_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return traj


# ── Combined multi-seed GIF ───────────────────────────────────────────────────

def make_combined_gif(seed_names, seed_grids, seed_colors, model, pca, bg_2d,
                      device, steps, fps, out_path):
    # pre-compute all trajectories
    all_grids, all_trajs = [], []
    for name, init in zip(seed_names, seed_grids):
        grids, embs = [], []
        g = init.copy()
        for _ in range(steps):
            grids.append(g.copy())
            embs.append(model.encode(
                torch.from_numpy(g).float().unsqueeze(0).unsqueeze(0).to(device)
            ).squeeze(0).detach().cpu().numpy())
            g = gol_step(g)
        all_grids.append(grids)
        all_trajs.append(pca.transform(np.stack(embs)))

    n_seeds = len(seed_names)
    ncols = min(4, n_seeds)
    nrows = (n_seeds + ncols - 1) // ncols
    fig_w = ncols * 2.8 + 4.5
    fig_h = max(nrows * 2.8, 5.0)

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG)
    # GoL axes: left grid
    gol_axes = []
    for i in range(n_seeds):
        row, col = divmod(i, ncols)
        ax = fig.add_axes([col / (ncols + 1.6) * (ncols / (ncols + 1.6)),
                           1 - (row + 1) / nrows * 0.92,
                           0.85 / (ncols + 1.6),
                           0.85 / nrows])
        ax.set_facecolor(BG)
        ax.axis("off")
        gol_axes.append(ax)

    # Shared embedding axis: right side
    ax_emb = fig.add_axes([0.62, 0.08, 0.36, 0.84])
    ax_emb.set_facecolor(BG)
    ax_emb.tick_params(colors=TEXT_C, labelsize=7)
    for sp in ax_emb.spines.values():
        sp.set_edgecolor("#333")
    ax_emb.set_title("Embedding space (PCA)", color=TEXT_C, fontsize=9,
                      fontweight="bold")
    ax_emb.set_xlabel("PC1", color=TEXT_C, fontsize=8)
    ax_emb.set_ylabel("PC2", color=TEXT_C, fontsize=8)

    pad = (bg_2d.max(0) - bg_2d.min(0)) * 0.05
    ax_emb.set_xlim(bg_2d[:, 0].min() - pad[0], bg_2d[:, 0].max() + pad[0])
    ax_emb.set_ylim(bg_2d[:, 1].min() - pad[1], bg_2d[:, 1].max() + pad[1])
    ax_emb.scatter(bg_2d[:, 0], bg_2d[:, 1], s=1, c="#2a2a2a", alpha=0.5,
                   rasterized=True, zorder=1)

    # per-seed: GoL image artist + embedding line + dot
    ims, lines, dots, titles = [], [], [], []
    for i, (name, color) in enumerate(zip(seed_names, seed_colors)):
        im = gol_axes[i].imshow(all_grids[i][0], cmap="binary",
                                 vmin=0, vmax=1, interpolation="nearest")
        t = gol_axes[i].set_title(f"{name}\nt=0", color=TEXT_C, fontsize=7,
                                    fontweight="bold", pad=2)
        ln, = ax_emb.plot([], [], lw=1.2, alpha=0.85, color=color, zorder=2)
        dt = ax_emb.scatter([], [], s=50, color=color, zorder=4,
                             edgecolors="white", linewidths=0.8)
        ims.append(im); lines.append(ln); dots.append(dt); titles.append(t)

    # legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=c, lw=2, label=n)
               for n, c in zip(seed_names, seed_colors)]
    ax_emb.legend(handles=handles, facecolor="#1c1f26", edgecolor="#444",
                  labelcolor=TEXT_C, fontsize=7, loc="upper left",
                  framealpha=0.85)

    step_text = fig.text(0.01, 0.01, "t=0", color=TEXT_C, fontsize=8)

    def update(t):
        artists = []
        for i in range(n_seeds):
            ims[i].set_data(all_grids[i][t])
            titles[i].set_text(f"{seed_names[i]}\nt={t}")
            lines[i].set_data(all_trajs[i][:t+1, 0], all_trajs[i][:t+1, 1])
            dots[i].set_offsets(all_trajs[i][t])
            artists += [ims[i], lines[i], dots[i], titles[i]]
        step_text.set_text(f"t={t}")
        artists.append(step_text)
        return artists

    ani = animation.FuncAnimation(fig, update, frames=steps,
                                  interval=1000 // fps, blit=True)
    ani.save(out_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

SEED_COLORS = ["#3a7bd5", "#2ecc71", "#e74c3c", "#f39c12",
               "#9b59b6", "#1abc9c", "#e67e22"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--data", default=os.path.join(HERE, "data", "trajectories.npz"))
    ap.add_argument("--ckpt", default=os.path.join(HERE, "results", "encoder.pt"))
    ap.add_argument("--config", default=os.path.join(HERE, "results", "train_config.json"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    with open(args.config) as f:
        cfg = json.load(f)
    model = TrajectoryEncoder(cfg["latent_dim"], cfg["proj_dim"]).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    print(f"Loaded encoder (latent={cfg['latent_dim']})")

    print("Building PCA background from training data …")
    bg_emb = encode_dataset_sample(model, args.data, device, n=8000)
    pca = PCA(n_components=2, random_state=0)
    bg_2d = pca.fit_transform(bg_emb)
    ev = pca.explained_variance_ratio_
    print(f"  PCA explained variance: {ev[0]:.1%}, {ev[1]:.1%}")

    seeds = make_seeds()
    names  = list(seeds.keys())
    grids  = [seeds[n] for n in names]
    colors = SEED_COLORS[:len(names)]

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\nGenerating {len(names)} individual GIFs ({args.steps} steps each) …")
    for name, grid, color in zip(names, grids, colors):
        out = os.path.join(args.out_dir, f"anim_{name.replace(' ', '_')}.gif")
        make_seed_gif(name, grid, model, pca, bg_2d, device,
                      args.steps, args.fps, out, color)

    print("\nGenerating combined GIF …")
    make_combined_gif(names, grids, colors, model, pca, bg_2d, device,
                      args.steps, args.fps,
                      os.path.join(args.out_dir, "anim_combined.gif"))

    print("\nDone.")


if __name__ == "__main__":
    main()
