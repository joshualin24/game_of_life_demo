"""
generate_checkpoint_gifs.py — Generate all GIF visualizations for a given checkpoint.

Produces the same PCA-embedding and raw-embedding GIFs as animate_embeddings.py
and animate_raw_embedding.py, but for any checkpoint, saved into a named subfolder.

Usage:
    python generate_checkpoint_gifs.py --ckpt results/encoder_epoch001.pt --label barely_trained
    python generate_checkpoint_gifs.py --ckpt results/encoder_best_val.pt  --label best_val
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


# ── GoL engine ────────────────────────────────────────────────────────────────

def gol_step(cells):
    n = sum(
        np.roll(np.roll(cells, i, 0), j, 1)
        for i in (-1, 0, 1) for j in (-1, 0, 1)
        if (i, j) != (0, 0)
    )
    return ((n == 3) | (cells & (n == 2))).astype(np.uint8)


# ── Seeds ─────────────────────────────────────────────────────────────────────

def _place(size, pattern, r, c):
    g = np.zeros((size, size), np.uint8)
    h, w = pattern.shape
    g[r:r+h, c:c+w] = pattern
    return g

def _make_blinkers(size=64):
    g = np.zeros((size, size), np.uint8)
    for r, c in [(10, 10), (10, 40), (40, 10), (40, 40)]:
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
    rng = np.random.default_rng(7)
    soup = (rng.random((size, size)) < 0.30).astype(np.uint8)
    for _ in range(5): soup = gol_step(soup)
    rng2 = np.random.default_rng(99)
    soup2 = (rng2.random((size, size)) < 0.40).astype(np.uint8)
    for _ in range(3): soup2 = gol_step(soup2)
    return {
        "glider":      _place(size, GLIDER, 2, 2),
        "lwss":        _place(size, LWSS, 4, 2),
        "blinker×4":   _make_blinkers(size),
        "pulsar":      _place(size, PULSAR, 25, 25),
        "r_pentomino": _place(size, R_PENT, 30, 30),
        "soup_A":      soup,
        "soup_B":      soup2,
    }


# ── Encode ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_frame(model, grid, device):
    x = torch.from_numpy(grid).float().unsqueeze(0).unsqueeze(0).to(device)
    return model.encode(x).squeeze(0).detach().cpu().numpy()

@torch.no_grad()
def encode_dataset_sample(model, data_path, device, n=8000, batch=512):
    d = np.load(data_path)
    frames = d["frames"]
    N = len(frames)
    rng = np.random.default_rng(0)
    idx = rng.choice(N, size=min(n, N), replace=False)
    sample = frames[idx, -1]
    out = []
    x = torch.from_numpy(sample).float().unsqueeze(1)
    for i in range(0, len(x), batch):
        out.append(model.encode(x[i:i+batch].to(device)).detach().cpu().numpy())
    return np.concatenate(out)


# ── PCA GIFs ──────────────────────────────────────────────────────────────────

def make_pca_gif(name, init_grid, model, pca, bg_2d, device,
                 steps, fps, out_path, color, label):
    grids, embs = [], []
    g = init_grid.copy()
    for _ in range(steps):
        grids.append(g.copy())
        embs.append(encode_frame(model, g, device))
        g = gol_step(g)
    traj = pca.transform(np.stack(embs))
    colors = [plt.cm.plasma(t / max(steps - 1, 1)) for t in range(steps)]

    fig, (ax_gol, ax_emb) = plt.subplots(1, 2, figsize=(10, 5))
    fig.patch.set_facecolor(BG)
    for ax in (ax_gol, ax_emb):
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT_C, labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#333")

    ax_emb.scatter(bg_2d[:, 0], bg_2d[:, 1], s=1.5, c="#333", alpha=0.4,
                   rasterized=True, zorder=1)
    ax_emb.set_xlabel("PC1", color=TEXT_C, fontsize=8)
    ax_emb.set_ylabel("PC2", color=TEXT_C, fontsize=8)

    im = ax_gol.imshow(grids[0], cmap="binary", vmin=0, vmax=1,
                       interpolation="nearest")
    ax_gol.axis("off")
    title_gol = ax_gol.set_title(f"{name}  t=0  [{label}]",
                                   color=TEXT_C, fontsize=9, fontweight="bold")
    ax_emb.set_title(f"Contrastive Embedding PCA  [{label}]",
                      color=TEXT_C, fontsize=9, fontweight="bold")

    line, = ax_emb.plot([], [], lw=1.5, alpha=0.8, color=color, zorder=2)
    dot = ax_emb.scatter([], [], s=80, color="white", zorder=4,
                          edgecolors=color, linewidths=1.5)
    pad = (bg_2d.max(0) - bg_2d.min(0)) * 0.05
    ax_emb.set_xlim(bg_2d[:, 0].min() - pad[0], bg_2d[:, 0].max() + pad[0])
    ax_emb.set_ylim(bg_2d[:, 1].min() - pad[1], bg_2d[:, 1].max() + pad[1])
    fig.tight_layout(pad=0.8)

    def update(t):
        im.set_data(grids[t])
        title_gol.set_text(f"{name}  t={t}  [{label}]")
        line.set_data(traj[:t+1, 0], traj[:t+1, 1])
        line.set_color(colors[t])
        dot.set_offsets(traj[t])
        return im, line, dot, title_gol

    ani = animation.FuncAnimation(fig, update, frames=steps,
                                  interval=1000 // fps, blit=True)
    ani.save(out_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"    {os.path.basename(out_path)}")


def make_pca_combined_gif(seed_names, seed_grids, seed_colors, model, pca,
                           bg_2d, device, steps, fps, out_path, label):
    all_grids, all_trajs = [], []
    for init in seed_grids:
        grids, embs = [], []
        g = init.copy()
        for _ in range(steps):
            grids.append(g.copy())
            embs.append(encode_frame(model, g, device))
            g = gol_step(g)
        all_grids.append(grids)
        all_trajs.append(pca.transform(np.stack(embs)))

    n = len(seed_names)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(ncols * 2.5 + 4.5, max(nrows * 2.8, 5.0)),
                     facecolor=BG)

    gol_axes = []
    col_w = 0.85 / (ncols + 1.6)
    for i in range(n):
        row, col = divmod(i, ncols)
        ax = fig.add_axes([col * col_w + 0.01,
                           1 - (row + 1) / nrows * 0.88,
                           col_w * 0.92, 0.80 / nrows])
        ax.set_facecolor(BG); ax.axis("off")
        gol_axes.append(ax)

    ax_emb = fig.add_axes([0.62, 0.08, 0.36, 0.84])
    ax_emb.set_facecolor(BG)
    ax_emb.tick_params(colors=TEXT_C, labelsize=7)
    for sp in ax_emb.spines.values(): sp.set_edgecolor("#333")
    ax_emb.set_title(f"Embedding PCA  [{label}]", color=TEXT_C, fontsize=9,
                      fontweight="bold")
    ax_emb.set_xlabel("PC1", color=TEXT_C, fontsize=8)
    ax_emb.set_ylabel("PC2", color=TEXT_C, fontsize=8)
    pad = (bg_2d.max(0) - bg_2d.min(0)) * 0.05
    ax_emb.set_xlim(bg_2d[:, 0].min() - pad[0], bg_2d[:, 0].max() + pad[0])
    ax_emb.set_ylim(bg_2d[:, 1].min() - pad[1], bg_2d[:, 1].max() + pad[1])
    ax_emb.scatter(bg_2d[:, 0], bg_2d[:, 1], s=1, c="#2a2a2a", alpha=0.5,
                   rasterized=True, zorder=1)

    ims, lines, dots, titles = [], [], [], []
    for i, (name, color) in enumerate(zip(seed_names, seed_colors)):
        im = gol_axes[i].imshow(all_grids[i][0], cmap="binary",
                                 vmin=0, vmax=1, interpolation="nearest")
        t = gol_axes[i].set_title(f"{name}\nt=0", color=TEXT_C,
                                    fontsize=7, fontweight="bold", pad=2)
        ln, = ax_emb.plot([], [], lw=1.2, alpha=0.85, color=color, zorder=2)
        dt = ax_emb.scatter([], [], s=50, color=color, zorder=4,
                             edgecolors="white", linewidths=0.8)
        ims.append(im); lines.append(ln); dots.append(dt); titles.append(t)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=c, lw=2, label=nm)
               for nm, c in zip(seed_names, seed_colors)]
    ax_emb.legend(handles=handles, facecolor="#1c1f26", edgecolor="#444",
                  labelcolor=TEXT_C, fontsize=7, loc="upper left", framealpha=0.85)
    step_text = fig.text(0.01, 0.01, "t=0", color=TEXT_C, fontsize=8)

    def update(t):
        artists = []
        for i in range(n):
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
    print(f"    {os.path.basename(out_path)}")


# ── Raw embedding GIFs ────────────────────────────────────────────────────────

def make_raw_gif(name, init_grid, model, device, steps, fps,
                 out_path, vmin, vmax, label):
    grids, embs = [], []
    g = init_grid.copy()
    for _ in range(steps):
        grids.append(g.copy())
        embs.append(encode_frame(model, g, device))
        g = gol_step(g)

    side = int(np.sqrt(len(embs[0])))
    fig, (ax_gol, ax_emb) = plt.subplots(1, 2, figsize=(10, 5))
    fig.patch.set_facecolor(BG)
    ax_gol.set_facecolor(BG); ax_gol.axis("off")
    ax_emb.set_facecolor(BG); ax_emb.axis("off")

    im_gol = ax_gol.imshow(grids[0], cmap="binary", vmin=0, vmax=1,
                            interpolation="nearest")
    title_gol = ax_gol.set_title(f"{name}  t=0  [{label}]",
                                   color=TEXT_C, fontsize=9, fontweight="bold")
    im_emb = ax_emb.imshow(embs[0].reshape(side, side), cmap="RdBu_r",
                            vmin=vmin, vmax=vmax, interpolation="nearest",
                            aspect="equal")
    ax_emb.set_title(f"Raw Embedding (64-dim → 8×8)  [{label}]",
                      color=TEXT_C, fontsize=9, fontweight="bold")
    cbar = fig.colorbar(im_emb, ax=ax_emb, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=TEXT_C, labelsize=7)
    for i in range(side):
        for j in range(side):
            ax_emb.text(j, i, str(i * side + j), ha="center", va="center",
                        color="white", fontsize=5, alpha=0.5)
    fig.tight_layout(pad=1.0)

    def update(t):
        im_gol.set_data(grids[t])
        title_gol.set_text(f"{name}  t={t}  [{label}]")
        im_emb.set_data(embs[t].reshape(side, side))
        return im_gol, im_emb, title_gol

    ani = animation.FuncAnimation(fig, update, frames=steps,
                                  interval=1000 // fps, blit=True)
    ani.save(out_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"    {os.path.basename(out_path)}")


def make_raw_combined_gif(seed_names, seed_grids, model, device, steps, fps,
                           out_path, vmin, vmax, label):
    n = len(seed_names)
    side = 8
    all_grids, all_embs = [], []
    for init in seed_grids:
        grids, embs = [], []
        g = init.copy()
        for _ in range(steps):
            grids.append(g.copy())
            embs.append(encode_frame(model, g, device))
            g = gol_step(g)
        all_grids.append(grids)
        all_embs.append(embs)

    fig, axes = plt.subplots(2, n, figsize=(n * 2.6, 5.5))
    fig.patch.set_facecolor(BG)

    ims_gol, ims_emb, titles = [], [], []
    for col, name in enumerate(seed_names):
        axes[0, col].set_facecolor(BG); axes[0, col].axis("off")
        axes[1, col].set_facecolor(BG); axes[1, col].axis("off")
        im_g = axes[0, col].imshow(all_grids[col][0], cmap="binary",
                                    vmin=0, vmax=1, interpolation="nearest")
        t = axes[0, col].set_title(f"{name}\nt=0", color=TEXT_C,
                                    fontsize=7, fontweight="bold", pad=2)
        im_e = axes[1, col].imshow(all_embs[col][0].reshape(side, side),
                                    cmap="RdBu_r", vmin=vmin, vmax=vmax,
                                    interpolation="nearest", aspect="equal")
        ims_gol.append(im_g); ims_emb.append(im_e); titles.append(t)

    fig.colorbar(ims_emb[0], ax=axes[1, :].tolist(),
                 fraction=0.015, pad=0.02).ax.tick_params(colors=TEXT_C, labelsize=7)
    fig.suptitle(f"GoL  |  Raw 64-dim Embedding (8×8)  [{label}]",
                 color=TEXT_C, fontsize=10, fontweight="bold", y=1.01)
    fig.tight_layout(pad=0.5)

    def update(t):
        artists = []
        for col in range(n):
            ims_gol[col].set_data(all_grids[col][t])
            ims_emb[col].set_data(all_embs[col][t].reshape(side, side))
            titles[col].set_text(f"{seed_names[col]}\nt={t}")
            artists += [ims_gol[col], ims_emb[col], titles[col]]
        return artists

    ani = animation.FuncAnimation(fig, update, frames=steps,
                                  interval=1000 // fps, blit=True)
    ani.save(out_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"    {os.path.basename(out_path)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",   required=True, help="path to encoder checkpoint")
    ap.add_argument("--label",  required=True, help="label for this checkpoint (e.g. barely_trained)")
    ap.add_argument("--steps",  type=int, default=80)
    ap.add_argument("--fps",    type=int, default=10)
    ap.add_argument("--data",   default=os.path.join(HERE, "data", "trajectories.npz"))
    ap.add_argument("--config", default=os.path.join(HERE, "results", "train_config.json"))
    ap.add_argument("--out-base", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    out_dir = os.path.join(args.out_base, args.label)
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  checkpoint: {args.ckpt}  |  label: {args.label}")

    with open(args.config) as f:
        cfg = json.load(f)
    model = TrajectoryEncoder(cfg["latent_dim"], cfg["proj_dim"]).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    seeds  = make_seeds()
    names  = list(seeds.keys())
    grids  = [seeds[n] for n in names]
    colors = SEED_COLORS[:len(names)]

    # ── PCA background ──
    print("Building PCA background …")
    bg_emb = encode_dataset_sample(model, args.data, device)
    pca = PCA(n_components=2, random_state=0)
    bg_2d = pca.fit_transform(bg_emb)

    # ── Global raw embedding range ──
    print("Computing embedding value range …")
    all_vals = []
    for init in grids:
        g = init.copy()
        for _ in range(args.steps):
            all_vals.append(encode_frame(model, g, device))
            g = gol_step(g)
    vabs = np.abs(np.stack(all_vals)).max()
    vmin, vmax = -vabs, vabs

    # ── PCA GIFs ──
    print(f"\n[{args.label}] PCA embedding GIFs …")
    for name, grid, color in zip(names, grids, colors):
        out = os.path.join(out_dir, f"anim_{name.replace(' ', '_')}.gif")
        make_pca_gif(name, grid, model, pca, bg_2d, device,
                     args.steps, args.fps, out, color, args.label)
    make_pca_combined_gif(names, grids, colors, model, pca, bg_2d, device,
                           args.steps, args.fps,
                           os.path.join(out_dir, "anim_combined.gif"), args.label)

    # ── Raw embedding GIFs ──
    print(f"\n[{args.label}] Raw embedding GIFs …")
    for name, grid in zip(names, grids):
        out = os.path.join(out_dir, f"raw_emb_{name.replace(' ', '_')}.gif")
        make_raw_gif(name, grid, model, device, args.steps, args.fps,
                     out, vmin, vmax, args.label)
    make_raw_combined_gif(names, grids, model, device, args.steps, args.fps,
                           os.path.join(out_dir, "raw_emb_combined.gif"),
                           vmin, vmax, args.label)

    print(f"\nAll GIFs saved to {out_dir}")


if __name__ == "__main__":
    main()
