"""
animate_raw_embedding.py — GIF showing GoL grid + raw embedding vector heatmap.

Left panel  — GoL grid evolving over time
Right panel — 64-dim embedding vector shown as an 8x8 heatmap, updating each step

One GIF per seed, plus a combined GIF with all seeds side by side.

Usage:
    python animate_raw_embedding.py [--steps 80] [--fps 10]
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


# ── Seeds (same as animate_embeddings.py) ────────────────────────────────────

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


# ── Encode one frame ──────────────────────────────────────────────────────────

@torch.no_grad()
def encode_frame(model, grid, device):
    x = torch.from_numpy(grid).float().unsqueeze(0).unsqueeze(0).to(device)
    return model.encode(x).squeeze(0).detach().cpu().numpy()  # (D,)


# ── Single-seed GIF ───────────────────────────────────────────────────────────

def make_single_gif(name, init_grid, model, device, steps, fps, out_path,
                    vmin, vmax):
    grids, embs = [], []
    g = init_grid.copy()
    for _ in range(steps):
        grids.append(g.copy())
        embs.append(encode_frame(model, g, device))
        g = gol_step(g)

    D = embs[0].shape[0]
    side = int(np.sqrt(D))   # 8 for D=64

    fig, (ax_gol, ax_emb) = plt.subplots(1, 2, figsize=(10, 5))
    fig.patch.set_facecolor(BG)
    ax_gol.set_facecolor(BG); ax_gol.axis("off")
    ax_emb.set_facecolor(BG); ax_emb.axis("off")

    im_gol = ax_gol.imshow(grids[0], cmap="binary", vmin=0, vmax=1,
                            interpolation="nearest")
    title_gol = ax_gol.set_title(f"{name}  t=0", color=TEXT_C,
                                   fontsize=11, fontweight="bold")

    emb_grid = embs[0].reshape(side, side)
    im_emb = ax_emb.imshow(emb_grid, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                            interpolation="nearest", aspect="equal")
    ax_emb.set_title("Raw Embedding (64-dim → 8×8)", color=TEXT_C,
                      fontsize=11, fontweight="bold")

    # colorbar
    cbar = fig.colorbar(im_emb, ax=ax_emb, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=TEXT_C, labelsize=7)

    # dimension index annotations
    for i in range(side):
        for j in range(side):
            ax_emb.text(j, i, str(i * side + j), ha="center", va="center",
                        color="white", fontsize=5, alpha=0.5)

    fig.tight_layout(pad=1.0)

    def update(t):
        im_gol.set_data(grids[t])
        title_gol.set_text(f"{name}  t={t}")
        im_emb.set_data(embs[t].reshape(side, side))
        return im_gol, im_emb, title_gol

    ani = animation.FuncAnimation(fig, update, frames=steps,
                                  interval=1000 // fps, blit=True)
    ani.save(out_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Combined GIF (all seeds, shared colormap) ─────────────────────────────────

def make_combined_gif(seed_names, seed_grids, model, device, steps, fps,
                      out_path, vmin, vmax):
    n = len(seed_names)
    D = 64
    side = int(np.sqrt(D))

    # pre-compute
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

    # layout: n columns, 2 rows (GoL top, embedding bottom)
    fig, axes = plt.subplots(2, n, figsize=(n * 2.6, 5.5))
    fig.patch.set_facecolor(BG)

    ims_gol, ims_emb, titles = [], [], []
    for col, name in enumerate(seed_names):
        ax_g = axes[0, col]
        ax_e = axes[1, col]
        ax_g.set_facecolor(BG); ax_g.axis("off")
        ax_e.set_facecolor(BG); ax_e.axis("off")

        im_g = ax_g.imshow(all_grids[col][0], cmap="binary", vmin=0, vmax=1,
                            interpolation="nearest")
        t = ax_g.set_title(f"{name}\nt=0", color=TEXT_C, fontsize=7,
                             fontweight="bold", pad=2)
        im_e = ax_e.imshow(all_embs[col][0].reshape(side, side),
                            cmap="RdBu_r", vmin=vmin, vmax=vmax,
                            interpolation="nearest", aspect="equal")
        ims_gol.append(im_g); ims_emb.append(im_e); titles.append(t)

    # shared colorbar
    fig.colorbar(ims_emb[0], ax=axes[1, :].tolist(),
                 fraction=0.015, pad=0.02).ax.tick_params(colors=TEXT_C, labelsize=7)

    fig.suptitle("GoL Evolution  |  Raw 64-dim Embedding (8×8 heatmap)",
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
    print(f"  Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

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

    seeds = make_seeds()
    names  = list(seeds.keys())
    grids  = [seeds[n] for n in names]

    # compute global vmin/vmax across all seeds and steps for consistent colormap
    print("Pre-computing embedding range …")
    all_vals = []
    for init in grids:
        g = init.copy()
        for _ in range(args.steps):
            all_vals.append(encode_frame(model, g, device))
            g = gol_step(g)
    all_vals = np.stack(all_vals)
    vmin, vmax = all_vals.min(), all_vals.max()
    # center on zero for RdBu
    vabs = max(abs(vmin), abs(vmax))
    vmin, vmax = -vabs, vabs
    print(f"  Embedding range: [{vmin:.3f}, {vmax:.3f}]")

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\nGenerating {len(names)} individual GIFs …")
    for name, grid in zip(names, grids):
        out = os.path.join(args.out_dir,
                           f"raw_emb_{name.replace(' ', '_')}.gif")
        make_single_gif(name, grid, model, device,
                        args.steps, args.fps, out, vmin, vmax)

    print("\nGenerating combined GIF …")
    make_combined_gif(names, grids, model, device, args.steps, args.fps,
                      os.path.join(args.out_dir, "raw_emb_combined.gif"),
                      vmin, vmax)

    print("\nDone.")


if __name__ == "__main__":
    main()
