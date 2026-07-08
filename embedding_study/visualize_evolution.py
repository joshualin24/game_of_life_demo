"""
Visualize Game of Life evolution alongside its VAE embedding trajectory.

For each of several seed patterns (glider, pulsar, soup), evolves the GoL
grid for N steps, encodes each frame through the trained VAE encoder, projects
all embeddings to 2D via PCA (fitted on the full dataset embeddings), and
saves a three-panel GIF:
  Left   — GoL grid at current step
  Center — VAE reconstruction of the current grid
  Right  — PCA embedding space (dataset as background scatter, trajectory
            as a colored path with the current point highlighted)

Usage:
    python visualize_evolution.py [--steps 60] [--out results/evolution.gif]
                                   [--models models/] [--results results/]
                                   [--fps 8]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
from sklearn.decomposition import PCA

from generate_dataset import gol_step
from vae import VAE

HERE = os.path.dirname(os.path.abspath(__file__))


# ── Seed patterns ─────────────────────────────────────────────────────────────

def make_glider_grid(size=64):
    g = np.zeros((size, size), np.uint8)
    # glider near top-left, room to travel
    g[2, 3] = g[3, 4] = g[4, 2] = g[4, 3] = g[4, 4] = 1
    return g


def make_pulsar_grid(size=64):
    g = np.zeros((size, size), np.uint8)
    pulsar = np.array([
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
    ], dtype=np.uint8)
    r, c = (size - 13) // 2, (size - 13) // 2
    g[r:r+13, c:c+13] = pulsar
    return g


def make_soup_grid(size=64, seed=42):
    rng = np.random.default_rng(seed)
    g = (rng.random((size, size)) < 0.35).astype(np.uint8)
    for _ in range(10):
        g = gol_step(g)
    return g


def make_r_pentomino(size=64):
    """R-pentomino: chaotic evolution for ~1100 generations."""
    g = np.zeros((size, size), np.uint8)
    r, c = size // 2, size // 2
    g[r,   c+1] = g[r,   c+2] = 1
    g[r+1, c  ] = g[r+1, c+1] = 1
    g[r+2, c+1] = 1
    return g


SEEDS = {
    "glider":      make_glider_grid,
    "pulsar":      make_pulsar_grid,
    "r_pentomino": make_r_pentomino,
    "soup":        make_soup_grid,
}


# ── Encoding helpers ──────────────────────────────────────────────────────────

@torch.no_grad()
def encode_grid(model, grid, device):
    x = torch.from_numpy(grid).float().unsqueeze(0).unsqueeze(0).to(device)
    mu, _ = model.encoder(x)
    return mu.squeeze(0).cpu().numpy()


@torch.no_grad()
def reconstruct_grid(model, grid, device):
    x = torch.from_numpy(grid).float().unsqueeze(0).unsqueeze(0).to(device)
    logits, _, _ = model(x)
    return torch.sigmoid(logits).squeeze().cpu().numpy()


# ── GIF builder ───────────────────────────────────────────────────────────────

def build_gif(seed_name, init_grid, model, pca, bg_2d, device,
              steps, fps, out_path):
    """Evolve GoL for `steps` steps and save a 3-panel animated GIF."""

    # Pre-compute all frames
    grids, embeddings, recons = [], [], []
    grid = init_grid.copy()
    for _ in range(steps):
        grids.append(grid.copy())
        embeddings.append(encode_grid(model, grid, device))
        recons.append(reconstruct_grid(model, grid, device))
        grid = gol_step(grid)

    traj_2d = pca.transform(np.stack(embeddings))  # (steps, 2)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor("#111")
    for ax in axes:
        ax.set_facecolor("#111")

    ax_gol, ax_recon, ax_emb = axes

    # Background scatter (dataset embeddings, gray)
    ax_emb.scatter(bg_2d[:, 0], bg_2d[:, 1], s=0.3, c="#444", alpha=0.3,
                   rasterized=True)
    ax_emb.set_title("Embedding space (PCA)", color="white", fontsize=9)
    ax_emb.tick_params(colors="white")
    for spine in ax_emb.spines.values():
        spine.set_edgecolor("#555")

    # Color trajectory by time
    colors = plt.cm.plasma(np.linspace(0, 1, steps))

    im_gol = ax_gol.imshow(grids[0], cmap="binary", vmin=0, vmax=1,
                            interpolation="nearest")
    ax_gol.set_title(f"GoL: {seed_name}  step 0", color="white", fontsize=9)
    ax_gol.axis("off")

    im_recon = ax_recon.imshow(recons[0], cmap="binary", vmin=0, vmax=1,
                                interpolation="nearest")
    ax_recon.set_title("VAE reconstruction", color="white", fontsize=9)
    ax_recon.axis("off")

    traj_line, = ax_emb.plot([], [], lw=1.0, color="cyan", alpha=0.6)
    dot = ax_emb.scatter([], [], s=60, color="red", zorder=5)

    fig.tight_layout(pad=0.5)

    def update(frame):
        im_gol.set_data(grids[frame])
        ax_gol.set_title(f"GoL: {seed_name}  step {frame}", color="white", fontsize=9)
        im_recon.set_data(recons[frame])
        traj_line.set_data(traj_2d[:frame+1, 0], traj_2d[:frame+1, 1])
        traj_line.set_color(colors[frame])
        dot.set_offsets(traj_2d[frame])
        return im_gol, im_recon, traj_line, dot

    ani = animation.FuncAnimation(fig, update, frames=steps,
                                  interval=1000 // fps, blit=True)
    ani.save(out_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--models", default=os.path.join(HERE, "models"))
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--seeds", nargs="+", default=list(SEEDS.keys()),
                    help="which seeds to animate: glider pulsar r_pentomino soup")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    import json
    cfg_path = os.path.join(args.models, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    latent_dim = cfg["latent_dim"]

    model = VAE(latent_dim).to(device)
    model.load_state_dict(torch.load(os.path.join(args.models, "vae.pt"),
                                     map_location=device))
    model.eval()
    print(f"Loaded VAE (latent_dim={latent_dim}) from {args.models}")

    # Load dataset embeddings and fit PCA
    emb_path = os.path.join(args.results, "embeddings.npz")
    d = np.load(emb_path)
    all_mu = d["mu"]                      # (N, latent_dim)
    print(f"Fitting PCA on {len(all_mu):,} embeddings …")
    pca = PCA(n_components=2, random_state=0)
    bg_2d = pca.fit_transform(all_mu)
    print(f"  Explained variance: {pca.explained_variance_ratio_}")

    os.makedirs(args.results, exist_ok=True)

    for name in args.seeds:
        if name not in SEEDS:
            print(f"Unknown seed '{name}', skipping.")
            continue
        print(f"\nAnimating '{name}' for {args.steps} steps …")
        init = SEEDS[name]()
        out = os.path.join(args.results, f"evolution_{name}.gif")
        build_gif(name, init, model, pca, bg_2d, device,
                  args.steps, args.fps, out)

    print("\nDone.")


if __name__ == "__main__":
    main()
