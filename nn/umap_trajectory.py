"""
UMAP embedding of GoL trajectories with trajectory-path tracking.

For each of the 10 named patterns (and random grids) we:
  1. Run T=60 GoL steps.
  2. Embed the PARTIAL trajectory [frame_0 … frame_t] at t = 5,10,…,60
     → each pattern traces a *path* through embedding space as the
       transformer sees more and more of its evolution.
  3. Collect all embeddings, run UMAP to 2-D.
  4. Plot:

  umap_01_background.png        – 500 full-traj embeddings coloured by category
  umap_02_paths.png             – paths of named patterns through embedding space
  umap_03_paths_overlay.png     – background + paths overlaid
  umap_04_perturbation_shift.png– how top perturbations shift embedding from baseline
  umap_05_path_animation.gif    – animated path growth step-by-step

Requires: umap-learn, matplotlib, numpy, torch
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import torch
import umap

from nn.models         import TrajectoryTransformer
from nn.trajectory_data import (TrajectoryDataset, PATTERN_CELLS,
                                 PATTERN_CATEGORY, PATTERN_NAMES,
                                 run_trajectory, _augment)
from nn.data_gen       import compute_sensitivity_map
from nn.utils          import CKPT_DIR, RESULTS_DIR, DEVICE

# ── Config ─────────────────────────────────────────────────────────────────────

GRID_SIZE   = 40
T           = 60
D_MODEL     = 128
CKPT        = os.path.join(CKPT_DIR, "traj_emb_d128_best.pt")
SEED        = 0
T_SNAPSHOTS = list(range(5, T + 1, 5))   # [5, 10, 15, …, 60]
N_AUG       = 8    # augmented copies per pattern for background cloud
N_RANDOM    = 200  # random grids for background

CATEGORY_COLORS = {
    "still_life":  "#4C72B0",
    "oscillator":  "#55A868",
    "spaceship":   "#C44E52",
    "methuselah":  "#8172B2",
    "random":      "#999999",
    "mixed":       "#CCB974",
}
PATTERN_MARKERS = {
    "block": "o", "beehive": "o",
    "blinker": "s", "toad": "s", "beacon": "s", "pulsar": "s",
    "glider": "^", "lwss": "^",
    "r_pentomino": "D", "acorn": "D",
}

rng = np.random.default_rng(SEED)


# ── Load model ─────────────────────────────────────────────────────────────────

def load_model():
    m = TrajectoryTransformer(d_model=D_MODEL, nhead=4, num_layers=4,
                               grid_size=GRID_SIZE, T=T, dropout=0.0).to(DEVICE)
    m.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    m.eval()
    return m


@torch.no_grad()
def embed_traj(model, traj: np.ndarray) -> np.ndarray:
    """traj: (T+1, H, W) uint8  →  (d_model,) float32 embedding."""
    t = torch.tensor(traj, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    emb, _ = model(t)
    return emb.squeeze(0).cpu().numpy()


@torch.no_grad()
def embed_partial(model, full_traj: np.ndarray, t_end: int) -> np.ndarray:
    """Embed frames [0 … t_end] of a trajectory (variable length)."""
    sub = full_traj[:t_end + 1]          # (t_end+1, H, W)
    t   = torch.tensor(sub, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    B, Tp1, H, W = t.shape
    frames = t.reshape(B * Tp1, 1, H, W)
    embs   = model.frame_encoder(frames).reshape(B, Tp1, D_MODEL)
    cls    = model.cls_token.expand(B, -1, -1)
    tokens = torch.cat([cls, embs], dim=1)
    tokens = model.pos_enc(tokens)
    out    = model.transformer(tokens)
    return out[:, 0].squeeze(0).cpu().numpy()


# ── Build background embedding cloud ──────────────────────────────────────────

def build_background(model):
    """
    Returns embeddings (N, d_model), categories (N,), pattern_names (N,).
    Includes N_AUG augmentations of each pattern + N_RANDOM random grids.
    """
    embs, cats, names = [], [], []

    for pname in PATTERN_NAMES:
        cells = PATTERN_CELLS[pname]
        cat   = PATTERN_CATEGORY[pname]
        for _ in range(N_AUG):
            g = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
            aug = _augment(cells.copy(), rng)
            h, w = aug.shape
            r0 = int(rng.integers(0, GRID_SIZE - h + 1))
            c0 = int(rng.integers(0, GRID_SIZE - w + 1))
            g[r0:r0+h, c0:c0+w] = aug
            traj = run_trajectory(g, T)
            embs.append(embed_traj(model, traj))
            cats.append(cat)
            names.append(pname)

    for _ in range(N_RANDOM):
        density = float(rng.uniform(0.2, 0.5))
        g = (rng.random((GRID_SIZE, GRID_SIZE)) < density).astype(np.uint8)
        traj = run_trajectory(g, T)
        embs.append(embed_traj(model, traj))
        cats.append("random")
        names.append("random")

    return np.array(embs), cats, names


# ── Build per-pattern paths through embedding space ────────────────────────────

def build_paths(model):
    """
    For each of the 10 named patterns, compute one canonical initial grid
    (centered, no augmentation) and embed its partial trajectories at each
    time snapshot in T_SNAPSHOTS.

    Returns: dict  pattern_name → (len(T_SNAPSHOTS), d_model) array
    """
    paths = {}
    for pname in PATTERN_NAMES:
        cells = PATTERN_CELLS[pname]
        g = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        h, w = cells.shape
        r0 = (GRID_SIZE - h) // 2
        c0 = (GRID_SIZE - w) // 2
        g[r0:r0+h, c0:c0+w] = cells
        full_traj = run_trajectory(g, T)

        path_embs = []
        for t in T_SNAPSHOTS:
            path_embs.append(embed_partial(model, full_traj, t))
        paths[pname] = np.array(path_embs)   # (len(T_SNAPSHOTS), d_model)
        print(f"  path built: {pname}")
    return paths


# ── Perturbation shift in embedding space ─────────────────────────────────────

def build_perturbation_embeddings(model, n_patterns=4):
    """
    For a subset of patterns, compute the embedding shift caused by the
    highest-impact single-cell perturbation.
    Returns: dict  pattern_name → {"base": emb, "perturbed": emb, "low": emb}
    """
    demo_patterns = ["glider", "pulsar", "r_pentomino", "acorn"][:n_patterns]
    result = {}

    SENS_STEPS = 30  # cheaper sensitivity sweep for demo

    for pname in demo_patterns:
        cells = PATTERN_CELLS[pname]
        g = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        h, w = cells.shape
        r0 = (GRID_SIZE - h) // 2
        c0 = (GRID_SIZE - w) // 2
        g[r0:r0+h, c0:c0+w] = cells

        # base trajectory embedding
        base_traj = run_trajectory(g, T)
        base_emb  = embed_traj(model, base_traj)

        # sensitivity sweep (vectorised)
        sens_map = compute_sensitivity_map(g, SENS_STEPS)   # (H, W)
        top_r, top_c = np.unravel_index(sens_map.argmax(), (GRID_SIZE, GRID_SIZE))
        low_r, low_c = np.unravel_index(
            np.where(sens_map == 0, -1, sens_map).argmin(), (GRID_SIZE, GRID_SIZE))
        # fallback if all cells have impact
        if sens_map[low_r, low_c] == 0:
            low_idx = np.argsort(sens_map.ravel())[0]
            low_r, low_c = np.unravel_index(low_idx, (GRID_SIZE, GRID_SIZE))

        g_top = g.copy(); g_top[top_r, top_c] ^= 1
        g_low = g.copy(); g_low[low_r, low_c] ^= 1

        top_emb = embed_traj(model, run_trajectory(g_top, T))
        low_emb = embed_traj(model, run_trajectory(g_low, T))

        result[pname] = {
            "base":      base_emb,
            "top_pert":  top_emb,
            "low_pert":  low_emb,
            "top_cell":  (top_r, top_c),
            "low_cell":  (low_r, low_c),
            "top_sens":  float(sens_map[top_r, top_c]),
            "low_sens":  float(sens_map[low_r, low_c]),
        }
        print(f"  perturbation done: {pname}  "
              f"top={sens_map[top_r,top_c]:.0f}  low={sens_map[low_r,low_c]:.0f}")

    return result


# ── UMAP fit ──────────────────────────────────────────────────────────────────

def fit_umap(all_embs: np.ndarray, seed: int = SEED):
    reducer = umap.UMAP(n_components=2, random_state=seed,
                        n_neighbors=20, min_dist=0.1, metric="cosine")
    return reducer.fit(all_embs)


def project(reducer, embs: np.ndarray) -> np.ndarray:
    return reducer.transform(embs)


# ── Plotting ───────────────────────────────────────────────────────────────────

def savefig(fig, name):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_background(umap_bg, cats, names, title="", ax=None, standalone=True):
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 8))
    unique_cats = sorted(set(cats))
    for cat in unique_cats:
        idx = [i for i, c in enumerate(cats) if c == cat]
        col = CATEGORY_COLORS.get(cat, "#aaaaaa")
        ax.scatter(umap_bg[idx, 0], umap_bg[idx, 1],
                   c=col, label=cat, alpha=0.45, s=18, linewidths=0)
    ax.legend(fontsize=9, markerscale=1.5, loc="upper right")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title(title or "Trajectory embeddings (UMAP)", fontweight="bold")
    if standalone:
        return fig


def plot_paths(umap_paths_dict, ax=None, standalone=True, show_legend=True):
    """Draw paths as arrows; start = star, end = circle."""
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("tab10")
    for i, (pname, path_2d) in enumerate(umap_paths_dict.items()):
        col  = cmap(i / 10)
        cat  = PATTERN_CATEGORY[pname]
        # draw arrows along path
        for j in range(len(path_2d) - 1):
            dx = path_2d[j+1, 0] - path_2d[j, 0]
            dy = path_2d[j+1, 1] - path_2d[j, 1]
            ax.annotate("", xy=(path_2d[j+1,0], path_2d[j+1,1]),
                        xytext=(path_2d[j,0], path_2d[j,1]),
                        arrowprops=dict(arrowstyle="-|>", color=col,
                                        lw=1.4, mutation_scale=10))
        ax.scatter(*path_2d[0],  marker="*", s=220, c=[col], zorder=6)   # start
        ax.scatter(*path_2d[-1], marker="o", s=80,  c=[col], zorder=6, label=f"{pname} [{cat}]")
    if show_legend:
        ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title("Named-pattern paths through embedding space (★=t=5, ●=t=60)",
                 fontweight="bold")
    if standalone:
        return fig


def plot_perturbation_shift(umap_pert_dict, ax=None, standalone=True):
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.get_cmap("Set1")
    for i, (pname, d) in enumerate(umap_pert_dict.items()):
        col = cmap(i / 4)
        base = d["umap_base"]
        top  = d["umap_top"]
        low  = d["umap_low"]

        # base → top (high-impact perturbation)
        ax.annotate("", xy=top, xytext=base,
                    arrowprops=dict(arrowstyle="-|>", color=col,
                                    lw=2, mutation_scale=14,
                                    linestyle="solid"))
        # base → low (low-impact perturbation)
        ax.annotate("", xy=low, xytext=base,
                    arrowprops=dict(arrowstyle="-|>", color=col,
                                    lw=1.2, mutation_scale=10,
                                    linestyle="dashed"))

        ax.scatter(*base, marker="*", s=300, c=[col], zorder=6,
                   label=f"{pname} (base)")
        ax.scatter(*top,  marker="X", s=100, c=[col], zorder=6)
        ax.scatter(*low,  marker=".",  s=100, c=[col], zorder=6)

    ax.legend(fontsize=8)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title("Perturbation shift in embedding space\n"
                 "★=baseline  ✕=high-impact flip  •=low-impact flip",
                 fontweight="bold")
    if standalone:
        return fig


# ── Animation ─────────────────────────────────────────────────────────────────

def make_path_animation(umap_bg, cats, umap_paths_dict, t_snapshots):
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")

    # static background (greyed out)
    unique_cats = sorted(set(cats))
    for cat in unique_cats:
        idx = [i for i, c in enumerate(cats) if c == cat]
        col = CATEGORY_COLORS.get(cat, "#888888")
        ax.scatter(umap_bg[idx, 0], umap_bg[idx, 1],
                   c=col, alpha=0.18, s=14, linewidths=0)

    cmap = plt.get_cmap("tab10")
    pnames = list(umap_paths_dict.keys())
    n = len(pnames)

    # pre-collect artists for animation
    lines  = []
    dots   = []
    stars  = []
    for i, pname in enumerate(pnames):
        col = cmap(i / n)
        path = umap_paths_dict[pname]
        ln,  = ax.plot([], [], "-", color=col, lw=1.5, alpha=0.8)
        dot, = ax.plot([], [], "o", color=col, markersize=7, zorder=6)
        star = ax.scatter(*path[0], marker="*", s=200, c=[col], zorder=7)
        lines.append(ln); dots.append(dot); stars.append(star)

    title = ax.set_title("", color="white", fontsize=12, fontweight="bold")

    # legend
    handles = [plt.Line2D([0],[0], color=cmap(i/n), lw=2,
                           label=f"{pname} [{PATTERN_CATEGORY[pname]}]")
               for i, pname in enumerate(pnames)]
    ax.legend(handles=handles, fontsize=7, loc="upper right",
              facecolor="#222222", labelcolor="white", ncol=2)

    ax.set_xlabel("UMAP-1", color="white")
    ax.set_ylabel("UMAP-2", color="white")

    def _update(frame):
        t_idx = frame  # frame = index into T_SNAPSHOTS
        for i, pname in enumerate(pnames):
            path = umap_paths_dict[pname]
            visible = path[:t_idx + 1]
            lines[i].set_data(visible[:, 0], visible[:, 1])
            dots[i].set_data([path[t_idx, 0]], [path[t_idx, 1]])
        title.set_text(f"Trajectory paths in embedding space — t = {t_snapshots[t_idx]}")
        return lines + dots + [title]

    ani = animation.FuncAnimation(fig, _update, frames=len(t_snapshots),
                                   interval=300, blit=True)
    path = os.path.join(RESULTS_DIR, "umap_05_path_animation.gif")
    ani.save(path, writer=animation.PillowWriter(fps=4))
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading model …")
    model = load_model()

    # 1. Background embeddings
    print("\n[1] Building background embeddings …")
    bg_embs, bg_cats, bg_names = build_background(model)
    print(f"  {len(bg_embs)} background embeddings")

    # 2. Paths for each named pattern
    print("\n[2] Building per-pattern paths …")
    paths = build_paths(model)   # {pname: (n_snapshots, d_model)}

    # 3. Perturbation shifts
    print("\n[3] Computing perturbation embeddings …")
    pert = build_perturbation_embeddings(model, n_patterns=4)

    # 4. Collect all embeddings for UMAP fit
    all_embs = [bg_embs]
    for path_arr in paths.values():
        all_embs.append(path_arr)
    for d in pert.values():
        all_embs.append(np.array([d["base"], d["top_pert"], d["low_pert"]]))
    all_embs = np.vstack(all_embs)

    print(f"\n[4] Fitting UMAP on {len(all_embs)} embeddings …")
    reducer = fit_umap(all_embs)

    # Project everything
    umap_bg = project(reducer, bg_embs)

    umap_paths = {}
    for pname, path_arr in paths.items():
        umap_paths[pname] = project(reducer, path_arr)

    for pname, d in pert.items():
        stacked = np.array([d["base"], d["top_pert"], d["low_pert"]])
        proj    = project(reducer, stacked)
        d["umap_base"] = proj[0]
        d["umap_top"]  = proj[1]
        d["umap_low"]  = proj[2]

    # 5. Fig 01 — background only
    print("\n[5] Plotting …")
    fig = plot_background(umap_bg, bg_cats, bg_names,
                          title="GoL trajectory embeddings — UMAP (500+ samples)")
    savefig(fig, "umap_01_background.png")

    # 6. Fig 02 — paths only
    fig = plot_paths(umap_paths)
    savefig(fig, "umap_02_paths.png")

    # 7. Fig 03 — paths overlaid on background
    fig, ax = plt.subplots(figsize=(12, 9))
    plot_background(umap_bg, bg_cats, bg_names, ax=ax, standalone=False)
    plot_paths(umap_paths, ax=ax, standalone=False, show_legend=False)
    # custom legend combining both
    bg_handles = [plt.Line2D([0],[0], marker="o", color="w",
                              markerfacecolor=CATEGORY_COLORS.get(c,"#aaa"),
                              markersize=8, label=c)
                  for c in sorted(set(bg_cats))]
    cmap10 = plt.get_cmap("tab10")
    path_handles = [plt.Line2D([0],[0], color=cmap10(i/10), lw=2,
                                label=f"{p} [{PATTERN_CATEGORY[p]}]")
                    for i, p in enumerate(PATTERN_NAMES)]
    ax.legend(handles=bg_handles + path_handles, fontsize=7,
              loc="upper right", ncol=2)
    ax.set_title("Trajectory embedding space — background + named-pattern paths",
                 fontweight="bold", fontsize=13)
    savefig(fig, "umap_03_paths_overlay.png")

    # 8. Fig 04 — perturbation shifts
    fig = plot_perturbation_shift(pert)
    savefig(fig, "umap_04_perturbation_shift.png")

    # 9. Fig 05 — animation
    make_path_animation(umap_bg, bg_cats, umap_paths, T_SNAPSHOTS)

    print("\nAll UMAP figures saved to nn/results/")


if __name__ == "__main__":
    main()
