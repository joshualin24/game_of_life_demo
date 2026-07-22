"""
Compare V4 (density=0.35 trained) vs V5 best (density-diverse trained)
on identical initial conditions to reveal the density-bias fix.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from nn.models   import CNNTransformerV4
from nn.data_gen import run_trajectory
from nn.utils    import CKPT_DIR, RESULTS_DIR, DEVICE

GRID_SIZE = 40

def load_model(ckpt_name):
    m = CNNTransformerV4(grid_size=GRID_SIZE, patch_size=4,
                         d_model=64, nhead=4, num_layers=4).to(DEVICE)
    path = os.path.join(CKPT_DIR, ckpt_name)
    m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    m.eval()
    return m

def rollout(model, init, steps=20):
    frames = [init.copy()]
    curr = torch.tensor(init, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        for _ in range(steps):
            curr = model.step(curr)
            frames.append(curr.squeeze().cpu().numpy().astype(np.uint8))
    return frames

def place_pattern(grid, pat, r0, c0):
    H, W = grid.shape
    ph, pw = pat.shape
    rows = (r0 + np.arange(ph)) % H
    cols = (c0 + np.arange(pw)) % W
    grid[np.ix_(rows, cols)] |= pat

def compare_plot(scenarios, v4, v5, steps=20, out_name="compare_v4_v5.png"):
    """
    Each scenario: (label, init_array).
    Rows: True GoL / V4 / V5.
    Columns: time steps.
    """
    show_at = np.linspace(0, steps, min(8, steps + 1), dtype=int)
    n_col   = len(show_at)
    n_scen  = len(scenarios)

    fig = plt.figure(figsize=(n_col * 1.8, n_scen * 3 * 1.8))
    outer = gridspec.GridSpec(n_scen, 1, figure=fig, hspace=0.35)

    for si, (label, init) in enumerate(scenarios):
        true_frames = run_trajectory(init, steps)
        v4_frames   = rollout(v4,  init, steps)
        v5_frames   = rollout(v5,  init, steps)

        inner = gridspec.GridSpecFromSubplotSpec(
            3, n_col, subplot_spec=outer[si], hspace=0.05, wspace=0.05)

        row_labels = ["True GoL", "V4 (density=0.35)", "V5 (density-diverse)"]
        for ri, frames in enumerate([true_frames, v4_frames, v5_frames]):
            for ci, t in enumerate(show_at):
                ax = fig.add_subplot(inner[ri, ci])
                ax.imshow(frames[t], cmap="inferno", vmin=0, vmax=1,
                          interpolation="nearest")
                ax.axis("off")
                if ci == 0:
                    ax.set_ylabel(row_labels[ri], fontsize=7, rotation=90,
                                  labelpad=4)
                if ri == 0:
                    ax.set_title(f"t={t}", fontsize=7)

        fig.text(0.5,
                 outer[si].get_position(fig).y1 + 0.005,
                 label, ha="center", va="bottom",
                 fontsize=9, fontweight="bold")

    fig.suptitle("V4 vs V5: density-bias comparison\n"
                 "V4: trained on density=0.35 only | "
                 "V5: trained on 8 densities + 13 named patterns + combos",
                 fontsize=9, y=1.01)
    path = os.path.join(RESULTS_DIR, out_name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")
    return path


def main():
    v4 = load_model("task12_cnn_transformer_v4_best.pt")
    v5 = load_model("task13_cnn_transformer_v5_best.pt")
    print("Models loaded.")

    rng = np.random.default_rng(7)

    # ── Named patterns ─────────────────────────────────────────────────────
    glider_pat = np.array([[0,1,0],[0,0,1],[1,1,1]], dtype=np.uint8)
    blinker_pat = np.array([[1,1,1]], dtype=np.uint8)
    pulsar_pat = np.array([
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
        [0,0,1,1,1,0,0,0,1,1,1,0,0]], dtype=np.uint8)

    def mk(pat, r, c, label):
        g = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        place_pattern(g, pat, r, c)
        return label, g

    def rand_grid(density, seed_offset):
        return (rng.random((GRID_SIZE, GRID_SIZE)) < density).astype(np.uint8)

    # Two gliders at different orientations
    two_g = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    place_pattern(two_g, glider_pat,              5,  5)
    place_pattern(two_g, np.rot90(glider_pat, 2), 25, 25)

    # ── Scenario groups ────────────────────────────────────────────────────
    sparse_scenarios = [
        mk(glider_pat,  8, 8,  "Single glider (5 cells, ~0.3% density)"),
        ("Two gliders — different orientations", two_g),
        mk(blinker_pat, 20, 18, "Blinker (3 cells, ~0.2% density)"),
        mk(pulsar_pat,  13, 13, "Pulsar (48 cells, ~3% density)"),
    ]

    density_scenarios = [
        ("Sparse random  (density=0.05)", rand_grid(0.05, 0)),
        ("Medium random  (density=0.35)", rand_grid(0.35, 1)),
        ("Dense random   (density=0.70)", rand_grid(0.70, 2)),
    ]

    paths = []
    paths.append(compare_plot(sparse_scenarios,  v4, v5, steps=20,
                               out_name="compare_v4_v5_patterns.png"))
    paths.append(compare_plot(density_scenarios, v4, v5, steps=20,
                               out_name="compare_v4_v5_densities.png"))
    print("Done.")
    return paths


if __name__ == "__main__":
    main()
