"""
Extra V7 examples + failure-case finder.

Part A: rollouts for patterns not shown before (LWSS, acorn, r-pentomino,
        toad, multi-glider collision, dense combo).
Part B: scan many grids for step-1 prediction errors; visualise the best
        (most interesting) failures as a 3-row figure:
        True GoL | V7 prediction | error map (FP=red, FN=blue).
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from nn.models   import CNNTransformerV4
from nn.data_gen import run_trajectory
from nn.utils    import CKPT_DIR, RESULTS_DIR, DEVICE

GRID_SIZE = 40
STEPS     = 30

# ── Load V7 best checkpoint ────────────────────────────────────────────────────

def load_v7():
    m = CNNTransformerV4(grid_size=GRID_SIZE, patch_size=4,
                         d_model=64, nhead=4, num_layers=4).to(DEVICE)
    path = os.path.join(CKPT_DIR, "task15_cnn_transformer_v7_best.pt")
    m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    m.eval()
    return m

# ── Helpers ────────────────────────────────────────────────────────────────────

def _p(*rows): return np.array(rows, dtype=np.uint8)

PATTERNS = {
    "lwss":       _p([0,1,0,0,1],[1,0,0,0,0],[1,0,0,0,1],[1,1,1,1,0]),
    "acorn":      _p([0,1,0,0,0,0,0],[0,0,0,1,0,0,0],[1,1,0,0,1,1,1]),
    "r_pent":     _p([0,1,1],[1,1,0],[0,1,0]),
    "toad":       _p([0,1,1,1],[1,1,1,0]),
    "beacon":     _p([1,1,0,0],[1,1,0,0],[0,0,1,1],[0,0,1,1]),
    "glider":     _p([0,1,0],[0,0,1],[1,1,1]),
}

def place(grid, pat, r0, c0):
    H, W = grid.shape
    ph, pw = pat.shape
    rows = (r0 + np.arange(ph)) % H
    cols = (c0 + np.arange(pw)) % W
    grid[np.ix_(rows, cols)] |= pat

def rollout(model, init, steps=STEPS):
    frames = [init.copy()]
    curr = torch.tensor(init, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        for _ in range(steps):
            curr = model.step(curr)
            frames.append(curr.squeeze().cpu().numpy().astype(np.uint8))
    return frames

def save_rollout(model, init, steps, label):
    true_traj = run_trajectory(init, steps)
    pred_traj = rollout(model, init, steps)
    show_at   = np.linspace(0, steps, min(10, steps + 1), dtype=int)

    fig, axes = plt.subplots(2, len(show_at), figsize=(len(show_at) * 2, 5))
    for col, t in enumerate(show_at):
        for row, (frames, title) in enumerate(
                [(true_traj, "True GoL"), (pred_traj, "V7")]):
            axes[row, col].imshow(frames[t], cmap="inferno",
                                  vmin=0, vmax=1, interpolation="nearest")
            axes[row, col].set_title(f"t={t}", fontsize=8)
            axes[row, col].axis("off")
        axes[0, col]
    axes[0, 0].set_ylabel("True GoL", fontsize=9)
    axes[1, 0].set_ylabel("V7",       fontsize=9)
    fig.suptitle(f"V7 — {label}", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, f"v7_extra_rollout_{label}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")
    return path

# ── Part A: extra rollouts ─────────────────────────────────────────────────────

def part_a(model, rng):
    paths = []
    print("\n[Part A] Extra rollouts …")

    # Individual named patterns not shown before
    for name in ["lwss", "acorn", "r_pent", "toad", "beacon"]:
        init = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        place(init, PATTERNS[name], 15, 15)
        paths.append(save_rollout(model, init, STEPS, name))

    # Four gliders on collision course
    g = PATTERNS["glider"]
    init = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    place(init, g,                  2,  2)
    place(init, np.rot90(g, 1),     2, 30)
    place(init, np.rot90(g, 2),    30, 30)
    place(init, np.rot90(g, 3),    30,  2)
    paths.append(save_rollout(model, init, STEPS, "four_gliders"))

    # Dense random (density=0.55, different seed)
    init = (rng.random((GRID_SIZE, GRID_SIZE)) < 0.55).astype(np.uint8)
    paths.append(save_rollout(model, init, STEPS, "random_dense_alt"))

    # Very sparse (density=0.02)
    init = (rng.random((GRID_SIZE, GRID_SIZE)) < 0.02).astype(np.uint8)
    paths.append(save_rollout(model, init, STEPS, "random_very_sparse"))

    return paths


# ── Part B: failure finder ─────────────────────────────────────────────────────

def step1_errors(model, init):
    """
    Returns (fp_cells, fn_cells) — pixel coords where V7 is wrong on step 1.
    fp: predicted alive, actually dead  (false positive)
    fn: predicted dead, actually alive  (false negative)
    """
    true_next = run_trajectory(init, 1)[1]          # ground truth step 1
    x = torch.tensor(init, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
    pred = (logits > 0).squeeze().cpu().numpy().astype(np.uint8)
    fp = np.argwhere((pred == 1) & (true_next == 0))
    fn = np.argwhere((pred == 0) & (true_next == 1))
    return fp, fn, pred, true_next


def save_failure(init, pred, true_next, fp, fn, label, step=1):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

    axes[0].imshow(true_next, cmap="inferno", vmin=0, vmax=1,
                   interpolation="nearest")
    axes[0].set_title(f"True GoL (t={step})", fontsize=10)

    axes[1].imshow(pred, cmap="inferno", vmin=0, vmax=1,
                   interpolation="nearest")
    axes[1].set_title(f"V7 prediction (t={step})", fontsize=10)

    # Error map: TP=white, FP=red, FN=blue, TN=black
    err = np.zeros((*init.shape, 3))
    err[(pred == 1) & (true_next == 1)] = [1.0, 1.0, 1.0]   # TP white
    err[(pred == 1) & (true_next == 0)] = [1.0, 0.2, 0.2]   # FP red
    err[(pred == 0) & (true_next == 1)] = [0.2, 0.4, 1.0]   # FN blue
    axes[2].imshow(err, interpolation="nearest")
    axes[2].set_title(
        f"Error map  FP={len(fp)} (red)  FN={len(fn)} (blue)", fontsize=10)

    for ax in axes:
        ax.axis("off")

    patches = [
        mpatches.Patch(color='white',          label='True positive'),
        mpatches.Patch(color=(1.0,0.2,0.2),    label=f'False positive ({len(fp)})'),
        mpatches.Patch(color=(0.2,0.4,1.0),    label=f'False negative ({len(fn)})'),
        mpatches.Patch(color='black',          label='True negative'),
    ]
    axes[2].legend(handles=patches, loc='lower right',
                   fontsize=7, framealpha=0.8)

    # Draw initial state inset on axes[0]
    inset = axes[0].inset_axes([0.0, -0.28, 1.0, 0.25])
    inset.imshow(init, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
    inset.set_title("Initial state (t=0)", fontsize=8)
    inset.axis("off")

    fig.suptitle(f"V7 failure case — {label}", fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, f"v7_failure_{label}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def part_b(model, rng):
    print("\n[Part B] Searching for failure cases …")
    candidates = []

    # Random grids across all densities
    for density in [0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80]:
        for _ in range(200):
            init = (rng.random((GRID_SIZE, GRID_SIZE)) < density).astype(np.uint8)
            fp, fn, pred, true_next = step1_errors(model, init)
            n_err = len(fp) + len(fn)
            if n_err > 0:
                candidates.append((n_err, "fp" if len(fp) >= len(fn) else "fn",
                                   density, init.copy(), pred, true_next, fp, fn))

    # Named and combo patterns
    for name, pat in PATTERNS.items():
        for trial in range(80):
            init = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
            r0 = int(rng.integers(0, GRID_SIZE))
            c0 = int(rng.integers(0, GRID_SIZE))
            place(init, pat, r0, c0)
            fp, fn, pred, true_next = step1_errors(model, init)
            n_err = len(fp) + len(fn)
            if n_err > 0:
                candidates.append((n_err, name, None, init.copy(),
                                   pred, true_next, fp, fn))

    print(f"  Found {len(candidates)} grids with ≥1 error out of "
          f"{200*8 + 80*len(PATTERNS)} tested")

    if not candidates:
        print("  V7 made zero errors on all tested grids!")
        return []

    # Sort by error count descending and pick 3 most interesting / diverse
    candidates.sort(key=lambda c: -c[0])

    # Pick: most errors, a FP case, a FN case (deduplicate by error type)
    chosen, seen_types = [], set()
    for cand in candidates:
        etype = "fp" if len(cand[6]) > 0 else "fn"
        if etype not in seen_types or len(chosen) < 3:
            chosen.append(cand)
            seen_types.add(etype)
        if len(chosen) == 3:
            break
    # If fewer than 3 unique types, just take top 3 by error count
    if len(chosen) < 3:
        chosen = candidates[:3]

    paths = []
    for i, (n_err, tag, density, init, pred, true_next, fp, fn) in enumerate(chosen):
        d_str = f"d{density:.2f}_" if density is not None else f"{tag}_"
        label = f"{i+1}_{d_str}fp{len(fp)}_fn{len(fn)}"
        paths.append(save_failure(init, pred, true_next, fp, fn, label))
    return paths


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    model = load_v7()
    print("V7 best checkpoint loaded.")
    rng = np.random.default_rng(99)

    paths_a = part_a(model, rng)
    paths_b = part_b(model, rng)

    print(f"\nDone.  {len(paths_a)} extra rollouts + {len(paths_b)} failure cases.")

if __name__ == "__main__":
    main()
