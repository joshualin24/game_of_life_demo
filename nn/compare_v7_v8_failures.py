"""
Reproduce the 3 V7 failure cases, then run V7 and V8 on the same grids
and display them in a single comparison figure per case.

Layout (one figure per failure case):
  Row 0 — initial state (inset) + True GoL at t=1
  Row 1 — V7 prediction   |  V7 error map  (FP=red, FN=blue)
  Row 2 — V8 prediction   |  V8 error map  (FP=red, FN=blue)
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

from nn.models   import CNNTransformerV4
from nn.data_gen import run_trajectory
from nn.utils    import CKPT_DIR, RESULTS_DIR, DEVICE

GRID_SIZE = 40

# ── Load models ────────────────────────────────────────────────────────────────

def load_model(ckpt_name):
    m = CNNTransformerV4(grid_size=GRID_SIZE, patch_size=4,
                         d_model=64, nhead=4, num_layers=4).to(DEVICE)
    m.load_state_dict(torch.load(os.path.join(CKPT_DIR, ckpt_name),
                                 map_location=DEVICE, weights_only=True))
    m.eval()
    return m

# ── Helpers ────────────────────────────────────────────────────────────────────

def _p(*rows): return np.array(rows, dtype=np.uint8)

PATTERNS = {
    "lwss":    _p([0,1,0,0,1],[1,0,0,0,0],[1,0,0,0,1],[1,1,1,1,0]),
    "acorn":   _p([0,1,0,0,0,0,0],[0,0,0,1,0,0,0],[1,1,0,0,1,1,1]),
    "r_pent":  _p([0,1,1],[1,1,0],[0,1,0]),
    "toad":    _p([0,1,1,1],[1,1,1,0]),
    "beacon":  _p([1,1,0,0],[1,1,0,0],[0,0,1,1],[0,0,1,1]),
    "glider":  _p([0,1,0],[0,0,1],[1,1,1]),
}

def place(grid, pat, r0, c0):
    H, W = grid.shape
    ph, pw = pat.shape
    rows = (r0 + np.arange(ph)) % H
    cols = (c0 + np.arange(pw)) % W
    grid[np.ix_(rows, cols)] |= pat

def predict(model, init):
    x = torch.tensor(init, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
    return (logits > 0).squeeze().cpu().numpy().astype(np.uint8)

def errors(pred, true_next):
    fp = np.argwhere((pred == 1) & (true_next == 0))
    fn = np.argwhere((pred == 0) & (true_next == 1))
    return fp, fn

def error_rgb(pred, true_next):
    img = np.zeros((*pred.shape, 3))
    img[(pred == 1) & (true_next == 1)] = [1.0, 1.0, 1.0]   # TP white
    img[(pred == 1) & (true_next == 0)] = [1.0, 0.2, 0.2]   # FP red
    img[(pred == 0) & (true_next == 1)] = [0.2, 0.4, 1.0]   # FN blue
    return img

# ── Replicate V7 failure search (same rng seed = 99) ──────────────────────────

def find_v7_failures(v7):
    rng = np.random.default_rng(99)
    candidates = []

    for density in [0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80]:
        for _ in range(200):
            init = (rng.random((GRID_SIZE, GRID_SIZE)) < density).astype(np.uint8)
            true_next = run_trajectory(init, 1)[1]
            pred = predict(v7, init)
            fp, fn = errors(pred, true_next)
            n_err = len(fp) + len(fn)
            if n_err > 0:
                candidates.append((n_err, "fp" if len(fp) >= len(fn) else "fn",
                                   density, init.copy(), true_next))

    for name, pat in PATTERNS.items():
        for _ in range(80):
            init = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
            r0 = int(rng.integers(0, GRID_SIZE))
            c0 = int(rng.integers(0, GRID_SIZE))
            place(init, pat, r0, c0)
            true_next = run_trajectory(init, 1)[1]
            pred = predict(v7, init)
            fp, fn = errors(pred, true_next)
            n_err = len(fp) + len(fn)
            if n_err > 0:
                candidates.append((n_err, name, None, init.copy(), true_next))

    candidates.sort(key=lambda c: -c[0])

    chosen, seen_types = [], set()
    for cand in candidates:
        etype = "fp" if cand[1] in ("fp",) else "fn"
        if etype not in seen_types or len(chosen) < 3:
            chosen.append(cand)
            seen_types.add(etype)
        if len(chosen) == 3:
            break
    if len(chosen) < 3:
        chosen = candidates[:3]

    return chosen

# ── Comparison figure ──────────────────────────────────────────────────────────

def save_comparison(init, true_next, v7, v8, label):
    pred_v7 = predict(v7, init)
    pred_v8 = predict(v8, init)
    fp7, fn7 = errors(pred_v7, true_next)
    fp8, fn8 = errors(pred_v8, true_next)
    err7 = error_rgb(pred_v7, true_next)
    err8 = error_rgb(pred_v8, true_next)

    legend_patches = [
        mpatches.Patch(color='white',       label='True positive'),
        mpatches.Patch(color=(1,.2,.2),     label='False positive'),
        mpatches.Patch(color=(.2,.4,1.),    label='False negative'),
        mpatches.Patch(color='black',       label='True negative'),
    ]

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.08)

    # ── Row 0: initial state | true next | (empty) ────────────────────────────
    ax_init = fig.add_subplot(gs[0, 0])
    ax_init.imshow(init, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
    ax_init.set_title("Initial state (t=0)", fontsize=9)
    ax_init.axis("off")

    ax_true = fig.add_subplot(gs[0, 1])
    ax_true.imshow(true_next, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
    ax_true.set_title("True GoL (t=1)", fontsize=9)
    ax_true.axis("off")

    ax_leg = fig.add_subplot(gs[0, 2])
    ax_leg.axis("off")
    ax_leg.legend(handles=legend_patches, loc="center", fontsize=9,
                  framealpha=0.9, title="Error map key", title_fontsize=9)

    # ── Row 1: V7 prediction | V7 error map ───────────────────────────────────
    ax_v7p = fig.add_subplot(gs[1, 0])
    ax_v7p.imshow(pred_v7, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
    ax_v7p.set_title("V7 prediction", fontsize=9)
    ax_v7p.set_ylabel("V7", fontsize=10, fontweight="bold", labelpad=6)
    ax_v7p.axis("off")

    ax_v7e = fig.add_subplot(gs[1, 1:])
    ax_v7e.imshow(err7, interpolation="nearest")
    ax_v7e.set_title(
        f"V7 error map   FP={len(fp7)}  FN={len(fn7)}  total={len(fp7)+len(fn7)}",
        fontsize=9)
    ax_v7e.axis("off")

    # ── Row 2: V8 prediction | V8 error map ───────────────────────────────────
    ax_v8p = fig.add_subplot(gs[2, 0])
    ax_v8p.imshow(pred_v8, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
    ax_v8p.set_title("V8 prediction", fontsize=9)
    ax_v8p.set_ylabel("V8", fontsize=10, fontweight="bold", labelpad=6)
    ax_v8p.axis("off")

    ax_v8e = fig.add_subplot(gs[2, 1:])
    ax_v8e.imshow(err8, interpolation="nearest")
    ax_v8e.set_title(
        f"V8 error map   FP={len(fp8)}  FN={len(fn8)}  total={len(fp8)+len(fn8)}",
        fontsize=9)
    ax_v8e.axis("off")

    fig.suptitle(
        f"V7 vs V8 — failure case {label}\n"
        f"V7: {len(fp7)} FP + {len(fn7)} FN = {len(fp7)+len(fn7)} errors    "
        f"V8: {len(fp8)} FP + {len(fn8)} FN = {len(fp8)+len(fn8)} errors",
        fontsize=10, fontweight="bold")

    path = os.path.join(RESULTS_DIR, f"compare_v7_v8_failure_{label}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return path

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    v7 = load_model("task15_cnn_transformer_v7_best.pt")
    v8 = load_model("task16_cnn_transformer_v8_best.pt")
    print("Models loaded.")

    print("Finding V7 failure cases …")
    failures = find_v7_failures(v7)
    print(f"  Found {len(failures)} cases.")

    paths = []
    for i, (n_err, tag, density, init, true_next) in enumerate(failures):
        d_str = f"d{density:.2f}" if density is not None else str(tag)
        label = f"{i+1}_{d_str}_v7err{n_err}"
        print(f"  Case {i+1}: tag={tag}  density={density}  V7 errors={n_err}")
        paths.append(save_comparison(init, true_next, v7, v8, label))

    print(f"\nDone. {len(paths)} comparison figures saved.")

if __name__ == "__main__":
    main()
