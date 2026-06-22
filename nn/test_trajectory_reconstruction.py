"""
Trajectory Reconstruction Test
--------------------------------
Evaluates the two trained TrajectoryTransformer models on a held-out test set
(never seen during training).

Test dataset composition:
  60% random grids  (different seed from training)
  40% named-pattern grids (single / dual / triple)

Metrics:
  - Per-trajectory cell accuracy  = fraction of (frame × cell) pairs
                                    correctly reconstructed after thresholding
  - Overall mean accuracy across all test samples

Outputs:
  nn/results/test_recon_d<N>_summary.txt
  nn/results/test_recon_d<N>_accuracy_hist.png
  nn/results/test_recon_d<N>_success_<i>.gif   (top-5 by accuracy)
  nn/results/test_recon_d<N>_failure_<i>.gif   (bottom-5 by accuracy)
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from nn.trajectory_data import TrajectoryDataset, run_trajectory
from nn.models           import TrajectoryTransformer
from nn.utils            import CKPT_DIR, RESULTS_DIR, DEVICE

# ── Config ─────────────────────────────────────────────────────────────────────

GRID_SIZE   = 40
T           = 60
N_TEST      = 500
TEST_SEED   = 999          # different from training seed (42)
RANDOM_FRAC = 0.60         # 60% random, 40% pattern
BATCH_SIZE  = 32
D_MODELS    = [64, 128]
N_GIF       = 5            # GIFs per category (success / failure)
FPS         = 8


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_all(model: TrajectoryTransformer,
                 ds: TrajectoryDataset) -> np.ndarray:
    """
    Returns (N,) array of per-trajectory cell accuracies in [0, 1].
    Accuracy = fraction of (frame × cell) pairs correctly predicted.
    """
    model.eval()
    accs = []
    with torch.no_grad():
        for i in range(0, len(ds), BATCH_SIZE):
            batch = torch.stack([ds[j] for j in range(i, min(i + BATCH_SIZE, len(ds)))])
            x = batch.to(DEVICE)                      # (B, T+1, H, W)
            _, logits = model(x)                      # (B, T+1, H, W)
            pred = (logits > 0).float()               # threshold at 0
            correct = (pred == x).float()
            acc = correct.mean(dim=(1, 2, 3))         # per-trajectory mean
            accs.extend(acc.cpu().numpy().tolist())
            if (i // BATCH_SIZE + 1) % 5 == 0:
                print(f"    {i + len(batch)}/{len(ds)}", flush=True)
    return np.array(accs)


# ── GIF generation ─────────────────────────────────────────────────────────────

def make_gif(out_path: str, original: np.ndarray, reconstructed: np.ndarray,
             label: str, accuracy: float):
    """
    3-panel animated GIF: original | reconstructed | difference.
    original / reconstructed: (T+1, H, W) uint8 arrays.
    Difference: red = missed alive cell, blue = false positive dead cell.
    """
    T_frames = original.shape[0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor("#111111")
    for ax in axes:
        ax.set_facecolor("black")
        ax.axis("off")
    axes[0].set_title("Original",       color="white", fontsize=11)
    axes[1].set_title("Reconstructed",  color="white", fontsize=11)
    axes[2].set_title("Difference",     color="white", fontsize=11)

    im0  = axes[0].imshow(original[0],      cmap="inferno", vmin=0, vmax=1,
                           interpolation="nearest")
    im1  = axes[1].imshow(reconstructed[0], cmap="inferno", vmin=0, vmax=1,
                           interpolation="nearest")
    diff0 = reconstructed[0].astype(int) - original[0].astype(int)
    im2  = axes[2].imshow(diff0, cmap="RdBu_r", vmin=-1, vmax=1,
                           interpolation="nearest")

    n_wrong0 = int(np.abs(diff0).sum())
    n_total  = original[0].size
    suptitle = fig.suptitle(
        f"{label}  |  overall acc={accuracy*100:.1f}%\n"
        f"t=0   frame errors={n_wrong0}/{n_total}",
        color="white", fontsize=10, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    def _update(t):
        im0.set_data(original[t])
        im1.set_data(reconstructed[t])
        diff = reconstructed[t].astype(int) - original[t].astype(int)
        im2.set_data(diff)
        n_wrong = int(np.abs(diff).sum())
        frame_acc = 1.0 - n_wrong / original[t].size
        suptitle.set_text(
            f"{label}  |  overall acc={accuracy*100:.1f}%\n"
            f"t={t}   frame errors={n_wrong}/{original[t].size}  "
            f"frame acc={frame_acc*100:.1f}%"
        )
        return im0, im1, im2, suptitle

    ani = animation.FuncAnimation(fig, _update, frames=T_frames,
                                  interval=1000 // FPS, blit=False)
    ani.save(out_path, writer=animation.PillowWriter(fps=FPS))
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Per-model analysis ─────────────────────────────────────────────────────────

def run_for_model(d_model: int, ds: TrajectoryDataset):
    ckpt = os.path.join(CKPT_DIR, f"traj_emb_d{d_model}_best.pt")
    if not os.path.exists(ckpt):
        print(f"[skip] Checkpoint not found: {ckpt}")
        return

    print(f"\n{'='*60}")
    print(f"  d_model={d_model}")
    print(f"{'='*60}")

    model = TrajectoryTransformer(
        d_model=d_model, nhead=4, num_layers=4,
        grid_size=GRID_SIZE, T=T,
    ).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))

    print("  Evaluating …")
    accs = evaluate_all(model, ds)                    # (N,) float in [0,1]

    mean_acc  = float(accs.mean())
    pct_95    = float((accs >= 0.95).mean()) * 100
    pct_99    = float((accs >= 0.99).mean()) * 100

    print(f"\n  Results (n={len(accs)}):")
    print(f"    Mean accuracy      : {mean_acc*100:.2f}%")
    print(f"    Trajectories ≥95%  : {pct_95:.1f}%")
    print(f"    Trajectories ≥99%  : {pct_99:.1f}%")
    print(f"    Min accuracy       : {accs.min()*100:.2f}%")
    print(f"    Max accuracy       : {accs.max()*100:.2f}%")

    # save summary
    tag  = f"traj_emb_d{d_model}"
    summary_path = os.path.join(RESULTS_DIR, f"test_recon_d{d_model}_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"TrajectoryTransformer d_model={d_model}  test n={len(accs)}\n")
        f.write(f"Mean accuracy      : {mean_acc*100:.2f}%\n")
        f.write(f"Trajectories ≥95%  : {pct_95:.1f}%\n")
        f.write(f"Trajectories ≥99%  : {pct_99:.1f}%\n")
        f.write(f"Min accuracy       : {accs.min()*100:.2f}%\n")
        f.write(f"Max accuracy       : {accs.max()*100:.2f}%\n")

    # accuracy histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(accs * 100, bins=50, color="#3498db", edgecolor="white", linewidth=0.4)
    ax.axvline(mean_acc * 100, color="#e74c3c", ls="--",
               label=f"mean={mean_acc*100:.1f}%")
    ax.set_xlabel("Per-trajectory cell accuracy (%)")
    ax.set_ylabel("Count")
    ax.set_title(f"d_model={d_model} — reconstruction accuracy distribution",
                 fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    hist_path = os.path.join(RESULTS_DIR, f"test_recon_d{d_model}_accuracy_hist.png")
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    print(f"  Histogram → {hist_path}")

    # ── Generate GIFs ──────────────────────────────────────────────────────────
    print(f"\n  Generating GIFs …")
    order      = np.argsort(accs)
    top_ids    = order[-N_GIF:][::-1]   # highest accuracy first
    bottom_ids = order[:N_GIF]           # lowest accuracy first

    model.eval()
    for rank, (ids, kind) in enumerate([
        (top_ids,    "success"),
        (bottom_ids, "failure"),
    ]):
        for i, idx in enumerate(ids):
            traj_np  = ds[idx].numpy()                       # (T+1, H, W) float
            original = traj_np.astype(np.uint8)

            x = torch.from_numpy(traj_np).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                _, logits = model(x)
            recon = (logits[0] > 0).cpu().numpy().astype(np.uint8)

            meta     = ds.meta[idx]
            pat_info = "/".join(meta["patterns"]) if meta["patterns"] else "random"
            label    = f"d{d_model} #{idx} [{meta['type']}:{pat_info}]"

            fname = os.path.join(RESULTS_DIR,
                                 f"test_recon_d{d_model}_{kind}_{i+1}.gif")
            make_gif(fname, original, recon, label, accs[idx])


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"[data] Building test dataset  n={N_TEST}  "
          f"random_frac={RANDOM_FRAC}  seed={TEST_SEED} …")
    ds = TrajectoryDataset(
        n_samples=N_TEST,
        grid_size=GRID_SIZE,
        T=T,
        random_frac=RANDOM_FRAC,
        seed=TEST_SEED,
        precompute=True,
    )
    print(f"  Types: { {t: sum(1 for m in ds.meta if m['type']==t) for t in ['random','single','dual','triple']} }")

    for d_model in D_MODELS:
        run_for_model(d_model, ds)

    print("\nDone.")


if __name__ == "__main__":
    main()
