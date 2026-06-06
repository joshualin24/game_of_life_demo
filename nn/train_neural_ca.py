"""
Task 4: Neural Cellular Automaton
-----------------------------------
Train a tiny fully-convolutional net to reproduce the GoL update rule.
After training we can:
  - Verify it matches GoL exactly on unseen grids
  - Unroll it for many steps and compare to true GoL
  - Visualise the learned conv weights

Input : current state grid  (1, H, W) float
Output: next state probs     (1, H, W) float  (binarised at threshold 0.5)
Model : NeuralCA  (Conv3x3 → ReLU → Conv1x1 → ReLU → Conv1x1 → Sigmoid)
Loss  : Binary cross-entropy
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from nn.data_gen  import generate_trajectory_dataset, load_dataset, run_trajectory
from nn.models    import NeuralCA
from nn.utils     import (Trainer, make_loaders, set_seed,
                          plot_loss_curves, plot_neural_ca_rollout,
                          DEVICE, RESULTS_DIR)

# ── Config ─────────────────────────────────────────────────────────────────────
TASK        = "task4_neural_ca"
N_INITS     = 300
TRAJ_STEPS  = 100
GRID_SIZE   = 32
EPOCHS      = 40
BATCH_SIZE  = 128
LR          = 1e-3
SEED        = 42


def plot_weight_vis(model: NeuralCA, task_name: str):
    """Visualise the first conv layer weights (3×3 filters)."""
    w = model.net[0].weight.data.cpu().numpy()  # (hidden, 1, 3, 3)
    n = min(32, w.shape[0])
    ncols = 8; nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.4, nrows * 1.4))
    axes = axes.ravel()
    vmax = np.abs(w[:n]).max()
    for i in range(n):
        axes[i].imshow(w[i, 0], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       interpolation="nearest")
        axes[i].axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"{task_name} — first conv layer weights (3×3)", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{task_name}_weights.png")
    fig.savefig(path, dpi=150)
    print(f"  Weight vis → {path}")
    plt.close(fig)


def evaluate_accuracy(model: NeuralCA, loader, device) -> float:
    """Cell-level accuracy: fraction of cells correctly predicted."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x_b, y_b in loader:
            pred = (model(x_b.to(device)) > 0.5).float()
            correct += (pred == y_b.to(device)).sum().item()
            total   += y_b.numel()
    return correct / total


def main():
    set_seed(SEED)

    # ── Data ──────────────────────────────────────────────────────────────────
    data_path = os.path.join(os.path.dirname(__file__), "data", "trajectories.npz")
    if os.path.exists(data_path):
        print("[task4] Loading cached trajectory dataset …")
        data = load_dataset("trajectories")
    else:
        data = generate_trajectory_dataset(
            n_inits=N_INITS, grid_size=GRID_SIZE, traj_steps=TRAJ_STEPS)

    states = data["states"].astype(np.float32)  # (N, H, W)
    nexts  = data["nexts"].astype(np.float32)

    X = torch.tensor(states[:, None])  # (N,1,H,W)
    Y = torch.tensor(nexts[:,  None])  # (N,1,H,W)
    print(f"  Pairs: {len(X):,}  grid={GRID_SIZE}×{GRID_SIZE}")

    train_dl, val_dl = make_loaders(X, Y, batch_size=BATCH_SIZE, seed=SEED)

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = NeuralCA(hidden=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS)
    loss_fn   = nn.BCELoss()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: NeuralCA  params={n_params:,}  device={DEVICE}")

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = Trainer(model, optimizer, loss_fn, TASK, scheduler)
    history = trainer.fit(train_dl, val_dl, epochs=EPOCHS, print_every=5)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    trainer.load_best()
    acc = evaluate_accuracy(model, val_dl, DEVICE)
    print(f"\n  Cell-level accuracy on val set: {acc*100:.3f}%")

    plot_loss_curves(history, TASK)
    plot_weight_vis(model, TASK)

    # rollout test on a fresh random grid and a known pattern (glider)
    rng = np.random.default_rng(999)
    rand_init = (rng.random((GRID_SIZE, GRID_SIZE)) < 0.35).astype(np.uint8)
    plot_neural_ca_rollout(model, rand_init, steps=60,
                           task_name=TASK + "_random", n_show=8)

    glider = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    r0, c0 = GRID_SIZE//2, GRID_SIZE//2
    for dr, dc in [(0,1),(1,2),(2,0),(2,1),(2,2)]:
        glider[r0+dr, c0+dc] = 1
    plot_neural_ca_rollout(model, glider, steps=60,
                           task_name=TASK + "_glider", n_show=8)

    print(f"  Done.  Results in nn/results/  checkpoints in nn/checkpoints/")


if __name__ == "__main__":
    main()
